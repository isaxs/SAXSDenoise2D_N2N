#!/usr/bin/env python3
"""
SAXS Denoising Inference GUI
=============================
Inference tool for models trained with SAXSAttentionUNet (Noise2Noise).

Features:
  - Load trained .pth checkpoints
  - Support multiple input formats: .h5, .edf, .npy, .tiff/.tif, .png/.jpg/.bmp
  - Real-time preview of noisy vs denoised images (linear/log display modes)
  - Batch-process multiple files
  - Save results as .npy / .tiff / .png and other formats

Usage:
  python inference_gui.py
"""

import os

# Fix OMP Error #15 on Windows:
# deepenv ships its own libiomp5md.dll in both conda mkl and the CPU-only
# torch build; loading both at once crashes the process (OMP: Error #15).
# This must be set before importing torch/numpy; it is PyTorch's official
# workaround.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import time
import traceback

import numpy as np
import torch
import torch.nn as nn

SUPPORTED_IMAGE_EXTS = frozenset({
    '.h5', '.hdf5', '.edf', '.edf.gz', '.npy', '.tiff', '.tif',
    '.png', '.jpg', '.jpeg', '.bmp',
})

# ---------------------------------------------------------------------------
# 1. Model definition (identical to training script SAXSDenoise2D_N2N_08.py)
# ---------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    """Squeeze-and-excitation channel attention."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 4), bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(max(channels // reduction, 4), channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.shape[:2]
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ImprovedResidualBlock(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3,
                               padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv2d(channels, channels, 3,
                               padding=dilation, dilation=dilation)
        self.se = ChannelAttention(channels)

    def forward(self, x):
        residual = x
        out = nn.functional.silu(self.conv1(x))
        out = self.conv2(out)
        out = self.se(out)
        return nn.functional.silu(out + residual)


class ASPPBottleneck(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.branch1 = ImprovedResidualBlock(channels, dilation=1)
        self.branch2 = ImprovedResidualBlock(channels, dilation=2)
        self.branch4 = ImprovedResidualBlock(channels, dilation=4)
        self.branch8 = ImprovedResidualBlock(channels, dilation=8)
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 4, channels * 2, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels * 2, channels, 1)
        )

    def forward(self, x):
        f1 = self.branch1(x)
        f2 = self.branch2(x)
        f4 = self.branch4(x)
        f8 = self.branch8(x)
        fused = torch.cat([f1, f2, f4, f8], dim=1)
        return x + self.fusion(fused)


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int=None):
        super().__init__()
        if F_int is None:
            F_int = max(F_l // 2, 8)
        self.W_g = nn.Conv2d(F_g, F_int, 1, bias=False)
        self.W_x = nn.Conv2d(F_l, F_int, 1, bias=False)
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, gating, skip):
        g1 = self.W_g(gating)
        x1 = self.W_x(skip)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = nn.functional.interpolate(g1, size=x1.shape[2:],
                                           mode='bilinear', align_corners=False)
        alpha = self.psi(nn.functional.silu(g1 + x1))
        return skip * alpha


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)
        self.res = ImprovedResidualBlock(out_ch)

    def forward(self, x):
        return self.res(self.down(x))


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, 3, stride=2,
                                     padding=1, output_padding=1)
        self.attn_gate = AttentionGate(in_ch, skip_ch)
        self.res = ImprovedResidualBlock(in_ch + skip_ch)
        self.out_conv = nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = nn.functional.interpolate(x, size=skip.shape[2:],
                                          mode='bilinear', align_corners=False)
        skip = self.attn_gate(x, skip)
        x = torch.cat([x, skip], dim=1)
        x = self.res(x)
        return self.out_conv(x)


class SAXSAttentionUNet(nn.Module):
    """SAXS-specific Attention U-Net (identical to the training script)."""
    def __init__(self, in_ch=1, out_ch=1, init_ch=32, use_log=True):
        super().__init__()
        C = init_ch
        self.use_log = use_log

        self.enc1 = EncoderBlock(in_ch, C)
        self.enc2 = EncoderBlock(C, C * 2)
        self.enc3 = EncoderBlock(C * 2, C * 4)
        self.enc4 = EncoderBlock(C * 4, C * 8)
        self.bottleneck = ASPPBottleneck(C * 8)
        self.dec4 = DecoderBlock(C * 8, C * 4, C * 4)
        self.dec3 = DecoderBlock(C * 4, C * 2, C * 2)
        self.dec2 = DecoderBlock(C * 2, C, C)
        self.dec1 = DecoderBlock(C, in_ch, C)
        # Matches training script v08: the log-domain correction must be
        # signed, otherwise the model can only add and collapses to the
        # identity output.
        self.out_conv = nn.Conv2d(C, out_ch, 3, padding=1)

    def forward(self, x):
        if self.use_log:
            x_log = torch.log1p(x)
        else:
            x_log = x

        e1 = self.enc1(x_log)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b = self.bottleneck(e4)
        d4 = self.dec4(b, e3)
        d3 = self.dec3(d4, e2)
        d2 = self.dec2(d3, e1)
        d1 = self.dec1(d2, x_log)
        log_correction = self.out_conv(d1)

        if self.use_log:
            out = torch.expm1(x_log + log_correction)
            # expm1 can be negative in low-intensity background; clamp to 0 to
            # keep the output physically non-negative.
            return torch.clamp(out, min=0)
        return nn.functional.softplus(log_correction)


# ---------------------------------------------------------------------------
# 2. Device and precision helpers
# ---------------------------------------------------------------------------

def get_best_device(verbose=True):
    """Automatically select the best device: CUDA > MPS (Apple Silicon) > CPU.

    Returns:
        (device, info_dict)
    """
    info = {}

    if torch.cuda.is_available():
        device = torch.device('cuda')
        props = torch.cuda.get_device_properties(device)
        info['name'] = props.name
        info['memory_gb'] = props.total_memory / 1e9
        info['compute_capability'] = f"{props.major}.{props.minor}"
        info['backend'] = 'CUDA'

        # Check bfloat16 support (cc >= 8.0, e.g. A100, RTX 30xx+).
        info['supports_bf16'] = (props.major >= 8)
        info['supports_fp16'] = (props.major >= 7)  # V100, RTX 20xx+.

    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        info['name'] = 'Apple Silicon (MPS)'
        info['memory_gb'] = None  # MPS does not expose total memory.
        info['backend'] = 'MPS'
        info['supports_bf16'] = False
        info['supports_fp16'] = False   # MPS fp16 is unstable; use fp32 conservatively.

    else:
        device = torch.device('cpu')
        info['name'] = 'CPU'
        info['memory_gb'] = None
        info['backend'] = 'CPU'
        info['supports_bf16'] = False
        info['supports_fp16'] = False

    if verbose:
        print(f"\n{'='*50}")
        print(f"Inference device: {info['name']}  ({info['backend']})")
        if info['memory_gb'] is not None:
            print(f"Memory:   {info['memory_gb']:.1f} GB")
        print(f"AMP fp16: {'YES' if info['supports_fp16'] else 'NO'}")
        print(f"AMP bf16: {'YES' if info['supports_bf16'] else 'NO'}")
        if info['backend'] == 'CPU':
            print(f"  Note: CPU inference uses float32 and may be slow for large images")
            print(f"  Suggestion: use tiled inference (tile_size) to reduce memory usage")
        print(f"{'='*50}\n")

    return device, info


def get_inference_autocast(device, device_info):
    """Return the autocast (dtype, enabled) used for inference."""
    if device.type == 'cuda':
        if device_info.get('supports_bf16', False):
            return torch.bfloat16, True
        elif device_info.get('supports_fp16', False):
            return torch.float16, True
    # CPU / MPS: disable AMP.
    return None, False


def detect_low_memory_gpu(device_info, threshold_gb=4.0):
    """Detect a low-memory GPU or CPU, used to decide whether to enable tiled inference.

    Returns:
        True means tiled inference should be enabled by default (memory < threshold or non-CUDA).
    """
    mem = device_info.get('memory_gb')
    if mem is None:
        return True   # CPU / MPS -> enable tiling by default
    return mem < threshold_gb


# ---------------------------------------------------------------------------
# 3. Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path, device, init_ch=32):
    """
    Load a SAXSAttentionUNet checkpoint.
    Always load weights on CPU first (compatible with checkpoints saved on
    different devices), then move the model to the target device.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    model = SAXSAttentionUNet(in_ch=1, out_ch=1, init_ch=init_ch, use_log=True)

    # Always load on CPU, compatible with CUDA -> CPU / MPS transfers.
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    # Strip the DDP 'module.' prefix.
    model_state = model.state_dict()
    if list(state_dict.keys())[0].startswith('module.') and \
       not list(model_state.keys())[0].startswith('module.'):
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
    elif not list(state_dict.keys())[0].startswith('module.') and \
         list(model_state.keys())[0].startswith('module.'):
        state_dict = {'module.' + k: v for k, v in state_dict.items()}

    # Only load keys with matching shapes (loose match, ignoring
    # training-specific keys).
    filtered_state = {}
    skipped = []
    for k, v in state_dict.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered_state[k] = v
        else:
            skipped.append(k)

    if skipped:
        print(f"  Skipped {len(skipped)} mismatched weight keys (possibly from a different architecture)")

    model.load_state_dict(filtered_state, strict=False)
    model = model.to(device)
    model.eval()

    # Print training info.
    epoch = checkpoint.get('epoch', '?')
    val_loss = checkpoint.get('best_val_loss', '?')
    print(f"  Model loaded | epoch={epoch} | best_val_loss={val_loss}")

    return model


# ---------------------------------------------------------------------------
# 3. Input image reading
# ---------------------------------------------------------------------------

def _pad_to_multiple(image, multiple=16):
    """Pad image dimensions to a multiple of 16 (U-Net's 4 downsampling levels need multiples of 16)."""
    h, w = image.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        # pad_info means (top, bottom, left, right) and must be all zeros when
        # unpadded (an earlier implementation wrongly used (0, 0, h, w), making
        # _unpad return an empty array for images already sized to a multiple of 16).
        return image, (0, 0, 0, 0)
    # Symmetric padding: less before, more after (keep padding near the edges).
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    padded = np.pad(image, ((top, bottom), (left, right)), mode='reflect')
    return padded, (top, bottom, left, right)


def _unpad(image, pad_info):
    """Remove padding and restore the original size."""
    top, bottom, left, right = pad_info
    h_orig = image.shape[0] - top - bottom
    w_orig = image.shape[1] - left - right
    return image[top:top + h_orig, left:left + w_orig]


def _check_frame_index(frame, total_frames, source_name):
    if frame < 0 or frame >= total_frames:
        raise ValueError(
            f"Frame {frame} out of range for {source_name} with {total_frames} frames")
    return frame


def _get_file_ext(filepath):
    """Return the lowercased extension, keeping a trailing '.gz' suffix."""
    base, ext = os.path.splitext(filepath)
    ext = ext.lower()
    if ext == '.gz':
        nested = os.path.splitext(base)[1].lower()
        ext = nested + ext if nested else ext
    return ext


def _parse_header_float(header, key):
    """Parse a numeric header value, tolerating comma thousand separators."""
    if key not in header:
        return None
    val = header[key]
    if isinstance(val, (int, float)):
        return float(val)
    try:
        return float(str(val).replace(',', ''))
    except (ValueError, TypeError):
        return None


def _get_center_from_header(header, shape):
    """Return the beam center from an EDF header, falling back to the frame center."""
    rows, cols = shape
    cx = _parse_header_float(header, 'Center_1')
    cy = _parse_header_float(header, 'Center_2')
    if cx is not None and cy is not None and 0 <= cx < cols and 0 <= cy < rows:
        return cx, cy

    lower_keys = {k.lower(): k for k in header.keys()}
    x_candidates = ['center_x', 'beam_x', 'centerx', 'xcenter', 'beam_center_x']
    y_candidates = ['center_y', 'beam_y', 'centery', 'ycenter', 'beam_center_y']
    for cand in x_candidates:
        if cand in lower_keys:
            cx = _parse_header_float(header, lower_keys[cand])
            if cx is not None:
                break
    for cand in y_candidates:
        if cand in lower_keys:
            cy = _parse_header_float(header, lower_keys[cand])
            if cy is not None:
                break
    if cx is not None and cy is not None and 0 <= cx < cols and 0 <= cy < rows:
        return cx, cy
    return cols / 2.0, rows / 2.0


def _find_beam_center(data, fallback_center):
    """Locate the beam-stop center by full-frame intensity centroid.

    The header Center_1/Center_2 is not always reliable (verified: GID frames
    are ~4 px off), so the beam center is computed from the loaded EDF frame
    instead. The beam-stop disk dominates the frame intensity (~99.9%), so the
    intensity centroid of the whole clipped frame points at the beam. Falls
    back to `fallback_center` (header/geometric) when the frame has no signal.
    """
    total = float(data.sum())
    if total <= 0:
        return fallback_center
    rows, cols = data.shape
    yy, xx = np.ogrid[:rows, :cols]
    cx = float((xx * data).sum()) / total
    cy = float((yy * data).sum()) / total
    return (cx, cy)


def display_center(meta, shape):
    """Beam center in display-image coordinates: (cx, cy).

    EDF: meta['edf_center'] is full-frame (col, row); the raw 1028-row frame
    is cropped 2 rows top/bottom before display, so cy shifts by -2.
    Non-EDF / missing center: geometric center (W/2, H/2).
    """
    h, w = shape
    if meta and 'edf_center' in meta:
        cx, cy = meta['edf_center']
        raw = meta.get('raw_shape')
        if raw is not None and raw[0] == 1028:
            cy -= 2
        return (float(np.clip(cx, 0.0, w - 1.0)),
                float(np.clip(cy, 0.0, h - 1.0)))
    return (w / 2.0, h / 2.0)


def polar_grids(shape, center):
    """Precompute (r, theta) grids for one image/center pair.

    theta in degrees, 0 = +x (right), CCW positive (up = 90), wrapped to
    [0, 360). Image y is downward, hence atan2(-(row-cy), col-cx).
    """
    h, w = shape
    ys = np.arange(h, dtype=np.float64)[:, None]
    xs = np.arange(w, dtype=np.float64)[None, :]
    dx = xs - center[0]
    dy = ys - center[1]
    r = np.hypot(dx, dy)
    theta = np.degrees(np.arctan2(-dy, dx)) % 360.0
    return r, theta


def radial_profile(image, r_grid, theta_grid, angle_deg, delta=0.5, rmax=None):
    """Mean intensity per integer-radius bin along azimuth angle_deg +- delta.

    Bins without any pixel are NaN. radius r maps to round(r) (nearest bin).
    """
    d = np.abs(((theta_grid - angle_deg + 180.0) % 360.0) - 180.0)
    mask = d <= delta
    if rmax is not None:
        mask &= r_grid <= rmax
    r_bin = np.rint(r_grid[mask]).astype(np.int64)
    vals = image[mask]
    n_bins = int(np.floor(rmax)) + 1 if rmax is not None else int(r_bin.max()) + 1
    sums = np.bincount(r_bin, weights=vals, minlength=n_bins)
    counts = np.bincount(r_bin, minlength=n_bins)
    prof = np.full(n_bins, np.nan)
    ok = counts > 0
    prof[ok] = sums[ok] / counts[ok]
    return np.arange(n_bins, dtype=np.float64), prof


def azimuthal_profile(image, r_grid, theta_grid, radius, n_bins=360):
    """Mean intensity per 1-deg angle bin over pixels with radius in [r-0.5, r+0.5].

    Bin i covers angles [i, i+1) degrees. Empty bins are NaN.
    """
    mask = np.abs(r_grid - radius) <= 0.5
    ang_bin = theta_grid[mask].astype(np.int64)          # [0, 359]
    vals = image[mask]
    sums = np.bincount(ang_bin, weights=vals, minlength=n_bins)
    counts = np.bincount(ang_bin, minlength=n_bins)
    prof = np.full(n_bins, np.nan)
    ok = counts > 0
    prof[ok] = sums[ok] / counts[ok]
    return np.arange(n_bins, dtype=np.float64), prof


def _normalize_edf_frame(data, center, radius=30):
    """Clip negatives and scale to the training-H5 intensity range.

    The training H5 datasets were built (by the original builder) as:
      clip -> zero the beam disk (r=25..30) -> divide by a per-group
      constant (~ beam-disk intensity / 1e8, rounded; GID used 500, S used
      180). That differs from the current h5_DataSet_Construct (divide by
      the raw beam-disk sum). To match the trained intensity range
      (0..~0.7, x1000 = 0..700), we reproduce the same scale dynamically:
      scale = max(1, round(beam-disk intensity / 1e8)).

    The returned mask is applied by the caller so the beam stop is excluded
    from denoising (matches the training H5s, which had a zeroed disk).
    """
    data = np.clip(data, 0, None)
    rows, cols = data.shape
    cx, cy = center
    y_grid, x_grid = np.ogrid[:rows, :cols]
    mask = (x_grid - cx) ** 2 + (y_grid - cy) ** 2 <= radius ** 2
    disk_total = float(data[mask].sum())
    if disk_total == 0:
        # Beam-stop region has been zeroed (e.g. re-reading our own denoised
        # output). Fall back to the full-frame sum so the file stays usable
        # and the restore scale stays consistent.
        disk_total = float(data.sum())
        if disk_total == 0:
            raise ValueError(f"EDF image is all zero; cannot normalize frame")
        print(f"  Warning: beam-center region sum is zero; "
              f"scaling by full-frame sum ({disk_total:.3e})")
    scale = max(1.0, float(round(disk_total / 1e8)))
    normalized = data.astype(np.float32) / scale
    return normalized, mask, scale


def restore_edf_output(denoised, meta, title=None):
    """Return (full_frame, header) restored to the raw EDF size for saving.

    The denoised array is kept in the NORMALIZED domain (same as the GUI
    display); the physical-count restore (x beam scale) happens here at save
    time. The clipped raw frame (meta['edf_raw']) is already physical counts
    and is passed through unscaled (noisy save).

    Output policy (user-approved):
      - Inference keeps the training crop (1024 rows); at save time the top and
        bottom 2 rows are filled back from the clipped raw frame, restoring the
        original 1028x512 size.
      - Beam-stop disk pixels are restored to the original input values at
        save time (stored in meta at read time), so the saved output is
        denoised only outside the disk. Real headers put Center_2 ~ 550, so
        the raw edge rows never intersect the disk.
      - Raw-frame dummy pixels (-1.5) were clipped to 0 at read time (edf_raw),
        so the saved output is all non-negative.
      - Non-1028-row EDFs were never cropped: returned unchanged (scaled).
      - meta missing / not an EDF: returns (denoised, None), callers keep the
        current minimal-header behavior.

    title: None = keep the original header Title/title verbatim (noisy save);
           str  = replace it (denoised save marker).
    """
    if not meta or meta.get('format') not in ('.edf', '.edf.gz'):
        return denoised, None
    raw = meta.get('edf_raw')
    header_src = meta.get('edf_header')
    if raw is None or header_src is None:
        return denoised, None          # old/partial meta: degrade gracefully

    scale = float(meta.get('edf_normalize_total', 1.0))
    raw_rows, raw_cols = raw.shape
    if denoised.shape == raw.shape:
        # Already full-size: non-1028 EDF (denoised, normalized) or the clipped
        # raw frame itself (noisy save, physical counts). Scale only when it is
        # not the raw frame.
        if denoised is not raw:
            denoised = denoised * scale
        full = np.ascontiguousarray(denoised, dtype=np.float32)
    elif raw_rows == 1028 and denoised.shape[0] == raw_rows - 4 \
            and denoised.shape[1] == raw_cols:
        full = np.empty((raw_rows, raw_cols), dtype=np.float32)
        full[:2, :] = raw[:2, :]        # original top 2 rows (clipped, >= 0)
        full[2:-2, :] = denoised * scale  # denoised 1024 rows (beam already 0)
        full[-2:, :] = raw[-2:, :]      # original bottom 2 rows
    else:
        # Shape mismatch (should not happen); keep denoised untouched.
        print(f"  [restore] unexpected shape {denoised.shape} vs raw "
              f"{raw.shape}; saving denoised as-is")
        full = denoised

    # Mask pixels (raw value <= 0 after clip, i.e. the Dummy=-1.5 pixels) are
    # forced to 0 in the saved output: the model emits small log-domain values
    # for zero inputs, which multiply up to huge numbers by the beam total.
    if full.shape == raw.shape:
        full[raw <= 0] = 0.0
    else:
        full[2:-2][raw[2:-2] <= 0] = 0.0

    # Put the original beam-disk pixels back: the output EDF is denoised
    # only OUTSIDE the disk; inside it stays exactly as the input.
    disk_mask = meta.get('edf_disk_mask')
    disk_values = meta.get('edf_disk_values')
    if disk_mask is not None and disk_values is not None \
            and disk_mask.shape == full.shape:
        full[disk_mask] = disk_values

    header = dict(header_src)
    if title is not None:
        # Original beamline files use lowercase 'title'; avoid duplicate keys.
        if 'title' in header:
            header['title'] = title
        else:
            header['Title'] = title
    # Dim_1/Dim_2/Size/DataType/ByteOrder/EDF_* are recomputed by fabio at
    # write time (get_edf_block), so nothing else needs updating here.
    return full, header


def read_image(filepath, h5_dataset=None, frame=0, crop_edf=True):
    """
    Read various image files and return (image_2d, metadata_dict).

    Supported formats:
      .h5 / .hdf5  -> HDF5 file, dataset name must be specified
      .edf / .edf.gz -> EDF image (fabio); raw 1028x512 data is cropped
                         to 1024x512 by removing the top/bottom 2 rows
                         (crop_edf=False keeps the full 1028 rows, with the
                         beam-stop disk still zeroed)
      .npy         -> NumPy array
      .tiff / .tif -> TIFF image
      .png / .jpg / .jpeg / .bmp -> generic image

    For 3D stacks (N, H, W) and 4D stacks (N, C, H, W), `frame` selects which
    stack index to use (default 0).
    """
    ext = _get_file_ext(filepath)
    meta = {'filepath': filepath, 'format': ext}

    if ext in ('.h5', '.hdf5'):
        import h5py
        with h5py.File(filepath, 'r') as f:
            # Find a usable dataset.
            image_keys = [k for k in f.keys() if isinstance(f[k], h5py.Dataset) and f[k].ndim >= 2]
            if not image_keys:
                raise ValueError(f"No >=2D dataset found in HDF5 file: {filepath}")

            if h5_dataset and h5_dataset in image_keys:
                key = h5_dataset
            elif len(image_keys) == 1:
                key = image_keys[0]
            else:
                # Prefer 'images'.
                if 'images' in image_keys:
                    key = 'images'
                else:
                    key = image_keys[0]
                print(f"  HDF5 contains multiple datasets {image_keys}, auto-selected '{key}'")

            data = f[key][:]
            meta['h5_key'] = key

            if data.ndim == 3:
                total_frames = data.shape[0]
                frame = _check_frame_index(frame, total_frames, key)
                data = data[frame]
                meta['h5_frame'] = frame
                meta['h5_total_frames'] = total_frames
                meta['from_3d'] = True
            elif data.ndim > 3:
                total_frames = data.shape[0]
                frame = _check_frame_index(frame, total_frames, key)
                data = data[(frame,) + (0,) * (data.ndim - 2)]  # (N, C, H, W) -> (H, W)
                meta['h5_frame'] = frame
                meta['h5_total_frames'] = total_frames
                meta['from_4d'] = True

            image = data.astype(np.float32)

    elif ext in ('.edf', '.edf.gz'):
        try:
            import fabio
        except ImportError:
            raise RuntimeError(
                "fabio is required to read EDF files; "
                "install it with: pip install fabio"
            )
        edf = fabio.open(filepath)
        total_frames = int(edf.nframes)
        if total_frames > 1:
            frame = _check_frame_index(frame, total_frames, os.path.basename(filepath))
            data = edf.getframe(frame).data
            meta['edf_frame'] = frame
            meta['edf_total_frames'] = total_frames
            header = dict(edf.getframe(frame).header)
        else:
            data = edf.data
            meta['edf_frame'] = 0
            meta['edf_total_frames'] = 1
            # Single-frame file: use the header fabio.open already read.
            # fabio.getframe() re-opens the file via jump_filename(), whose
            # numstem regex mangles names with "-<digits>" right before a
            # non-digit suffix (e.g. "...T25-150-S_noisy.edf" becomes
            # "...T250000-S_noisy.edf"), so avoid it entirely here.
            header = dict(edf.header)
        # Keep the clipped raw full frame + per-frame header for save-time
        # restoration (stitch the 2+2 edge rows back to 1028).
        clipped = np.clip(data, 0, None)
        if clipped.dtype != np.float32:
            clipped = clipped.astype(np.float32)
        meta['edf_raw'] = clipped          # ~2.1 MB for 1028x512 float32
        meta['edf_header'] = dict(header)  # per-frame header

        if data.ndim != 2:
            raise ValueError(f"EDF frame is not 2D: shape={data.shape}")

        # Beam center: header values are not always reliable (GID frames are
        # ~4 px off), so locate the beam by the intensity centroid of the
        # clipped frame.
        center = _find_beam_center(clipped, _get_center_from_header(header, data.shape))
        meta['edf_center'] = center
        meta['edf_header_center'] = _get_center_from_header(header, data.shape)

        # The scale must match the training H5s, which the original builder
        # produced as clip -> zero beam disk -> / (beam-disk intensity ~ 1e8).
        normalized, beam_mask, scale = _normalize_edf_frame(data, center)
        meta['edf_normalize_total'] = scale
        meta['raw_shape'] = data.shape

        # Keep the original beam-disk pixels (physical counts) so the saved
        # denoised EDF can put them back verbatim: only the area outside the
        # disk is denoised, the disk itself stays identical to the input.
        meta['edf_disk_mask'] = beam_mask               # full-frame bool
        meta['edf_disk_values'] = clipped[beam_mask]    # physical counts

        if data.shape[0] == 1028 and crop_edf:
            # Raw detector frames are 1028 rows tall; the top and bottom edge
            # rows are detector artifacts and are dropped before inference.
            image = normalized[2:-2, :]
            beam_mask = beam_mask[2:-2, :]
        else:
            # crop_edf=False keeps the full frame size (Tab3 compares images
            # at their original size); the beam disk is still zeroed below.
            image = normalized
        # Zero the beam-stop disk (radius 30 around the center) before denoising.
        image = np.where(beam_mask, 0.0, image)
        meta['edf_beam_mask'] = beam_mask

    elif ext == '.npy':
        data = np.load(filepath)
        if data.ndim == 3:
            total_frames = data.shape[0]
            frame = _check_frame_index(frame, total_frames, os.path.basename(filepath))
            data = data[frame]
            meta['npy_frame'] = frame
            meta['npy_total_frames'] = total_frames
            meta['from_3d'] = True
        elif data.ndim > 3:
            total_frames = data.shape[0]
            frame = _check_frame_index(frame, total_frames, os.path.basename(filepath))
            data = data[(frame,) + (0,) * (data.ndim - 2)]
            meta['npy_frame'] = frame
            meta['npy_total_frames'] = total_frames
            meta['from_4d'] = True
        image = data.astype(np.float32)

    elif ext in ('.tiff', '.tif'):
        try:
            import tifffile
            image = tifffile.imread(filepath).astype(np.float32)
        except ImportError:
            from PIL import Image
            image = np.array(Image.open(filepath)).astype(np.float32)

    elif ext in ('.png', '.jpg', '.jpeg', '.bmp', '.gif'):
        from PIL import Image
        img = Image.open(filepath)
        # Convert to grayscale.
        if img.mode != 'L':
            img = img.convert('L')
        image = np.array(img).astype(np.float32)

    else:
        raise ValueError(f"Unsupported file format: {ext}")

    meta['original_shape'] = image.shape
    meta['min'] = float(image.min())
    meta['max'] = float(image.max())
    meta['mean'] = float(image.mean())

    return image, meta


# ---------------------------------------------------------------------------
# 4. Denoising inference
# ---------------------------------------------------------------------------

def denoise_image(model, image, device, device_info=None,
                  tile_size=None, tile_overlap=32, use_amp=True,
                  restore_scale=1.0, restore_mask=None):
    """
    Denoise a single 2D image, with optional tiled inference for large images
    to save GPU memory.

    Args:
        model:         loaded SAXSAttentionUNet
        image:         (H, W) float32 numpy array
        device:        torch device
        device_info:   device info dict returned by get_best_device()
        tile_size:     tile size; None means direct inference;
                       'auto' picks 512 on low-memory devices
        tile_overlap:  tile overlap in pixels
        use_amp:       whether to enable automatic mixed precision (CUDA fp16/bf16)
        restore_scale: multiply the normalized output by this value to convert
                       it back to raw counts (1.0 for already-scaled inputs)
        restore_mask:  boolean array matching the image shape; True pixels are
                       kept as zero in the output (e.g. the beam-stop disk)

    Returns:
        denoised: (H, W) float32 numpy array
    """
    original_h, original_w = image.shape

    # Pad to a multiple of 16.
    padded, pad_info = _pad_to_multiple(image, multiple=16)

    # Match training preprocessing: x1000 (the model trained on scaled data).
    padded = padded * 1000.0

    # Decide automatically whether to use tiles.
    if tile_size == 'auto':
        if device_info and detect_low_memory_gpu(device_info):
            tile_size = 512
        else:
            tile_size = None

    # Resolve autocast settings.
    amp_dtype, amp_enabled = None, False
    if use_amp and device_info:
        amp_dtype, amp_enabled = get_inference_autocast(device, device_info)

    model.eval()
    with torch.no_grad():
        if tile_size is None:
            # Direct inference.
            tensor = torch.from_numpy(padded).float().unsqueeze(0).unsqueeze(0).to(device)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                output = model(tensor)
            denoised_padded = output[0, 0].cpu().numpy()
        else:
            # Tiled inference.
            denoised_padded = _tiled_inference(
                model, padded, device, tile_size, tile_overlap,
                amp_dtype=amp_dtype, amp_enabled=amp_enabled
            )

    # Undo the scaling: divide by 1000 to match training preprocessing.
    denoised_padded = denoised_padded / 1000.0

    # Remove padding.
    denoised = _unpad(denoised_padded, pad_info)
    # Restore the raw intensity scale for EDF imports (normalize_total).
    denoised = denoised * restore_scale
    if restore_mask is not None:
        denoised = np.where(restore_mask, 0.0, denoised)
    # Keep the output non-negative.
    denoised = np.maximum(denoised, 0)
    return denoised.astype(np.float32)


def _tiled_inference(model, image, device, tile_size, overlap,
                     amp_dtype=None, amp_enabled=False):
    """Tiled inference with overlap-weighted fusion and AMP support."""
    h, w = image.shape
    # Use Hann windows for weighted fusion to reduce tiling artifacts.
    hann_y = np.hanning(tile_size)
    hann_x = np.hanning(tile_size)
    weight_map = np.outer(hann_y, hann_x)

    result = np.zeros((h, w), dtype=np.float32)
    weight_sum = np.zeros((h, w), dtype=np.float32)

    step = tile_size - overlap

    y_starts = list(range(0, h - overlap, step))
    x_starts = list(range(0, w - overlap, step))
    # Ensure the right/bottom edges are covered.
    if y_starts[-1] + tile_size < h:
        y_starts.append(h - tile_size)
    if x_starts[-1] + tile_size < w:
        x_starts.append(w - tile_size)

    total_tiles = len(y_starts) * len(x_starts)
    for i, y0 in enumerate(y_starts):
        for j, x0 in enumerate(x_starts):
            tile = image[y0:y0 + tile_size, x0:x0 + tile_size]
            # Edge tiles are clipped automatically by numpy slicing (size < tile_size).
            tile_h, tile_w = tile.shape
            tensor = torch.from_numpy(tile).float().unsqueeze(0).unsqueeze(0).to(device)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                output = model(tensor)
            out_tile = output[0, 0].cpu().numpy()

            # The weight window must match the tile's actual size, otherwise
            # multiplying by the (tile_size, tile_size) weight_map fails to
            # broadcast (small image + large tile).
            wm = weight_map[:tile_h, :tile_w]
            result[y0:y0 + tile_h, x0:x0 + tile_w] += out_tile * wm
            weight_sum[y0:y0 + tile_h, x0:x0 + tile_w] += wm

    # Avoid division by zero.
    weight_sum = np.maximum(weight_sum, 1e-8)
    return result / weight_sum


def denoise_batch(model, images, device, progress_callback=None):
    """Denoise a batch of images with an optional progress callback."""
    results = []
    for i, img in enumerate(images):
        denoised = denoise_image(model, img, device)
        results.append(denoised)
        if progress_callback:
            progress_callback(i + 1, len(images))
    return results


# ---------------------------------------------------------------------------
# 5. GUI application
# ---------------------------------------------------------------------------

import queue
import threading

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# matplotlib backend.
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import Circle

# Display-area text uses Arial by default; individual calls can override it.
plt.rcParams['font.family'] = 'Arial'


class DenoiseApp:
    """Main SAXS denoising inference GUI application."""

    DEFAULT_MODEL_DIR = "/root/xuke/shixin/models"
    DEFAULT_OUTPUT_DIR = "/root/xuke/shixin/results"

    def __init__(self, root):
        self.root = root
        self.root.title("SAXS Denoising - Noise2Noise Inference Tool")
        self.root.geometry("1400x850")
        self.root.minsize(1000, 650)

        # ---- Device detection ----
        self.device, self.device_info = get_best_device(verbose=True)

        # ---- State ----
        self.model = None
        self.checkpoint_path = None
        self.noisy_image = None       # (H, W) float32
        self.denoised_image = None    # (H, W) float32
        self.image_meta = None
        self.current_file = None
        self.current_h5_dataset = None
        self.current_total_frames = None
        self._updating_frame = False
        self.batch_files = []         # Batch file list.
        # Batch denoise saves each result to disk immediately; only the saved
        # paths are kept in memory (no image data).
        self.batch_saved = set()
        # Threaded batch-processing state: the worker thread only puts messages
        # on the queue; every tkinter/matplotlib call happens in the GUI thread.
        self._batch_running = False
        self._batch_stop = False
        self._msg_queue = queue.Queue()
        # GUI-thread snapshots read by the batch worker (Tcl is not
        # thread-safe, so the worker never touches tkinter variables).
        self._batch_frame = 0
        self._batch_pause_sec = 1.0
        # Both intensity images share one colormap/colorbar; the residual gets its own.
        self._shared_cbar = None
        self._residual_cbar = None
        self._cut_lines = {}
        self._cut_markers = {}
        self._r_grid = None
        self._theta_grid = None
        # ---- Tab2 (EDF comparison) state ----
        self.tab2_raw_image = None          # (H, W) float32, raw EDF
        self.tab2_denoised_image = None     # (H, W) float32, denoised EDF
        self.tab2_long_image = None         # (H, W) float32, long-exposure EDF
        self.tab2_raw_meta = None
        self.tab2_denoised_meta = None
        self.tab2_long_meta = None
        self.tab2_raw_path = None
        self.tab2_denoised_path = None
        self.tab2_long_path = None
        self.tab2_shared_cbar = None
        self.tab2_long_cbar = None
        self.tab2_cut_lines = {}
        self.tab2_cut_markers = {}
        self.tab2_r_grid = None
        self.tab2_theta_grid = None
        # ---- Tab3 (raw/denoised + residual comparison) state ----
        self.tab3_raw_image = None          # (H, W) float32, raw EDF
        self.tab3_denoised_image = None     # (H, W) float32, denoised EDF
        self.tab3_raw_meta = None
        self.tab3_denoised_meta = None
        self.tab3_raw_path = None
        self.tab3_denoised_path = None
        self.tab3_shared_cbar = None
        self.tab3_residual_cbar = None
        self.tab3_cut_lines = {}
        self.tab3_cut_markers = {}
        self.tab3_r_grid = None
        self.tab3_theta_grid = None

        # ---- Variables ----
        self.var_checkpoint = tk.StringVar(value="")
        self.var_status = tk.StringVar(value="Ready - load a model first")
        self.var_display_mode = tk.StringVar(value="log")
        self.var_cmap = tk.StringVar(value="inferno")
        self.var_range_low = tk.StringVar(value="1.0")
        self.var_range_high = tk.StringVar(value="99.0")
        self.var_cut_pos_h = tk.DoubleVar(value=0.0)
        self.var_cut_pos_v = tk.DoubleVar(value=0.0)
        self.var_cut_pos_h_text = tk.StringVar(value="Row: 0")
        self.var_cut_pos_v_text = tk.StringVar(value="Col: 0")
        self.var_profile_mode = tk.StringVar(value="H / V")
        self.var_radial_angle = tk.DoubleVar(value=0.0)
        self.var_azimuthal_radius = tk.DoubleVar(value=30.0)
        self.var_radial_angle_text = tk.StringVar(value="Angle: 0 deg")
        self.var_azimuthal_radius_text = tk.StringVar(value="Radius: 30 px")
        self.var_frame = tk.StringVar(value="0")
        self.var_frame_info = tk.StringVar(value="Frame 0 / 0")
        self.var_export_dpi = tk.IntVar(value=300)
        self.var_init_ch = tk.IntVar(value=32)

        # ---- Tab2 variables (mirror Tab1 defaults) ----
        self.var_tab2_display_mode = tk.StringVar(value="log")
        self.var_tab2_cmap = tk.StringVar(value="inferno")
        self.var_tab2_range_mode = tk.StringVar(value="percentile")
        self.var_tab2_range_low = tk.StringVar(value="1.0")
        self.var_tab2_range_high = tk.StringVar(value="99.0")
        self.var_tab2_range_min = tk.DoubleVar(value=0.0)
        self.var_tab2_range_max = tk.DoubleVar(value=0.7)
        self.var_tab2_profile_mode = tk.StringVar(value="H / V")
        self.var_tab2_cut_pos_h = tk.DoubleVar(value=0.0)
        self.var_tab2_cut_pos_v = tk.DoubleVar(value=0.0)
        self.var_tab2_cut_pos_h_text = tk.StringVar(value="Row: 0")
        self.var_tab2_cut_pos_v_text = tk.StringVar(value="Col: 0")
        self.var_tab2_radial_angle = tk.DoubleVar(value=0.0)
        self.var_tab2_azimuthal_radius = tk.DoubleVar(value=30.0)
        self.var_tab2_radial_angle_text = tk.StringVar(value="Angle: 0 deg")
        self.var_tab2_azimuthal_radius_text = tk.StringVar(value="Radius: 30 px")
        self.var_tab2_export_dpi = tk.IntVar(value=300)
        self.var_tab2_file_info = tk.StringVar(value="No files loaded")

        # ---- Tab3 variables (mirror Tab2 defaults) ----
        self.var_tab3_display_mode = tk.StringVar(value="log")
        self.var_tab3_cmap = tk.StringVar(value="inferno")
        self.var_tab3_range_mode = tk.StringVar(value="percentile")
        self.var_tab3_range_low = tk.StringVar(value="1.0")
        self.var_tab3_range_high = tk.StringVar(value="99.0")
        self.var_tab3_range_min = tk.DoubleVar(value=0.0)
        self.var_tab3_range_max = tk.DoubleVar(value=0.7)
        self.var_tab3_profile_mode = tk.StringVar(value="H / V")
        self.var_tab3_cut_pos_h = tk.DoubleVar(value=0.0)
        self.var_tab3_cut_pos_v = tk.DoubleVar(value=0.0)
        self.var_tab3_cut_pos_h_text = tk.StringVar(value="Row: 0")
        self.var_tab3_cut_pos_v_text = tk.StringVar(value="Col: 0")
        self.var_tab3_radial_angle = tk.DoubleVar(value=0.0)
        self.var_tab3_azimuthal_radius = tk.DoubleVar(value=30.0)
        self.var_tab3_radial_angle_text = tk.StringVar(value="Angle: 0 deg")
        self.var_tab3_azimuthal_radius_text = tk.StringVar(value="Radius: 30 px")
        self.var_tab3_export_dpi = tk.IntVar(value=300)
        self.var_tab3_file_info = tk.StringVar(value="No files loaded")
        # Pause between batch frames (seconds), adjustable while a run is active.
        self.var_batch_pause = tk.StringVar(value="1.0")
        # Batch output toggles (read at start; the worker never touches Tcl).
        self.var_batch_save_edf = tk.BooleanVar(value=True)
        self.var_batch_save_png = tk.BooleanVar(value=False)
        self.var_batch_save_profile = tk.BooleanVar(value=False)

        # Device info string.
        if self.device_info['backend'] == 'CUDA':
            dev_str = f"CUDA: {self.device_info['name']}  ({self.device_info['memory_gb']:.1f} GB)"
        else:
            dev_str = f"{self.device_info['backend']}: {self.device_info['name']}"
        self.var_device_str = tk.StringVar(value=dev_str)
        self.var_file_info = tk.StringVar(value="")

        self._build_ui()
        self._update_status("Ready - select a model checkpoint (.pth)")

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        # ---- Bottom status bar (pack first so the left/right columns are not squeezed) ----
        status_frame = ttk.Frame(self.root, relief=tk.SUNKEN, padding=2)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)

        ttk.Label(status_frame, textvariable=self.var_status,
                  font=('sans-serif', 9)).pack(side=tk.LEFT, padx=8)

        # Progress bar.
        self.progress = ttk.Progressbar(status_frame, mode='determinate', length=200)
        self.progress.pack(side=tk.RIGHT, padx=8)

        # Prominent Stop button, shown next to the progress bar only while a
        # batch run is active (the Batch panel's "Denoise All" button also
        # turns into "Stop", but this one is impossible to miss).
        self.btn_batch_stop = tk.Button(status_frame, text="Stop",
                                        fg='white', bg='#c0392b',
                                        activebackground='#e74c3c',
                                        font=('sans-serif', 9, 'bold'),
                                        command=self._on_batch_stop)
        self.btn_batch_stop.pack_forget()

        # ---- Tab container: Tab1 (denoise) + Tab2 (EDF comparison) ----
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.tab1_frame = ttk.Frame(self.notebook)
        self.tab2_frame = ttk.Frame(self.notebook)
        self.tab3_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.tab1_frame, text="Tab1: Denoise")
        self.notebook.add(self.tab2_frame, text="Tab2: Compare")
        self.notebook.add(self.tab3_frame, text="Tab3: Compare Large")
        self.notebook.bind('<<NotebookTabChanged>>', self._on_tab_changed)
        # Single global mouse-wheel handler for the scrollable left columns.
        self.root.bind_all('<MouseWheel>', self._on_mousewheel)

        # ---- Left: toolbar + info panel + batch list (fixed width, vertical) ----
        left_panel = ttk.Frame(self.tab1_frame, width=320)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(6, 0), pady=4)
        left_panel.pack_propagate(False)   # Fixed width, not resized by children.

        # The left column holds many controls; make it scrollable so no button
        # is ever clipped by the window height. The scrollbar must be packed
        # FIRST so it keeps its ~15px width; packing it after the canvas would
        # leave it no space and push it out of view.
        self._left_canvas_t1 = tk.Canvas(left_panel, width=320,
                                         highlightthickness=0)
        self._scrollbar_t1 = ttk.Scrollbar(left_panel, orient=tk.VERTICAL,
                                           command=self._left_canvas_t1.yview)
        self._scrollbar_t1.pack(side=tk.RIGHT, fill=tk.Y)
        self._left_canvas_t1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._left_canvas_t1.configure(yscrollcommand=self._scrollbar_t1.set)
        left_inner = ttk.Frame(self._left_canvas_t1)
        left_win = self._left_canvas_t1.create_window((0, 0), window=left_inner,
                                                      anchor='nw')
        left_inner.bind('<Configure>', lambda e: self._left_canvas_t1.configure(
            scrollregion=self._left_canvas_t1.bbox('all')))
        self._left_canvas_t1.bind(
            '<Configure>',
            lambda e: self._left_canvas_t1.itemconfigure(left_win, width=e.width))

        # 1) Toolbar (vertical).
        toolbar = ttk.LabelFrame(left_inner, text="Tools", padding=6)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        toolbar_grid = ttk.Frame(toolbar)
        toolbar_grid.pack(fill=tk.X)
        toolbar_grid.columnconfigure(1, weight=1)

        ttk.Label(toolbar_grid, text="Device:").grid(row=0, column=0,
                                                     sticky='w', padx=(0, 4))
        ttk.Label(toolbar_grid, textvariable=self.var_device_str,
                  foreground='green').grid(row=0, column=1, sticky='w')

        ttk.Label(toolbar_grid, text="init_ch:").grid(row=1, column=0,
                                                      sticky='w', padx=(0, 4), pady=2)
        spin_ch = ttk.Spinbox(toolbar_grid, textvariable=self.var_init_ch,
                              from_=8, to=128, width=10, increment=8)
        spin_ch.grid(row=1, column=1, sticky='we', pady=2)

        row = 2
        button_rows = (
            (("Load Model ...", self._on_load_model), ("Open File ...", self._on_open_file)),
            (("Open Folder ...", self._on_open_folder), ("Denoise", self._on_denoise)),
            (("Save Noisy Input ...", self._on_save_noisy), ("Save Result ...", self._on_save)),
        )
        # Buttons disabled while a batch run is active (collected here).
        self._batch_ui_buttons = []
        for left_button, right_button in button_rows:
            btn = ttk.Button(toolbar_grid, text=left_button[0], command=left_button[1])
            btn.grid(row=row, column=0, sticky='we', padx=(0, 2), pady=1)
            self._batch_ui_buttons.append(btn)
            btn = ttk.Button(toolbar_grid, text=right_button[0], command=right_button[1])
            btn.grid(row=row, column=1, sticky='we', padx=(2, 0), pady=1)
            self._batch_ui_buttons.append(btn)
            row += 1
        btn = ttk.Button(toolbar_grid, text="Open Files (multi-select) ...",
                         command=self._on_open_files)
        btn.grid(row=row, column=0, columnspan=2, sticky='we', pady=1)
        self._batch_ui_buttons.append(btn)
        row += 1
        btn = ttk.Button(toolbar_grid, text="Import EDF ...",
                         command=self._on_import_edf)
        btn.grid(row=row, column=0, columnspan=2, sticky='we', pady=1)
        self._batch_ui_buttons.append(btn)
        row += 1

        ttk.Label(toolbar_grid, text="Display:").grid(row=row, column=0,
                                                      sticky='w', padx=(0, 4), pady=(4, 2))
        cbox_mode = ttk.Combobox(toolbar_grid, textvariable=self.var_display_mode,
                                 values=['log', 'linear'], width=10, state='readonly')
        cbox_mode.grid(row=row, column=1, sticky='we', pady=(4, 2))
        row += 1

        ttk.Label(toolbar_grid, text="Colormap:").grid(row=row, column=0,
                                                       sticky='w', padx=(0, 4), pady=2)
        cmaps = ['inferno', 'viridis', 'plasma', 'magma', 'gray', 'hot', 'jet']
        cbox_cmap = ttk.Combobox(toolbar_grid, textvariable=self.var_cmap,
                                 values=cmaps, width=10, state='readonly')
        cbox_cmap.grid(row=row, column=1, sticky='we', pady=2)
        row += 1

        ttk.Label(toolbar_grid, text="Range Low %:").grid(row=row, column=0,
                                                          sticky='w', padx=(0, 4), pady=2)
        spin_low = ttk.Spinbox(toolbar_grid, textvariable=self.var_range_low,
                               from_=0.0, to=99.9, width=10, increment=0.5)
        spin_low.grid(row=row, column=1, sticky='we', pady=2)
        row += 1
        ttk.Label(toolbar_grid, text="Range High %:").grid(row=row, column=0,
                                                           sticky='w', padx=(0, 4), pady=2)
        spin_high = ttk.Spinbox(toolbar_grid, textvariable=self.var_range_high,
                                from_=0.1, to=100.0, width=10, increment=0.5)
        spin_high.grid(row=row, column=1, sticky='we', pady=2)
        row += 1

        ttk.Label(toolbar_grid, text="Profile Mode:").grid(
            row=row, column=0, sticky='w', padx=(0, 4), pady=(4, 2))
        self.cbox_profile_mode = ttk.Combobox(
            toolbar_grid, textvariable=self.var_profile_mode,
            values=['H / V', 'Radial / Azimuthal'], width=15, state='readonly')
        self.cbox_profile_mode.grid(row=row, column=1, sticky='we', pady=(4, 2))
        row += 1

        self.lbl_cut_h = ttk.Label(toolbar_grid, text="Horizontal Profile:")
        self.lbl_cut_h.grid(row=row, column=0, sticky='w', padx=(0, 4), pady=(4, 0))
        self.lbl_cut_h_text = ttk.Label(toolbar_grid,
                                        textvariable=self.var_cut_pos_h_text)
        self.lbl_cut_h_text.grid(row=row, column=1, sticky='e')
        row += 1
        self.scale_cut_h = ttk.Scale(toolbar_grid, from_=0, to=100,
                                     variable=self.var_cut_pos_h,
                                     command=lambda *a: self._on_cut_change())
        self.scale_cut_h.grid(row=row, column=0, columnspan=2, sticky='we', pady=(0, 4))
        row += 1

        self.lbl_cut_v = ttk.Label(toolbar_grid, text="Vertical Profile:")
        self.lbl_cut_v.grid(row=row, column=0, sticky='w', padx=(0, 4))
        self.lbl_cut_v_text = ttk.Label(toolbar_grid,
                                        textvariable=self.var_cut_pos_v_text)
        self.lbl_cut_v_text.grid(row=row, column=1, sticky='e')
        row += 1
        self.scale_cut_v = ttk.Scale(toolbar_grid, from_=0, to=100,
                                     variable=self.var_cut_pos_v,
                                     command=lambda *a: self._on_cut_change())
        self.scale_cut_v.grid(row=row, column=0, columnspan=2, sticky='we', pady=(0, 4))
        row += 1

        # Radial/Azimuthal profile controls (hidden unless R/A mode is active).
        self.lbl_ra_angle = ttk.Label(toolbar_grid, text="Radial Angle:")
        self.lbl_ra_angle.grid(row=row, column=0, sticky='w', padx=(0, 4), pady=(4, 0))
        self.lbl_ra_angle_text = ttk.Label(toolbar_grid,
                                           textvariable=self.var_radial_angle_text)
        self.lbl_ra_angle_text.grid(row=row, column=1, sticky='e')
        row += 1
        self.scale_ra_angle = ttk.Scale(toolbar_grid, from_=0, to=360,
                                        variable=self.var_radial_angle,
                                        command=lambda *a: self._on_ra_change())
        self.scale_ra_angle.grid(row=row, column=0, columnspan=2, sticky='we', pady=(0, 4))
        row += 1

        self.lbl_ra_radius = ttk.Label(toolbar_grid, text="Azimuthal Radius:")
        self.lbl_ra_radius.grid(row=row, column=0, sticky='w', padx=(0, 4))
        self.lbl_ra_radius_text = ttk.Label(toolbar_grid,
                                            textvariable=self.var_azimuthal_radius_text)
        self.lbl_ra_radius_text.grid(row=row, column=1, sticky='e')
        row += 1
        self.scale_ra_radius = ttk.Scale(toolbar_grid, from_=0, to=100,
                                         variable=self.var_azimuthal_radius,
                                         command=lambda *a: self._on_ra_change())
        self.scale_ra_radius.grid(row=row, column=0, columnspan=2, sticky='we', pady=(0, 4))
        row += 1

        ttk.Label(toolbar_grid, text="Frame:").grid(row=row, column=0,
                                                    sticky='w', padx=(0, 4), pady=2)
        self.spin_frame = ttk.Spinbox(toolbar_grid, textvariable=self.var_frame,
                                      from_=0, to=0, width=10, increment=1,
                                      command=self._on_frame_change, state='disabled')
        self.spin_frame.grid(row=row, column=1, sticky='we', pady=2)
        ttk.Label(toolbar_grid, textvariable=self.var_frame_info).grid(
            row=row, column=1, sticky='e', pady=2)
        self.spin_frame.bind('<Return>', lambda *a: self._on_frame_change())
        self.spin_frame.bind('<FocusOut>', lambda *a: self._on_frame_change())
        row += 1

        ttk.Label(toolbar_grid, text="Pause (s):").grid(row=row, column=0,
                                                        sticky='w', padx=(0, 4), pady=(4, 2))
        ttk.Spinbox(toolbar_grid, textvariable=self.var_batch_pause,
                    from_=0.0, to=10.0, width=10, increment=0.5).grid(
                        row=row, column=1, sticky='we', pady=(4, 2))
        row += 1

        ttk.Label(toolbar_grid, text="Export DPI:").grid(row=row, column=0,
                                                         sticky='w', padx=(0, 4), pady=(4, 2))
        spin_dpi = ttk.Spinbox(toolbar_grid, textvariable=self.var_export_dpi,
                               from_=72, to=1200, width=10, increment=50)
        spin_dpi.grid(row=row, column=1, sticky='we', pady=(4, 2))
        row += 1
        btn = ttk.Button(toolbar_grid, text="Export Figure ...",
                         command=self._on_export_figure)
        btn.grid(row=row, column=0, columnspan=2, sticky='we', pady=1)
        self._batch_ui_buttons.append(btn)
        row += 1
        btn = ttk.Button(toolbar_grid, text="Clear", command=self._on_clear)
        btn.grid(row=row, column=0, columnspan=2, sticky='we', pady=1)
        self._batch_ui_buttons.append(btn)

        # Refresh the display when the colormap/display mode changes.
        self.var_cmap.trace_add('write', lambda *a: self._refresh_display())
        self.var_display_mode.trace_add('write', lambda *a: self._refresh_display())
        self.var_range_low.trace_add('write', lambda *a: self._refresh_display())
        self.var_range_high.trace_add('write', lambda *a: self._refresh_display())
        self.var_profile_mode.trace_add('write', lambda *a: self._on_profile_mode_change())
        # Start with the H/V controls visible (R/A hidden).
        self._apply_profile_mode_visibility()

        # 2) Model/file info.
        info_group = ttk.LabelFrame(left_inner, text="File Info", padding=6)
        info_group.pack(side=tk.TOP, fill=tk.X, pady=(6, 0))

        self.text_info = tk.Text(info_group, height=6, wrap=tk.WORD,
                                 font=('monospace', 9), state=tk.DISABLED)
        self.text_info.pack(fill=tk.X)

        # 3) Batch file list (list only; the action buttons live in their own
        # section below so they are never clipped by the list's expansion).
        batch_group = ttk.LabelFrame(left_inner, text="Batch Files", padding=6)
        batch_group.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(6, 0))

        self.listbox_batch = tk.Listbox(batch_group, selectmode=tk.EXTENDED,
                                         font=('monospace', 9))
        self.listbox_batch.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll_batch = ttk.Scrollbar(batch_group, orient=tk.VERTICAL,
                                     command=self.listbox_batch.yview)
        scroll_batch.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox_batch.configure(yscrollcommand=scroll_batch.set)
        self.listbox_batch.bind('<<ListboxSelect>>', self._on_batch_select)

        # 4) Batch actions: output toggles + action buttons, in their own
        # section below the file list.
        batch_actions_group = ttk.LabelFrame(left_inner, text="Batch Actions",
                                             padding=6)
        batch_actions_group.pack(side=tk.TOP, fill=tk.X, pady=(6, 0))

        self.chk_batch_save_edf = ttk.Checkbutton(
            batch_actions_group, text="Save EDF",
            variable=self.var_batch_save_edf)
        self.chk_batch_save_edf.pack(side=tk.LEFT, padx=2)
        self.chk_batch_save_png = ttk.Checkbutton(
            batch_actions_group, text="Save PNG/frame",
            variable=self.var_batch_save_png)
        self.chk_batch_save_png.pack(side=tk.LEFT, padx=2)
        self.chk_batch_save_profile = ttk.Checkbutton(
            batch_actions_group, text="Save Profile (CSV)",
            variable=self.var_batch_save_profile)
        self.chk_batch_save_profile.pack(side=tk.LEFT, padx=2)
        self._batch_ui_buttons.append(self.chk_batch_save_edf)
        self._batch_ui_buttons.append(self.chk_batch_save_png)
        self._batch_ui_buttons.append(self.chk_batch_save_profile)
        btn_row = ttk.Frame(batch_actions_group)
        btn_row.pack(side=tk.BOTTOM, fill=tk.X, pady=(2, 0))
        # Becomes "Stop" while a batch run is active.
        self.btn_denoise_all = ttk.Button(btn_row, text="Denoise All",
                                          command=self._on_batch_denoise)
        self.btn_denoise_all.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(btn_row, text="Save All",
                   command=self._on_batch_save).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        # ---- Right: image display area (two subplots) ----
        right_frame = ttk.Frame(self.tab1_frame)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=4)

        # Slightly taller figure so the enlarged labels/titles do not clip.
        self.fig = Figure(figsize=(11, 15), dpi=100)
        # Top: noisy | shared colorbar | denoised | residual | residual colorbar.
        # Below: horizontal profile, then vertical profile (each full width).
        grid_spec = self.fig.add_gridspec(3, 5, height_ratios=[3, 1, 1],
                                          width_ratios=[3, 0.4, 3, 3, 0.4],
                                          hspace=0.45, wspace=0.3)
        self.ax_noisy = self.fig.add_subplot(grid_spec[0, 0])
        self.ax_cbar_shared = self.fig.add_subplot(grid_spec[0, 1])
        self.ax_denoised = self.fig.add_subplot(grid_spec[0, 2])
        self.ax_residual = self.fig.add_subplot(grid_spec[0, 3])
        self.ax_cbar_residual = self.fig.add_subplot(grid_spec[0, 4])
        self.ax_cbar_shared.set_visible(False)
        self.ax_cbar_residual.set_visible(False)
        self.ax_profile_h = self.fig.add_subplot(grid_spec[1, :])
        self.ax_profile_v = self.fig.add_subplot(grid_spec[2, :])
        # No fig.tight_layout() here: the gridspec already fixes hspace/wspace
        # explicitly, so tight_layout is a no-op AND matplotlib >= 3.8 warns
        # ("Axes are not compatible with tight_layout") about exactly that.
        # Layout is fully determined by the gridspec parameters above.

        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # matplotlib navigation toolbar.
        nav_frame = ttk.Frame(right_frame)
        nav_frame.pack(side=tk.BOTTOM, fill=tk.X)
        NavigationToolbar2Tk(self.canvas, nav_frame)
        # Shrink the toolbar buttons.
        for child in nav_frame.winfo_children():
            try:
                child.configure(width=20)
            except Exception:
                pass

        # ---- Tab2 (EDF comparison page) ----
        self._build_tab2_ui(self.tab2_frame)
        # ---- Tab3 (raw/denoised + residual comparison page) ----
        self._build_tab3_ui(self.tab3_frame)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def _update_status(self, msg):
        self.var_status.set(msg)
        self.root.update_idletasks()

    def _set_progress(self, value, maximum=100):
        self.progress['maximum'] = maximum
        self.progress['value'] = value
        self.root.update_idletasks()

    def _on_load_model(self):
        """Load a model checkpoint."""
        initial_dir = self.DEFAULT_MODEL_DIR if os.path.isdir(self.DEFAULT_MODEL_DIR) else os.getcwd()
        path = filedialog.askopenfilename(
            title="Select Model Checkpoint",
            initialdir=initial_dir,
            filetypes=[("PyTorch Checkpoint", "*.pth *.pt"), ("All Files", "*.*")]
        )
        if not path:
            return

        self._update_status(f"Loading model: {os.path.basename(path)} ...")
        self.root.configure(cursor='watch')
        self.root.update_idletasks()

        try:
            self.model = load_model(path, self.device, init_ch=self.var_init_ch.get())
            self.checkpoint_path = path
            self.var_checkpoint.set(os.path.basename(path))
            self._update_status(f"Model loaded: {os.path.basename(path)}  |  epoch info in terminal output")
        except Exception as e:
            messagebox.showerror("Model Load Failed", f"{e}\n\n{traceback.format_exc()}")
            self._update_status("Model load failed")
        finally:
            self.root.configure(cursor='')

    def _on_open_file(self):
        """Open a single image file."""
        path = filedialog.askopenfilename(
            title="Open Image File",
            filetypes=[
                ("All Supported Files", "*.h5 *.hdf5 *.edf *.edf.gz *.npy *.tiff *.tif *.png *.jpg *.jpeg *.bmp"),
                ("HDF5 Files", "*.h5 *.hdf5"),
                ("EDF Files", "*.edf *.edf.gz"),
                ("NumPy Files", "*.npy"),
                ("TIFF Files", "*.tiff *.tif"),
                ("Image Files", "*.png *.jpg *.jpeg *.bmp"),
                ("All Files", "*.*"),
            ]
        )
        if not path:
            return

        # Handle HDF5 dataset selection. Frame selection happens in the toolbar.
        h5_dataset = None
        ext = os.path.splitext(path)[1].lower()
        if ext in ('.h5', '.hdf5'):
            import h5py
            from tkinter import simpledialog
            with h5py.File(path, 'r') as f:
                datasets = [k for k in f.keys() if isinstance(f[k], h5py.Dataset) and f[k].ndim >= 2]
                if not datasets:
                    messagebox.showerror("File Read Failed", "No >=2D dataset found in HDF5 file")
                    return
                if len(datasets) > 1:
                    # Ask the user to choose in a dialog.
                    choice = simpledialog.askstring(
                        "Select HDF5 Dataset",
                        f"The file contains multiple datasets:\n{', '.join(datasets)}\n\nEnter dataset name:",
                        initialvalue='images' if 'images' in datasets else datasets[0]
                    )
                    if choice and choice in datasets:
                        h5_dataset = choice
                    elif choice:
                        messagebox.showwarning("Invalid Selection", f"'{choice}' does not exist; will auto-select")

        self.current_h5_dataset = h5_dataset
        self._load_single_file(path, h5_dataset, 0)

    def _on_import_edf(self):
        """Import a raw EDF image (1028x512 is cropped to 1024x512)."""
        path = filedialog.askopenfilename(
            title="Import EDF Image",
            filetypes=[
                ("EDF Files", "*.edf *.edf.gz"),
                ("All Files", "*.*"),
            ]
        )
        if not path:
            return
        if _get_file_ext(path) not in ('.edf', '.edf.gz'):
            messagebox.showwarning("Invalid File", "Please select an EDF (.edf / .edf.gz) file")
            return
        self.current_h5_dataset = None
        self._load_single_file(path, None, 0)

    def _load_single_file(self, path, h5_dataset=None, frame=0):
        """Load and display a single file."""
        self.root.configure(cursor='watch')
        self.root.update_idletasks()
        try:
            self.noisy_image, self.image_meta = read_image(path, h5_dataset, frame)
            self._display_loaded(path, h5_dataset, frame)
        except Exception as e:
            messagebox.showerror("File Read Failed", f"{os.path.basename(path)}:\n{e}")
        finally:
            self.root.configure(cursor='')

    def _display_loaded(self, path, h5_dataset=None, frame=0, total_frames=None):
        """Display the current noisy_image (no file I/O).

        Shared by the single-file path and the batch worker: the worker reads
        files in a background thread and only sends the arrays here, so the
        display logic runs in the GUI thread.
        """
        self.denoised_image = None      # Clear the previous frame's result.
        self.current_file = path
        self.current_h5_dataset = h5_dataset
        if total_frames is None:
            total_frames = self.image_meta.get(
                'h5_total_frames',
                self.image_meta.get('npy_total_frames',
                                    self.image_meta.get('edf_total_frames', 1))
            )
        self.current_total_frames = total_frames
        self.spin_frame.configure(from_=0, to=total_frames - 1,
                                  state='normal' if total_frames > 1 else 'disabled')
        self._set_frame_var(frame)
        self._update_info_panel()
        self._remove_cbar('both')
        self._setup_cut_controls()
        self.ax_profile_h.clear()
        self.ax_profile_v.clear()
        self._set_profile_titles()
        self._display_image(self.noisy_image, self.ax_noisy, "Noisy Input")
        self.ax_denoised.clear()
        self.ax_denoised.set_title("Denoised Output\n(click Denoise to start)",
                                   fontsize=18, fontweight='bold',
                                   fontfamily='Arial')
        self.ax_denoised.text(0.5, 0.5, 'Click the Denoise button', transform=self.ax_denoised.transAxes,
                              ha='center', va='center', fontsize=21, color='gray',
                              fontfamily='Arial')
        self.ax_residual.clear()
        self.ax_residual.set_title("Residual (log)", fontsize=18,
                                   fontweight='bold', fontfamily='Arial')
        self._draw_profiles()
        self.canvas.draw()
        self._update_status(f"Loaded: {os.path.basename(path)}  "
                            f"{self.noisy_image.shape}  "
                            f"[{self.image_meta['min']:.2f}, {self.image_meta['max']:.2f}]")

    def _selected_frame(self):
        """Return the toolbar frame value, clamped to the current stack."""
        total = self.current_total_frames or 1
        try:
            frame = int(float(self.var_frame.get()))
        except (TypeError, ValueError):
            return 0
        return max(0, min(frame, total - 1))

    def _set_frame_var(self, frame):
        """Set the toolbar frame value without re-triggering the change handler."""
        self._updating_frame = True
        try:
            self.var_frame.set(str(frame))
            self.var_frame_info.set(f"Frame {frame} / {self.current_total_frames or 1}")
        finally:
            self._updating_frame = False

    def _on_frame_change(self):
        """Reload the current file when the toolbar frame changes."""
        if self._updating_frame or not self.current_file:
            return
        frame = self._selected_frame()
        self._load_single_file(self.current_file, self.current_h5_dataset, frame)

    def _on_open_folder(self):
        """Load image files from a folder in batch."""
        folder = filedialog.askdirectory(title="Select Folder with Image Files")
        if not folder:
            return

        files = sorted([
            os.path.join(folder, f) for f in os.listdir(folder)
            if _get_file_ext(f) in SUPPORTED_IMAGE_EXTS
        ])
        if not files:
            messagebox.showinfo("Notice", f"No supported image files found in folder:\n{folder}")
            return

        self.batch_files = files
        self._refresh_batch_list()
        self._update_status(f"Batch loaded: {len(files)} files")

        # Auto-load the first file for preview.
        if files:
            self._load_single_file(files[0])

    def _on_open_files(self):
        """Multi-select image files directly (not by folder) into the batch list."""
        ext_patterns = " ".join(f"*{ext}" for ext in sorted(SUPPORTED_IMAGE_EXTS))
        paths = filedialog.askopenfilenames(
            title="Select Image Files (Ctrl/Shift to multi-select)",
            filetypes=[("Image Files", ext_patterns), ("All Files", "*.*")])
        if not paths:
            return

        unsupported = [p for p in paths
                       if _get_file_ext(p) not in SUPPORTED_IMAGE_EXTS]
        if unsupported:
            messagebox.showwarning(
                "Unsupported Files",
                "Skipped unsupported files:\n" + "\n".join(
                    os.path.basename(p) for p in unsupported))
        paths = [p for p in paths
                 if _get_file_ext(p) in SUPPORTED_IMAGE_EXTS]
        if not paths:
            return

        # Merge into the batch list, preserving order and removing duplicates.
        existing = set(self.batch_files)
        self.batch_files = list(self.batch_files) + \
            [p for p in paths if p not in existing]
        self._refresh_batch_list()
        self._update_status(f"Batch loaded: {len(self.batch_files)} files")

        # Auto-load the first newly selected file for preview.
        self._load_single_file(paths[0])

    def _refresh_batch_list(self):
        """Refresh the batch file list display."""
        self.listbox_batch.delete(0, tk.END)
        for f in self.batch_files:
            self.listbox_batch.insert(tk.END, f"  {os.path.basename(f)}")
        # Mark processed files when results exist.
        processed = set(self.batch_saved)
        for i, f in enumerate(self.batch_files):
            if f in processed:
                current = self.listbox_batch.get(i)
                if not current.endswith(' [done]'):
                    self.listbox_batch.delete(i)
                    self.listbox_batch.insert(i, f"  {os.path.basename(f)} [done]")

    def _on_batch_select(self, event):
        """Batch list click handler: preview the selected file."""
        if self._batch_running:
            return      # Preview would clash with the batch frame display.
        selection = self.listbox_batch.curselection()
        if not selection:
            return
        idx = selection[0]
        if idx < len(self.batch_files):
            f = self.batch_files[idx]
            self._load_single_file(f)

    def _on_denoise(self):
        """Denoise the currently loaded single image."""
        if self.model is None:
            messagebox.showwarning("Notice", "Please load a model checkpoint first")
            return
        if self.noisy_image is None:
            messagebox.showwarning("Notice", "Please open an image file first")
            return

        self._update_status("Denoising ...")
        self.root.configure(cursor='watch')
        self.root.update_idletasks()
        t0 = time.time()

        try:
            self.denoised_image = denoise_image(
                self.model, self.noisy_image, self.device,
                device_info=self.device_info, tile_size='auto',
                # Display stays in the normalized domain (same as the noisy
                # input); the physical-count restore happens at save time.
                restore_mask=self.image_meta.get('edf_beam_mask')
            )
            elapsed = time.time() - t0
            # Recolor both images with the shared range after denoising.
            self._refresh_display()
            self._update_status(
                f"Denoising complete  |  {elapsed:.2f}s  |  "
                f"output range [{self.denoised_image.min():.2f}, {self.denoised_image.max():.2f}]"
            )
        except Exception as e:
            messagebox.showerror("Denoising Failed", f"{e}\n\n{traceback.format_exc()}")
            self._update_status("Denoising failed")
        finally:
            self.root.configure(cursor='')

    def _on_batch_denoise(self):
        """Start the threaded batch denoise run.

        Each file is read/denoised/saved in a background worker thread; the
        GUI thread only renders frames from the message queue, so the window
        stays responsive and every frame's result is visible (with a pause).
        """
        if self._batch_running:
            return
        if self.model is None:
            messagebox.showwarning("Notice", "Please load a model checkpoint first")
            return
        if not self.batch_files:
            messagebox.showwarning("Notice", "Please load files via Open Folder / Open Files first")
            return

        # Read the output toggles on the GUI thread (worker never touches Tcl).
        save_edf = self.var_batch_save_edf.get()
        save_png = self.var_batch_save_png.get()
        save_profile = self.var_batch_save_profile.get()
        self._batch_save_png = save_png
        self._batch_save_edf = save_edf
        self._batch_save_profile = save_profile

        # Confirmation.
        confirm_msg = (
            f"This will denoise {len(self.batch_files)} files one by one"
            + (" and save each result as '{name}_denoised.edf'" if save_edf else "")
            + (" and save each frame figure as PNG" if save_png else "")
            + (" and save each frame's profile curves as CSV" if save_profile else "")
            + ".\n\nThe GUI stays responsive; watch each frame and press "
              "Stop any time. Continue?"
        )
        ok = messagebox.askokcancel("Batch Denoise", confirm_msg)
        if not ok:
            return

        # Choose the output folder FIRST (only when something is written to
        # disk), so results are saved immediately and memory stays bounded.
        folder = None
        if save_edf or save_png or save_profile:
            folder = filedialog.askdirectory(
                title="Select Output Folder for Denoised Results")
            if not folder:
                return
        self._batch_folder = folder

        self._batch_running = True
        self._batch_stop = False
        # Bounded queue: the worker blocks on put when the GUI thread is busy,
        # so large per-frame arrays cannot pile up in memory.
        self._msg_queue = queue.Queue(maxsize=8)
        self.batch_saved = set()
        # Snapshot tkinter variables on the GUI thread only: the worker thread
        # never touches tkinter objects (Tcl is not thread-safe).
        self._batch_frame = self._selected_frame()
        self._batch_pause_sec = self._read_batch_pause()
        self._set_batch_ui_state(True)
        self._set_progress(0, len(self.batch_files))
        self._update_status(f"Batch denoise started: {len(self.batch_files)} files"
                            + (f" -> {os.path.basename(folder)}" if folder else ""))
        threading.Thread(target=self._batch_worker,
                         args=(folder, self._batch_frame, save_edf),
                         daemon=True).start()
        self.root.after(50, self._poll_batch_queue)

    def _read_batch_pause(self):
        """Snapshot of the Pause spinbox value (GUI thread only)."""
        try:
            pause = float(self.var_batch_pause.get())
        except (TypeError, ValueError):
            pause = 1.0
        return max(0.0, min(pause, 10.0))

    def _batch_worker(self, folder, frame, save_edf):
        """Background loop: read -> denoise -> (save EDF?) -> pause, one file
        at a time.

        Only puts messages on the queue; every tkinter/matplotlib call happens
        in the GUI thread (see _poll_batch_queue). Must not touch any tkinter
        object - frame, pause and save_edf come from GUI-thread snapshots.
        """
        saved = failed = 0
        try:
            for i, fpath in enumerate(self.batch_files):
                if self._batch_stop:                 # Stop check: before each file.
                    break
                try:
                    try:
                        img, meta = read_image(fpath, frame=frame)
                    except ValueError as e:
                        if 'out of range' not in str(e):
                            raise
                        print(f"  Frame {frame} out of range for {os.path.basename(fpath)}, using frame 0")
                        img, meta = read_image(fpath, frame=0)
                    self._msg_queue.put(('noisy', i, fpath, img, meta))
                    denoised = denoise_image(self.model, img, self.device,
                                             device_info=self.device_info,
                                             tile_size='auto',
                                             # Kept in the normalized domain; the
                                             # physical-count restore happens at save.
                                             restore_mask=meta.get('edf_beam_mask'))
                    self._msg_queue.put(('denoised', i, fpath, denoised))
                    if save_edf:
                        # Save to disk immediately (memory holds at most one
                        # result).
                        out_path = self._save_denoised_to(fpath, denoised, meta,
                                                          folder)
                        self._msg_queue.put(('saved', i, fpath, out_path))
                        print(f"  [saved] {os.path.basename(out_path)}")
                    else:
                        self._msg_queue.put(('done', i, fpath))
                        print(f"  [denoised] {os.path.basename(fpath)}")
                    saved += 1
                    # Pause for inspection: the GUI is showing this frame's
                    # result while the worker sleeps. The pause snapshot is
                    # refreshed by the GUI thread in _poll_batch_queue, so
                    # changing Pause mid-run takes effect immediately.
                    time.sleep(self._batch_pause_sec)
                except Exception as e:
                    self._msg_queue.put(('failed', i, fpath, str(e)))
                    failed += 1
        finally:
            # Always send a terminal message so the poll loop can stop.
            if self._batch_stop:
                self._msg_queue.put(('stopped', saved, failed))
            else:
                self._msg_queue.put(('finished', saved, failed))

    def _poll_batch_queue(self):
        """GUI-thread poller: render frames/messages from the batch worker."""
        if not self._batch_running:
            return
        # Refresh the pause snapshot (Pause changes take effect immediately).
        self._batch_pause_sec = self._read_batch_pause()
        try:
            while True:
                kind, *rest = self._msg_queue.get_nowait()
                if kind == 'noisy':
                    i, fpath, img, meta = rest
                    self.noisy_image = img
                    self.image_meta = meta
                    self._display_loaded(fpath, None, frame=self._batch_frame)
                elif kind == 'denoised':
                    i, fpath, denoised = rest
                    self.denoised_image = denoised
                    self._refresh_display()
                    # Optionally save the full right-side figure (3 x 2D +
                    # 2 profiles) as a PNG for this frame.
                    if self._batch_save_png and self._batch_folder:
                        try:
                            dpi = int(self.var_export_dpi.get())
                            dpi = max(72, min(dpi, 1200))
                        except (TypeError, ValueError):
                            dpi = 300
                        try:
                            base = os.path.splitext(
                                os.path.basename(fpath))[0]
                            self.fig.savefig(
                                os.path.join(self._batch_folder,
                                             f"{base}_denoised.png"),
                                format='png', dpi=dpi)
                        except Exception as e:
                            print(f"  [png failed] {os.path.basename(fpath)}: {e}")
                    # Optionally save the two profile curves as CSV.
                    if self._batch_save_profile and self._batch_folder:
                        try:
                            self._save_profile_csvs(fpath, self._batch_folder)
                        except Exception as e:
                            print(f"  [csv failed] {os.path.basename(fpath)}: {e}")
                    self._update_status(
                        f"Batch denoise [{i+1}/{len(self.batch_files)}]: "
                        f"{os.path.basename(fpath)} denoised")
                elif kind == 'saved':
                    i, fpath, out_path = rest
                    self.batch_saved.add(fpath)
                    self._refresh_batch_list()
                    self._set_progress(i + 1, len(self.batch_files))
                    self._update_status(
                        f"Batch denoise [{i+1}/{len(self.batch_files)}]: saved "
                        f"{os.path.basename(out_path)}")
                elif kind == 'done':
                    i, fpath = rest
                    self._set_progress(i + 1, len(self.batch_files))
                    self._update_status(
                        f"Batch denoise [{i+1}/{len(self.batch_files)}]: "
                        f"{os.path.basename(fpath)} done")
                elif kind == 'failed':
                    i, fpath, err = rest
                    self._update_status(
                        f"Batch denoise [{i+1}/{len(self.batch_files)}]: "
                        f"{os.path.basename(fpath)} FAILED - {err}")
                elif kind in ('finished', 'stopped'):
                    self._finish_batch(stopped=(kind == 'stopped'),
                                       saved=rest[0], failed=rest[1])
                    return
        except queue.Empty:
            pass
        if self._batch_running:
            self.root.after(50, self._poll_batch_queue)

    def _finish_batch(self, stopped, saved, failed):
        """Stop polling, restore the UI and report the batch outcome."""
        self._batch_running = False
        self._msg_queue = queue.Queue()
        self._set_batch_ui_state(False)
        self._set_progress(0, 1)
        self._refresh_batch_list()
        total = len(self.batch_files)
        verb = "stopped" if stopped else "complete"
        outcome = "saved" if self._batch_save_edf else "denoised"
        self._update_status(f"Batch {verb}: {saved}/{total} {outcome}")
        messagebox.showinfo(
            "Batch " + ("Stopped" if stopped else "Complete"),
            f"Batch {verb}:\n{saved} / {total} files {outcome}"
            f"{f' ({failed} failed)' if failed else ''}")

    def _on_batch_stop(self):
        """Request a stop after the current frame finishes denoising."""
        if not self._batch_running:
            return
        self._batch_stop = True
        self.btn_denoise_all.configure(state=tk.DISABLED)
        self.btn_batch_stop.configure(state=tk.DISABLED)
        self._update_status("Stopping batch ... (current frame finishes first)")

    def _set_batch_ui_state(self, running):
        """Enable/disable Tab1 controls while a batch run is active."""
        state = tk.DISABLED if running else tk.NORMAL
        for btn in self._batch_ui_buttons:
            btn.configure(state=state)
        self.btn_denoise_all.configure(
            text="Stop" if running else "Denoise All",
            command=self._on_batch_stop if running else self._on_batch_denoise,
            state=tk.NORMAL)
        # Show/hide the prominent status-bar Stop button.
        if running:
            self.btn_batch_stop.configure(state=tk.NORMAL)
            self.btn_batch_stop.pack(side=tk.RIGHT, padx=(0, 8))
        else:
            self.btn_batch_stop.pack_forget()

    def _save_denoised_to(self, fpath, denoised, meta, folder):
        """Save one denoised result into `folder` (EDF: restored 1028x512 frame;
        other formats: .npy). Returns the output path, or raises on failure."""
        base = os.path.splitext(os.path.basename(fpath))[0]
        if meta and meta.get('format') in ('.edf', '.edf.gz'):
            from fabio.edfimage import edfimage
            eh = meta.get('edf_header') or {}
            orig_title = eh.get('title') or eh.get('Title') or base
            full, header = restore_edf_output(
                denoised, meta, title=f"{orig_title} [SAXS denoised]")
            out_path = os.path.join(folder, f"{base}_denoised.edf")
            edfimage(data=full, header=header).write(out_path)
        else:
            out_path = os.path.join(folder, f"{base}_denoised.npy")
            np.save(out_path, denoised)
        return out_path

    def _save_profile_csvs(self, fpath, folder):
        """Save the two current profile curves as CSV (raw intensity, no log).

        H/V mode: {base}_profile_H.csv / _V.csv (pixel_index, raw, denoised).
        R/A mode: {base}_profile_Radial.csv / _Azimuthal.csv.
        NaN bins (empty radial/azimuthal bins) are written as `nan` and
        understood by Excel/Origin/matplotlib.
        """
        if self.noisy_image is None or self.denoised_image is None:
            return
        base = os.path.splitext(os.path.basename(fpath))[0]

        def _write(name, x, raw, den, header):
            path = os.path.join(folder, f"{base}_{name}.csv")
            np.savetxt(path, np.column_stack([x, raw, den]), delimiter=',',
                       header=header, comments='')

        if self._is_radial_mode():
            angle = self._radial_angle()
            x, raw = radial_profile(self.noisy_image, self._r_grid,
                                    self._theta_grid, angle, delta=0.5,
                                    rmax=self._rmax())
            _, den = radial_profile(self.denoised_image, self._r_grid,
                                    self._theta_grid, angle, delta=0.5,
                                    rmax=self._rmax())
            _write("profile_Radial", x, raw, den, "radius_px,raw,denoised")
            radius = self._azimuthal_radius()
            x, raw = azimuthal_profile(self.noisy_image, self._r_grid,
                                       self._theta_grid, radius)
            _, den = azimuthal_profile(self.denoised_image, self._r_grid,
                                       self._theta_grid, radius)
            _write("profile_Azimuthal", x, raw, den, "angle_deg,raw,denoised")
        else:
            pos = self._cut_pos('horizontal')
            x = np.arange(self.noisy_image.shape[1])
            _write("profile_H", x, self.noisy_image[pos, :],
                   self.denoised_image[pos, :], "pixel_index,raw,denoised")
            pos = self._cut_pos('vertical')
            x = np.arange(self.noisy_image.shape[0])
            _write("profile_V", x, self.noisy_image[:, pos],
                   self.denoised_image[:, pos], "pixel_index,raw,denoised")

    def _save_array(self, array, title, default_name, status_label, meta=None):
        """Save a 2D array through the file dialog (EDF/TIFF/NPY/PNG/JPG).

        meta: optional read_image metadata; for EDF inputs the saved .edf is
        restored to the raw size/header (see restore_edf_output).
        """
        if array is None:
            messagebox.showwarning("Notice", f"No {status_label} to save")
            return
        default_dir = self.DEFAULT_OUTPUT_DIR if os.path.isdir(self.DEFAULT_OUTPUT_DIR) else os.getcwd()

        path = filedialog.asksaveasfilename(
            title=title,
            initialdir=default_dir,
            initialfile=default_name,
            defaultextension=".edf",
            filetypes=[
                ("EDF Image (fabio)", "*.edf"),
                ("TIFF Float Image", "*.tiff *.tif"),
                ("NumPy Array", "*.npy"),
                ("PNG Image (8-bit)", "*.png"),
                ("JPEG Image (8-bit)", "*.jpg"),
            ]
        )
        if not path:
            return

        try:
            ext = os.path.splitext(path)[1].lower()
            if ext == '.edf':
                try:
                    from fabio.edfimage import edfimage
                    if meta and meta.get('format') in ('.edf', '.edf.gz'):
                        eh = meta.get('edf_header') or {}
                        orig_title = eh.get('title') or eh.get('Title') or default_name
                        full, header = restore_edf_output(
                            array, meta,
                            title=f"{orig_title} [SAXS {status_label}]")
                        edfimage(data=full, header=header or {
                            'Title': f"SAXS denoising - {status_label}",
                        }).write(path)
                    else:
                        # Non-EDF input: unchanged minimal-header behavior.
                        edfimage(data=array, header={
                            'Title': f"SAXS denoising - {status_label}",
                            'DataType': 'Float',
                        }).write(path)
                except ImportError:
                    raise RuntimeError(
                        "fabio is required to save EDF files; "
                        "install it with: pip install fabio"
                    )
            elif ext in ('.tiff', '.tif'):
                try:
                    import tifffile
                    tifffile.imwrite(path, array)
                except ImportError:
                    from PIL import Image
                    # Normalize to 0-65535.
                    data = array
                    data = (data - data.min()) / (data.max() - data.min() + 1e-8) * 65535
                    Image.fromarray(data.astype(np.uint16)).save(path)
            elif ext == '.npy':
                np.save(path, array)
            elif ext in ('.png', '.jpg', '.jpeg'):
                from PIL import Image
                # Stretch linearly or in log space to 0-255.
                data = array.copy()
                if self.var_display_mode.get() == 'log':
                    data = np.log1p(np.maximum(data, 0))
                lo, hi = data.min(), data.max()
                data = ((data - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)
                Image.fromarray(data).save(path)

            self._update_status(f"Saved: {path}")
        except Exception as e:
            messagebox.showerror("Save Failed", f"{e}")

    def _on_save_noisy(self):
        """Save the current noisy input array."""
        if self.noisy_image is None:
            messagebox.showwarning("Notice", "No noisy input to save; open a file first")
            return
        fname = os.path.basename(self.current_file) if self.current_file else ""
        ext = _get_file_ext(fname) if fname else ""
        base = fname[:-len(ext)] if ext in SUPPORTED_IMAGE_EXTS \
            else (os.path.splitext(fname)[0] if fname else "noisy_input")
        meta = self.image_meta
        if meta and meta.get('format') in ('.edf', '.edf.gz'):
            # EDF input: save the restored ORIGINAL frame (clipped full frame +
            # original header) instead of the normalized display array.
            self._save_array(meta['edf_raw'], "Save Noisy Input",
                             f"{base}_noisy", "noisy input", meta=meta)
        else:
            self._save_array(self.noisy_image, "Save Noisy Input",
                             f"{base}_noisy", "noisy input")

    def _on_save(self):
        """Save the current denoised result."""
        if self.denoised_image is None:
            messagebox.showwarning("Notice", "No denoised result to save; run denoise first")
            return
        base = os.path.splitext(os.path.basename(self.current_file))[0] if self.current_file else "denoised_output"
        self._save_array(self.denoised_image, "Save Denoised Result",
                         f"{base}_denoised", "denoised output",
                         meta=self.image_meta)

    def _on_export_figure(self):
        """Export the current figure as SVG or PDF at the requested DPI."""
        try:
            dpi = int(self.var_export_dpi.get())
        except (TypeError, ValueError):
            dpi = 300
        dpi = max(72, min(dpi, 1200))

        if self.current_file:
            base = os.path.splitext(os.path.basename(self.current_file))[0]
            default_name = f"{base}_figure"
        else:
            default_name = "saxs_denoising_figure"

        path = filedialog.asksaveasfilename(
            title="Export Figure",
            initialdir=os.getcwd(),
            initialfile=default_name,
            defaultextension=".svg",
            filetypes=[
                ("SVG Vector Image", "*.svg"),
                ("PDF Vector Document", "*.pdf"),
            ]
        )
        if not path:
            return

        try:
            ext = os.path.splitext(path)[1].lower()
            file_format = 'svg' if ext == '.svg' else 'pdf'
            self.fig.savefig(path, format=file_format, dpi=dpi)
            self._update_status(
                f"Figure exported: {os.path.basename(path)} ({file_format}, {dpi} dpi)"
            )
        except Exception as e:
            messagebox.showerror("Export Failed", f"{e}")

    def _on_batch_save(self):
        """Batch results are saved to disk during 'Denoise All' (the output
        folder is chosen there); this button only reports the saved status."""
        if not self.batch_saved:
            messagebox.showinfo(
                "Notice",
                "Batch results are saved during 'Denoise All': run it and "
                "choose the output folder there.")
            return
        messagebox.showinfo(
            "Complete",
            f"{len(self.batch_saved)} files were saved during batch denoise.")

    def _on_clear(self):
        """Clear the current display and results."""
        self.noisy_image = None
        self.denoised_image = None
        self.image_meta = None
        self.current_file = None
        self.batch_files = []
        self.batch_saved = set()
        self.listbox_batch.delete(0, tk.END)
        self._remove_cbar('both')
        self._cut_lines = {}
        self._cut_markers = {}
        self._r_grid = None
        self._theta_grid = None
        self.ax_noisy.clear()
        self.ax_noisy.set_title("Noisy Input", fontsize=18, fontweight='bold',
                                fontfamily='Arial')
        self.ax_denoised.clear()
        self.ax_denoised.set_title("Denoised Output", fontsize=18,
                                   fontweight='bold', fontfamily='Arial')
        self.ax_residual.clear()
        self.ax_residual.set_title("Residual (log)", fontsize=18,
                                   fontweight='bold', fontfamily='Arial')
        self.ax_profile_h.clear()
        self.ax_profile_v.clear()
        self._set_profile_titles()
        self.canvas.draw()
        self._update_info_panel()
        self._update_status("Cleared")

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def _remove_cbar(self, key='both'):
        """Hide the dedicated colorbar axes without deleting them, keeping the
        three 2D plots exactly the same size.

        The Colorbar objects are created once and reused via update_normal():
        re-creating them per frame leaks the callbacks they register on the
        figure (observed ~7 MB per batch frame in CallbackRegistry).
        """
        if key in ('shared', 'both'):
            self.ax_cbar_shared.set_visible(False)
        if key in ('residual', 'both'):
            self.ax_cbar_residual.set_visible(False)

    def _get_percentiles(self):
        """Return the user display percentiles (low, high), falling back to 1~99 when out of range."""
        try:
            low = float(self.var_range_low.get())
            high = float(self.var_range_high.get())
        except (TypeError, ValueError):
            low, high = 1.0, 99.0
        low = max(0.0, min(low, 99.9))
        high = max(low + 0.1, min(high, 100.0))
        return low, high

    def _get_shared_range(self, mode=None):
        """Return the shared percentile range (vmin, vmax) for both images, robust to outlier values."""
        if mode is None:
            mode = self.var_display_mode.get()
        images = []
        if self.noisy_image is not None:
            images.append(self.noisy_image)
        if self.denoised_image is not None:
            images.append(self.denoised_image)
        if not images:
            return None, None
        finite = np.concatenate([img[np.isfinite(img)] for img in images])
        if finite.size == 0:
            return None, None
        low, high = self._get_percentiles()
        if mode == 'log':
            finite = finite[finite > 0]
            if finite.size == 0:
                return None, None
        return float(np.percentile(finite, low)), float(np.percentile(finite, high))

    def _display_image(self, img, ax, title, mode=None):
        """Display a 2D image on the given matplotlib Axes."""
        if mode is None:
            mode = self.var_display_mode.get()
        ax.clear()
        ax.set_title(title, fontsize=18, fontweight='bold', fontfamily='Arial')

        cmap = self.var_cmap.get()
        vmin, vmax = self._get_shared_range(mode)

        if mode == 'log':
            # Log display uses the shared percentile range; values at 0 fall
            # below vmin and show as the colormap's bottom color.
            if vmin is not None and vmax is not None and vmax > vmin:
                norm = LogNorm(vmin=vmin, vmax=vmax)
                display_data = np.nan_to_num(img, nan=0.0,
                                             posinf=vmax, neginf=0.0)
            else:
                norm = None
                display_data = np.nan_to_num(img, nan=0.0,
                                             posinf=1.0, neginf=0.0)
        else:
            # Linear display also uses the shared range, otherwise matplotlib
            # auto-scales each image separately.
            if vmin is not None and vmax is not None and vmax > vmin:
                norm = Normalize(vmin=vmin, vmax=vmax)
            else:
                norm = None
            display_data = img

        cmap_obj = plt.get_cmap(cmap).copy()
        if mode == 'log':
            # Outliers above the shared max are shown in magenta so clipped
            # regions are visible.
            cmap_obj.set_over('magenta')
        else:
            cmap_obj.set_under('gray')
            cmap_obj.set_over('magenta')

        im = ax.imshow(display_data, cmap=cmap_obj, norm=norm, aspect='equal',
                       interpolation='bilinear', origin='upper')
        # Create the colorbar once and reuse it via update_normal: re-creating
        # it per frame leaks figure callbacks (see _remove_cbar).
        if self._shared_cbar is None:
            self._shared_cbar = self.fig.colorbar(im, cax=self.ax_cbar_shared,
                                                  location='left')
        else:
            self._shared_cbar.update_normal(im)
        self.ax_cbar_shared.set_visible(True)
        cbar = self._shared_cbar
        cbar.set_label('Intensity (log scale)' if mode == 'log' else 'Intensity',
                       fontsize=13.5, fontfamily='Arial')
        cbar.ax.tick_params(labelsize=12, labelfontfamily='Arial')
        ax.tick_params(labelsize=12, labelfontfamily='Arial')
        ax.set_xlabel(f"W = {img.shape[1]} px", fontsize=13.5,
                      fontfamily='Arial')
        ax.set_ylabel(f"H = {img.shape[0]} px", fontsize=13.5,
                      fontfamily='Arial')
        # Fixed detector-window range (1024x512): keeps the image geometry
        # identical across batch frames regardless of the current frame's
        # shape (imshow(aspect='equal') re-lays-out the axes box from the
        # limits on every draw).
        ax.set_xlim(-0.5, 511.5)
        ax.set_ylim(1023.5, -0.5)
        self._draw_overlays(ax)

    def _display_residual(self):
        """Plot the log-domain residual on the right subplot with a diverging
        colormap to emphasize what denoising changed."""
        ax = self.ax_residual
        ax.clear()
        ax.set_title("Residual (log)", fontsize=18, fontweight='bold',
                     fontfamily='Arial')

        noisy = np.log1p(np.nan_to_num(self.noisy_image, nan=0.0, posinf=0.0, neginf=0.0))
        denoised = np.log1p(np.nan_to_num(self.denoised_image, nan=0.0, posinf=0.0, neginf=0.0))
        residual = denoised - noisy
        finite = residual[np.isfinite(residual)]
        if finite.size == 0:
            norm = None
            display_data = residual
        else:
            low, high = self._get_percentiles()
            clip = max(abs(float(np.percentile(finite, low))),
                       abs(float(np.percentile(finite, high))), 1e-12)
            norm = Normalize(vmin=-clip, vmax=clip)
            display_data = np.clip(residual, -clip, clip)

        im = ax.imshow(display_data, cmap=plt.get_cmap('coolwarm'), norm=norm,
                       aspect='equal', interpolation='bilinear', origin='upper')
        # Reuse the colorbar object (see _display_image).
        if self._residual_cbar is None:
            self._residual_cbar = self.fig.colorbar(im, cax=self.ax_cbar_residual,
                                                    location='left')
        else:
            self._residual_cbar.update_normal(im)
        self.ax_cbar_residual.set_visible(True)
        cbar = self._residual_cbar
        cbar.set_label('dlog(1+I)', fontsize=13.5, fontfamily='Arial')
        cbar.ax.tick_params(labelsize=12, labelfontfamily='Arial')
        ax.tick_params(labelsize=12, labelfontfamily='Arial')
        ax.set_xlabel(f"W = {self.noisy_image.shape[1]} px", fontsize=13.5,
                      fontfamily='Arial')
        ax.set_ylabel(f"H = {self.noisy_image.shape[0]} px", fontsize=13.5,
                      fontfamily='Arial')
        # Fixed detector-window range (1024x512); see _display_image.
        ax.set_xlim(-0.5, 511.5)
        ax.set_ylim(1023.5, -0.5)
        self._draw_overlays(ax)

    def _setup_cut_controls(self):
        """Set both 1D profile slider ranges based on the current image size."""
        if self.noisy_image is None:
            return
        height, width = self.noisy_image.shape
        self.scale_cut_h.configure(from_=0, to=height - 1)
        self.scale_cut_v.configure(from_=0, to=width - 1)
        if self._batch_running:
            # During a batch run keep the user's cut positions (clamped to
            # the current frame size) instead of resetting to the center,
            # so every frame shows the profile at the same row/column.
            self.var_cut_pos_h.set(max(0.0, min(self.var_cut_pos_h.get(),
                                                height - 1)))
            self.var_cut_pos_v.set(max(0.0, min(self.var_cut_pos_v.get(),
                                                width - 1)))
        else:
            self.var_cut_pos_h.set((height - 1) / 2)
            self.var_cut_pos_v.set((width - 1) / 2)
        self.var_cut_pos_h_text.set(f"Row: {int(round(self.var_cut_pos_h.get()))}")
        self.var_cut_pos_v_text.set(f"Col: {int(round(self.var_cut_pos_v.get()))}")

        # R/A: precompute the polar grids for the current beam center and set
        # the radius slider range (keep the user's angle/radius, clamped).
        center = self._get_profile_center()
        self._r_grid, self._theta_grid = polar_grids((height, width), center)
        rmax = self._rmax()
        self.scale_ra_radius.configure(from_=0, to=max(rmax, 1e-6))
        self.var_azimuthal_radius.set(min(self.var_azimuthal_radius.get(), rmax))
        self.var_radial_angle.set(np.clip(self.var_radial_angle.get(), 0.0, 360.0))
        self.var_radial_angle_text.set(f"Angle: {self._radial_angle():.0f} deg")
        self.var_azimuthal_radius_text.set(f"Radius: {self._azimuthal_radius():.1f} px")

    def _cut_pos(self, direction):
        """Return the profile position for the given direction, clamped to bounds."""
        if self.noisy_image is None:
            return 0
        height, width = self.noisy_image.shape
        if direction == 'horizontal':
            pos = int(round(self.var_cut_pos_h.get()))
            limit = height - 1
        else:
            pos = int(round(self.var_cut_pos_v.get()))
            limit = width - 1
        return max(0, min(pos, limit))

    def _is_radial_mode(self):
        return self.var_profile_mode.get().startswith('Radial')

    def _get_profile_center(self):
        """Beam center in display coordinates for the current image."""
        if self.noisy_image is None:
            return None
        return display_center(self.image_meta, self.noisy_image.shape)

    def _rmax(self):
        """Largest integer radius fully inside the image (inscribed circle)."""
        if self.noisy_image is None:
            return 0.0
        h, w = self.noisy_image.shape
        cx, cy = self._get_profile_center()
        return float(np.floor(min(cx, w - 1 - cx, cy, h - 1 - cy)))

    def _radial_angle(self):
        return float(np.clip(self.var_radial_angle.get(), 0.0, 360.0))

    def _azimuthal_radius(self):
        return float(np.clip(self.var_azimuthal_radius.get(), 0.0, self._rmax()))

    def _ray_endpoint(self, cx, cy, angle_deg):
        """One-sided ray endpoint from (cx, cy) to the image border at angle_deg."""
        h, w = self.noisy_image.shape
        a = np.deg2rad(angle_deg)
        ca, sa = np.cos(a), np.sin(a)
        t = []
        if ca > 1e-12:
            t.append((w - 1 - cx) / ca)
        elif ca < -1e-12:
            t.append((0 - cx) / ca)
        if sa > 1e-12:
            t.append((0 - cy) / (-sa))          # upward: dy = -t*sa
        elif sa < -1e-12:
            t.append((h - 1 - cy) / (-sa))      # downward
        tmax = max(t) if t else 0.0
        return cx + tmax * ca, cy - tmax * sa

    def _draw_cut_line(self, ax, direction):
        """Draw the profile line for the given direction on the 2D plot."""
        if self.noisy_image is None:
            return
        pos = self._cut_pos(direction)
        if direction == 'horizontal':
            line = ax.axhline(y=pos, color='cyan', lw=1.0, ls='--')
        else:
            line = ax.axvline(x=pos, color='yellow', lw=1.0, ls='--')
        self._cut_lines[(id(ax), direction)] = line

    def _draw_overlays(self, ax):
        """Draw the mode-appropriate overlay (H/V cut lines or R/A markers)."""
        if self._is_radial_mode():
            self._draw_ra_markers(ax)
        else:
            self._draw_cut_line(ax, 'horizontal')
            self._draw_cut_line(ax, 'vertical')

    def _draw_ra_markers(self, ax):
        """Center marker + azimuthal-radius circle + radial-angle ray (dashed)."""
        if self.noisy_image is None:
            return
        cx, cy = self._get_profile_center()
        radius = self._azimuthal_radius()
        angle = self._radial_angle()

        center_marker, = ax.plot([cx], [cy], marker='+', color='lime',
                                 ms=11, mew=1.5, ls='none')
        self._cut_markers[(id(ax), 'center')] = center_marker

        circ = Circle((cx, cy), radius, fill=False, edgecolor='cyan',
                      lw=1.0, ls='--')
        ax.add_patch(circ)
        self._cut_markers[(id(ax), 'circle')] = circ

        x1, y1 = self._ray_endpoint(cx, cy, angle)
        ray, = ax.plot([cx, x1], [cy, y1], color='yellow', lw=1.0, ls='--')
        self._cut_markers[(id(ax), 'ray')] = ray

    def _refresh_ra_markers(self):
        """In-place update of R/A markers after slider drags (no ax.clear)."""
        if self.noisy_image is None:
            return
        cx, cy = self._get_profile_center()
        radius = self._azimuthal_radius()
        angle = self._radial_angle()
        for (_, kind), art in list(self._cut_markers.items()):
            if art.axes is None:
                continue
            if kind == 'circle':
                if art not in art.axes.patches:
                    continue
                art.set_center((cx, cy))
                art.set_radius(radius)
            elif kind == 'center':
                if art not in art.axes.lines:
                    continue
                art.set_data([cx], [cy])
            elif kind == 'ray':
                if art not in art.axes.lines:
                    continue
                x1, y1 = self._ray_endpoint(cx, cy, angle)
                art.set_data([cx, x1], [cy, y1])

    def _refresh_1d_only(self):
        """Update the profile lines and both 1D curves after a slider change."""
        if self.noisy_image is None:
            return
        if self._is_radial_mode():
            self._refresh_ra_markers()
        else:
            for (_, direction), line in list(self._cut_lines.items()):
                if line.axes is None or line not in line.axes.lines:
                    continue
                if direction == 'horizontal':
                    pos = self._cut_pos('horizontal')
                    line.set_ydata([pos, pos])
                else:
                    pos = self._cut_pos('vertical')
                    line.set_xdata([pos, pos])
        self._draw_profiles()
        self.canvas.draw()

    def _draw_profile(self, direction):
        """Plot the 1D intensity curve for h/v/radial/azimuthal profiles."""
        if self.noisy_image is None:
            return
        if direction == 'horizontal':
            ax = self.ax_profile_h
            pos = self._cut_pos('horizontal')
            title = f"Horizontal Profile (row {pos})"
            noisy_vals = self.noisy_image[pos, :]
            denoised_vals = self.denoised_image[pos, :] if self.denoised_image is not None else None
            x = np.arange(noisy_vals.size)
            xlabel = 'Pixel index'
        elif direction == 'vertical':
            ax = self.ax_profile_v
            pos = self._cut_pos('vertical')
            title = f"Vertical Profile (col {pos})"
            noisy_vals = self.noisy_image[:, pos]
            denoised_vals = self.denoised_image[:, pos] if self.denoised_image is not None else None
            x = np.arange(noisy_vals.size)
            xlabel = 'Pixel index'
        elif direction == 'radial':
            ax = self.ax_profile_h
            angle = self._radial_angle()
            title = f"Radial Profile (angle {angle:.0f} deg)"
            x, noisy_vals = radial_profile(self.noisy_image, self._r_grid,
                                           self._theta_grid, angle, delta=0.5,
                                           rmax=self._rmax())
            if self.denoised_image is not None:
                _, denoised_vals = radial_profile(self.denoised_image, self._r_grid,
                                                  self._theta_grid, angle,
                                                  delta=0.5, rmax=self._rmax())
            else:
                denoised_vals = None
            xlabel = 'Radius (px)'
        else:  # 'azimuthal'
            ax = self.ax_profile_v
            radius = self._azimuthal_radius()
            title = f"Azimuthal Profile (r = {radius:.1f} px)"
            x, noisy_vals = azimuthal_profile(self.noisy_image, self._r_grid,
                                              self._theta_grid, radius)
            if self.denoised_image is not None:
                _, denoised_vals = azimuthal_profile(self.denoised_image,
                                                     self._r_grid,
                                                     self._theta_grid, radius)
            else:
                denoised_vals = None
            xlabel = 'Angle (deg, CCW from +x)'
        ax.clear()
        if self.var_display_mode.get() == 'log':
            noisy_vals = np.log1p(np.maximum(noisy_vals, 0))
            if denoised_vals is not None:
                denoised_vals = np.log1p(np.maximum(denoised_vals, 0))
            ylabel = 'log1p(I)'
        else:
            ylabel = 'Intensity'
        ax.plot(x, noisy_vals, label='Noisy', color='tab:blue', lw=1.0)
        if denoised_vals is not None:
            ax.plot(x, denoised_vals, label='Denoised', color='tab:red', lw=1.0)
        ax.set_title(title, fontsize=15, fontweight='bold', fontfamily='Arial')
        ax.set_xlabel(xlabel, fontsize=13.5, fontfamily='Arial')
        ax.set_ylabel(ylabel, fontsize=13.5, fontfamily='Arial')
        if direction == 'azimuthal':
            ax.set_xlim(0, 360)      # fixed 0~360, NaN edges must not shift the axis
        ax.grid(alpha=0.3)
        ax.legend(prop={'family': 'Arial', 'size': 12})
        ax.tick_params(labelsize=12, labelfontfamily='Arial')

    def _draw_profiles(self):
        """Draw both 1D profile plots for the active mode."""
        if self._is_radial_mode():
            self._draw_profile('radial')
            self._draw_profile('azimuthal')
        else:
            self._draw_profile('horizontal')
            self._draw_profile('vertical')

    def _on_cut_change(self):
        """Refresh the position labels and curves when either profile slider changes."""
        if self.noisy_image is None:
            return
        self.var_cut_pos_h_text.set(f"Row: {self._cut_pos('horizontal')}")
        self.var_cut_pos_v_text.set(f"Col: {self._cut_pos('vertical')}")
        self._refresh_1d_only()

    def _on_ra_change(self):
        """Refresh R/A labels and plots when the angle/radius sliders change."""
        if self.noisy_image is None:
            return
        self.var_radial_angle_text.set(f"Angle: {self._radial_angle():.0f} deg")
        self.var_azimuthal_radius_text.set(
            f"Radius: {self._azimuthal_radius():.1f} px")
        self._refresh_1d_only()

    def _apply_profile_mode_visibility(self):
        """Show the H/V slider block or the R/A slider block (mutually exclusive)."""
        radial = self._is_radial_mode()
        hv_widgets = (self.lbl_cut_h, self.lbl_cut_h_text, self.scale_cut_h,
                      self.lbl_cut_v, self.lbl_cut_v_text, self.scale_cut_v)
        ra_widgets = (self.lbl_ra_angle, self.lbl_ra_angle_text, self.scale_ra_angle,
                      self.lbl_ra_radius, self.lbl_ra_radius_text, self.scale_ra_radius)
        for w in hv_widgets:
            w.grid_remove() if radial else w.grid()
        for w in ra_widgets:
            w.grid() if radial else w.grid_remove()

    def _on_profile_mode_change(self):
        """Switch between H/V and Radial/Azimuthal profile mode."""
        self._apply_profile_mode_visibility()
        if self.noisy_image is not None:
            self._setup_cut_controls()
            self._refresh_display()
        self.canvas.draw()

    def _set_profile_titles(self):
        """Mode-aware placeholder titles for the two profile axes."""
        if self._is_radial_mode():
            titles = ("Radial Profile", "Azimuthal Profile")
        else:
            titles = ("Horizontal Profile", "Vertical Profile")
        for ax, t in zip((self.ax_profile_h, self.ax_profile_v), titles):
            ax.set_title(t, fontsize=15, fontweight='bold', fontfamily='Arial')

    def _refresh_display(self):
        """Refresh the currently displayed images (called when colormap/mode/range changes)."""
        if self.noisy_image is None:
            return
        self._display_image(self.noisy_image, self.ax_noisy, "Noisy Input")
        if self.denoised_image is not None:
            self._display_image(self.denoised_image, self.ax_denoised, "Denoised Output")
            self._display_residual()
        else:
            self.ax_denoised.clear()
            self.ax_denoised.set_title("Denoised Output\n(click Denoise to start)",
                                       fontsize=18, fontweight='bold',
                                       fontfamily='Arial')
            self.ax_residual.clear()
            self.ax_residual.set_title("Residual (log)", fontsize=18,
                                       fontweight='bold', fontfamily='Arial')
        self._draw_profiles()
        self.canvas.draw()

    def _update_info_panel(self):
        """Update the right-side info panel."""
        self.text_info.configure(state=tk.NORMAL)
        self.text_info.delete('1.0', tk.END)

        if self.checkpoint_path:
            self.text_info.insert(tk.END, f"Model: {os.path.basename(self.checkpoint_path)}\n")
        else:
            self.text_info.insert(tk.END, "Model: (not loaded)\n")

        if self.current_file:
            self.text_info.insert(tk.END, f"File: {os.path.basename(self.current_file)}\n")
        else:
            self.text_info.insert(tk.END, "File: (not opened)\n")

        if self.image_meta:
            m = self.image_meta
            self.text_info.insert(tk.END, f"Size: {m['original_shape']}\n")
            self.text_info.insert(tk.END, f"Format: {m['format']}\n")
            if 'h5_frame' in m:
                self.text_info.insert(tk.END, f"Frame: {m['h5_frame']} of {m['h5_total_frames']} (0-based)\n")
            elif 'npy_frame' in m:
                self.text_info.insert(tk.END, f"Frame: {m['npy_frame']} of {m['npy_total_frames']} (0-based)\n")
            elif 'edf_frame' in m:
                self.text_info.insert(tk.END, f"Frame: {m['edf_frame']} of {m['edf_total_frames']} (0-based)\n")
            self.text_info.insert(tk.END, f"Range: [{m['min']:.4g}, {m['max']:.4g}]\n")
            self.text_info.insert(tk.END, f"Mean: {m['mean']:.4g}\n")
            if 'edf_normalize_total' in m:
                raw_min = m['min'] * m['edf_normalize_total']
                raw_max = m['max'] * m['edf_normalize_total']
                self.text_info.insert(
                    tk.END,
                    f"EDF normalized (x1000 for model); raw range: "
                    f"[{raw_min:.4g}, {raw_max:.4g}]\n"
                )

        if self.denoised_image is not None:
            d = self.denoised_image
            self.text_info.insert(tk.END, "\n--- Denoised Result ---\n")
            self.text_info.insert(tk.END, f"Range: [{d.min():.4g}, {d.max():.4g}]\n")
            self.text_info.insert(tk.END, f"Mean: {d.mean():.4g}\n")

        self.text_info.configure(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # Tab2: EDF comparison (raw / denoised / long-exposure)
    # ------------------------------------------------------------------

    def _build_tab2_ui(self, tab2_frame):
        """Build the Tab2 layout: left toolbar + right figure."""
        # ---- Left: toolbar ----
        tab2_left = ttk.Frame(tab2_frame, width=320)
        tab2_left.pack(side=tk.LEFT, fill=tk.Y, padx=(6, 0), pady=4)
        tab2_left.pack_propagate(False)   # Fixed width, not resized by children.

        # Scrollable left column (see the Tab1 build for the same pattern).
        self._left_canvas_t2 = tk.Canvas(tab2_left, width=320,
                                         highlightthickness=0)
        self._scrollbar_t2 = ttk.Scrollbar(tab2_left, orient=tk.VERTICAL,
                                           command=self._left_canvas_t2.yview)
        self._scrollbar_t2.pack(side=tk.RIGHT, fill=tk.Y)
        self._left_canvas_t2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._left_canvas_t2.configure(yscrollcommand=self._scrollbar_t2.set)
        tab2_inner = ttk.Frame(self._left_canvas_t2)
        tab2_win = self._left_canvas_t2.create_window((0, 0), window=tab2_inner,
                                                      anchor='nw')
        tab2_inner.bind('<Configure>', lambda e: self._left_canvas_t2.configure(
            scrollregion=self._left_canvas_t2.bbox('all')))
        self._left_canvas_t2.bind(
            '<Configure>',
            lambda e: self._left_canvas_t2.itemconfigure(tab2_win, width=e.width))

        toolbar2 = ttk.LabelFrame(tab2_inner, text="Compare EDF", padding=6)
        toolbar2.pack(side=tk.TOP, fill=tk.X)

        g = ttk.Frame(toolbar2)
        g.pack(fill=tk.X)
        g.columnconfigure(1, weight=1)
        row = 0

        # Import rows: label + Import button; the file status label below shows
        # what is currently loaded in each of the three slots.
        for text, cmd in (
            ("Raw EDF:", self._on_tab2_import_raw),
            ("Denoised:", self._on_tab2_import_denoised),
            ("Long Exposure:", self._on_tab2_import_long),
        ):
            ttk.Label(g, text=text).grid(row=row, column=0,
                                         sticky='w', padx=(0, 4), pady=1)
            ttk.Button(g, text="Import ...", command=cmd).grid(
                row=row, column=1, sticky='we', pady=1)
            row += 1
        ttk.Label(g, textvariable=self.var_tab2_file_info, foreground='#555555',
                  font=('sans-serif', 8), wraplength=280, justify=tk.LEFT).grid(
                      row=row, column=0, columnspan=2, sticky='w', pady=(0, 4))
        row += 1

        ttk.Label(g, text="Display:").grid(row=row, column=0,
                                           sticky='w', padx=(0, 4), pady=(4, 2))
        ttk.Combobox(g, textvariable=self.var_tab2_display_mode,
                     values=['log', 'linear'], width=10, state='readonly').grid(
                         row=row, column=1, sticky='we', pady=(4, 2))
        row += 1

        ttk.Label(g, text="Colormap:").grid(row=row, column=0,
                                            sticky='w', padx=(0, 4), pady=2)
        ttk.Combobox(g, textvariable=self.var_tab2_cmap,
                     values=['inferno', 'viridis', 'plasma', 'magma',
                             'gray', 'hot', 'jet'],
                     width=10, state='readonly').grid(
                         row=row, column=1, sticky='we', pady=2)
        row += 1

        # Range mode radio buttons: Percentile (default) or direct Value input.
        ttk.Label(g, text="Range Mode:").grid(row=row, column=0,
                                              sticky='w', padx=(0, 4), pady=(4, 2))
        mode_frame = ttk.Frame(g)
        mode_frame.grid(row=row, column=1, sticky='we', pady=(4, 2))
        ttk.Radiobutton(mode_frame, text="Percentile", value='percentile',
                        variable=self.var_tab2_range_mode).pack(side=tk.LEFT)
        ttk.Radiobutton(mode_frame, text="Value", value='value',
                        variable=self.var_tab2_range_mode).pack(side=tk.LEFT, padx=(6, 0))
        row += 1

        # Percentile group: Low%/High% inputs (visible in percentile mode).
        lbl_pct_low = ttk.Label(g, text="Range Low %:")
        lbl_pct_low.grid(row=row, column=0, sticky='w', padx=(0, 4), pady=2)
        spin_pct_low = ttk.Spinbox(g, textvariable=self.var_tab2_range_low,
                                   from_=0.0, to=99.9, width=10, increment=0.5)
        spin_pct_low.grid(row=row, column=1, sticky='we', pady=2)
        row += 1
        lbl_pct_high = ttk.Label(g, text="Range High %:")
        lbl_pct_high.grid(row=row, column=0, sticky='w', padx=(0, 4), pady=2)
        spin_pct_high = ttk.Spinbox(g, textvariable=self.var_tab2_range_high,
                                    from_=0.1, to=100.0, width=10, increment=0.5)
        spin_pct_high.grid(row=row, column=1, sticky='we', pady=2)
        row += 1
        self._tab2_pct_widgets = (lbl_pct_low, spin_pct_low,
                                  lbl_pct_high, spin_pct_high)

        # Value group: Min/Max numeric boxes + Auto button (visible in value mode).
        lbl_val_min = ttk.Label(g, text="Range Min:")
        lbl_val_min.grid(row=row, column=0, sticky='w', padx=(0, 4), pady=2)
        spin_val_min = ttk.Spinbox(g, textvariable=self.var_tab2_range_min,
                                   from_=0.0, to=10.0, width=10, increment=0.01)
        spin_val_min.grid(row=row, column=1, sticky='we', pady=2)
        row += 1
        lbl_val_max = ttk.Label(g, text="Range Max:")
        lbl_val_max.grid(row=row, column=0, sticky='w', padx=(0, 4), pady=2)
        spin_val_max = ttk.Spinbox(g, textvariable=self.var_tab2_range_max,
                                   from_=0.0, to=10.0, width=10, increment=0.01)
        spin_val_max.grid(row=row, column=1, sticky='we', pady=2)
        row += 1
        btn_auto = ttk.Button(g, text="Auto (0.5~99.5 pct of images)",
                              command=self._tab2_auto_range)
        btn_auto.grid(row=row, column=0, columnspan=2, sticky='we', pady=1)
        row += 1
        self._tab2_value_widgets = (lbl_val_min, spin_val_min,
                                    lbl_val_max, spin_val_max, btn_auto)

        ttk.Label(g, text="Profile Mode:").grid(row=row, column=0,
                                                sticky='w', padx=(0, 4), pady=(4, 2))
        ttk.Combobox(g, textvariable=self.var_tab2_profile_mode,
                     values=['H / V', 'Radial / Azimuthal'],
                     width=15, state='readonly').grid(
                         row=row, column=1, sticky='we', pady=(4, 2))
        row += 1

        self.lbl_tab2_cut_h = ttk.Label(g, text="Horizontal Profile:")
        self.lbl_tab2_cut_h.grid(row=row, column=0, sticky='w', padx=(0, 4), pady=(4, 0))
        self.lbl_tab2_cut_h_text = ttk.Label(g,
                                             textvariable=self.var_tab2_cut_pos_h_text)
        self.lbl_tab2_cut_h_text.grid(row=row, column=1, sticky='e')
        row += 1
        self.scale_tab2_cut_h = ttk.Scale(g, from_=0, to=100,
                                          variable=self.var_tab2_cut_pos_h,
                                          command=lambda *a: self._tab2_on_cut_change())
        self.scale_tab2_cut_h.grid(row=row, column=0, columnspan=2, sticky='we', pady=(0, 4))
        row += 1

        self.lbl_tab2_cut_v = ttk.Label(g, text="Vertical Profile:")
        self.lbl_tab2_cut_v.grid(row=row, column=0, sticky='w', padx=(0, 4))
        self.lbl_tab2_cut_v_text = ttk.Label(g,
                                             textvariable=self.var_tab2_cut_pos_v_text)
        self.lbl_tab2_cut_v_text.grid(row=row, column=1, sticky='e')
        row += 1
        self.scale_tab2_cut_v = ttk.Scale(g, from_=0, to=100,
                                          variable=self.var_tab2_cut_pos_v,
                                          command=lambda *a: self._tab2_on_cut_change())
        self.scale_tab2_cut_v.grid(row=row, column=0, columnspan=2, sticky='we', pady=(0, 4))
        row += 1

        # Radial/Azimuthal profile controls (hidden unless R/A mode is active).
        self.lbl_tab2_ra_angle = ttk.Label(g, text="Radial Angle:")
        self.lbl_tab2_ra_angle.grid(row=row, column=0, sticky='w', padx=(0, 4), pady=(4, 0))
        self.lbl_tab2_ra_angle_text = ttk.Label(g,
                                                textvariable=self.var_tab2_radial_angle_text)
        self.lbl_tab2_ra_angle_text.grid(row=row, column=1, sticky='e')
        row += 1
        self.scale_tab2_ra_angle = ttk.Scale(g, from_=0, to=360,
                                             variable=self.var_tab2_radial_angle,
                                             command=lambda *a: self._tab2_on_ra_change())
        self.scale_tab2_ra_angle.grid(row=row, column=0, columnspan=2, sticky='we', pady=(0, 4))
        row += 1

        self.lbl_tab2_ra_radius = ttk.Label(g, text="Azimuthal Radius:")
        self.lbl_tab2_ra_radius.grid(row=row, column=0, sticky='w', padx=(0, 4))
        self.lbl_tab2_ra_radius_text = ttk.Label(
            g, textvariable=self.var_tab2_azimuthal_radius_text)
        self.lbl_tab2_ra_radius_text.grid(row=row, column=1, sticky='e')
        row += 1
        self.scale_tab2_ra_radius = ttk.Scale(g, from_=0, to=100,
                                              variable=self.var_tab2_azimuthal_radius,
                                              command=lambda *a: self._tab2_on_ra_change())
        self.scale_tab2_ra_radius.grid(row=row, column=0, columnspan=2, sticky='we', pady=(0, 4))
        row += 1

        ttk.Label(g, text="Export DPI:").grid(row=row, column=0,
                                              sticky='w', padx=(0, 4), pady=(4, 2))
        ttk.Spinbox(g, textvariable=self.var_tab2_export_dpi,
                    from_=72, to=1200, width=10, increment=50).grid(
                        row=row, column=1, sticky='we', pady=(4, 2))
        row += 1
        ttk.Button(g, text="Export Figure ...",
                   command=self._on_tab2_export_figure).grid(
                       row=row, column=0, columnspan=2, sticky='we', pady=1)
        row += 1
        ttk.Button(g, text="Clear", command=self._on_tab2_clear).grid(
            row=row, column=0, columnspan=2, sticky='we', pady=1)

        # ---- Right: figure ----
        tab2_right = ttk.Frame(tab2_frame)
        tab2_right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=4)

        # Same gridspec layout as Tab1, but the third 2D slot is the
        # long-exposure image instead of the residual.
        self.tab2_fig = Figure(figsize=(11, 15), dpi=100)
        grid_spec = self.tab2_fig.add_gridspec(3, 5, height_ratios=[3, 1, 1],
                                               width_ratios=[3, 0.4, 3, 3, 0.4],
                                               hspace=0.45, wspace=0.3)
        self.tab2_ax_raw = self.tab2_fig.add_subplot(grid_spec[0, 0])
        self.tab2_ax_cbar_shared = self.tab2_fig.add_subplot(grid_spec[0, 1])
        self.tab2_ax_denoised = self.tab2_fig.add_subplot(grid_spec[0, 2])
        self.tab2_ax_long = self.tab2_fig.add_subplot(grid_spec[0, 3])
        self.tab2_ax_cbar_long = self.tab2_fig.add_subplot(grid_spec[0, 4])
        self.tab2_ax_cbar_shared.set_visible(False)
        self.tab2_ax_cbar_long.set_visible(False)
        self.tab2_ax_profile_h = self.tab2_fig.add_subplot(grid_spec[1, :])
        self.tab2_ax_profile_v = self.tab2_fig.add_subplot(grid_spec[2, :])

        self.tab2_canvas = FigureCanvasTkAgg(self.tab2_fig, master=tab2_right)
        self.tab2_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # matplotlib navigation toolbar.
        nav_frame2 = ttk.Frame(tab2_right)
        nav_frame2.pack(side=tk.BOTTOM, fill=tk.X)
        NavigationToolbar2Tk(self.tab2_canvas, nav_frame2)
        # Shrink the toolbar buttons.
        for child in nav_frame2.winfo_children():
            try:
                child.configure(width=20)
            except Exception:
                pass

        # ---- Bindings ----
        self.var_tab2_cmap.trace_add('write', lambda *a: self._tab2_refresh_display())
        self.var_tab2_display_mode.trace_add('write', lambda *a: self._tab2_refresh_display())
        self.var_tab2_range_mode.trace_add('write',
                                           lambda *a: self._tab2_on_range_mode_change())
        self.var_tab2_range_low.trace_add('write', lambda *a: self._tab2_refresh_display())
        self.var_tab2_range_high.trace_add('write', lambda *a: self._tab2_refresh_display())
        self.var_tab2_range_min.trace_add('write', lambda *a: self._tab2_refresh_display())
        self.var_tab2_range_max.trace_add('write', lambda *a: self._tab2_refresh_display())
        self.var_tab2_profile_mode.trace_add('write',
                                             lambda *a: self._tab2_on_profile_mode_change())

        # Start with the H/V controls and the Percentile range group visible.
        self._tab2_apply_profile_mode_visibility()
        self._tab2_on_range_mode_change()
        self._tab2_refresh_display()

    def _on_tab2_import_raw(self):
        self._tab2_import_file('raw')

    def _on_tab2_import_denoised(self):
        self._tab2_import_file('denoised')

    def _on_tab2_import_long(self):
        self._tab2_import_file('long')

    def _tab2_import_file(self, kind):
        """File dialog for one of the three Tab2 EDF slots."""
        path = filedialog.askopenfilename(
            title="Import EDF Image",
            filetypes=[("EDF Files", "*.edf *.edf.gz"), ("All Files", "*.*")])
        if not path:
            return
        if _get_file_ext(path) not in ('.edf', '.edf.gz'):
            messagebox.showwarning("Invalid File",
                                   "Please select an EDF (.edf / .edf.gz) file")
            return
        self._tab2_load_image(kind, path)

    def _tab2_load_image(self, kind, path):
        """Load an EDF into the given slot (raw/denoised/long) and refresh."""
        self.root.configure(cursor='watch')
        self.root.update_idletasks()
        try:
            image, meta = read_image(path, None, 0)
            if kind == 'raw':
                self.tab2_raw_image = image
                self.tab2_raw_meta = meta
                self.tab2_raw_path = path
                # The raw EDF is the profile base: reset slider ranges/polar grids.
                self._tab2_setup_cut_controls()
            elif kind == 'denoised':
                self.tab2_denoised_image = image
                self.tab2_denoised_meta = meta
                self.tab2_denoised_path = path
            else:
                self.tab2_long_image = image
                self.tab2_long_meta = meta
                self.tab2_long_path = path
            if kind != 'raw' and self.tab2_raw_image is not None \
                    and image.shape != self.tab2_raw_image.shape:
                messagebox.showwarning(
                    "Shape Mismatch",
                    f"{os.path.basename(path)} is {image.shape}, but the raw EDF "
                    f"is {self.tab2_raw_image.shape}.\nProfiles use the raw EDF as "
                    "the base; mismatched curves may be clipped or skipped.")
            names = [os.path.basename(p) if p else "-"
                     for p in (self.tab2_raw_path, self.tab2_denoised_path,
                               self.tab2_long_path)]
            self.var_tab2_file_info.set(
                f"Raw: {names[0]}\nDenoised: {names[1]}\nLong: {names[2]}")
            self._tab2_refresh_display()
            self._update_status(f"[Tab2] Loaded {kind}: {os.path.basename(path)}  "
                                f"{image.shape}")
        except Exception as e:
            messagebox.showerror("File Read Failed", f"{os.path.basename(path)}:\n{e}")
        finally:
            self.root.configure(cursor='')

    def _tab2_get_percentiles(self):
        """Return the user display percentiles (low, high) for Tab2."""
        try:
            low = float(self.var_tab2_range_low.get())
            high = float(self.var_tab2_range_high.get())
        except (TypeError, ValueError):
            low, high = 1.0, 99.0
        low = max(0.0, min(low, 99.9))
        high = max(low + 0.1, min(high, 100.0))
        return low, high

    def _tab2_get_range(self, mode=None):
        """Return the shared (vmin, vmax) for the three 2D plots.

        Percentile mode: joint percentile over all loaded images.
        Value mode: direct numeric input (clamped, max > min enforced).
        In log mode vmin is raised to a small positive value, because LogNorm
        requires a positive vmin.
        """
        if mode is None:
            mode = self.var_tab2_display_mode.get()
        images = [im for im in (self.tab2_raw_image, self.tab2_denoised_image,
                                self.tab2_long_image) if im is not None]
        if self.var_tab2_range_mode.get() == 'value':
            try:
                vmin = float(self.var_tab2_range_min.get())
                vmax = float(self.var_tab2_range_max.get())
            except (TypeError, ValueError):
                vmin, vmax = 0.0, 0.7
            vmin = max(0.0, min(vmin, 10.0))
            vmax = max(vmin + 1e-6, min(vmax, 10.0))
            if mode == 'log' and vmin <= 0:
                vmin = 1e-6
            return vmin, vmax
        # Percentile mode (default).
        if not images:
            return None, None
        finite = np.concatenate([im[np.isfinite(im)] for im in images])
        if finite.size == 0:
            return None, None
        low, high = self._tab2_get_percentiles()
        if mode == 'log':
            finite = finite[finite > 0]
            if finite.size == 0:
                return None, None
        vmin = float(np.percentile(finite, low))
        vmax = float(np.percentile(finite, high))
        if mode == 'log' and vmin <= 0:
            vmin = 1e-6
        return vmin, vmax

    def _tab2_auto_range(self):
        """Auto button: fill the Min/Max boxes from joint image percentiles."""
        images = [im for im in (self.tab2_raw_image, self.tab2_denoised_image,
                                self.tab2_long_image) if im is not None]
        if not images:
            messagebox.showwarning("Notice", "Import an EDF first")
            return
        finite = np.concatenate([im[np.isfinite(im)] for im in images])
        finite = finite[finite > 0] if finite.size else finite
        if finite.size == 0:
            messagebox.showwarning("Notice",
                                   "Loaded images contain no positive values")
            return
        vmin = float(np.percentile(finite, 0.5))
        vmax = float(np.percentile(finite, 99.5))
        self.var_tab2_range_min.set(max(0.0, vmin))
        self.var_tab2_range_max.set(max(vmin + 1e-6, vmax))
        self._update_status(f"[Tab2] Auto range: [{vmin:.4g}, {vmax:.4g}]")

    def _tab2_remove_cbar(self, key='both'):
        """Hide the dedicated Tab2 colorbar axes without deleting them.

        The Colorbar objects are reused via update_normal (see _remove_cbar).
        """
        if key in ('shared', 'both'):
            self.tab2_ax_cbar_shared.set_visible(False)
        if key in ('long', 'both'):
            self.tab2_ax_cbar_long.set_visible(False)

    def _tab2_display_image(self, img, ax, title, cbar_key, mode=None):
        """Display one 2D image; all three images share the same vmin/vmax."""
        if mode is None:
            mode = self.var_tab2_display_mode.get()
        ax.clear()
        ax.set_title(title, fontsize=18, fontweight='bold', fontfamily='Arial')

        cmap = self.var_tab2_cmap.get()
        vmin, vmax = self._tab2_get_range(mode)

        if mode == 'log':
            # Log display uses the shared range; values at 0 fall below vmin and
            # show as the colormap's bottom color.
            if vmin is not None and vmax is not None and vmax > vmin:
                norm = LogNorm(vmin=vmin, vmax=vmax)
                display_data = np.nan_to_num(img, nan=0.0,
                                             posinf=vmax, neginf=0.0)
            else:
                norm = None
                display_data = np.nan_to_num(img, nan=0.0,
                                             posinf=1.0, neginf=0.0)
        else:
            if vmin is not None and vmax is not None and vmax > vmin:
                norm = Normalize(vmin=vmin, vmax=vmax)
            else:
                norm = None
            display_data = img

        cmap_obj = plt.get_cmap(cmap).copy()
        if mode == 'log':
            # Outliers above the shared max are shown in magenta so clipped
            # regions are visible.
            cmap_obj.set_over('magenta')
        else:
            cmap_obj.set_under('gray')
            cmap_obj.set_over('magenta')

        im = ax.imshow(display_data, cmap=cmap_obj, norm=norm, aspect='equal',
                       interpolation='bilinear', origin='upper')
        # Reuse the colorbar object (see _remove_cbar).
        ax_cbar = (self.tab2_ax_cbar_shared if cbar_key == 'shared'
                   else self.tab2_ax_cbar_long)
        if cbar_key == 'shared':
            if self.tab2_shared_cbar is None:
                self.tab2_shared_cbar = self.tab2_fig.colorbar(
                    im, cax=ax_cbar, location='left')
            else:
                self.tab2_shared_cbar.update_normal(im)
            cbar = self.tab2_shared_cbar
        else:
            if self.tab2_long_cbar is None:
                self.tab2_long_cbar = self.tab2_fig.colorbar(
                    im, cax=ax_cbar, location='left')
            else:
                self.tab2_long_cbar.update_normal(im)
            cbar = self.tab2_long_cbar
        ax_cbar.set_visible(True)
        cbar.set_label('Intensity (log scale)' if mode == 'log' else 'Intensity',
                       fontsize=13.5, fontfamily='Arial')
        cbar.ax.tick_params(labelsize=12, labelfontfamily='Arial')
        ax.tick_params(labelsize=12, labelfontfamily='Arial')
        ax.set_xlabel(f"W = {img.shape[1]} px", fontsize=13.5,
                      fontfamily='Arial')
        ax.set_ylabel(f"H = {img.shape[0]} px", fontsize=13.5,
                      fontfamily='Arial')
        self._tab2_draw_overlays(ax)

    def _tab2_placeholder(self, ax, title):
        """Placeholder title + hint on an axis with no image loaded."""
        ax.clear()
        ax.set_title(title, fontsize=18, fontweight='bold', fontfamily='Arial')
        ax.text(0.5, 0.5, 'Not loaded', transform=ax.transAxes,
                ha='center', va='center', fontsize=21, color='gray',
                fontfamily='Arial')

    def _tab2_refresh_display(self):
        """Redraw the three 2D images and both profile plots."""
        if self.tab2_raw_image is not None:
            self._tab2_display_image(self.tab2_raw_image, self.tab2_ax_raw,
                                     "Raw", 'shared')
        else:
            self._tab2_placeholder(self.tab2_ax_raw, "Raw")
        if self.tab2_denoised_image is not None:
            self._tab2_display_image(self.tab2_denoised_image,
                                     self.tab2_ax_denoised,
                                     "Denoised", 'shared')
        else:
            self._tab2_placeholder(self.tab2_ax_denoised, "Denoised")
        if self.tab2_long_image is not None:
            self._tab2_display_image(self.tab2_long_image, self.tab2_ax_long,
                                     "Long Exposure", 'long')
        else:
            self._tab2_placeholder(self.tab2_ax_long, "Long Exposure")
        self._tab2_draw_profiles()
        self.tab2_canvas.draw()

    def _tab2_setup_cut_controls(self):
        """Set Tab2 slider ranges based on the raw EDF size (raw is the base)."""
        if self.tab2_raw_image is None:
            return
        height, width = self.tab2_raw_image.shape
        self.scale_tab2_cut_h.configure(from_=0, to=height - 1)
        self.var_tab2_cut_pos_h.set((height - 1) / 2)
        self.scale_tab2_cut_v.configure(from_=0, to=width - 1)
        self.var_tab2_cut_pos_v.set((width - 1) / 2)
        self.var_tab2_cut_pos_h_text.set(f"Row: {int(round(self.var_tab2_cut_pos_h.get()))}")
        self.var_tab2_cut_pos_v_text.set(f"Col: {int(round(self.var_tab2_cut_pos_v.get()))}")

        # R/A: precompute the polar grids for the raw beam center and set the
        # radius slider range (keep the user's angle/radius, clamped).
        center = self._tab2_get_profile_center()
        self.tab2_r_grid, self.tab2_theta_grid = polar_grids((height, width), center)
        rmax = self._tab2_rmax()
        self.scale_tab2_ra_radius.configure(from_=0, to=max(rmax, 1e-6))
        self.var_tab2_azimuthal_radius.set(
            min(self.var_tab2_azimuthal_radius.get(), rmax))
        self.var_tab2_radial_angle.set(np.clip(self.var_tab2_radial_angle.get(), 0.0, 360.0))
        self.var_tab2_radial_angle_text.set(f"Angle: {self._tab2_radial_angle():.0f} deg")
        self.var_tab2_azimuthal_radius_text.set(
            f"Radius: {self._tab2_azimuthal_radius():.1f} px")

    def _tab2_cut_pos(self, direction):
        """Profile position for the given direction, clamped to the raw bounds."""
        if self.tab2_raw_image is None:
            return 0
        height, width = self.tab2_raw_image.shape
        if direction == 'horizontal':
            pos = int(round(self.var_tab2_cut_pos_h.get()))
            limit = height - 1
        else:
            pos = int(round(self.var_tab2_cut_pos_v.get()))
            limit = width - 1
        return max(0, min(pos, limit))

    def _tab2_is_radial_mode(self):
        return self.var_tab2_profile_mode.get().startswith('Radial')

    def _tab2_get_profile_center(self):
        """Beam center in display coordinates for the raw EDF."""
        if self.tab2_raw_image is None:
            return None
        return display_center(self.tab2_raw_meta, self.tab2_raw_image.shape)

    def _tab2_rmax(self):
        """Largest integer radius fully inside the raw image (inscribed circle)."""
        if self.tab2_raw_image is None:
            return 0.0
        h, w = self.tab2_raw_image.shape
        cx, cy = self._tab2_get_profile_center()
        return float(np.floor(min(cx, w - 1 - cx, cy, h - 1 - cy)))

    def _tab2_radial_angle(self):
        return float(np.clip(self.var_tab2_radial_angle.get(), 0.0, 360.0))

    def _tab2_azimuthal_radius(self):
        return float(np.clip(self.var_tab2_azimuthal_radius.get(), 0.0,
                             self._tab2_rmax()))

    def _tab2_ray_endpoint(self, cx, cy, angle_deg):
        """One-sided ray endpoint from (cx, cy) to the raw image border."""
        h, w = self.tab2_raw_image.shape
        a = np.deg2rad(angle_deg)
        ca, sa = np.cos(a), np.sin(a)
        t = []
        if ca > 1e-12:
            t.append((w - 1 - cx) / ca)
        elif ca < -1e-12:
            t.append((0 - cx) / ca)
        if sa > 1e-12:
            t.append((0 - cy) / (-sa))          # upward: dy = -t*sa
        elif sa < -1e-12:
            t.append((h - 1 - cy) / (-sa))      # downward
        tmax = max(t) if t else 0.0
        return cx + tmax * ca, cy - tmax * sa

    def _tab2_draw_cut_line(self, ax, direction):
        """Draw the profile line for the given direction on a 2D plot."""
        if self.tab2_raw_image is None:
            return
        pos = self._tab2_cut_pos(direction)
        if direction == 'horizontal':
            line = ax.axhline(y=pos, color='cyan', lw=1.0, ls='--')
        else:
            line = ax.axvline(x=pos, color='yellow', lw=1.0, ls='--')
        self.tab2_cut_lines[(id(ax), direction)] = line

    def _tab2_draw_overlays(self, ax):
        """Draw the mode-appropriate overlay (H/V cut lines or R/A markers)."""
        if self._tab2_is_radial_mode():
            self._tab2_draw_ra_markers(ax)
        else:
            self._tab2_draw_cut_line(ax, 'horizontal')
            self._tab2_draw_cut_line(ax, 'vertical')

    def _tab2_draw_ra_markers(self, ax):
        """Center marker + azimuthal-radius circle + radial-angle ray."""
        if self.tab2_raw_image is None:
            return
        cx, cy = self._tab2_get_profile_center()
        radius = self._tab2_azimuthal_radius()
        angle = self._tab2_radial_angle()

        center_marker, = ax.plot([cx], [cy], marker='+', color='lime',
                                 ms=11, mew=1.5, ls='none')
        self.tab2_cut_markers[(id(ax), 'center')] = center_marker

        circ = Circle((cx, cy), radius, fill=False, edgecolor='cyan',
                      lw=1.0, ls='--')
        ax.add_patch(circ)
        self.tab2_cut_markers[(id(ax), 'circle')] = circ

        x1, y1 = self._tab2_ray_endpoint(cx, cy, angle)
        ray, = ax.plot([cx, x1], [cy, y1], color='yellow', lw=1.0, ls='--')
        self.tab2_cut_markers[(id(ax), 'ray')] = ray

    def _tab2_refresh_ra_markers(self):
        """In-place update of R/A markers after slider drags (no ax.clear)."""
        if self.tab2_raw_image is None:
            return
        cx, cy = self._tab2_get_profile_center()
        radius = self._tab2_azimuthal_radius()
        angle = self._tab2_radial_angle()
        for (_, kind), art in list(self.tab2_cut_markers.items()):
            if art.axes is None:
                continue
            if kind == 'circle':
                if art not in art.axes.patches:
                    continue
                art.set_center((cx, cy))
                art.set_radius(radius)
            elif kind == 'center':
                if art not in art.axes.lines:
                    continue
                art.set_data([cx], [cy])
            elif kind == 'ray':
                if art not in art.axes.lines:
                    continue
                x1, y1 = self._tab2_ray_endpoint(cx, cy, angle)
                art.set_data([cx, x1], [cy, y1])

    def _tab2_refresh_1d_only(self):
        """Update the profile lines and both 1D curves after a slider change."""
        if self.tab2_raw_image is None:
            return
        if self._tab2_is_radial_mode():
            self._tab2_refresh_ra_markers()
        else:
            for (_, direction), line in list(self.tab2_cut_lines.items()):
                if line.axes is None or line not in line.axes.lines:
                    continue
                if direction == 'horizontal':
                    pos = self._tab2_cut_pos('horizontal')
                    line.set_ydata([pos, pos])
                else:
                    pos = self._tab2_cut_pos('vertical')
                    line.set_xdata([pos, pos])
        self._tab2_draw_profiles()
        self.tab2_canvas.draw()

    def _tab2_draw_profile(self, direction):
        """Plot the 1D intensity curves (Raw/Denoised/Long) for one profile.

        Profiles are based on the raw EDF; denoised/long images are clipped to
        their own shapes for H/V cuts and skipped when their shape differs for
        the radial/azimuthal profiles.
        """
        if self.tab2_raw_image is None:
            self._tab2_set_profile_titles()
            return
        log_mode = self.var_tab2_display_mode.get() == 'log'

        def to_y(vals):
            return np.log1p(np.maximum(vals, 0)) if log_mode else vals

        series = []   # (label, color, x, y)
        images = (('Raw', 'tab:blue', self.tab2_raw_image),
                  ('Denoised', 'tab:red', self.tab2_denoised_image),
                  ('Long Exposure', 'tab:green', self.tab2_long_image))
        if direction == 'horizontal':
            ax = self.tab2_ax_profile_h
            pos = self._tab2_cut_pos('horizontal')
            title = f"Horizontal Profile (row {pos})"
            for label, color, img in images:
                if img is None:
                    continue
                p = min(pos, img.shape[0] - 1)
                vals = img[p, :]
                series.append((label, color, np.arange(vals.size), to_y(vals)))
            xlabel = 'Pixel index'
        elif direction == 'vertical':
            ax = self.tab2_ax_profile_v
            pos = self._tab2_cut_pos('vertical')
            title = f"Vertical Profile (col {pos})"
            for label, color, img in images:
                if img is None:
                    continue
                p = min(pos, img.shape[1] - 1)
                vals = img[:, p]
                series.append((label, color, np.arange(vals.size), to_y(vals)))
            xlabel = 'Pixel index'
        elif direction == 'radial':
            ax = self.tab2_ax_profile_h
            angle = self._tab2_radial_angle()
            title = f"Radial Profile (angle {angle:.0f} deg)"
            for label, color, img in images:
                if img is None or img.shape != self.tab2_raw_image.shape:
                    continue
                x, vals = radial_profile(img, self.tab2_r_grid,
                                         self.tab2_theta_grid, angle,
                                         delta=0.5, rmax=self._tab2_rmax())
                series.append((label, color, x, to_y(vals)))
            xlabel = 'Radius (px)'
        else:  # 'azimuthal'
            ax = self.tab2_ax_profile_v
            radius = self._tab2_azimuthal_radius()
            title = f"Azimuthal Profile (r = {radius:.1f} px)"
            for label, color, img in images:
                if img is None or img.shape != self.tab2_raw_image.shape:
                    continue
                x, vals = azimuthal_profile(img, self.tab2_r_grid,
                                            self.tab2_theta_grid, radius)
                series.append((label, color, x, to_y(vals)))
            xlabel = 'Angle (deg, CCW from +x)'

        ax.clear()
        for label, color, x, y in series:
            ax.plot(x, y, label=label, color=color, lw=1.0)
        ax.set_title(title, fontsize=15, fontweight='bold', fontfamily='Arial')
        ax.set_xlabel(xlabel, fontsize=13.5, fontfamily='Arial')
        ax.set_ylabel('log1p(I)' if log_mode else 'Intensity',
                      fontsize=13.5, fontfamily='Arial')
        if direction == 'azimuthal':
            ax.set_xlim(0, 360)      # fixed 0~360, NaN edges must not shift the axis
        ax.grid(alpha=0.3)
        if series:
            ax.legend(prop={'family': 'Arial', 'size': 12})
        ax.tick_params(labelsize=12, labelfontfamily='Arial')

    def _tab2_draw_profiles(self):
        """Draw both 1D profile plots for the active mode."""
        if self._tab2_is_radial_mode():
            self._tab2_draw_profile('radial')
            self._tab2_draw_profile('azimuthal')
        else:
            self._tab2_draw_profile('horizontal')
            self._tab2_draw_profile('vertical')

    def _tab2_set_profile_titles(self):
        """Mode-aware placeholder titles for the two Tab2 profile axes."""
        if self._tab2_is_radial_mode():
            titles = ("Radial Profile", "Azimuthal Profile")
        else:
            titles = ("Horizontal Profile", "Vertical Profile")
        for ax, t in zip((self.tab2_ax_profile_h, self.tab2_ax_profile_v), titles):
            ax.clear()
            ax.set_title(t, fontsize=15, fontweight='bold', fontfamily='Arial')

    def _tab2_on_cut_change(self):
        """Refresh the position labels and curves when either profile slider changes."""
        if self.tab2_raw_image is None:
            return
        self.var_tab2_cut_pos_h_text.set(f"Row: {self._tab2_cut_pos('horizontal')}")
        self.var_tab2_cut_pos_v_text.set(f"Col: {self._tab2_cut_pos('vertical')}")
        self._tab2_refresh_1d_only()

    def _tab2_on_ra_change(self):
        """Refresh R/A labels and plots when the angle/radius sliders change."""
        if self.tab2_raw_image is None:
            return
        self.var_tab2_radial_angle_text.set(f"Angle: {self._tab2_radial_angle():.0f} deg")
        self.var_tab2_azimuthal_radius_text.set(
            f"Radius: {self._tab2_azimuthal_radius():.1f} px")
        self._tab2_refresh_1d_only()

    def _tab2_apply_profile_mode_visibility(self):
        """Show the H/V slider block or the R/A slider block (mutually exclusive)."""
        radial = self._tab2_is_radial_mode()
        hv_widgets = (self.lbl_tab2_cut_h, self.lbl_tab2_cut_h_text,
                      self.scale_tab2_cut_h,
                      self.lbl_tab2_cut_v, self.lbl_tab2_cut_v_text,
                      self.scale_tab2_cut_v)
        ra_widgets = (self.lbl_tab2_ra_angle, self.lbl_tab2_ra_angle_text,
                      self.scale_tab2_ra_angle,
                      self.lbl_tab2_ra_radius, self.lbl_tab2_ra_radius_text,
                      self.scale_tab2_ra_radius)
        for w in hv_widgets:
            w.grid_remove() if radial else w.grid()
        for w in ra_widgets:
            w.grid() if radial else w.grid_remove()

    def _tab2_on_profile_mode_change(self):
        """Switch between H/V and Radial/Azimuthal profile mode."""
        self._tab2_apply_profile_mode_visibility()
        if self.tab2_raw_image is not None:
            self._tab2_setup_cut_controls()
            self._tab2_refresh_display()
        self.tab2_canvas.draw()

    def _tab2_on_range_mode_change(self):
        """Toggle between the percentile and value range controls."""
        value_mode = self.var_tab2_range_mode.get() == 'value'
        for w in self._tab2_pct_widgets:
            w.grid_remove() if value_mode else w.grid()
        for w in self._tab2_value_widgets:
            w.grid() if value_mode else w.grid_remove()
        self._tab2_refresh_display()

    def _on_tab2_export_figure(self):
        """Export the Tab2 figure as SVG or PDF at the requested DPI."""
        try:
            dpi = int(self.var_tab2_export_dpi.get())
        except (TypeError, ValueError):
            dpi = 300
        dpi = max(72, min(dpi, 1200))

        if self.tab2_raw_path:
            base = os.path.splitext(os.path.basename(self.tab2_raw_path))[0]
            default_name = f"{base}_compare"
        else:
            default_name = "saxs_compare_figure"

        path = filedialog.asksaveasfilename(
            title="Export Figure",
            initialdir=os.getcwd(),
            initialfile=default_name,
            defaultextension=".svg",
            filetypes=[
                ("SVG Vector Image", "*.svg"),
                ("PDF Vector Document", "*.pdf"),
            ]
        )
        if not path:
            return

        try:
            ext = os.path.splitext(path)[1].lower()
            file_format = 'svg' if ext == '.svg' else 'pdf'
            self.tab2_fig.savefig(path, format=file_format, dpi=dpi)
            self._update_status(
                f"[Tab2] Figure exported: {os.path.basename(path)} "
                f"({file_format}, {dpi} dpi)"
            )
        except Exception as e:
            messagebox.showerror("Export Failed", f"{e}")

    def _on_tab2_clear(self):
        """Clear the three Tab2 slots and redraw placeholders."""
        self.tab2_raw_image = None
        self.tab2_denoised_image = None
        self.tab2_long_image = None
        self.tab2_raw_meta = None
        self.tab2_denoised_meta = None
        self.tab2_long_meta = None
        self.tab2_raw_path = None
        self.tab2_denoised_path = None
        self.tab2_long_path = None
        self._tab2_remove_cbar('both')
        self.tab2_cut_lines = {}
        self.tab2_cut_markers = {}
        self.tab2_r_grid = None
        self.tab2_theta_grid = None
        self.var_tab2_file_info.set("No files loaded")
        self._tab2_refresh_display()
        self._update_status("[Tab2] Cleared")

    # ------------------------------------------------------------------
    # Tab3: raw/denoised comparison with a residual plot
    # ------------------------------------------------------------------

    def _build_tab3_ui(self, tab3_frame):
        """Build the Tab3 layout: left toolbar + right figure."""
        # ---- Left: toolbar ----
        tab3_left = ttk.Frame(tab3_frame, width=320)
        tab3_left.pack(side=tk.LEFT, fill=tk.Y, padx=(6, 0), pady=4)
        tab3_left.pack_propagate(False)   # Fixed width, not resized by children.

        # Scrollable left column (see the Tab1 build for the same pattern).
        self._left_canvas_t3 = tk.Canvas(tab3_left, width=320,
                                         highlightthickness=0)
        self._scrollbar_t3 = ttk.Scrollbar(tab3_left, orient=tk.VERTICAL,
                                           command=self._left_canvas_t3.yview)
        self._scrollbar_t3.pack(side=tk.RIGHT, fill=tk.Y)
        self._left_canvas_t3.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._left_canvas_t3.configure(yscrollcommand=self._scrollbar_t3.set)
        tab3_inner = ttk.Frame(self._left_canvas_t3)
        tab3_win = self._left_canvas_t3.create_window((0, 0), window=tab3_inner,
                                                      anchor='nw')
        tab3_inner.bind('<Configure>', lambda e: self._left_canvas_t3.configure(
            scrollregion=self._left_canvas_t3.bbox('all')))
        self._left_canvas_t3.bind(
            '<Configure>',
            lambda e: self._left_canvas_t3.itemconfigure(tab3_win, width=e.width))

        toolbar3 = ttk.LabelFrame(tab3_inner, text="Compare (Raw/Denoised)",
                                  padding=6)
        toolbar3.pack(side=tk.TOP, fill=tk.X)

        g = ttk.Frame(toolbar3)
        g.pack(fill=tk.X)
        g.columnconfigure(1, weight=1)
        row = 0

        # Import rows: label + Import button (no long-exposure slot).
        for text, cmd in (
            ("Raw EDF:", self._on_tab3_import_raw),
            ("Denoised:", self._on_tab3_import_denoised),
        ):
            ttk.Label(g, text=text).grid(row=row, column=0,
                                         sticky='w', padx=(0, 4), pady=1)
            ttk.Button(g, text="Import ...", command=cmd).grid(
                row=row, column=1, sticky='we', pady=1)
            row += 1
        ttk.Label(g, textvariable=self.var_tab3_file_info, foreground='#555555',
                  font=('sans-serif', 8), wraplength=280, justify=tk.LEFT).grid(
                      row=row, column=0, columnspan=2, sticky='w', pady=(0, 4))
        row += 1

        ttk.Label(g, text="Display:").grid(row=row, column=0,
                                           sticky='w', padx=(0, 4), pady=(4, 2))
        ttk.Combobox(g, textvariable=self.var_tab3_display_mode,
                     values=['log', 'linear'], width=10, state='readonly').grid(
                         row=row, column=1, sticky='we', pady=(4, 2))
        row += 1

        ttk.Label(g, text="Colormap:").grid(row=row, column=0,
                                            sticky='w', padx=(0, 4), pady=2)
        ttk.Combobox(g, textvariable=self.var_tab3_cmap,
                     values=['inferno', 'viridis', 'plasma', 'magma',
                             'gray', 'hot', 'jet'],
                     width=10, state='readonly').grid(
                         row=row, column=1, sticky='we', pady=2)
        row += 1

        # Range mode radio buttons: Percentile (default) or direct Value input.
        ttk.Label(g, text="Range Mode:").grid(row=row, column=0,
                                              sticky='w', padx=(0, 4), pady=(4, 2))
        mode_frame = ttk.Frame(g)
        mode_frame.grid(row=row, column=1, sticky='we', pady=(4, 2))
        ttk.Radiobutton(mode_frame, text="Percentile", value='percentile',
                        variable=self.var_tab3_range_mode).pack(side=tk.LEFT)
        ttk.Radiobutton(mode_frame, text="Value", value='value',
                        variable=self.var_tab3_range_mode).pack(side=tk.LEFT, padx=(6, 0))
        row += 1

        # Percentile group: Low%/High% inputs (visible in percentile mode).
        lbl_pct_low = ttk.Label(g, text="Range Low %:")
        lbl_pct_low.grid(row=row, column=0, sticky='w', padx=(0, 4), pady=2)
        spin_pct_low = ttk.Spinbox(g, textvariable=self.var_tab3_range_low,
                                   from_=0.0, to=99.9, width=10, increment=0.5)
        spin_pct_low.grid(row=row, column=1, sticky='we', pady=2)
        row += 1
        lbl_pct_high = ttk.Label(g, text="Range High %:")
        lbl_pct_high.grid(row=row, column=0, sticky='w', padx=(0, 4), pady=2)
        spin_pct_high = ttk.Spinbox(g, textvariable=self.var_tab3_range_high,
                                    from_=0.1, to=100.0, width=10, increment=0.5)
        spin_pct_high.grid(row=row, column=1, sticky='we', pady=2)
        row += 1
        self._tab3_pct_widgets = (lbl_pct_low, spin_pct_low,
                                  lbl_pct_high, spin_pct_high)

        # Value group: Min/Max numeric boxes + Auto button (value mode).
        lbl_val_min = ttk.Label(g, text="Range Min:")
        lbl_val_min.grid(row=row, column=0, sticky='w', padx=(0, 4), pady=2)
        spin_val_min = ttk.Spinbox(g, textvariable=self.var_tab3_range_min,
                                   from_=0.0, to=10.0, width=10, increment=0.01)
        spin_val_min.grid(row=row, column=1, sticky='we', pady=2)
        row += 1
        lbl_val_max = ttk.Label(g, text="Range Max:")
        lbl_val_max.grid(row=row, column=0, sticky='w', padx=(0, 4), pady=2)
        spin_val_max = ttk.Spinbox(g, textvariable=self.var_tab3_range_max,
                                   from_=0.0, to=10.0, width=10, increment=0.01)
        spin_val_max.grid(row=row, column=1, sticky='we', pady=2)
        row += 1
        btn_auto = ttk.Button(g, text="Auto (0.5~99.5 pct of images)",
                              command=self._tab3_auto_range)
        btn_auto.grid(row=row, column=0, columnspan=2, sticky='we', pady=1)
        row += 1
        self._tab3_value_widgets = (lbl_val_min, spin_val_min,
                                    lbl_val_max, spin_val_max, btn_auto)

        ttk.Label(g, text="Profile Mode:").grid(row=row, column=0,
                                                sticky='w', padx=(0, 4), pady=(4, 2))
        ttk.Combobox(g, textvariable=self.var_tab3_profile_mode,
                     values=['H / V', 'Radial / Azimuthal'],
                     width=15, state='readonly').grid(
                         row=row, column=1, sticky='we', pady=(4, 2))
        row += 1

        self.lbl_tab3_cut_h = ttk.Label(g, text="Horizontal Profile:")
        self.lbl_tab3_cut_h.grid(row=row, column=0, sticky='w', padx=(0, 4), pady=(4, 0))
        self.lbl_tab3_cut_h_text = ttk.Label(g,
                                             textvariable=self.var_tab3_cut_pos_h_text)
        self.lbl_tab3_cut_h_text.grid(row=row, column=1, sticky='e')
        row += 1
        self.scale_tab3_cut_h = ttk.Scale(g, from_=0, to=100,
                                          variable=self.var_tab3_cut_pos_h,
                                          command=lambda *a: self._tab3_on_cut_change())
        self.scale_tab3_cut_h.grid(row=row, column=0, columnspan=2, sticky='we', pady=(0, 4))
        row += 1

        self.lbl_tab3_cut_v = ttk.Label(g, text="Vertical Profile:")
        self.lbl_tab3_cut_v.grid(row=row, column=0, sticky='w', padx=(0, 4))
        self.lbl_tab3_cut_v_text = ttk.Label(g,
                                             textvariable=self.var_tab3_cut_pos_v_text)
        self.lbl_tab3_cut_v_text.grid(row=row, column=1, sticky='e')
        row += 1
        self.scale_tab3_cut_v = ttk.Scale(g, from_=0, to=100,
                                          variable=self.var_tab3_cut_pos_v,
                                          command=lambda *a: self._tab3_on_cut_change())
        self.scale_tab3_cut_v.grid(row=row, column=0, columnspan=2, sticky='we', pady=(0, 4))
        row += 1

        # Radial/Azimuthal profile controls (hidden unless R/A mode is active).
        self.lbl_tab3_ra_angle = ttk.Label(g, text="Radial Angle:")
        self.lbl_tab3_ra_angle.grid(row=row, column=0, sticky='w', padx=(0, 4), pady=(4, 0))
        self.lbl_tab3_ra_angle_text = ttk.Label(g,
                                                textvariable=self.var_tab3_radial_angle_text)
        self.lbl_tab3_ra_angle_text.grid(row=row, column=1, sticky='e')
        row += 1
        self.scale_tab3_ra_angle = ttk.Scale(g, from_=0, to=360,
                                             variable=self.var_tab3_radial_angle,
                                             command=lambda *a: self._tab3_on_ra_change())
        self.scale_tab3_ra_angle.grid(row=row, column=0, columnspan=2, sticky='we', pady=(0, 4))
        row += 1

        self.lbl_tab3_ra_radius = ttk.Label(g, text="Azimuthal Radius:")
        self.lbl_tab3_ra_radius.grid(row=row, column=0, sticky='w', padx=(0, 4))
        self.lbl_tab3_ra_radius_text = ttk.Label(
            g, textvariable=self.var_tab3_azimuthal_radius_text)
        self.lbl_tab3_ra_radius_text.grid(row=row, column=1, sticky='e')
        row += 1
        self.scale_tab3_ra_radius = ttk.Scale(g, from_=0, to=100,
                                              variable=self.var_tab3_azimuthal_radius,
                                              command=lambda *a: self._tab3_on_ra_change())
        self.scale_tab3_ra_radius.grid(row=row, column=0, columnspan=2, sticky='we', pady=(0, 4))
        row += 1

        ttk.Label(g, text="Export DPI:").grid(row=row, column=0,
                                              sticky='w', padx=(0, 4), pady=(4, 2))
        ttk.Spinbox(g, textvariable=self.var_tab3_export_dpi,
                    from_=72, to=1200, width=10, increment=50).grid(
                        row=row, column=1, sticky='we', pady=(4, 2))
        row += 1
        ttk.Button(g, text="Export Figure ...",
                   command=self._on_tab3_export_figure).grid(
                       row=row, column=0, columnspan=2, sticky='we', pady=1)
        row += 1
        ttk.Button(g, text="Clear", command=self._on_tab3_clear).grid(
            row=row, column=0, columnspan=2, sticky='we', pady=1)

        # ---- Right: figure ----
        tab3_right = ttk.Frame(tab3_frame)
        tab3_right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=4)

        # Same gridspec layout as Tab2, but the third 2D slot is the
        # log-domain residual instead of the long-exposure image.
        self.tab3_fig = Figure(figsize=(11, 15), dpi=100)
        grid_spec = self.tab3_fig.add_gridspec(3, 5, height_ratios=[3, 1, 1],
                                               width_ratios=[3, 0.4, 3, 3, 0.4],
                                               hspace=0.45, wspace=0.3)
        self.tab3_ax_raw = self.tab3_fig.add_subplot(grid_spec[0, 0])
        self.tab3_ax_cbar_shared = self.tab3_fig.add_subplot(grid_spec[0, 1])
        self.tab3_ax_denoised = self.tab3_fig.add_subplot(grid_spec[0, 2])
        self.tab3_ax_residual = self.tab3_fig.add_subplot(grid_spec[0, 3])
        self.tab3_ax_cbar_res = self.tab3_fig.add_subplot(grid_spec[0, 4])
        self.tab3_ax_cbar_shared.set_visible(False)
        self.tab3_ax_cbar_res.set_visible(False)
        self.tab3_ax_profile_h = self.tab3_fig.add_subplot(grid_spec[1, :])
        self.tab3_ax_profile_v = self.tab3_fig.add_subplot(grid_spec[2, :])

        self.tab3_canvas = FigureCanvasTkAgg(self.tab3_fig, master=tab3_right)
        self.tab3_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # matplotlib navigation toolbar.
        nav_frame3 = ttk.Frame(tab3_right)
        nav_frame3.pack(side=tk.BOTTOM, fill=tk.X)
        NavigationToolbar2Tk(self.tab3_canvas, nav_frame3)
        for child in nav_frame3.winfo_children():
            try:
                child.configure(width=20)
            except Exception:
                pass

        # ---- Bindings ----
        self.var_tab3_cmap.trace_add('write', lambda *a: self._tab3_refresh_display())
        self.var_tab3_display_mode.trace_add('write', lambda *a: self._tab3_refresh_display())
        self.var_tab3_range_mode.trace_add('write',
                                           lambda *a: self._tab3_on_range_mode_change())
        self.var_tab3_range_low.trace_add('write', lambda *a: self._tab3_refresh_display())
        self.var_tab3_range_high.trace_add('write', lambda *a: self._tab3_refresh_display())
        self.var_tab3_range_min.trace_add('write', lambda *a: self._tab3_refresh_display())
        self.var_tab3_range_max.trace_add('write', lambda *a: self._tab3_refresh_display())
        self.var_tab3_profile_mode.trace_add('write',
                                             lambda *a: self._tab3_on_profile_mode_change())

        # Start with the H/V controls and the Percentile range group visible.
        self._tab3_apply_profile_mode_visibility()
        self._tab3_on_range_mode_change()
        self._tab3_refresh_display()

    def _on_tab3_import_raw(self):
        self._tab3_import_file('raw')

    def _on_tab3_import_denoised(self):
        self._tab3_import_file('denoised')

    def _tab3_import_file(self, kind):
        """File dialog for one of the two Tab3 EDF slots."""
        path = filedialog.askopenfilename(
            title="Import EDF Image",
            filetypes=[("EDF Files", "*.edf *.edf.gz"), ("All Files", "*.*")])
        if not path:
            return
        if _get_file_ext(path) not in ('.edf', '.edf.gz'):
            messagebox.showwarning("Invalid File",
                                   "Please select an EDF (.edf / .edf.gz) file")
            return
        self._tab3_load_image(kind, path)

    def _tab3_load_image(self, kind, path):
        """Load an EDF into the given slot (raw/denoised) and refresh."""
        self.root.configure(cursor='watch')
        self.root.update_idletasks()
        try:
            # Tab3 compares images at their ORIGINAL size: keep the full
            # 1028-row frame (only the beam-stop disk is zeroed), no cropping.
            image, meta = read_image(path, None, 0, crop_edf=False)
            if kind == 'raw':
                self.tab3_raw_image = image
                self.tab3_raw_meta = meta
                self.tab3_raw_path = path
                # The raw EDF is the profile base: reset slider ranges/polar grids.
                self._tab3_setup_cut_controls()
            else:
                self.tab3_denoised_image = image
                self.tab3_denoised_meta = meta
                self.tab3_denoised_path = path
            if kind != 'raw' and self.tab3_raw_image is not None \
                    and image.shape != self.tab3_raw_image.shape:
                messagebox.showwarning(
                    "Shape Mismatch",
                    f"{os.path.basename(path)} is {image.shape}, but the raw EDF "
                    f"is {self.tab3_raw_image.shape}.\nProfiles use the raw EDF as "
                    "the base; mismatched curves may be clipped or skipped.")
            names = [os.path.basename(p) if p else "-"
                     for p in (self.tab3_raw_path, self.tab3_denoised_path)]
            self.var_tab3_file_info.set(
                f"Raw: {names[0]}\nDenoised: {names[1]}")
            self._tab3_refresh_display()
            self._update_status(f"[Tab3] Loaded {kind}: {os.path.basename(path)}  "
                                f"{image.shape}")
        except Exception as e:
            messagebox.showerror("File Read Failed", f"{os.path.basename(path)}:\n{e}")
        finally:
            self.root.configure(cursor='')

    def _tab3_get_percentiles(self):
        """Return the user display percentiles (low, high) for Tab3."""
        try:
            low = float(self.var_tab3_range_low.get())
            high = float(self.var_tab3_range_high.get())
        except (TypeError, ValueError):
            low, high = 1.0, 99.0
        low = max(0.0, min(low, 99.9))
        high = max(low + 0.1, min(high, 100.0))
        return low, high

    def _tab3_get_range(self, mode=None):
        """Return the shared (vmin, vmax) for the raw/denoised 2D plots.

        Percentile mode: joint percentile over both loaded images.
        Value mode: direct numeric input (clamped, max > min enforced).
        The residual plot does NOT participate (it has its own symmetric
        range, see _tab3_display_residual).
        """
        if mode is None:
            mode = self.var_tab3_display_mode.get()
        images = [im for im in (self.tab3_raw_image, self.tab3_denoised_image)
                  if im is not None]
        if self.var_tab3_range_mode.get() == 'value':
            try:
                vmin = float(self.var_tab3_range_min.get())
                vmax = float(self.var_tab3_range_max.get())
            except (TypeError, ValueError):
                vmin, vmax = 0.0, 0.7
            vmin = max(0.0, min(vmin, 10.0))
            vmax = max(vmin + 1e-6, min(vmax, 10.0))
            if mode == 'log' and vmin <= 0:
                vmin = 1e-6
            return vmin, vmax
        # Percentile mode (default).
        if not images:
            return None, None
        finite = np.concatenate([im[np.isfinite(im)] for im in images])
        if finite.size == 0:
            return None, None
        low, high = self._tab3_get_percentiles()
        if mode == 'log':
            finite = finite[finite > 0]
            if finite.size == 0:
                return None, None
        vmin = float(np.percentile(finite, low))
        vmax = float(np.percentile(finite, high))
        if mode == 'log' and vmin <= 0:
            vmin = 1e-6
        return vmin, vmax

    def _tab3_auto_range(self):
        """Auto button: fill the Min/Max boxes from joint image percentiles."""
        images = [im for im in (self.tab3_raw_image, self.tab3_denoised_image)
                  if im is not None]
        if not images:
            messagebox.showwarning("Notice", "Import an EDF first")
            return
        finite = np.concatenate([im[np.isfinite(im)] for im in images])
        finite = finite[finite > 0] if finite.size else finite
        if finite.size == 0:
            messagebox.showwarning("Notice",
                                   "Loaded images contain no positive values")
            return
        vmin = float(np.percentile(finite, 0.5))
        vmax = float(np.percentile(finite, 99.5))
        self.var_tab3_range_min.set(max(0.0, vmin))
        self.var_tab3_range_max.set(max(vmin + 1e-6, vmax))
        self._update_status(f"[Tab3] Auto range: [{vmin:.4g}, {vmax:.4g}]")

    def _tab3_remove_cbar(self, key='both'):
        """Hide the dedicated Tab3 colorbar axes without deleting them.

        The Colorbar objects are reused via update_normal (see _remove_cbar).
        """
        if key in ('shared', 'both'):
            self.tab3_ax_cbar_shared.set_visible(False)
        if key in ('residual', 'both'):
            self.tab3_ax_cbar_res.set_visible(False)

    def _tab3_display_image(self, img, ax, title, cbar_key, mode=None):
        """Display one 2D image; raw/denoised share the same vmin/vmax."""
        if mode is None:
            mode = self.var_tab3_display_mode.get()
        ax.clear()
        ax.set_title(title, fontsize=18, fontweight='bold', fontfamily='Arial')

        cmap = self.var_tab3_cmap.get()
        vmin, vmax = self._tab3_get_range(mode)

        if mode == 'log':
            if vmin is not None and vmax is not None and vmax > vmin:
                norm = LogNorm(vmin=vmin, vmax=vmax)
                display_data = np.nan_to_num(img, nan=0.0,
                                             posinf=vmax, neginf=0.0)
            else:
                norm = None
                display_data = np.nan_to_num(img, nan=0.0,
                                             posinf=1.0, neginf=0.0)
        else:
            if vmin is not None and vmax is not None and vmax > vmin:
                norm = Normalize(vmin=vmin, vmax=vmax)
            else:
                norm = None
            display_data = img

        cmap_obj = plt.get_cmap(cmap).copy()
        if mode == 'log':
            cmap_obj.set_over('magenta')
        else:
            cmap_obj.set_under('gray')
            cmap_obj.set_over('magenta')

        im = ax.imshow(display_data, cmap=cmap_obj, norm=norm, aspect='equal',
                       interpolation='bilinear', origin='upper')
        # Reuse the colorbar object (see _remove_cbar).
        ax_cbar = (self.tab3_ax_cbar_shared if cbar_key == 'shared'
                   else self.tab3_ax_cbar_res)
        if cbar_key == 'shared':
            if self.tab3_shared_cbar is None:
                self.tab3_shared_cbar = self.tab3_fig.colorbar(
                    im, cax=ax_cbar, location='left')
            else:
                self.tab3_shared_cbar.update_normal(im)
            cbar = self.tab3_shared_cbar
        else:
            if self.tab3_residual_cbar is None:
                self.tab3_residual_cbar = self.tab3_fig.colorbar(
                    im, cax=ax_cbar, location='left')
            else:
                self.tab3_residual_cbar.update_normal(im)
            cbar = self.tab3_residual_cbar
        ax_cbar.set_visible(True)
        cbar.set_label('Intensity (log scale)' if mode == 'log' else 'Intensity',
                       fontsize=13.5, fontfamily='Arial')
        cbar.ax.tick_params(labelsize=12, labelfontfamily='Arial')
        ax.tick_params(labelsize=12, labelfontfamily='Arial')
        ax.set_xlabel(f"W = {img.shape[1]} px", fontsize=13.5,
                      fontfamily='Arial')
        ax.set_ylabel(f"H = {img.shape[0]} px", fontsize=13.5,
                      fontfamily='Arial')
        # Tab3 shows images at their original size (1028x512 for EDFs), so the
        # axes auto-fit each frame instead of locking a fixed detector window.
        self._tab3_draw_overlays(ax)

    def _tab3_display_residual(self):
        """Plot the log-domain residual (same math as Tab1) on the third 2D
        slot with a diverging colormap."""
        ax = self.tab3_ax_residual
        ax.clear()
        ax.set_title("Residual (log)", fontsize=18, fontweight='bold',
                     fontfamily='Arial')

        noisy = np.log1p(np.nan_to_num(self.tab3_raw_image, nan=0.0,
                                       posinf=0.0, neginf=0.0))
        denoised = np.log1p(np.nan_to_num(self.tab3_denoised_image, nan=0.0,
                                          posinf=0.0, neginf=0.0))
        residual = denoised - noisy
        finite = residual[np.isfinite(residual)]
        if finite.size == 0:
            norm = None
            display_data = residual
        else:
            low, high = self._tab3_get_percentiles()
            clip = max(abs(float(np.percentile(finite, low))),
                       abs(float(np.percentile(finite, high))), 1e-12)
            norm = Normalize(vmin=-clip, vmax=clip)
            display_data = np.clip(residual, -clip, clip)

        im = ax.imshow(display_data, cmap=plt.get_cmap('coolwarm'), norm=norm,
                       aspect='equal', interpolation='bilinear', origin='upper')
        if self.tab3_residual_cbar is None:
            self.tab3_residual_cbar = self.tab3_fig.colorbar(
                im, cax=self.tab3_ax_cbar_res, location='left')
        else:
            self.tab3_residual_cbar.update_normal(im)
        self.tab3_ax_cbar_res.set_visible(True)
        cbar = self.tab3_residual_cbar
        cbar.set_label('dlog(1+I)', fontsize=13.5, fontfamily='Arial')
        cbar.ax.tick_params(labelsize=12, labelfontfamily='Arial')
        ax.tick_params(labelsize=12, labelfontfamily='Arial')
        ax.set_xlabel(f"W = {self.tab3_raw_image.shape[1]} px", fontsize=13.5,
                      fontfamily='Arial')
        ax.set_ylabel(f"H = {self.tab3_raw_image.shape[0]} px", fontsize=13.5,
                      fontfamily='Arial')
        # Tab3 keeps the original frame size; axes auto-fit (see _tab3_display_image).
        self._tab3_draw_overlays(ax)

    def _tab3_placeholder(self, ax, title):
        """Placeholder title + hint on an axis with no image loaded."""
        ax.clear()
        ax.set_title(title, fontsize=18, fontweight='bold', fontfamily='Arial')
        ax.text(0.5, 0.5, 'Not loaded', transform=ax.transAxes,
                ha='center', va='center', fontsize=21, color='gray',
                fontfamily='Arial')

    def _tab3_refresh_display(self):
        """Redraw the three 2D plots (raw / denoised / residual) and profiles."""
        if self.tab3_raw_image is not None:
            self._tab3_display_image(self.tab3_raw_image, self.tab3_ax_raw,
                                     "Raw", 'shared')
        else:
            self._tab3_placeholder(self.tab3_ax_raw, "Raw")
        if self.tab3_denoised_image is not None:
            self._tab3_display_image(self.tab3_denoised_image,
                                     self.tab3_ax_denoised,
                                     "Denoised", 'shared')
        else:
            self._tab3_placeholder(self.tab3_ax_denoised, "Denoised")
        if self.tab3_raw_image is not None and self.tab3_denoised_image is not None:
            self._tab3_display_residual()
        else:
            self._tab3_placeholder(self.tab3_ax_residual, "Residual (log)")
        self._tab3_draw_profiles()
        self.tab3_canvas.draw()

    def _tab3_setup_cut_controls(self):
        """Set Tab3 slider ranges based on the raw EDF size (raw is the base)."""
        if self.tab3_raw_image is None:
            return
        height, width = self.tab3_raw_image.shape
        self.scale_tab3_cut_h.configure(from_=0, to=height - 1)
        self.var_tab3_cut_pos_h.set((height - 1) / 2)
        self.scale_tab3_cut_v.configure(from_=0, to=width - 1)
        self.var_tab3_cut_pos_v.set((width - 1) / 2)
        self.var_tab3_cut_pos_h_text.set(f"Row: {int(round(self.var_tab3_cut_pos_h.get()))}")
        self.var_tab3_cut_pos_v_text.set(f"Col: {int(round(self.var_tab3_cut_pos_v.get()))}")

        # R/A: precompute the polar grids for the raw beam center and set the
        # radius slider range (keep the user's angle/radius, clamped).
        center = self._tab3_get_profile_center()
        self.tab3_r_grid, self.tab3_theta_grid = polar_grids((height, width), center)
        rmax = self._tab3_rmax()
        self.scale_tab3_ra_radius.configure(from_=0, to=max(rmax, 1e-6))
        self.var_tab3_azimuthal_radius.set(
            min(self.var_tab3_azimuthal_radius.get(), rmax))
        self.var_tab3_radial_angle.set(np.clip(self.var_tab3_radial_angle.get(), 0.0, 360.0))
        self.var_tab3_radial_angle_text.set(f"Angle: {self._tab3_radial_angle():.0f} deg")
        self.var_tab3_azimuthal_radius_text.set(
            f"Radius: {self._tab3_azimuthal_radius():.1f} px")

    def _tab3_cut_pos(self, direction):
        """Profile position for the given direction, clamped to the raw bounds."""
        if self.tab3_raw_image is None:
            return 0
        height, width = self.tab3_raw_image.shape
        if direction == 'horizontal':
            pos = int(round(self.var_tab3_cut_pos_h.get()))
            limit = height - 1
        else:
            pos = int(round(self.var_tab3_cut_pos_v.get()))
            limit = width - 1
        return max(0, min(pos, limit))

    def _tab3_is_radial_mode(self):
        return self.var_tab3_profile_mode.get().startswith('Radial')

    def _tab3_get_profile_center(self):
        """Beam center in display coordinates for the raw EDF."""
        if self.tab3_raw_image is None:
            return None
        return display_center(self.tab3_raw_meta, self.tab3_raw_image.shape)

    def _tab3_rmax(self):
        """Largest integer radius fully inside the raw image (inscribed circle)."""
        if self.tab3_raw_image is None:
            return 0.0
        h, w = self.tab3_raw_image.shape
        cx, cy = self._tab3_get_profile_center()
        return float(np.floor(min(cx, w - 1 - cx, cy, h - 1 - cy)))

    def _tab3_radial_angle(self):
        return float(np.clip(self.var_tab3_radial_angle.get(), 0.0, 360.0))

    def _tab3_azimuthal_radius(self):
        return float(np.clip(self.var_tab3_azimuthal_radius.get(), 0.0,
                             self._tab3_rmax()))

    def _tab3_ray_endpoint(self, cx, cy, angle_deg):
        """One-sided ray endpoint from (cx, cy) to the raw image border."""
        h, w = self.tab3_raw_image.shape
        a = np.deg2rad(angle_deg)
        ca, sa = np.cos(a), np.sin(a)
        t = []
        if ca > 1e-12:
            t.append((w - 1 - cx) / ca)
        elif ca < -1e-12:
            t.append((0 - cx) / ca)
        if sa > 1e-12:
            t.append((0 - cy) / (-sa))          # upward: dy = -t*sa
        elif sa < -1e-12:
            t.append((h - 1 - cy) / (-sa))      # downward
        tmax = max(t) if t else 0.0
        return cx + tmax * ca, cy - tmax * sa

    def _tab3_draw_cut_line(self, ax, direction):
        """Draw the profile line for the given direction on a 2D plot."""
        if self.tab3_raw_image is None:
            return
        pos = self._tab3_cut_pos(direction)
        if direction == 'horizontal':
            line = ax.axhline(y=pos, color='cyan', lw=1.0, ls='--')
        else:
            line = ax.axvline(x=pos, color='yellow', lw=1.0, ls='--')
        self.tab3_cut_lines[(id(ax), direction)] = line

    def _tab3_draw_overlays(self, ax):
        """Draw the mode-appropriate overlay (H/V cut lines or R/A markers)."""
        if self._tab3_is_radial_mode():
            self._tab3_draw_ra_markers(ax)
        else:
            self._tab3_draw_cut_line(ax, 'horizontal')
            self._tab3_draw_cut_line(ax, 'vertical')

    def _tab3_draw_ra_markers(self, ax):
        """Center marker + azimuthal-radius circle + radial-angle ray."""
        if self.tab3_raw_image is None:
            return
        cx, cy = self._tab3_get_profile_center()
        radius = self._tab3_azimuthal_radius()
        angle = self._tab3_radial_angle()

        center_marker, = ax.plot([cx], [cy], marker='+', color='lime',
                                 ms=11, mew=1.5, ls='none')
        self.tab3_cut_markers[(id(ax), 'center')] = center_marker

        circ = Circle((cx, cy), radius, fill=False, edgecolor='cyan',
                      lw=1.0, ls='--')
        ax.add_patch(circ)
        self.tab3_cut_markers[(id(ax), 'circle')] = circ

        x1, y1 = self._tab3_ray_endpoint(cx, cy, angle)
        ray, = ax.plot([cx, x1], [cy, y1], color='yellow', lw=1.0, ls='--')
        self.tab3_cut_markers[(id(ax), 'ray')] = ray

    def _tab3_refresh_ra_markers(self):
        """In-place update of R/A markers after slider drags (no ax.clear)."""
        if self.tab3_raw_image is None:
            return
        cx, cy = self._tab3_get_profile_center()
        radius = self._tab3_azimuthal_radius()
        angle = self._tab3_radial_angle()
        for (_, kind), art in list(self.tab3_cut_markers.items()):
            if art.axes is None:
                continue
            if kind == 'circle':
                if art not in art.axes.patches:
                    continue
                art.set_center((cx, cy))
                art.set_radius(radius)
            elif kind == 'center':
                if art not in art.axes.lines:
                    continue
                art.set_data([cx], [cy])
            elif kind == 'ray':
                if art not in art.axes.lines:
                    continue
                x1, y1 = self._tab3_ray_endpoint(cx, cy, angle)
                art.set_data([cx, x1], [cy, y1])

    def _tab3_refresh_1d_only(self):
        """Update the profile lines and both 1D curves after a slider change."""
        if self.tab3_raw_image is None:
            return
        if self._tab3_is_radial_mode():
            self._tab3_refresh_ra_markers()
        else:
            for (_, direction), line in list(self.tab3_cut_lines.items()):
                if line.axes is None or line not in line.axes.lines:
                    continue
                if direction == 'horizontal':
                    pos = self._tab3_cut_pos('horizontal')
                    line.set_ydata([pos, pos])
                else:
                    pos = self._tab3_cut_pos('vertical')
                    line.set_xdata([pos, pos])
        self._tab3_draw_profiles()
        self.tab3_canvas.draw()

    def _tab3_draw_profile(self, direction):
        """Plot the 1D intensity curves (Raw/Denoised) for one profile.

        Profiles are based on the raw EDF; the denoised image is clipped to
        its own shape for H/V cuts and skipped when the shape differs for the
        radial/azimuthal profiles.
        """
        if self.tab3_raw_image is None:
            self._tab3_set_profile_titles()
            return
        log_mode = self.var_tab3_display_mode.get() == 'log'

        def to_y(vals):
            return np.log1p(np.maximum(vals, 0)) if log_mode else vals

        series = []   # (label, color, x, y)
        images = (('Raw', 'tab:blue', self.tab3_raw_image),
                  ('Denoised', 'tab:red', self.tab3_denoised_image))
        if direction == 'horizontal':
            ax = self.tab3_ax_profile_h
            pos = self._tab3_cut_pos('horizontal')
            title = f"Horizontal Profile (row {pos})"
            for label, color, img in images:
                if img is None:
                    continue
                p = min(pos, img.shape[0] - 1)
                vals = img[p, :]
                series.append((label, color, np.arange(vals.size), to_y(vals)))
            xlabel = 'Pixel index'
        elif direction == 'vertical':
            ax = self.tab3_ax_profile_v
            pos = self._tab3_cut_pos('vertical')
            title = f"Vertical Profile (col {pos})"
            for label, color, img in images:
                if img is None:
                    continue
                p = min(pos, img.shape[1] - 1)
                vals = img[:, p]
                series.append((label, color, np.arange(vals.size), to_y(vals)))
            xlabel = 'Pixel index'
        elif direction == 'radial':
            ax = self.tab3_ax_profile_h
            angle = self._tab3_radial_angle()
            title = f"Radial Profile (angle {angle:.0f} deg)"
            for label, color, img in images:
                if img is None or img.shape != self.tab3_raw_image.shape:
                    continue
                x, vals = radial_profile(img, self.tab3_r_grid,
                                         self.tab3_theta_grid, angle,
                                         delta=0.5, rmax=self._tab3_rmax())
                series.append((label, color, x, to_y(vals)))
            xlabel = 'Radius (px)'
        else:  # 'azimuthal'
            ax = self.tab3_ax_profile_v
            radius = self._tab3_azimuthal_radius()
            title = f"Azimuthal Profile (r = {radius:.1f} px)"
            for label, color, img in images:
                if img is None or img.shape != self.tab3_raw_image.shape:
                    continue
                x, vals = azimuthal_profile(img, self.tab3_r_grid,
                                            self.tab3_theta_grid, radius)
                series.append((label, color, x, to_y(vals)))
            xlabel = 'Angle (deg, CCW from +x)'

        ax.clear()
        for label, color, x, y in series:
            ax.plot(x, y, label=label, color=color, lw=1.0)
        ax.set_title(title, fontsize=15, fontweight='bold', fontfamily='Arial')
        ax.set_xlabel(xlabel, fontsize=13.5, fontfamily='Arial')
        ax.set_ylabel('log1p(I)' if log_mode else 'Intensity',
                      fontsize=13.5, fontfamily='Arial')
        if direction == 'azimuthal':
            ax.set_xlim(0, 360)      # fixed 0~360, NaN edges must not shift the axis
        ax.grid(alpha=0.3)
        if series:
            ax.legend(prop={'family': 'Arial', 'size': 12})
        ax.tick_params(labelsize=12, labelfontfamily='Arial')

    def _tab3_draw_profiles(self):
        """Draw both 1D profile plots for the active mode."""
        if self._tab3_is_radial_mode():
            self._tab3_draw_profile('radial')
            self._tab3_draw_profile('azimuthal')
        else:
            self._tab3_draw_profile('horizontal')
            self._tab3_draw_profile('vertical')

    def _tab3_set_profile_titles(self):
        """Mode-aware placeholder titles for the two Tab3 profile axes."""
        if self._tab3_is_radial_mode():
            titles = ("Radial Profile", "Azimuthal Profile")
        else:
            titles = ("Horizontal Profile", "Vertical Profile")
        for ax, t in zip((self.tab3_ax_profile_h, self.tab3_ax_profile_v), titles):
            ax.clear()
            ax.set_title(t, fontsize=15, fontweight='bold', fontfamily='Arial')

    def _tab3_on_cut_change(self):
        """Refresh the position labels and curves when either profile slider changes."""
        if self.tab3_raw_image is None:
            return
        self.var_tab3_cut_pos_h_text.set(f"Row: {self._tab3_cut_pos('horizontal')}")
        self.var_tab3_cut_pos_v_text.set(f"Col: {self._tab3_cut_pos('vertical')}")
        self._tab3_refresh_1d_only()

    def _tab3_on_ra_change(self):
        """Refresh R/A labels and plots when the angle/radius sliders change."""
        if self.tab3_raw_image is None:
            return
        self.var_tab3_radial_angle_text.set(f"Angle: {self._tab3_radial_angle():.0f} deg")
        self.var_tab3_azimuthal_radius_text.set(
            f"Radius: {self._tab3_azimuthal_radius():.1f} px")
        self._tab3_refresh_1d_only()

    def _tab3_apply_profile_mode_visibility(self):
        """Show the H/V slider block or the R/A slider block (mutually exclusive)."""
        radial = self._tab3_is_radial_mode()
        hv_widgets = (self.lbl_tab3_cut_h, self.lbl_tab3_cut_h_text,
                      self.scale_tab3_cut_h,
                      self.lbl_tab3_cut_v, self.lbl_tab3_cut_v_text,
                      self.scale_tab3_cut_v)
        ra_widgets = (self.lbl_tab3_ra_angle, self.lbl_tab3_ra_angle_text,
                      self.scale_tab3_ra_angle,
                      self.lbl_tab3_ra_radius, self.lbl_tab3_ra_radius_text,
                      self.scale_tab3_ra_radius)
        for w in hv_widgets:
            w.grid_remove() if radial else w.grid()
        for w in ra_widgets:
            w.grid() if radial else w.grid_remove()

    def _tab3_on_profile_mode_change(self):
        """Switch between H/V and Radial/Azimuthal profile mode."""
        self._tab3_apply_profile_mode_visibility()
        if self.tab3_raw_image is not None:
            self._tab3_setup_cut_controls()
            self._tab3_refresh_display()
        self.tab3_canvas.draw()

    def _tab3_on_range_mode_change(self):
        """Toggle between the percentile and value range controls."""
        value_mode = self.var_tab3_range_mode.get() == 'value'
        for w in self._tab3_pct_widgets:
            w.grid_remove() if value_mode else w.grid()
        for w in self._tab3_value_widgets:
            w.grid() if value_mode else w.grid_remove()
        self._tab3_refresh_display()

    def _on_tab3_export_figure(self):
        """Export the Tab3 figure as SVG or PDF at the requested DPI."""
        try:
            dpi = int(self.var_tab3_export_dpi.get())
        except (TypeError, ValueError):
            dpi = 300
        dpi = max(72, min(dpi, 1200))

        if self.tab3_raw_path:
            base = os.path.splitext(os.path.basename(self.tab3_raw_path))[0]
            default_name = f"{base}_compare_residual"
        else:
            default_name = "saxs_compare_figure"

        path = filedialog.asksaveasfilename(
            title="Export Figure",
            initialdir=os.getcwd(),
            initialfile=default_name,
            defaultextension=".svg",
            filetypes=[
                ("SVG Vector Image", "*.svg"),
                ("PDF Vector Document", "*.pdf"),
            ]
        )
        if not path:
            return

        try:
            ext = os.path.splitext(path)[1].lower()
            file_format = 'svg' if ext == '.svg' else 'pdf'
            self.tab3_fig.savefig(path, format=file_format, dpi=dpi)
            self._update_status(
                f"[Tab3] Figure exported: {os.path.basename(path)} "
                f"({file_format}, {dpi} dpi)"
            )
        except Exception as e:
            messagebox.showerror("Export Failed", f"{e}")

    def _on_tab3_clear(self):
        """Clear the two Tab3 slots and redraw placeholders."""
        self.tab3_raw_image = None
        self.tab3_denoised_image = None
        self.tab3_raw_meta = None
        self.tab3_denoised_meta = None
        self.tab3_raw_path = None
        self.tab3_denoised_path = None
        self._tab3_remove_cbar('both')
        self.tab3_cut_lines = {}
        self.tab3_cut_markers = {}
        self.tab3_r_grid = None
        self.tab3_theta_grid = None
        self.var_tab3_file_info.set("No files loaded")
        self._tab3_refresh_display()
        self._update_status("[Tab3] Cleared")

    def _on_mousewheel(self, event):
        """Scroll whichever left column is under the pointer (single global
        binding; per-widget bindings would add seconds of Tk startup time)."""
        x, y = event.x_root, event.y_root
        for canvas in (self._left_canvas_t1, self._left_canvas_t2,
                       self._left_canvas_t3):
            if canvas.winfo_viewable():
                x0 = canvas.winfo_rootx()
                y0 = canvas.winfo_rooty()
                w, h = canvas.winfo_width(), canvas.winfo_height()
                if x0 <= x <= x0 + w and y0 <= y <= y0 + h:
                    canvas.yview_scroll(-1 * (event.delta // 120), 'units')
                    return

    def _on_tab_changed(self, event):
        """Redraw the canvas of the tab that just became visible."""
        try:
            selected = self.notebook.select()
            if selected and str(selected) == str(self.tab1_frame):
                self.canvas.draw()
            elif selected and str(selected) == str(self.tab2_frame):
                self.tab2_canvas.draw()
            elif selected and str(selected) == str(self.tab3_frame):
                self.tab3_canvas.draw()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 6. CLI command-line batch inference mode
# ---------------------------------------------------------------------------

def cli_main():
    """Command-line batch inference entry point.

    Usage:
      # GUI mode (default)
      python inference_gui.py

      # CLI batch processing
      python inference_gui.py --cli \\
          --checkpoint models/best.pth \\
          --input /path/to/files/ \\
          --output /path/to/results/ \\
          [--tile-size 512] [--device cpu] [--no-amp] [--pattern "*.h5"] [--frame 0]

      # CLI single file
      python inference_gui.py --cli \\
          --checkpoint models/best.pth \\
          --input single_file.h5 \\
          --output result.npy
    """
    import argparse
    import glob as glob_mod

    parser = argparse.ArgumentParser(
        description="SAXS Denoising - Noise2Noise batch inference (CLI mode)")

    parser.add_argument('--cli', action='store_true',
                        help='enable command-line mode (no GUI)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='model checkpoint path (.pth)')
    parser.add_argument('--input', type=str, required=True,
                        help='input file/folder path')
    parser.add_argument('--output', type=str, required=True,
                        help='output file/folder path (folder for batch, file for single)')
    parser.add_argument('--init-ch', type=int, default=32,
                        help='model init_ch (default 32)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu', 'mps'],
                        help='inference device (default auto: best available)')
    parser.add_argument('--tile-size', type=int, default=0,
                        help='tile size (0=auto, -1=no tiling, or explicit like 512/256)')
    parser.add_argument('--tile-overlap', type=int, default=32,
                        help='tile overlap in pixels (default 32)')
    parser.add_argument('--no-amp', action='store_true',
                        help='disable automatic mixed precision (recommended to enable for CPU inference)')
    parser.add_argument('--pattern', type=str, default='*',
                        help='file match pattern (default *), e.g. "*.h5" or "*.npy"')
    parser.add_argument('--h5-dataset', type=str, default=None,
                        help='HDF5 dataset name (only needed when the file has multiple datasets)')
    parser.add_argument('--frame', type=int, default=0,
                        help='frame index for 3D/4D HDF5 or npy files (default 0)')

    args = parser.parse_args()

    # Start the GUI unless --cli is given.
    if not args.cli:
        return False  # Let the caller start the GUI.

    # ===================== CLI mode =====================
    print("\n" + "=" * 60)
    print("SAXS Denoising - CLI batch inference")
    print("=" * 60)

    # ---- Device ----
    if args.device == 'auto':
        device, device_info = get_best_device(verbose=True)
    elif args.device == 'cuda':
        if not torch.cuda.is_available():
            print("Error: --device cuda specified but CUDA is not available")
            sys.exit(1)
        device = torch.device('cuda')
        device_info = {'backend': 'CUDA', 'name': torch.cuda.get_device_name(0),
                       'memory_gb': torch.cuda.get_device_properties(0).total_memory / 1e9,
                       'supports_bf16': torch.cuda.get_device_capability(0)[0] >= 8,
                       'supports_fp16': torch.cuda.get_device_capability(0)[0] >= 7}
    elif args.device == 'mps':
        device = torch.device('mps')
        device_info = {'backend': 'MPS', 'name': 'Apple Silicon',
                       'memory_gb': None, 'supports_bf16': False, 'supports_fp16': False}
    elif args.device == 'cpu':
        device = torch.device('cpu')
        device_info = {'backend': 'CPU', 'name': 'CPU',
                       'memory_gb': None, 'supports_bf16': False, 'supports_fp16': False}
    else:
        raise ValueError(f"Unknown device: {args.device}")

    # ---- Load model ----
    print(f"\nLoading model: {args.checkpoint}")
    t0 = time.time()
    model = load_model(args.checkpoint, device, init_ch=args.init_ch)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # ---- Tiling settings ----
    use_amp = not args.no_amp
    if args.tile_size == 0:
        tile_size = 'auto'
    elif args.tile_size < 0:
        tile_size = None
    else:
        tile_size = args.tile_size

    if tile_size == 'auto':
        actual_tile = 512 if detect_low_memory_gpu(device_info) else None
        if device_info['backend'] == 'CPU':
            print(f"\nTiled inference: auto-enabled (tile=512, overlap={args.tile_overlap})")
        elif actual_tile is not None:
            mem_str = f"{device_info.get('memory_gb', 0):.1f} GB" if device_info.get('memory_gb') else 'unknown'
            print(f"\nTiled inference: auto-enabled (memory: {mem_str} < 4GB threshold, tile={actual_tile})")
        else:
            print(f"\nTiled inference: direct inference (enough memory)")
    elif tile_size is not None:
        print(f"\nTiled inference: manual tile={tile_size}, overlap={args.tile_overlap}")
    else:
        print(f"\nTiled inference: disabled (--tile-size -1)")

    if use_amp and device_info.get('supports_fp16'):
        dtype_name = 'bfloat16' if device_info.get('supports_bf16') else 'float16'
        print(f"Mixed precision: {dtype_name} (AMP)")
    else:
        print(f"Mixed precision: disabled (float32)")

    # ---- Collect files ----
    input_path = args.input
    if os.path.isdir(input_path):
        files = sorted([
            os.path.join(input_path, f) for f in os.listdir(input_path)
            if _get_file_ext(f) in SUPPORTED_IMAGE_EXTS
        ])
        # Apply the pattern filter.
        if args.pattern != '*':
            import fnmatch
            files = [f for f in files if fnmatch.fnmatch(os.path.basename(f), args.pattern)]
        if not files:
            print(f"\nError: no matching image files in directory: {input_path}")
            sys.exit(1)
    else:
        if not os.path.isfile(input_path):
            print(f"\nError: file not found: {input_path}")
            sys.exit(1)
        files = [input_path]

    print(f"\nFiles to process: {len(files)}")
    for f in files[:10]:
        print(f"  - {os.path.basename(f)}")
    if len(files) > 10:
        print(f"  ... and {len(files) - 10} more files")

    # ---- Output directory ----
    if len(files) > 1 or os.path.isdir(args.output):
        os.makedirs(args.output, exist_ok=True)
        output_is_dir = True
    else:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        output_is_dir = False

    # ---- Batch inference ----
    print(f"\nStarting inference ...")
    total_time = 0
    success = 0

    for i, fpath in enumerate(files):
        print(f"\n[{i+1}/{len(files)}] {os.path.basename(fpath)}")
        try:
            # Read the image.
            t1 = time.time()
            img, meta = read_image(fpath, h5_dataset=args.h5_dataset, frame=args.frame)
            print(f"  Read: {img.shape}  [{meta['min']:.2f}, {meta['max']:.2f}]")

            # Denoise.
            denoised = denoise_image(
                model, img, device, device_info=device_info,
                tile_size=tile_size, tile_overlap=args.tile_overlap,
                use_amp=use_amp,
                # Kept in the normalized domain; the physical-count restore
                # happens at save time (restore_edf_output).
                restore_mask=meta.get('edf_beam_mask')
            )
            elapsed = time.time() - t1
            total_time += elapsed
            print(f"  Done: {elapsed:.1f}s  ->  [{denoised.min():.2f}, {denoised.max():.2f}]")

            # Save.
            fname = os.path.basename(fpath)
            ext_in = _get_file_ext(fname)
            base = fname[:-len(ext_in)] if ext_in in SUPPORTED_IMAGE_EXTS \
                else os.path.splitext(fname)[0]
            is_edf = meta.get('format') in ('.edf', '.edf.gz')

            if output_is_dir:
                if is_edf:
                    out_path = os.path.join(args.output, f"{base}_denoised.edf")
                else:
                    out_path = os.path.join(args.output, f"{base}_denoised.npy")
            else:
                out_path = args.output

            ext = os.path.splitext(out_path)[1].lower()
            if ext == '.edf':
                try:
                    from fabio.edfimage import edfimage
                    if is_edf:
                        eh = meta.get('edf_header') or {}
                        orig_title = eh.get('title') or eh.get('Title') or base
                        full, header = restore_edf_output(
                            denoised, meta,
                            title=f"{orig_title} [SAXS denoised]")
                        edfimage(data=full, header=header).write(out_path)
                    else:
                        edfimage(data=denoised, header={
                            'Title': f"SAXS denoising - {fname}",
                        }).write(out_path)
                except ImportError:
                    np.save(out_path + '.npy', denoised)
                    print("  Note: fabio is not installed, saved as .npy instead")
            elif ext == '.npy':
                np.save(out_path, denoised)
            elif ext in ('.tiff', '.tif'):
                try:
                    import tifffile
                    tifffile.imwrite(out_path, denoised)
                except ImportError:
                    np.save(out_path + '.npy', denoised)
                    print(f"  Note: tifffile is not installed, saved as .npy instead")
            else:
                np.save(out_path, denoised)

            success += 1

        except Exception as e:
            print(f"  X failed: {e}")
            if hasattr(args, 'verbose') and args.verbose:
                traceback.print_exc()
            continue

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"Inference complete: {success}/{len(files)} files")
    print(f"Total time: {total_time:.1f}s | average: {total_time/max(success,1):.1f}s/file")
    print(f"Output directory: {args.output}")
    print(f"{'='*60}")

    return True


# ---------------------------------------------------------------------------
# 7. Main entry
# ---------------------------------------------------------------------------

def main():
    # Try CLI mode first (parse --cli).
    if '--cli' in sys.argv:
        cli_main()
        return

    # Default to the GUI.
    root = tk.Tk()

    # Enable high-DPI support (optional).
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    # Set the theme style.
    style = ttk.Style()
    try:
        style.theme_use('clam')  # A theme that looks decent across platforms.
    except Exception:
        pass

    app = DenoiseApp(root)

    def on_close():
        # Ask the batch worker to stop before tearing down the window
        # (the daemon thread exits with the process anyway).
        app._batch_stop = True
        root.destroy()

    root.protocol('WM_DELETE_WINDOW', on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
