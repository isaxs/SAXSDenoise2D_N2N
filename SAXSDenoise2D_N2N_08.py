import os
import sys
import random
import time
import math
import threading
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import h5py
import json
from torch.utils.tensorboard import SummaryWriter

# v04 changes for the RTX 3060 (12 GB):
# - bf16 autocast by default: fp16 overflows in this unnormalized residual U-Net
#   (decoder activations exceed 65504); bf16 keeps fp32-like range and is ~1% slower
# - log-domain training loss; no expm1/log1p roundtrip
# - float32 in-RAM dataset, num_workers=0 (Windows spawn would duplicate arrays)
# - channels_last inputs/model for faster cuDNN convs
# - loss accumulation without per-batch .item() GPU sync
# - torch.compile disabled by default: this host has no Triton, and the failure
#   surfaces only at the first forward call, crashing the v03 warmup
# v06 changes:
# - per-epoch JSONL diagnostics: train/val loss, grad/weight norms, correction
#   stats, attention-gate alpha stats, identity baseline; no architecture change
# v07 changes:
# - weight_l2 is the true L2 norm (was sum of squares), so update/weight ratio
#   is meaningful
# - weight snapshots now capture grads before zero_grad
# v08 changes:
# - dataset is stored in dataset/train, dataset/val and dataset/test
#   subdirectories; the split is fixed on disk instead of recomputed at startup
# - full English/ASCII source (comments, docstrings, messages, GUI strings)
# - cosine annealing with warm restarts (T_0=50, T_mult=2, eta_min=1e-6) and
#   early stopping patience 40 over 500 epochs
# - periodic test loss evaluation with a persistent train/val/test loss curve
#   plot that survives checkpoint resume

# ---------------------------- 1. Image Cropping ----------------------------
def crop_1028_to_1024(image):
    return image[..., 2:-2, :]

# ---------------------------- 2. In-Memory HDF5 Dataset ----------------------------
class Noise2NoiseH5Dataset(Dataset):
    def __init__(self, h5_paths, crop_func=None):
        self.samples = []
        t0 = time.time()
        self.images = []
        self.pairs = []
        for fi, path in enumerate(h5_paths):
            with h5py.File(path, 'r') as f:
                imgs = f['images'][:]
                if crop_func is not None:
                    imgs = crop_func(imgs)
                # Keep float32 in RAM; float64 doubles the preload and forces a
                # per-sample float64->float32 copy in __getitem__.
                imgs = np.ascontiguousarray(imgs, dtype=np.float32)
                imgs *= np.float32(1000.0)
                self.images.append(imgs)
                prs = f['pairs'][:]
                self.pairs.append(prs)
                for pi in range(prs.shape[0]):
                    self.samples.append((fi, pi))
        elapsed = time.time() - t0
        total_images = sum(len(imgs) for imgs in self.images)
        total_pairs = len(self.samples)
        size_gb = sum(imgs.nbytes for imgs in self.images) / 1e9
        print(f"  Preloaded: {len(h5_paths)} files, {total_images} images, "
              f"{total_pairs} pairs, {size_gb:.2f} GB, {elapsed:.1f}s")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_idx, pair_idx = self.samples[idx]
        idx_i, idx_j = self.pairs[file_idx][pair_idx]
        img_i = self.images[file_idx][idx_i]
        img_j = self.images[file_idx][idx_j]
        # Return (1, H, W).
        return (torch.from_numpy(img_i).unsqueeze(0),
                torch.from_numpy(img_j).unsqueeze(0))

# ---------------------------- 3. SAXS-specific Attention U-Net ----------------------------
# Designed around three core SAXS properties:
#   (a) intensity spans many orders of magnitude (10^0 ~ 10^5+) -> log-domain
#       operations + multi-scale receptive fields
#   (b) signal-dependent noise (high SNR in bright regions, low SNR in faint
#       regions) -> Attention Gate adapts per location
#   (c) sharp peaks and diffuse scattering coexist -> ASPP multi-scale
#       bottleneck captures local and global features
#
# Noise2Noise theory: MSE loss converges to E[clean | noisy], the statistically
# optimal denoiser.

class ChannelAttention(nn.Module):
    """Squeeze-and-excitation channel attention: learns the importance of each feature channel."""
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
    """SiLU activation + channel-attention residual block.

    SiLU's smooth gradients help handle the wide dynamic range of SAXS data.
    """
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
    """Multi-scale dilated convolution bottleneck (simplified DenseASPP).

    Runs parallel residual blocks with dilation in {1, 2, 4, 8} and fuses the
    multi-scale features, letting one forward pass see:
      - dilation=1: local sharp diffraction peaks
      - dilation=2,4: mid-scale scattering features
      - dilation=8: slowly varying global diffuse background

    This matters for SAXS data that contains both sharp Bragg peaks and broad
    diffuse scattering.
    """
    def __init__(self, channels):
        super().__init__()
        self.branch1 = ImprovedResidualBlock(channels, dilation=1)
        self.branch2 = ImprovedResidualBlock(channels, dilation=2)
        self.branch4 = ImprovedResidualBlock(channels, dilation=4)
        self.branch8 = ImprovedResidualBlock(channels, dilation=8)
        # Fuse the 4 branches back to the original channel count.
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
        return x + self.fusion(fused)  # Residual connection keeps the gradient flow.


class AttentionGate(nn.Module):
    """Attention gate (Oktay et al., MIDL 2018).

    A natural fit for SAXS's signal-dependent noise:
      the decoder provides global semantics (which regions are signal), the
      encoder provides local detail,
      the gate outputs a [0, 1] weight map:
        - high-intensity diffraction peaks -> alpha ~= 1 (keep detail, light denoising)
        - low-intensity diffuse scattering -> alpha ~= 0.2-0.5 (rely on decoder, strong denoising)
        - pure background noise -> alpha ~= 0 (fully suppressed)

    This adaptive mechanism is central for SAXS data spanning several orders of
    magnitude.
    """
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
    """Upsample -> attention-gated skip connection -> residual block -> output."""
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
    """SAXS-specific Attention U-Net: log domain + multi-scale + attention gates.

    Pipeline:
      input x (raw intensity, range 0 ~ 10^5+)
        | log(1 + x)        compress magnitude range, stabilize gradients
        | 4-level encoder   extract multi-scale features
        | ASPP bottleneck   fuse peak / mid-range / global features
        | 4-level decoder + AttnGate
                            adaptive reconstruction (low intensity relies on semantics,
                            high intensity keeps detail)
        | exp(y) - 1        back to linear intensity space
      output y_hat

    Loss design: compute MSE(log1p(y_hat), log1p(target)) in the log domain.
    Since log1p(expm1(y)) == y, log1p(y_hat) recovers the network's log-domain
    prediction exactly, giving equal weight to every order of magnitude without
    changing the architecture. This matters for SAXS, where errors in the 10^0
    and 10^5 regions should be treated equally.

    Args:
        in_ch, out_ch: input/output channel count (default 1)
        init_ch:       first-layer channel count (default 32)
        use_log:       operate in the log domain (default True, strongly recommended for SAXS)
    """
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
        # The output head emits a log-domain correction that must be signed:
        # Softplus would force the correction positive, collapsing the model to
        # the identity and preventing denoising.
        self.out_conv = nn.Conv2d(C, out_ch, 3, padding=1)

    def forward(self, x, log_output=False):
        # ---- Log-domain transform: compress magnitude range, stabilize gradients ----
        if self.use_log:
            x_log = torch.log1p(x)
        else:
            x_log = x

        e1 = self.enc1(x_log)       # (B, C,   H/2,  W/2)
        e2 = self.enc2(e1)          # (B, 2C,  H/4,  W/4)
        e3 = self.enc3(e2)          # (B, 4C,  H/8,  W/8)
        e4 = self.enc4(e3)          # (B, 8C,  H/16, W/16)
        b  = self.bottleneck(e4)     # (B, 8C, H/16, W/16) <- multi-scale fusion
        d4 = self.dec4(b, e3)       # (B, 4C,  H/8,  W/8)
        d3 = self.dec3(d4, e2)      # (B, 2C,  H/4,  W/4)
        d2 = self.dec2(d3, e1)      # (B, C,   H/2,  W/2)
        d1 = self.dec1(d2, x_log)   # (B, C,   H,    W)
        # The correction is in the log domain and can be positive or negative;
        # it is always computed in fp32 while the conv backbone stays in bf16.
        log_correction = self.out_conv(d1).float()

        if self.use_log:
            # Residual connection: output = exp(log_input + log_correction) - 1
            # At initialization log_correction ~= 0, so output ~= input and the
            # model starts from the identity.
            log_out = x_log.float() + log_correction
            if log_output:
                return log_out
            out = torch.expm1(log_out)
            # expm1 can be negative in low-intensity background; clamp to 0 to
            # keep the output physically non-negative.
            return torch.clamp(out, min=0)
        return nn.functional.softplus(log_correction)

# ---------------------------- 4. Checkpoints ----------------------------
def save_checkpoint(state, filename):
    torch.save(state, filename)

def nested_tensors_finite(obj):
    if isinstance(obj, torch.Tensor):
        if torch.is_floating_point(obj):
            return bool(torch.isfinite(obj).all())
        return True
    if isinstance(obj, dict):
        return all(nested_tensors_finite(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return all(nested_tensors_finite(v) for v in obj)
    return True

def load_checkpoint(filename, model, optimizer=None, scheduler=None):
    if not os.path.isfile(filename):
        return 0, float('inf')

    checkpoint = torch.load(filename, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model_state_dict']
    model_state = model.state_dict()

    # Handle the DDP 'module.' prefix.
    if list(state_dict.keys())[0].startswith('module.') and not list(model_state.keys())[0].startswith('module.'):
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
    elif not list(state_dict.keys())[0].startswith('module.') and list(model_state.keys())[0].startswith('module.'):
        state_dict = {'module.' + k: v for k, v in state_dict.items()}

    # Check architecture compatibility: compare state_dict keys with model keys.
    ckpt_keys = set(state_dict.keys())
    model_keys = set(model_state.keys())
    missing = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys

    if missing or unexpected:
        print(f"\n  Warning: checkpoint architecture mismatch (possibly an older model), training from scratch")
        if missing:
            print(f"    layers added in the model ({len(missing)}): "
                  f"{list(missing)[:3]}...")
        if unexpected:
            print(f"    layers dropped from the checkpoint ({len(unexpected)}): "
                  f"{list(unexpected)[:3]}...")
        return 0, float('inf')

    # Check shape compatibility.
    shape_mismatch = False
    for k in ckpt_keys & model_keys:
        if state_dict[k].shape != model_state[k].shape:
            print(f"\n  Warning: checkpoint architecture mismatch (layer '{k}' shape "
                  f"{list(state_dict[k].shape)} -> {list(model_state[k].shape)}), training from scratch")
            shape_mismatch = True
            break
    if shape_mismatch:
        return 0, float('inf')

    # A previous NaN run may have written bad weights into the checkpoint;
    # restart from scratch if detected.
    if not nested_tensors_finite(state_dict):
        print("\n  Warning: checkpoint weights contain NaN/Inf, ignoring it and training from scratch")
        return 0, float('inf')

    # All checks passed, so load the checkpoint.
    model.load_state_dict(state_dict)
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        opt_state = checkpoint['optimizer_state_dict']
        if not nested_tensors_finite(opt_state):
            print("  Warning: optimizer state contains NaN/Inf, skipping it (continue with the initial optimizer)")
        else:
            try:
                optimizer.load_state_dict(opt_state)
            except (ValueError, KeyError):
                pass  # Skip incompatible optimizer states and continue with the initial state.
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except (ValueError, KeyError):
            pass

    return start_epoch, best_val_loss

# ---------------------------- 5. GPU Memory Monitoring ----------------------------
def get_autocast_dtype(device):
    """Return the best autocast dtype for the device.

    - CUDA cc>=8.0 (RTX30xx/RTX40xx/A100/H100): bfloat16
    - CUDA cc>=7.0 (V100/RTX20xx): float16
    - CPU / MPS / older GPUs: None (caller should disable autocast)
    """
    if device.type == 'cuda':
        major = torch.cuda.get_device_capability(device)[0]
        # This unnormalized residual U-Net's intermediate activations exceed the
        # fp16 limit (decoder overflows to inf in practice); bf16 has the same
        # dynamic range as fp32, and on the 3060 its per-step cost is nearly
        # identical to fp16.
        if major >= 8:
            return torch.bfloat16
        elif major >= 7:
            return torch.float16
    return None


def print_gpu_memory_usage(device, prefix=""):
    if device.type != 'cuda':
        return
    allocated = torch.cuda.memory_allocated(device) / 1e9
    reserved = torch.cuda.memory_reserved(device) / 1e9
    max_allocated = torch.cuda.max_memory_allocated(device) / 1e9
    print(f"{prefix}GPU memory: {allocated:.3f}G / {reserved:.3f}G / peak {max_allocated:.3f}G")

# ---------------------------- 6. Smart Batch Size Search ----------------------------
def find_optimal_batch_size(model, dataset, device, safety_margin=0.90,
                            amp_dtype=None, channels_last=False):
    if device.type != 'cuda':
        print("No GPU detected, using default batch size 2")
        return 2

    device_idx = device.index if device.index is not None else 0
    props = torch.cuda.get_device_properties(device_idx)
    total_mem = props.total_memory
    threshold = total_mem * safety_margin
    gpu_name = props.name

    # Cap this process's memory at the threshold so the caching allocator raises
    # OOM immediately, avoiding long stalls or paging on Windows.
    try:
        torch.cuda.set_per_process_memory_fraction(safety_margin)
    except Exception as e:
        print(f"  Failed to set the memory limit: {e}")

    print(f"\n{'='*60}")
    print(f"[GPU {device_idx}] {gpu_name}  {total_mem/1e9:.2f} GB")
    print(f"Smart batch size search (threshold {safety_margin*100:.0f}% = {threshold/1e9:.2f} GB)")
    print(f"{'='*60}")

    # Collect samples.
    sample_list = []
    for i in range(64):
        img_i, img_j = dataset[i]
        sample_list.append((img_i, img_j))

    mem_fmt = torch.channels_last if channels_last else None

    def to_device_tensor(t):
        if mem_fmt is not None:
            return t.to(device, non_blocking=True, memory_format=mem_fmt)
        return t.to(device, non_blocking=True)

    def test_batch(bs):
        inputs = torch.stack([sample_list[i % len(sample_list)][0] for i in range(bs)], dim=0)
        targets = torch.stack([sample_list[i % len(sample_list)][1] for i in range(bs)], dim=0)
        inputs, targets = to_device_tensor(inputs), targets.to(device, non_blocking=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            opt = optim.Adam(model.parameters(), lr=1e-4)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype,
                                    enabled=(device.type == 'cuda' and amp_dtype is not None)):
                outputs = model(inputs, log_output=True)
                loss = log_domain_loss(outputs, targets, nn.MSELoss())
            loss.backward()
            opt.step()
            peak = torch.cuda.max_memory_allocated(device)
            return peak
        except RuntimeError as e:
            if "out of memory" in str(e):
                return None
            raise
        finally:
            del inputs, targets
            if 'outputs' in locals(): del outputs
            torch.cuda.empty_cache()

    # Phase 1: exponential search. Memory grows roughly linearly with batch size,
    # so extrapolate from the previous measured peak and skip sizes guaranteed to
    # exceed the threshold, avoiding long allocation stalls on Windows.
    last_ok = None
    last_ok_peak = None
    first_fail = None
    bs = 2
    while bs <= 1024:
        if last_ok_peak is not None:
            predicted = last_ok_peak * (bs / last_ok)
            if predicted >= threshold:
                first_fail = bs
                print(f"  bs={bs:4d}  -> predicted {predicted/1e9:.3f} GB  X (skipped test)")
                break
        peak = test_batch(bs)
        if peak is None:
            first_fail = bs
            print(f"  bs={bs:4d}  -> OOM")
            break
        if peak < threshold:
            last_ok = bs
            last_ok_peak = peak
            print(f"  bs={bs:4d}  -> {peak/1e9:.3f} GB  OK")
            bs *= 2
        else:
            first_fail = bs
            print(f"  bs={bs:4d}  -> {peak/1e9:.3f} GB  X")
            break
    if last_ok is None:
        last_ok = 1
        last_ok_peak = 0.0

    # Phase 2: binary search. Stop probing larger sizes once the current one uses
    # more than 85% of the threshold, avoiding long near-OOM waits or paging.
    if (first_fail is not None and first_fail > last_ok + 2
            and (last_ok_peak is None or last_ok_peak < threshold * 0.85)):
        lo, hi = last_ok, first_fail
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if last_ok_peak is not None:
                predicted = last_ok_peak * (mid / last_ok)
                if predicted >= threshold:
                    hi = mid
                    print(f"  bs={mid:4d}  -> predicted {predicted/1e9:.3f} GB  X (skipped test)")
                    continue
            peak = test_batch(mid)
            if peak is None:
                hi = mid
            elif peak < threshold:
                lo = mid
                last_ok_peak = peak
                print(f"  bs={mid:4d}  -> {peak/1e9:.3f} GB  OK")
            else:
                hi = mid
                print(f"  bs={mid:4d}  -> {peak/1e9:.3f} GB  X")
        last_ok = lo

    # Round down to a power of two.
    best_bs = last_ok if last_ok < 4 else (2 ** int(math.log2(last_ok)))
    print(f"\n  -> recommended per-GPU batch size: {best_bs}")
    print(f"  -> effective total batch: {best_bs} x {torch.cuda.device_count() if dist.is_initialized() else 1}")
    return best_bs

# ---------------------------- 7. TensorBoard Image Logging ----------------------------
def log_validation_images(writer, model, val_loader, epoch, device, num_samples=4,
                          channels_last=False):
    model.eval()
    with torch.inference_mode():
        for i, (input_img, target_img) in enumerate(val_loader):
            if i >= num_samples:
                break
            input_img = move_to_device(input_img, device, channels_last)
            output_img = model(input_img)
            inp_np = input_img[0, 0].cpu().numpy()
            out_np = output_img[0, 0].cpu().numpy()
            tgt_np = target_img[0, 0].cpu().numpy()
            def stretch(img):
                lo, hi = img.min(), img.max()
                return (img - lo) / (hi - lo) if hi > lo else img
            writer.add_image(f'Val/input_{i}', stretch(inp_np), epoch, dataformats='HW')
            writer.add_image(f'Val/output_{i}', stretch(out_np), epoch, dataformats='HW')
            writer.add_image(f'Val/target_{i}', stretch(tgt_np), epoch, dataformats='HW')

# ---------------------------- 8. Helper Functions ----------------------------
def print_file_statistics(file_list, dataset_name):
    print(f"\n========== {dataset_name} ==========")
    total_pairs = 0
    for fpath in file_list:
        with h5py.File(fpath, 'r') as f:
            num_pairs = f['pairs'].shape[0]
            total_pairs += num_pairs
            print(f"  {os.path.basename(fpath)}: {num_pairs} pairs")
    print(f"Total: {len(file_list)} files, {total_pairs} pairs")
    return total_pairs

def gather_val_loss(local_loss_sum, local_count, device, world_size):
    if world_size <= 1:
        return local_loss_sum / local_count if local_count > 0 else 0.0
    t = torch.tensor([local_loss_sum, float(local_count)], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    total_loss, total_count = t[0].item(), int(t[1].item())
    return total_loss / total_count if total_count > 0 else 0.0


def move_to_device(tensor, device, channels_last):
    if channels_last:
        return tensor.to(device, non_blocking=True, memory_format=torch.channels_last)
    return tensor.to(device, non_blocking=True)


def log_domain_loss(outputs, targets, criterion):
    # Always compute MSE in fp32 because fp16 loss easily overflows or loses
    # gradient precision.
    return criterion(outputs.float(), torch.log1p(targets).float())


def evaluate_loader_loss(model, loader, device, amp_dtype, channels_last, criterion, world_size):
    """Compute the mean loss over a loader, aggregated across DDP ranks."""
    model.eval()
    loss_sum = torch.zeros((), device=device)
    count = 0
    with torch.inference_mode():
        for inputs, targets in loader:
            inputs = move_to_device(inputs, device, channels_last)
            targets = targets.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype,
                                    enabled=(amp_dtype is not None)):
                outputs = model(inputs, log_output=True)
                loss = log_domain_loss(outputs, targets, criterion)
            loss_sum += loss.detach().float() * inputs.size(0)
            count += inputs.size(0)
    return gather_val_loss(loss_sum.item(), count, device, world_size)


def load_loss_history(path):
    """Load loss-history JSONL rows, skipping any partial or corrupt lines."""
    rows = []
    if os.path.isfile(path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def append_loss_history(path, row):
    """Append one loss-history row in JSONL format."""
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')


def plot_loss_curves(path, rows):
    """Persist a train/val/test loss curve plot; silently skips without matplotlib."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    epochs = [r['epoch'] for r in rows]
    train_losses = [r.get('train_loss') for r in rows]
    val_losses = [r.get('val_loss') for r in rows]
    test_rows = [(r['epoch'], r['test_loss']) for r in rows if r.get('test_loss') is not None]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, label='train', color='tab:blue')
    ax.plot(epochs, val_losses, label='val', color='tab:orange')
    if test_rows:
        ax.plot([e for e, _ in test_rows], [v for _, v in test_rows],
                label='test', color='tab:green', marker='o', markersize=3, linewidth=1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Train / Val / Test Loss')
    all_values = train_losses + val_losses + [v for _, v in test_rows]
    if all_values and all(isinstance(v, (int, float)) and v > 0 for v in all_values):
        ax.set_yscale('log')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_dataloader(dataset, batch_size, shuffle, sampler, num_workers,
                     pin_memory, prefetch_factor):
    kwargs = dict(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                  sampler=sampler, num_workers=num_workers, pin_memory=pin_memory)
    if num_workers > 0:
        kwargs['prefetch_factor'] = prefetch_factor
    return DataLoader(**kwargs)


def unwrap_compiled(model):
    if hasattr(model, '_orig_mod'):
        return model._orig_mod
    if hasattr(model, 'module') and hasattr(model.module, '_orig_mod'):
        model.module = model.module._orig_mod
    return model


def make_cpu_checkpoint(model_state, optimizer, scheduler, epoch, best_val_loss,
                        batch_size, world_size, eff_bs, accum_steps):
    opt_state = optimizer.state_dict()
    opt_state_cpu = {
        'state': {k: {sk: (sv.detach().cpu() if torch.is_tensor(sv) else sv)
                      for sk, sv in v.items()}
                  for k, v in opt_state['state'].items()},
        'param_groups': opt_state['param_groups'],
    }
    return {'epoch': epoch,
            'model_state_dict': {k: v.detach().cpu() for k, v in model_state.items()},
            'optimizer_state_dict': opt_state_cpu,
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'batch_size': batch_size,
            'world_size': world_size,
            'effective_batch_size': eff_bs * accum_steps}


def save_checkpoint_async(state, filename, previous_thread=None):
    if previous_thread is not None:
        previous_thread.join()

    def _save():
        tmp = filename + '.tmp'
        torch.save(state, tmp)
        os.replace(tmp, filename)

    thread = threading.Thread(target=_save, name='checkpoint-save', daemon=True)
    thread.start()
    return thread


def print_model_summary(model, input_shape=(1, 1024, 512)):
    """Print per-layer parameter counts for the model.

    Strips the DDP 'module.' prefix and only shows layers with trainable parameters.
    """
    if hasattr(model, 'module'):
        model = model.module

    print(f"\n{'='*65}")
    print(f"Model parameter details (input: {input_shape})")
    print(f"{'='*65}")
    print(f"  {'Layer':<48} {'Parameters':>12}")
    print(f"  {'-'*60}")

    total = 0
    for name, mod in model.named_modules():
        n = sum(p.numel() for p in mod.parameters(recurse=False))
        if n == 0:
            continue
        print(f"  {name:<48} {n:>12,}")
        total += n

    print(f"  {'-'*60}")
    print(f"  {'Total':<48} {total:>12,}  ({total/1e6:.2f}M)")
    print(f"{'='*65}\n")

# ---------------------------- 8.5 Training Diagnostics Log ----------------------------
class TrainingDiagnostics:
    """Append per-epoch aggregate statistics to JSONL for offline analysis."""

    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        self.train_path = os.path.join(log_dir, 'train_diagnostics.jsonl')
        self.weight_path = os.path.join(log_dir, 'weight_diagnostics.jsonl')

    @staticmethod
    def _append(path, row):
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    def write_config(self, config):
        row = {'type': 'config', 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'), **config}
        self._append(self.train_path, row)

    def log_epoch(self, epoch, metrics):
        self._append(self.train_path, {'type': 'epoch', 'epoch': epoch, **metrics})

    def log_final(self, metrics):
        self._append(self.train_path, {'type': 'final', **metrics})

    def log_weights(self, epoch, layer_stats):
        self._append(self.weight_path, {'type': 'weights', 'epoch': epoch, 'layers': layer_stats})


def compute_grad_norm_squared(model):
    total = torch.zeros((), device=next(model.parameters()).device)
    for param in model.parameters():
        if param.grad is not None:
            total += param.grad.detach().float().pow(2).sum()
    return total


def compute_weight_norm(model):
    total = torch.zeros((), device=next(model.parameters()).device)
    for param in model.parameters():
        total += param.detach().float().pow(2).sum()
    return total.sqrt()


def collect_weight_layer_stats(model):
    raw = unwrap_compiled(model)
    if hasattr(raw, 'module'):
        raw = raw.module
    layer_stats = []
    for name, param in raw.named_parameters():
        weight = param.detach().float()
        grad = param.grad.detach().float() if param.grad is not None else None
        weight_std = float(weight.std()) if weight.numel() > 1 else 0.0
        grad_std = float(grad.std()) if grad is not None and grad.numel() > 1 else 0.0
        layer_stats.append({
            'name': name,
            'numel': weight.numel(),
            'weight_mean': float(weight.mean()),
            'weight_std': weight_std,
            'weight_l2': float(weight.pow(2).sum().sqrt()),
            'grad_l2': float(grad.pow(2).sum().sqrt()) if grad is not None else None,
            'grad_std': grad_std,
        })
    return layer_stats


def collect_attention_gate_stats(model, sample_input, amp_dtype):
    raw = unwrap_compiled(model)
    if hasattr(raw, 'module'):
        raw = raw.module
    gate_stats = {}
    hooks = []
    for name, module in raw.named_modules():
        if not isinstance(module, AttentionGate):
            continue
        gate_name = name

        def make_hook(gate_name):
            def hook(_module, _args, alpha):
                alpha = alpha.float()
                gate_stats[gate_name] = {
                    'mean': float(alpha.mean()),
                    'std': float(alpha.std()),
                    'min': float(alpha.min()),
                    'max': float(alpha.max()),
                    'frac_high': float((alpha > 0.5).float().mean()),
                }
            return hook

        hooks.append(module.psi.register_forward_hook(make_hook(gate_name)))
    with torch.inference_mode():
        with torch.amp.autocast(device_type=sample_input.device.type, dtype=amp_dtype,
                                enabled=(amp_dtype is not None)):
            model(sample_input, log_output=True)
    for hook in hooks:
        hook.remove()
    return gate_stats

# ---------------------------- 9. DDP Worker Main Function ----------------------------
def main_worker(local_rank, world_size):
    # ========== DDP initialization ==========
    if world_size > 1:
        os.environ.setdefault('MASTER_ADDR', 'localhost')
        os.environ.setdefault('MASTER_PORT', '12355')
        dist.init_process_group(backend='nccl', init_method='env://',
                                world_size=world_size, rank=local_rank,
                                device_id=torch.device(f'cuda:{local_rank}'))
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        rank = local_rank
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        rank = 0
    is_main = (rank == 0)
    amp_dtype_override = None   # 'fp16'/'bf16' force AMP; None=auto-select by GPU

    # ========== Global performance tuning ==========
    torch.set_float32_matmul_precision('high')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        # Allow relaxed fp16 matmul reductions for slightly faster
        # linear/attention matmuls.
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    amp_dtype = get_autocast_dtype(device)
    if amp_dtype_override == 'fp16':
        amp_dtype = torch.float16
    elif amp_dtype_override == 'bf16':
        amp_dtype = torch.bfloat16
    if is_main:
        print(f"\n{'='*60}")
        print(f"Training engine tuning:")
        if device.type == 'cuda':
            print(f"  TF32 matmul:    high-precision mode (~5x matmul speedup)")
            print(f"  cuDNN benchmark: enabled (auto-selects best convolutions)")
        print(f"  AMP dtype:      {amp_dtype}")
        print(f"  World size:      {world_size} (DDP)")
        print(f"{'='*60}\n")

    if is_main and device.type == 'cuda':
        for i in range(world_size):
            p = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {p.name}  {p.total_memory/1e9:.1f} GB")
    if world_size > 1:
        dist.barrier()

    # ========== Configuration ==========
    data_dir = "./dataset"
    output_dir = "./models"
    log_dir = "./runs/noise2noise"
    resume_checkpoint = True  # True=auto-load latest.pth; None/False=train from scratch; str=explicit path

    epochs = 500
    learning_rate = 1e-4
    min_lr = 1e-6                 # floor for each cosine annealing cycle
    weight_decay = 1e-5
    scheduler_t0 = 50             # first cosine warm-restart cycle length
    scheduler_t_mult = 2          # cycle length multiplier after each restart
    test_eval_interval = 10       # evaluate test loss every N epochs during training
    num_workers = 0            # data is preloaded into RAM; Windows spawn pickles ~5 GB of
                               # numpy arrays into every worker, duplicating memory
    prefetch_factor = 4        # only used when num_workers > 0
    patience = 40
    seed = 42
    progress_print_interval = 50
    warmup_steps = 3
    use_channels_last = True   # NHWC input + params; cuDNN convolutions are faster on Ampere
    use_compile = False        # no Triton on this host; torch.compile failures surface only at first forward
    manual_batch_size = None   # when set, skip the automatic batch size search (8 is a good starting point here)

    # init_ch=32 gives ~12.4M parameters (SAXS Attention U-Net + ASPP bottleneck).
    init_channels = 32

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ========== Load dataset ==========
    # The split is fixed on disk: each H5 file lives in exactly one of
    # dataset/train, dataset/val and dataset/test. This keeps the partition
    # stable across runs and makes it impossible to train on held-out files.
    if not os.path.isdir(data_dir):
        if is_main: print(f"Error: {data_dir} does not exist")
        sys.exit(1)
    split_dirs = {
        'train': os.path.join(data_dir, 'train'),
        'val': os.path.join(data_dir, 'val'),
        'test': os.path.join(data_dir, 'test'),
    }
    for split_name, split_dir in split_dirs.items():
        if not os.path.isdir(split_dir):
            if is_main: print(f"Error: {split_dir} does not exist")
            sys.exit(1)
    # Refuse to run while unsplit .h5 files are still in the dataset root.
    root_h5_files = [f for f in os.listdir(data_dir) if f.endswith('.h5')]
    if root_h5_files:
        if is_main:
            print(f"Error: {len(root_h5_files)} .h5 files remain in {data_dir}; "
                  f"move them into train/val/test subdirectories first")
        sys.exit(1)

    split_files = {}
    for split_name, split_dir in split_dirs.items():
        files = sorted([os.path.join(split_dir, f) for f in os.listdir(split_dir)
                        if f.endswith('.h5')])
        if not files:
            if is_main: print(f"Error: no .h5 files found in {split_dir}")
            sys.exit(1)
        split_files[split_name] = files

    # Guard against accidental duplicates, e.g. a file copied into two splits.
    seen_basenames = {}
    for split_name, files in split_files.items():
        for fpath in files:
            base = os.path.basename(fpath)
            if base in seen_basenames:
                if is_main:
                    print(f"Error: {base} appears in both {seen_basenames[base]} and {split_name}")
                sys.exit(1)
            seen_basenames[base] = split_name

    train_files = split_files['train']
    val_files = split_files['val']
    test_files = split_files['test']

    if is_main:
        print_file_statistics(train_files, "Training set")
        print_file_statistics(val_files, "Validation set")
        print_file_statistics(test_files, "Test set")

    # Preload into memory.
    train_dataset = Noise2NoiseH5Dataset(train_files, crop_func=crop_1028_to_1024)
    val_dataset = Noise2NoiseH5Dataset(val_files, crop_func=crop_1028_to_1024)
    test_dataset = Noise2NoiseH5Dataset(test_files, crop_func=crop_1028_to_1024)

    # ========== Phase 1: search batch size with a temporary model (no compile) ==========
    need_search = is_main and not manual_batch_size
    if need_search:
        search_model = SAXSAttentionUNet(in_ch=1, out_ch=1, init_ch=init_channels).to(device)
        if use_channels_last and device.type == 'cuda':
            search_model = search_model.to(memory_format=torch.channels_last)
        optimal_batch_size = find_optimal_batch_size(
            search_model, train_dataset, device, safety_margin=0.85,
            amp_dtype=amp_dtype, channels_last=use_channels_last
        )
    elif is_main:
        optimal_batch_size = int(manual_batch_size)
        print(f"\nManual batch size: {optimal_batch_size} (skipping automatic search)")
    else:
        optimal_batch_size = 2
    if world_size > 1:
        bs_t = torch.tensor([optimal_batch_size], device=device)
        dist.broadcast(bs_t, src=0)
        optimal_batch_size = int(bs_t.item())
    batch_size = optimal_batch_size
    eff_bs = batch_size * world_size

    # Automatic gradient accumulation.
    target_eff_bs = max(64, eff_bs * 2)
    grad_accum = max(1, (target_eff_bs + eff_bs - 1) // eff_bs)
    if is_main:
        print(f"\nPer-GPU batch size: {batch_size}  |  total batch: {eff_bs} x {grad_accum} accum "
              f"= {eff_bs * grad_accum} effective")

    # ---------- Release the search model + CUDA cache (prevent fragmentation OOM) ----------
    if need_search:
        del search_model
        torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    if is_main:
        print_gpu_memory_usage(device, "after batch search (cleaned): ")

    # ========== Phase 2: build the training model + torch.compile + DDP ==========
    model = SAXSAttentionUNet(in_ch=1, out_ch=1, init_ch=init_channels).to(device)
    if use_channels_last and device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)

    if use_compile:
        # torch.compile failures surface only at the first forward call, so
        # warmup handles the fallback; this block only wraps the model.
        if is_main:
            print("\nCompiling model (torch.compile)...")
        try:
            model = torch.compile(model, mode='default')
        except Exception as e:
            model = unwrap_compiled(model)
            if is_main:
                print(f"  Warning: torch.compile setup failed ({e}), using the uncompiled version")
    else:
        if is_main:
            print("\nCompiling model: skipped (no Triton on this host, using eager mode)")
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # ========== DataLoader ==========
    use_pin = (device.type == 'cuda')
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank,
                                           shuffle=True, seed=seed)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        train_loader = build_dataloader(train_dataset, batch_size, shuffle=False, sampler=train_sampler,
                                        num_workers=num_workers, pin_memory=use_pin,
                                        prefetch_factor=prefetch_factor)
        val_loader = build_dataloader(val_dataset, batch_size, shuffle=False, sampler=val_sampler,
                                      num_workers=num_workers, pin_memory=use_pin,
                                      prefetch_factor=prefetch_factor)
        test_loader = build_dataloader(test_dataset, batch_size, shuffle=False, sampler=test_sampler,
                                       num_workers=num_workers, pin_memory=use_pin,
                                       prefetch_factor=prefetch_factor)
    else:
        train_loader = build_dataloader(train_dataset, batch_size, shuffle=True, sampler=None,
                                        num_workers=num_workers, pin_memory=use_pin,
                                        prefetch_factor=prefetch_factor)
        val_loader = build_dataloader(val_dataset, batch_size, shuffle=False, sampler=None,
                                      num_workers=num_workers, pin_memory=use_pin,
                                      prefetch_factor=prefetch_factor)
        test_loader = build_dataloader(test_dataset, batch_size, shuffle=False, sampler=None,
                                       num_workers=num_workers, pin_memory=use_pin,
                                       prefetch_factor=prefetch_factor)

    # ========== Loss / optimizer / scheduler ==========
    # Noise2Noise theory: MSE converges to E[clean|noisy], optimal without
    # clean targets.
    criterion = nn.MSELoss()
    try:
        # Fused Adam merges the optimizer update into a single kernel and is
        # noticeably faster with many parameters.
        optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                               weight_decay=weight_decay,
                               fused=(device.type == 'cuda'))
    except (RuntimeError, ValueError):
        optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                               weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=scheduler_t0, T_mult=scheduler_t_mult, eta_min=min_lr)

    scaler = torch.amp.GradScaler(device=device.type,
                                  enabled=(device.type == 'cuda' and amp_dtype == torch.float16))

    if is_main:
        print_model_summary(model, input_shape=(1, 1024, 512))

    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        if is_main:
            print_gpu_memory_usage(device, "after initialization: ")

    # ========== Phase 3: torch.compile + DDP warmup ==========
    # Trigger the compile trace and CUDA allocations with dummy batches before
    # real training, preventing OOM from transient trace memory on the first
    # real batch.
    if is_main:
        print("\nCompile warmup...")
    model.train()
    warmup_bs = batch_size  # Warm up with the real batch size so the compile trace shapes match.
    warmup_in = torch.rand(warmup_bs, 1, 1024, 512).to(device)   # Positive values, compatible with log1p.
    warmup_tgt = torch.rand(warmup_bs, 1, 1024, 512).to(device)
    if use_channels_last and device.type == 'cuda':
        warmup_in = warmup_in.to(memory_format=torch.channels_last)
    try:
        for _ in range(warmup_steps):
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype,
                                    enabled=(amp_dtype is not None)):
                warmup_out = model(warmup_in, log_output=True)
                warmup_loss = log_domain_loss(warmup_out, warmup_tgt, criterion)
            scaler.scale(warmup_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
    except Exception as e:
        # compile failures (e.g. no Triton) surface only at the first forward;
        # fall back to eager here.
        model = unwrap_compiled(model)
        if is_main:
            print(f"  warmup failed ({type(e).__name__}: {str(e)[:160]}), fell back to eager")
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=(amp_dtype is not None)):
            warmup_out = model(warmup_in, log_output=True)
            warmup_loss = log_domain_loss(warmup_out, warmup_tgt, criterion)
        scaler.scale(warmup_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    del warmup_in, warmup_tgt, warmup_out
    torch.cuda.empty_cache()
    if is_main:
        print_gpu_memory_usage(device, "  after warmup: ")

    # ========== TensorBoard / checkpoint resume ==========
    writer = None
    diag = None
    if is_main:
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        diag = TrainingDiagnostics(log_dir)

    # Resolve checkpoint path: True=auto, str=explicit path, None/False=skip.
    if resume_checkpoint is True:
        checkpoint_path = os.path.join(output_dir, 'latest.pth')
    elif isinstance(resume_checkpoint, str):
        checkpoint_path = resume_checkpoint
    else:
        checkpoint_path = None

    start_epoch, best_val_loss = 0, float('inf')
    if checkpoint_path and os.path.isfile(checkpoint_path):
        if is_main:
            print(f"\nResuming from checkpoint: {checkpoint_path}")
        start_epoch, best_val_loss = load_checkpoint(checkpoint_path, model, optimizer, scheduler)
        if is_main:
            if start_epoch == 0:
                print("    Checkpoint invalid or contains NaN/Inf, training from scratch")
            else:
                print(f"    Resuming from epoch {start_epoch}, best validation loss: {best_val_loss:.4e}")
    else:
        if is_main:
            if resume_checkpoint is True:
                print(f"\nCheckpoint {checkpoint_path} not found, training from scratch")
            elif not resume_checkpoint:
                print(f"\nCheckpoint resume is disabled, training from scratch")
    if world_size > 1:
        dist.barrier()

    loss_history_path = os.path.join(log_dir, 'loss_history.jsonl')
    loss_plot_path = os.path.join(log_dir, 'loss_curves.png')
    loss_history = []
    if is_main:
        if start_epoch == 0:
            with open(loss_history_path, 'w', encoding='utf-8') as f:
                pass
        else:
            loss_history = [r for r in load_loss_history(loss_history_path)
                            if r.get('epoch', 0) < start_epoch + 1]
            with open(loss_history_path, 'w', encoding='utf-8') as f:
                for row in loss_history:
                    f.write(json.dumps(row, ensure_ascii=False) + '\n')

    if is_main and diag is not None:
        diag.write_config({
            'script_version': 'SAXSDenoise2D_N2N_08',
            'train_dir': split_dirs['train'],
            'val_dir': split_dirs['val'],
            'test_dir': split_dirs['test'],
            'output_dir': output_dir,
            'log_dir': log_dir,
            'epochs': epochs,
            'learning_rate': learning_rate,
            'min_lr': min_lr,
            'scheduler': 'CosineAnnealingWarmRestarts',
            'scheduler_t0': scheduler_t0,
            'scheduler_t_mult': scheduler_t_mult,
            'weight_decay': weight_decay,
            'test_eval_interval': test_eval_interval,
            'num_workers': num_workers,
            'patience': patience,
            'seed': seed,
            'init_channels': init_channels,
            'batch_size': batch_size,
            'world_size': world_size,
            'effective_batch_size': eff_bs * grad_accum,
            'grad_accum': grad_accum,
            'amp_dtype': str(amp_dtype),
            'use_channels_last': use_channels_last,
            'start_epoch': start_epoch,
            'resume_checkpoint': resume_checkpoint,
        })

    # ========== Training loop ==========
    os.makedirs(output_dir, exist_ok=True)
    no_improve_epochs = 0
    accum_steps = grad_accum
    save_thread = None

    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()
        if world_size > 1:
            train_sampler.set_epoch(epoch)
        if is_main:
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{epochs}  "
                  f"[batch {batch_size} x {world_size} GPU x {accum_steps} accum = {batch_size*world_size*accum_steps}]")

        # ---- Training ----
        model.train()
        running_loss = torch.zeros((), device=device)
        samples_seen = 0
        num_batches = len(train_loader)
        optimizer.zero_grad()
        grad_norm_sum = torch.zeros((), device=device)
        grad_norm_count = torch.zeros((), device=device)
        last_epoch_weight_stats = []

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = move_to_device(inputs, device, use_channels_last)
            targets = targets.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=(amp_dtype is not None)):
                # Log-domain loss; the model emits log-domain corrections
                # directly, avoiding expm1/log1p round trips.
                outputs = model(inputs, log_output=True)
                loss = log_domain_loss(outputs, targets, criterion)
                loss = loss / accum_steps
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"NaN/Inf loss at batch {batch_idx + 1}, training aborted")

            scaler.scale(loss).backward()

            if (batch_idx + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                if is_main:
                    grad_norm_sum += compute_grad_norm_squared(model).sqrt()
                    grad_norm_count += 1
                    if (batch_idx + 1) == num_batches:
                        last_epoch_weight_stats = collect_weight_layer_stats(model)
                optimizer.zero_grad()

            running_loss += loss.detach().float() * inputs.size(0) * accum_steps
            samples_seen += inputs.size(0)

            if is_main and ((batch_idx + 1) % progress_print_interval == 0 or (batch_idx + 1) == num_batches):
                print(f"  [train] {batch_idx+1}/{num_batches}  loss: "
                      f"{(running_loss / samples_seen):.4e}")

        # Handle the final incomplete accumulation step.
        if (batch_idx + 1) % accum_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            if is_main:
                grad_norm_sum += compute_grad_norm_squared(model).sqrt()
                grad_norm_count += 1
                last_epoch_weight_stats = collect_weight_layer_stats(model)
            optimizer.zero_grad()

        # Aggregate training loss with a single sync at epoch end.
        if world_size > 1:
            dist.all_reduce(running_loss, op=dist.ReduceOp.SUM)
        train_loss = (running_loss / len(train_dataset)).item()

        # ---- Validation ----
        model.eval()
        val_loss_sum = torch.zeros((), device=device)
        val_count = 0
        if is_main:
            input_log_sum = torch.zeros((), device=device)
            input_log_sq_sum = torch.zeros((), device=device)
            output_log_sum = torch.zeros((), device=device)
            output_log_sq_sum = torch.zeros((), device=device)
            input_output_prod_sum = torch.zeros((), device=device)
            correction_sum = torch.zeros((), device=device)
            correction_abs_sum = torch.zeros((), device=device)
            correction_positive = torch.zeros((), device=device)
            correction_negative = torch.zeros((), device=device)
            correction_min = torch.full((), float('inf'), device=device)
            correction_max = torch.full((), float('-inf'), device=device)
            nonfinite_output_count = torch.zeros((), device=device)
            pixel_count = torch.zeros((), device=device)
            identity_loss_sum = torch.zeros((), device=device)
        with torch.inference_mode():
            for inputs, targets in val_loader:
                inputs = move_to_device(inputs, device, use_channels_last)
                targets = targets.to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=(amp_dtype is not None)):
                    outputs = model(inputs, log_output=True)
                    loss = log_domain_loss(outputs, targets, criterion)
                val_loss_sum += loss.detach().float() * inputs.size(0)
                val_count += inputs.size(0)
                if is_main:
                    input_log = torch.log1p(inputs).float()
                    output_log = outputs.float()
                    correction = output_log - input_log
                    pixel_count += input_log.numel()
                    input_log_sum += input_log.sum()
                    input_log_sq_sum += (input_log * input_log).sum()
                    output_log_sum += output_log.sum()
                    output_log_sq_sum += (output_log * output_log).sum()
                    input_output_prod_sum += (input_log * output_log).sum()
                    correction_sum += correction.sum()
                    correction_abs_sum += correction.abs().sum()
                    correction_positive += (correction > 0).sum()
                    correction_negative += (correction < 0).sum()
                    correction_min = torch.minimum(correction_min, correction.min())
                    correction_max = torch.maximum(correction_max, correction.max())
                    nonfinite_output_count += (~torch.isfinite(output_log)).sum()
                    if epoch == start_epoch:
                        identity_loss_sum += criterion(input_log, torch.log1p(targets).float()).detach() * inputs.size(0)
        val_loss = gather_val_loss(val_loss_sum.item(), val_count, device, world_size)

        gate_stats = {}
        if is_main and diag is not None and val_count > 0:
            first_input, _ = next(iter(val_loader))
            first_input = move_to_device(first_input, device, use_channels_last)
            gate_stats = collect_attention_gate_stats(model, first_input, amp_dtype)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start_time

        test_loss = None
        if (epoch + 1) % test_eval_interval == 0 or epoch + 1 == epochs:
            test_loss = evaluate_loader_loss(
                model, test_loader, device, amp_dtype, use_channels_last, criterion, world_size)

        if is_main and diag is not None:
            grad_norm_avg = (grad_norm_sum / grad_norm_count).item() if grad_norm_count.item() > 0 else 0.0
            weight_l2 = compute_weight_norm(model).item()
            n_pixels = max(int(pixel_count.item()), 1)
            input_mean = input_log_sum.item() / n_pixels
            output_mean = output_log_sum.item() / n_pixels
            input_var = max(input_log_sq_sum.item() / n_pixels - input_mean ** 2, 0.0)
            output_var = max(output_log_sq_sum.item() / n_pixels - output_mean ** 2, 0.0)
            covariance = input_output_prod_sum.item() / n_pixels - input_mean * output_mean
            correlation = covariance / math.sqrt(input_var * output_var) if input_var > 0 and output_var > 0 else 0.0
            metrics = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': current_lr,
                'epoch_time_s': epoch_time,
                'samples_per_second': len(train_dataset) / epoch_time,
                'grad_norm_mean': grad_norm_avg,
                'weight_l2': weight_l2,
                'update_to_weight_ratio': (current_lr * grad_norm_avg) / (weight_l2 + 1e-12),
                'input_log_mean': input_mean,
                'input_log_std': math.sqrt(input_var),
                'output_log_mean': output_mean,
                'output_log_std': math.sqrt(output_var),
                'output_input_correlation': correlation,
                'correction_mean': correction_sum.item() / n_pixels,
                'correction_abs_mean': correction_abs_sum.item() / n_pixels,
                'correction_positive_frac': correction_positive.item() / n_pixels,
                'correction_negative_frac': correction_negative.item() / n_pixels,
                'correction_min': correction_min.item(),
                'correction_max': correction_max.item(),
                'nonfinite_output_count': int(nonfinite_output_count.item()),
                'attention_gates': gate_stats,
            }
            if epoch == start_epoch:
                identity_val = (identity_loss_sum.item() / val_count) if val_count > 0 else None
                metrics['identity_val_loss'] = identity_val
                metrics['val_noise_floor_estimate'] = (identity_val / 2) if identity_val is not None else None
            diag.log_epoch(epoch, metrics)

        if is_main:
            loss_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'test_loss': test_loss,
            })
            append_loss_history(loss_history_path, loss_history[-1])
            plot_loss_curves(loss_plot_path, loss_history)

        if is_main and writer is not None:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            if test_loss is not None:
                writer.add_scalar('Loss/test', test_loss, epoch)
            writer.add_scalar('LR', current_lr, epoch)
            writer.add_scalar('Time/epoch', epoch_time, epoch)
            if diag is not None:
                writer.add_scalar('Diagnostics/grad_norm', grad_norm_avg, epoch)
                writer.add_scalar('Diagnostics/weight_l2', weight_l2, epoch)
                writer.add_scalar('Diagnostics/correction_mean', correction_sum.item() / n_pixels, epoch)
                writer.add_scalar('Diagnostics/correction_positive_frac', correction_positive.item() / n_pixels, epoch)
            if epoch % 10 == 0:
                log_validation_images(writer, model, val_loader, epoch, device,
                                      num_samples=4, channels_last=use_channels_last)

        if is_main:
            print(f"Epoch {epoch+1}/{epochs}  |  {epoch_time:.1f}s  |  "
                  f"train {train_loss:.4e}  val {val_loss:.4e}  lr {current_lr:.2e}  "
                  f"({len(train_dataset) / epoch_time:.1f} samples/s)")
            if test_loss is not None:
                print(f"  Test loss: {test_loss:.4e}")
            if device.type == 'cuda':
                print_gpu_memory_usage(device, "  ")

        # Save.
        if is_main:
            ms = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            cp = make_cpu_checkpoint(ms, optimizer, scheduler, epoch, best_val_loss,
                                     batch_size, world_size, eff_bs, accum_steps)
            save_thread = save_checkpoint_async(cp, os.path.join(output_dir, 'latest.pth'),
                                                save_thread)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_thread = save_checkpoint_async(cp, os.path.join(output_dir, 'best.pth'),
                                                    save_thread)
                no_improve_epochs = 0
                print(f"  * New best validation loss: {best_val_loss:.4e}")
            else:
                no_improve_epochs += 1
            if diag is not None and (epoch % 10 == 0 or val_loss == best_val_loss):
                diag.log_weights(epoch, last_epoch_weight_stats)

        if world_size > 1:
            st = torch.tensor([no_improve_epochs], device=device)
            dist.broadcast(st, src=0)
            no_improve_epochs = int(st.item())
        if no_improve_epochs >= patience:
            if is_main:
                print(f"\nEarly stopping: no improvement for {patience} epochs")
            break

    # ========== Final test evaluation (runs once after training) ==========
    # The test set stays fully isolated during training and gives an unbiased
    # estimate of generalization.
    test_loss = evaluate_loader_loss(
        model, test_loader, device, amp_dtype, use_channels_last, criterion, world_size)

    if is_main:
        if save_thread is not None:
            save_thread.join()
        if diag is not None:
            diag.log_final({
                'test_loss': test_loss,
                'best_val_loss': best_val_loss,
                'final_epoch': epoch + 1,
            })
        for row in reversed(loss_history):
            if row['epoch'] == epoch + 1:
                row['test_loss'] = test_loss
                break
        with open(loss_history_path, 'w', encoding='utf-8') as f:
            for row in loss_history:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        plot_loss_curves(loss_plot_path, loss_history)
        if writer is not None:
            writer.add_scalar('Loss/test_final', test_loss, start_epoch if start_epoch > 0 else epochs)
            # Log test-set sample images.
            log_validation_images(writer, model, test_loader, epochs, device,
                                  num_samples=4, channels_last=use_channels_last)
        print(f"\n{'='*50}")
        print(f"Final test evaluation (unbiased estimate): test loss = {test_loss:.4e}")

    if writer is not None:
        writer.close()
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    if is_main:
        # Export inference-only weights (state_dict only, no optimizer/scheduler,
        # portable across platforms).
        ms = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        inference_path = os.path.join(output_dir, 'model_inference.pth')
        torch.save(ms, inference_path)
        print(f"\nTraining complete | best validation loss: {best_val_loss:.4e} | test loss: {test_loss:.4e}")
        print(f"Full checkpoint:  {os.path.join(output_dir, 'latest.pth')}  "
              f"(includes optimizer/scheduler, resumable)")
        print(f"Inference model:  {inference_path}  "
              f"(weights only, {sum(p.numel() for p in ms.values())/1e6:.2f}M params, portable)")

# ---------------------------- 10. Main Entry ----------------------------
def main():
    if 'WORLD_SIZE' in os.environ and 'RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        main_worker(local_rank, world_size)
        return

    if not torch.cuda.is_available():
        print("No CUDA GPU detected, training on CPU")
        main_worker(0, 1)
        return

    n_gpus = torch.cuda.device_count()
    print(f"\nDetected {n_gpus} GPU(s):")
    for i in range(n_gpus):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {p.name}  {p.total_memory/1e9:.1f} GB")

    if n_gpus == 1:
        main_worker(0, 1)
    else:
        print(f"\nLaunching {n_gpus}-process DDP distributed training ...")
        os.environ.setdefault('MASTER_ADDR', 'localhost')
        os.environ.setdefault('MASTER_PORT', '12355')
        torch.multiprocessing.spawn(main_worker, args=(n_gpus,), nprocs=n_gpus, join=True)

if __name__ == "__main__":
    main()
