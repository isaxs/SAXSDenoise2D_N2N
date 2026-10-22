"""Reference-free denoising metrics for 2D SAXS images.

These metrics are intended for routine use in the inference GUI, where only the
noisy input and the denoised output are available. They do not measure the error
against a clean signal, which is unavailable. They measure three things that can
be measured without a reference:

  * how much background noise was removed (background sigma, SNR);
  * whether the residual looks like noise rather than removed signal
    (residual mean, residual standard deviation, lag-1 autocorrelation);
  * whether high-frequency noise was suppressed without erasing low-frequency
    signal (high-frequency and low-frequency power ratios).

When a second low-exposure frame of the same static sample is available, the
Noise2Noise proxy estimates the error against the unknown clean signal from the
pair. That estimate is the closest reference-free substitute for a clean-target
MSE.

All functions accept an optional boolean `exclude` mask (True = exclude), which
the GUI uses for the beam-stop disk.
"""

from __future__ import annotations

import numpy as np


def robust_sigma(values) -> float:
    """Median-absolute-deviation estimate of the standard deviation."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return 1.4826 * mad


def background_mask(image, block=32, quantile=0.20, exclude=None):
    """Select low-intensity blocks as a background proxy.

    The image is divided into `block` x `block` tiles. The tiles whose median
    intensity lies in the lowest `quantile` are treated as background. This is
    more robust than a fixed corner: a corner that happens to contain scattering
    signal will not be selected if brighter than the true background tiles.
    """
    image = np.asarray(image, dtype=np.float64)
    h, w = image.shape
    n_y, n_x = h // block, w // block
    if n_y == 0 or n_x == 0:
        mask = np.isfinite(image)
        if exclude is not None:
            mask &= ~exclude
        return mask

    medians = []
    boxes = []
    for iy in range(n_y):
        for ix in range(n_x):
            y0, y1 = iy * block, (iy + 1) * block
            x0, x1 = ix * block, (ix + 1) * block
            patch = image[y0:y1, x0:x1]
            valid = np.isfinite(patch)
            if exclude is not None:
                valid &= ~exclude[y0:y1, x0:x1]
            if not valid.any():
                continue
            med = float(np.median(patch[valid]))
            medians.append(med)
            boxes.append((y0, y1, x0, x1, med))

    if not medians:
        return np.zeros_like(image, dtype=bool)

    # Prefer positive blocks. Zero-median blocks are usually the zeroed beam
    # stop or clipped pixels, and including them would make the background
    # sigma exactly zero even when the real background carries counting noise.
    positive = [m for m in medians if m > 0]
    pool = positive if positive else medians
    cutoff = float(np.quantile(pool, quantile))
    mask = np.zeros_like(image, dtype=bool)
    for y0, y1, x0, x1, med in boxes:
        if med <= cutoff and (med > 0 or not positive):
            mask[y0:y1, x0:x1] = True
    if exclude is not None:
        mask &= ~exclude
    return mask


def signal_mask(image, quantile=0.99, exclude=None):
    """Select the brightest pixels as a signal proxy."""
    image = np.asarray(image, dtype=np.float64)
    valid = np.isfinite(image)
    if exclude is not None:
        valid &= ~exclude
    values = image[valid]
    if values.size == 0:
        return np.zeros_like(image, dtype=bool)
    cutoff = float(np.quantile(values, quantile))
    return valid & (image >= cutoff)


def snr_metrics(raw, den, exclude=None, block=32, bg_quantile=0.20,
                sig_quantile=0.99):
    """Background sigma, signal level, SNR and SNR gain.

    The same background and signal pixel sets are used for the noisy and
    denoised images, so the two SNRs are comparable.
    """
    raw = np.asarray(raw, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    bg = background_mask(raw, block=block, quantile=bg_quantile,
                         exclude=exclude)
    sig = signal_mask(raw, quantile=sig_quantile, exclude=exclude)
    bg &= np.isfinite(den)
    sig &= np.isfinite(den)

    if not bg.any() or not sig.any():
        return {
            "sigma_bg_raw": float("nan"),
            "sigma_bg_den": float("nan"),
            "sigma_ratio": float("nan"),
            "mu_sig_raw": float("nan"),
            "mu_sig_den": float("nan"),
            "snr_raw": float("nan"),
            "snr_den": float("nan"),
            "snr_gain": float("nan"),
            "n_bg": int(bg.sum()),
            "n_sig": int(sig.sum()),
        }

    sigma_bg_raw = robust_sigma(raw[bg])
    sigma_bg_den = robust_sigma(den[bg])
    mu_sig_raw = float(np.median(raw[sig]))
    mu_sig_den = float(np.median(den[sig]))
    snr_raw = mu_sig_raw / sigma_bg_raw if sigma_bg_raw > 0 else float("nan")
    snr_den = mu_sig_den / sigma_bg_den if sigma_bg_den > 0 else float("nan")

    return {
        "sigma_bg_raw": sigma_bg_raw,
        "sigma_bg_den": sigma_bg_den,
        "sigma_ratio": (sigma_bg_den / sigma_bg_raw
                        if sigma_bg_raw > 0 else float("nan")),
        "mu_sig_raw": mu_sig_raw,
        "mu_sig_den": mu_sig_den,
        "snr_raw": snr_raw,
        "snr_den": snr_den,
        "snr_gain": snr_den / snr_raw if snr_raw and snr_raw > 0 else float("nan"),
        "n_bg": int(bg.sum()),
        "n_sig": int(sig.sum()),
    }


def residual_metrics(raw, den, exclude=None):
    """Residual mean, standard deviation and lag-1 autocorrelation.

    A good denoiser leaves a residual with near-zero mean, a standard deviation
    comparable to the noise level, and little spatial correlation. Structure in
    the residual means signal was removed.
    """
    raw = np.asarray(raw, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    valid = np.isfinite(raw) & np.isfinite(den)
    if exclude is not None:
        valid &= ~exclude

    if not valid.any():
        return {
            "res_mean": float("nan"),
            "res_std": float("nan"),
            "res_lag1": float("nan"),
        }

    residual = np.where(valid, den - raw, 0.0)
    values = residual[valid]
    signal = den[valid]
    mean = float(np.mean(values))
    std = float(np.std(values))

    # Lag-1 autocorrelation over horizontal and vertical neighbours. Invalid
    # pixels are zero in `residual`, so their products do not contribute.
    numerator = float(np.sum(residual[:, :-1] * residual[:, 1:])
                      + np.sum(residual[:-1, :] * residual[1:, :]))
    denominator = float(np.sum(residual * residual))
    lag1 = numerator / denominator if denominator > 0 else float("nan")

    # Correlation between the residual and the denoised signal. A denoiser that
    # removes only noise leaves a residual uncorrelated with the signal. A
    # denoiser that over-smooths leaves part of the signal in the residual, so
    # this correlation moves away from zero.
    if (values.size > 1 and np.std(values) > 0
            and np.std(signal) > 0):
        signal_corr = float(np.corrcoef(values, signal)[0, 1])
    else:
        signal_corr = float("nan")

    return {
        "res_mean": mean,
        "res_std": std,
        "res_lag1": lag1,
        "res_signal_corr": signal_corr,
    }


def residual_whiteness(raw, den, exclude=None):
    """Spectral flatness of the residual.

    W is close to 1 when the residual is white noise and falls when the
    residual retains structure (rings, streaks, peaks). It measures the shape
    of the residual spectrum, not its amplitude; report it together with the
    residual standard deviation only if the amplitude matters.
    """
    raw = np.asarray(raw, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    valid = np.isfinite(raw) & np.isfinite(den)
    if exclude is not None:
        valid &= ~exclude
    if not valid.any():
        return float("nan")

    residual = np.where(valid, den - raw, 0.0)
    residual = residual - np.mean(residual[valid])
    window = np.outer(np.hanning(residual.shape[0]),
                      np.hanning(residual.shape[1]))
    residual = residual * window
    power = np.abs(np.fft.fftshift(np.fft.fft2(residual))) ** 2

    h, w = residual.shape
    yy, xx = np.ogrid[:h, :w]
    radius = np.sqrt((yy - (h - 1) / 2.0) ** 2
                     + (xx - (w - 1) / 2.0) ** 2)
    keep = radius > 2.0  # drop DC and its immediate neighbours
    power = power[keep]
    if power.size == 0:
        return float("nan")

    # A floor keeps zero bins out of the logarithm; it is far below any
    # meaningful power in these images.
    power = np.maximum(power, 1e-30)
    geometric = float(np.exp(np.mean(np.log(power))))
    arithmetic = float(np.mean(power))
    return geometric / arithmetic if arithmetic > 0 else float("nan")


def high_frequency_metrics(raw, den, fc_fraction=0.25):
    """High- and low-frequency power ratios between the denoised and raw images.

    The cutoff is `fc_fraction` of half the shorter image dimension. A Hann
    window suppresses edge leakage. HFPR < 1 means high-frequency noise was
    removed; LFPR close to 1 means low-frequency signal was preserved.
    """
    raw = np.asarray(raw, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    den = np.nan_to_num(den, nan=0.0, posinf=0.0, neginf=0.0)
    raw = raw - raw.mean()
    den = den - den.mean()

    window = np.outer(np.hanning(raw.shape[0]), np.hanning(raw.shape[1]))
    raw = raw * window
    den = den * window

    p_raw = np.abs(np.fft.fftshift(np.fft.fft2(raw))) ** 2
    p_den = np.abs(np.fft.fftshift(np.fft.fft2(den))) ** 2

    h, w = raw.shape
    yy, xx = np.ogrid[:h, :w]
    radius = np.sqrt((yy - (h - 1) / 2.0) ** 2
                     + (xx - (w - 1) / 2.0) ** 2)
    fc = fc_fraction * 0.5 * min(h, w)
    high = radius > fc
    low = radius <= fc

    hf_raw = float(p_raw[high].sum())
    hf_den = float(p_den[high].sum())
    lf_raw = float(p_raw[low].sum())
    lf_den = float(p_den[low].sum())

    hfpr = hf_den / hf_raw if hf_raw > 0 else float("nan")
    lfpr = lf_den / lf_raw if lf_raw > 0 else float("nan")
    ratio = (hfpr / lfpr
             if np.isfinite(hfpr) and np.isfinite(lfpr) and lfpr > 0
             else float("nan"))

    return {
        "hfpr": hfpr,
        "lfpr": lfpr,
        "hfpr_over_lfpr": ratio,
        "fc_cutoff_px": float(fc),
    }


def n2n_proxy(raw, den, paired, exclude=None):
    """Noise2Noise estimate of the clean-target MSE from a second noisy frame.

    The two low-exposure frames are assumed to share the same clean signal and
    to have independent zero-mean noise of equal variance. Under those
    assumptions:

        sigma^2 = 0.5 * MSE(raw, paired)
        MSE_clean(den) = MSE(den, paired) - sigma^2

    and the identity mapping has MSE_clean = sigma^2. A positive `n2n_gain`
    means the denoiser improves on the noisy input.
    """
    raw = np.asarray(raw, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    paired = np.asarray(paired, dtype=np.float64)

    if raw.shape != paired.shape:
        raise ValueError("paired frame shape does not match the noisy input")

    valid = np.isfinite(raw) & np.isfinite(den) & np.isfinite(paired)
    if exclude is not None:
        valid &= ~exclude
    if not valid.any():
        return {
            "n2n_sigma2": float("nan"),
            "n2n_clean_mse": float("nan"),
            "n2n_gain": float("nan"),
            "n2n_identity_mse": float("nan"),
        }

    raw_v = raw[valid]
    den_v = den[valid]
    paired_v = paired[valid]

    sigma2 = 0.5 * float(np.mean((raw_v - paired_v) ** 2))
    mse_den_paired = float(np.mean((den_v - paired_v) ** 2))
    clean_mse = mse_den_paired - sigma2
    gain = sigma2 - clean_mse

    return {
        "n2n_sigma2": sigma2,
        "n2n_clean_mse": clean_mse,
        "n2n_gain": gain,
        "n2n_identity_mse": sigma2,
    }


def compute_all(raw, den, exclude=None, block=32,
                bg_quantile=0.20, sig_quantile=0.99):
    """Compute the minimal reference-free metric set for one frame.

    The GUI reports background sigma, SNR and residual whiteness.
    residual_metrics(), high_frequency_metrics() and n2n_proxy() remain
    available as optional diagnostics but are deliberately not part of the
    minimal panel.
    """
    out = {}
    out.update(snr_metrics(raw, den, exclude=exclude, block=block,
                           bg_quantile=bg_quantile,
                           sig_quantile=sig_quantile))
    out['res_whiteness'] = residual_whiteness(raw, den, exclude=exclude)
    return out
