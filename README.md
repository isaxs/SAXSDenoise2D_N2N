This Python script trains a **Noise2Noise (N2N) denoising neural network** specifically tailored for **2D Small-Angle X-ray Scattering (SAXS)** data. Here's a concise overview of its functionality:

**Core Architecture & Domain**
- Implements a custom **SAXSAttentionUNet** that operates in the **log-domain** (`log1p`/`expm1`) to handle SAXS's intrinsic multi-order-of-magnitude intensity range (from background noise to sharp Bragg peaks).
- Integrates **multi-scale ASPP (dilated convolutions)** to capture both local sharp peaks and global diffuse scattering simultaneously.
- Uses **Attention Gates** (Oktay et al.) to adaptively weight features based on signal-dependent noise—preserving detail in high-intensity regions while aggressively denoising faint background areas.
- Employs **Channel Attention (Squeeze-and-Excitation)** and **SiLU activations** for stable gradients across the dynamic range.

**Data Handling**
- Preloads entire HDF5 datasets into RAM as `float32` (cropped from 1028×1028 to 1024×1024) for maximum I/O efficiency.
- Requires a fixed disk split into `dataset/train`, `dataset/val`, and `dataset/test` subdirectories, ensuring strict isolation of test data (v08 feature).

**Training Strategy**
- Supports **Distributed Data Parallel (DDP)** for multi-GPU training with NCCL backend.
- Performs an **automatic, memory-aware batch size search** (exponential + binary search with prediction) using a temporary model, respecting a safety margin to prevent OOM errors.
- Uses **bfloat16 mixed precision** by default (v08) to avoid overflow in the unnormalized residual decoder, while maintaining FP32-like dynamic range.
- Implements **gradient accumulation** to achieve a larger effective batch size without increasing memory footprint.

**Optimization & Scheduling**
- Employs **Adam optimizer** (with fused kernel support when available) and **Cosine Annealing with Warm Restarts** (`T_0=50`, `T_mult=2`, `eta_min=1e-6`).
- Includes **early stopping** with a patience of 40 epochs over a total of 500 epochs.

**Diagnostics & Monitoring**
- Logs detailed per-epoch diagnostics to **JSONL**, including train/val loss, gradient L2 norm, weight L2 norm, update-to-weight ratio, attention-gate alpha statistics (mean/std/min/max/fraction > 0.5), and correction statistics (mean/abs/positive fraction).
- Writes scalars and sample images to **TensorBoard**.
- Persists a **train/val/test loss curve plot** (via `matplotlib`) that survives checkpoint resumes (v08 feature).

**Checkpointing & Resilience**
- Performs **asynchronous checkpoint saving** to avoid stalling training.
- Robust checkpoint **resume logic** that checks for architecture mismatches (missing/unexpected layers, shape mismatches), detects corrupted weights (NaN/Inf), and gracefully falls back to training from scratch or skipping the optimizer state.
- Final inference weights are exported as a standalone, portable state dictionary.

**Key v08 Improvements**
- Fixed train/val/test disk split (no more runtime recomputation).
- Full English/ASCII source code.
- Periodic test loss evaluation integrated into the persistent loss curve.
- Cosine annealing with warm restarts replacing the previous scheduler.

In essence, this is a production-grade, memory-optimized, self-tuning training pipeline designed to deliver statistically optimal denoising for 2D SAXS images under the Noise2Noise theory (converging to the expected clean signal).
