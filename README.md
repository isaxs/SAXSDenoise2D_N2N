# SAXSDenoise2D_N2N

Self-supervised **Noise2Noise** denoising for low-exposure 2D small-angle X-ray
scattering (SAXS) patterns.

## Repository contents

| File | Description |
|---|---|
| `SAXSDenoise2D_N2N_10.py` | Training script, v10 (linear input, variance-matched loss) |
| `inference_gui_10.py` | Inference GUI for v10 checkpoints |
| `evaluate_denoising_gui.py` | Evaluation tool: agreement metrics, ratio/difference panels, per-*q* residuals |
| `denoise_metrics.py` | Metric helpers used by the inference and evaluation GUIs |
| `model_inference.zip.001` ... `.003` | Split archive of the trained v10 inference weights |
| `PVDF_60s.edf` | Raw 60 s PVDF frame (Figure 2A) |
| `PVDF_3600s_div60.edf` | 3600 s reference divided by 60, matched to the 60 s intensity scale (Figure 2C) |
| `PVDF_0_00004.edf` | Additional 1200 s PVDF frame |
| `SAXSDenoise2D_N2N_08.py`, `SAXSDenoise2D_N2N_09.py` | Earlier training versions |
| `inference_gui_08.py`, `inference_gui_09.py` | Earlier inference builds |

The long-exposure frame is a **less noisy measurement, not ground truth**. It
carries its own counting noise, and the evaluation tool reports that noise floor
together with a floor-corrected RMSE.

## Method summary

- Training pairs are two consecutive low-exposure frames of the same static
  specimen. No clean target is used at any stage.
- The network operates on the **linear intensity scale**. No logarithmic or
  other nonlinear transform is applied to the input or the target.
- The objective is a variance-matched relative loss:

  ```text
  L = mean( (f(y1) - y2)^2 / (sg[f(y1)] + eps) )
  ```

  where `sg[.]` is a stop-gradient, `eps = 1.0` in model-domain intensity
  units, `y2` enters untransformed, and the denominator is detached. One power
  of the prediction matches Poisson counting statistics, whose variance is
  proportional to the mean.
- Architecture: a four-level attention-gated U-Net with channel counts
  32/64/128/256, residual blocks with SiLU activations and squeeze-and-excitation
  channel attention (reduction 8), an ASPP bottleneck with dilation rates
  {1, 2, 4, 8}, and attention gates on the skip connections. Group normalization
  is used in the deeper encoder and decoder levels; the full-resolution levels
  are unnormalized. The output head applies a softplus activation, and a
  non-negative skip gain (initialized at zero) adds a direct linear path from
  input to output. The model has approximately 12.4 million parameters.

## Installation

The code was developed with Python 3.10 and PyTorch 2.x on an NVIDIA GPU.

```bash
pip install numpy h5py fabio pillow matplotlib tensorboard
# Install the PyTorch build that matches your CUDA version, for example:
# pip install torch --index-url https://download.pytorch.org/whl/cu121
```

The inference and evaluation GUIs use `tkinter`, which is included in the
standard Python distribution on Windows.

## Weights

The trained v10 weights are stored as a split archive. Reassemble and extract it
before running inference.

Windows:

```bat
copy /b model_inference.zip.001 + model_inference.zip.002 + model_inference.zip.003 model_inference.zip
```

Linux or macOS:

```bash
cat model_inference.zip.00* > model_inference.zip
```

Then unzip `model_inference.zip`. The extracted checkpoint is
`model_inference.pth`.

## Inference

```bash
python inference_gui_10.py
```

In the GUI:

1. Load `model_inference.pth` as the model checkpoint.
2. Select a noisy frame (`.edf`, `.h5`, `.npy`, `.tiff` or image formats).
3. Run denoising and save the output.

The GUI also supports batch processing and linear/log display modes. The beam
stop is masked during inference and restored in the saved output.

## Reproducing the Figure 2 evaluation

1. Denoise `PVDF_60s.edf` with `model_inference.pth` in `inference_gui_10.py`
   and save the result as `PVDF_60s_denoised.edf`.
2. Run the evaluation tool against the long-exposure reference:

```bash
python evaluate_denoising_gui.py \
  --low PVDF_60s.edf \
  --reference PVDF_3600s_div60.edf \
  --denoised PVDF_60s_denoised.edf \
  --denoised-scale counts \
  --out ./eval_PVDF_60s
```

The output directory contains:

- `metrics.txt` - RMSE in the linear and log domains, median relative error,
  bias, PSNR, the reference counting-noise floor and the floor-corrected RMSE;
- `q_residual.csv` - mean, RMS, Pearson residual and analytic envelope per
  radial *q* band;
- `count_levels.csv` - the same statistics grouped by count level;
- ratio and difference panels with the counting-statistics envelope;
- a per-*q* residual plot.

The `--checkpoint` option of the evaluation tool uses the v09 model loader. For
v10, denoise with `inference_gui_10.py` first and pass the result with
`--denoised`, as above.

## Training

The training script uses a configuration block near the bottom of
`SAXSDenoise2D_N2N_10.py` rather than command-line flags. Set at least:

```python
data_dir = "./dataset"
output_dir = "./models_v10"
log_dir = "./runs/noise2noise_v10"
resume_checkpoint = True   # False for a fresh run
```

The dataset directory must contain three subdirectories:

```text
dataset/
  train/
  val/
  test/
```

Each `.h5` file stores:

- `images`: an array of shape `(N, 1028, 512)` containing frames already divided
  by the per-group constant `S` (counts / `S`);
- `pairs`: an array of shape `(M, 2)` containing frame-index pairs. Both
  `(i, j)` and `(j, i)` may be stored; only the undirected pair (`i < j`) is used.

The loader crops frames from 1028 x 512 to 1024 x 512 and multiplies the
intensities by `10^3`, so the model sees approximately `10^3 * counts / S`
(model-domain values of about 0 to 1200). No nonlinear transform is applied.
The split is at the file level, so both frames of a pair always remain in the
same subset. The script refuses to run while unsplit `.h5` files are present in
the dataset root.

Run training with:

```bash
python SAXSDenoise2D_N2N_10.py
```

A single GPU is used automatically; multiple GPUs are launched as a DDP run.
The configuration uses Adam with a short linear warm-up, a single cosine decay
to `1e-6`, gradient clipping at a norm of 1.0, early stopping with a patience of
40 epochs, and a memory-bounded automatic batch-size search targeting an
effective batch size of 16.

Checkpoints written to `output_dir`:

- `best.pth` - best validation loss;
- `best_signal.pth` - best bright-pixel signal metric;
- `latest.pth` - latest epoch, used for resume;
- `model_inference.pth` - exported weights for inference.

## Key differences from v08 and v10

- Linear input and target; the `log1p`/`expm1` transform is removed.
- `VarianceMatchedLoss` replaces the log-domain MSE and the two-power
  high-dynamic-range loss.
- Group normalization in the deeper encoder and decoder levels.
- A non-negative skip gain, initialized at zero, instead of the v10 gain
  convention.
- A single cosine decay instead of cosine warm restarts, with gradient clipping.
- Model selection uses both the validation loss and the bright-pixel metric.

## Data and code availability

The code, the trained v10 weights and representative data are available in this
repository. The full training corpus and the remaining experimental data are
available from the corresponding authors upon reasonable request.
