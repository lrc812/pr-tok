# PAFR-VQ: Predictability-Aware Frequency-Adaptive Residual VQ

PAFR-VQ is a research prototype for testing whether a fixed budget of residual
tokens should be assigned to high-error/high-frequency locations rather than
uniformly everywhere. It is deliberately small and deterministic enough for a
CPU smoke test; it does not download data, weights, or start distributed jobs.

```text
image ──> frozen base tokenizer ──> base IDs + base reconstruction
   │                  │
   └─ residual/error scorer ─> exact top-B mask ─> residual VQ IDs
                                                    │
base features + sparse residual features ─> CNN residual decoder ─> final reconstruction
```

The base tokenizer remains fixed by default. Sparse and dense residual modes
use the same encoder, codebook, and decoder; dense mode simply selects every
latent position. Inactive positions are represented by `NULL_CODE_ID == K_r`.
Mask combinatorial bits are reported separately from residual-code bits.

## Installation and data

Use the repository environment (`uv sync` or the existing Conda environment).
The included commands run on synthetic images. For a real small experiment,
set `data.type: image_folder` and `data.root: /path/to/images`; that directory
may stay outside Git and is never downloaded by these scripts. Images are read
recursively and resized to `image_size`.

`TinyBaseTokenizer` is only a no-checkpoint fallback. To use a trained VQGAN,
construct `VQGANAdapter.from_checkpoint(config_yaml, checkpoint_path)` in a
custom experiment, call `freeze()`, and pass it to `PAFRTokenizer`. Checkpoints
are loaded with `weights_only=True` and must be tensor-only state dictionaries.

## Commands

Run all commands from this repository root:

```bash
python scripts/smoke_test.py --device cpu
python scripts/train_residual_tokenizer.py --config configs/residual_stage1.yaml --max-steps 10
python scripts/evaluate_tokenizer.py --config configs/residual_stage1.yaml --max-batches 2
python scripts/train_residual_prior.py --config configs/prior_stage2.yaml --max-steps 10
python scripts/train_joint.py --config configs/joint_stage3.yaml --max-steps 10
pytest -q tests
```

The phase-3 command is a minimal, stable post-training path: the residual AR
prior is frozen and its code logits supervise the quantizer's soft assignment.
For a real run, first train and load the Phase-2 prior rather than relying on
the randomly initialized smoke-test prior.

## Configuration

- `scorer.type`: `random`, `pixel`, `sobel`, `haar_dwt`, or `hybrid`.
- `selector.active_ratio` / `selector.budget`: exactly one allocation budget.
- `residual.dim`, `residual.codebook_size`: residual token capacity.
- `loss.*`: L1, DWT high-frequency, gradient and VQ weights; omitted terms are
  disabled. LPIPS is intentionally not imported by the smoke path.
- `prior.*`: transformer width/depth/heads for `MaskPrior` and `ResidualARPrior`.

Outputs contain a tensor-only checkpoint and JSON scalar metrics. Metrics
include L1/MSE/PSNR, DWT-band errors, Sobel error, codebook utilization,
perplexity, active ratio, base/residual/mask bits and bits/pixel.

## Implemented scope and limitations

Implemented: fixed Haar DWT, pixel/Sobel/DWT/hybrid/random allocation, exact
top-k selection, sparse/dense residual VQ, NULL-grid packing, bitrate bounds,
MaskPrior, causal packed ResidualARPrior, a soft predictability post-training
path, CPU smoke training and safe checkpoint round-trip tests.

Not implemented: LPIPS/rFID, inverse DWT, EMA residual codebook, learned gate,
KV-cache sampling, full VQGAN config CLI plumbing, and large ImageNet/gFID
experiments. Those require a real base tokenizer checkpoint and controlled
experiment setup; they are not silently attempted by this prototype.

## First real experiment

Freeze one 256px VQGAN, retain the residual decoder architecture, and compare
`random`, `pixel`, `sobel`, `haar_dwt`, and `hybrid` at active ratios 0.10,
0.25, 0.50 and 1.00. Report reconstruction, DWT/Sobel errors, residual energy
capture, code utilization, base/residual/mask bits and prior NLL before making
claims about downstream generator FID or convergence.
