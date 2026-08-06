# VQ tokenizer reconstruction grid

This evaluation reuses the dataset target and preprocessing parameters saved by
training in a VQ log directory. It randomly selects 100 dataset entries and
writes one image with three columns:

1. tokenizer input (the training/evaluation preprocessing result)
2. decoder output from the continuous encoder latent, without codebook lookup
3. decoder output after vector quantization

Run from `taming-transformers`:

```bash
python evaluation/vae/reconstruction_grid.py \
  --vq-log-dir logs/2026-07-31T00-03-10_imagenet_larp_vqgan
```

The default dataset selection is `test`, falling back to `validation`. To pick
a split explicitly, use `--dataset-split train|validation|test`. A timestamped
result folder under `evaluation/vae/test_results/` contains both
`reconstruction_comparison.png` and `run_info.json`. The JSON records the VQ log
directory, project config, checkpoint, dataset target and parameters, random
indices, source paths, and output paths.

Useful options:

```bash
python evaluation/vae/reconstruction_grid.py \
  --vq-log-dir /path/to/vq-log-dir \
  --dataset-split validation \
  --seed 23 \
  --batch-size 8 \
  --device cuda:0 \
  --precision fp32 \
  --output-dir evaluation/vae/test_results/my_run
```
