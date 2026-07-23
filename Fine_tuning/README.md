# HighSS Boltz-1 fine-tuning code

This package contains the cleaned training entry point, monomer and complex
fine-tuning configurations, data-processing utilities and evaluation scripts
associated with HighSS.

Associated manuscript: *Programmable disulfide topology enables de novo design
of multi-disulfide peptide binders*.

Public inference and design code:
https://github.com/hongliangduan/HighSS

## What is included

```text
scripts/train/train.py
scripts/train/configs/highss_mon_ft.yaml
scripts/train/configs/highss_com_ft.yaml
scripts/train/validate_config.py
scripts/process/
scripts/eval/
docs/PARAMETER_SOURCES.md
data/README.md
```

No training structures, MSAs or processed tensors are included.

## Training sequence

1. Place the released base Boltz-1 checkpoint at `checkpoints/boltz1.ckpt`.
2. Prepare the monomer dataset and its validation IDs as described in
   `data/README.md`.
3. Run monomer fine-tuning:

```bash
python scripts/train/validate_config.py scripts/train/configs/highss_mon_ft.yaml
python scripts/train/train.py scripts/train/configs/highss_mon_ft.yaml
```

The best validation checkpoint is copied to `outputs/mon_ft/mon-FT.ckpt`.

4. Copy or link that file to `checkpoints/mon-FT.ckpt`.
5. Prepare the complex dataset and validation IDs.
6. Run complex fine-tuning:

```bash
python scripts/train/validate_config.py scripts/train/configs/highss_com_ft.yaml
python scripts/train/train.py scripts/train/configs/highss_com_ft.yaml
```

The best validation checkpoint is copied to `outputs/com_ft/com-FT.ckpt`.

## Path overrides

OmegaConf dot-list overrides avoid storing machine-specific paths in YAML:

```bash
python scripts/train/train.py scripts/train/configs/highss_mon_ft.yaml \
  pretrained=/path/to/boltz1.ckpt \
  data.datasets.0.target_dir=/path/to/monomer/processed \
  data.datasets.0.msa_dir=/path/to/monomer/processed/msa \
  data.datasets.0.split=/path/to/monomer/validation_ids.txt \
  output=/path/to/output/mon_ft
```

## Software compatibility

Use Python 3.10 and the Boltz-1 source/runtime vendored with the validated HighSS
release. The training code uses the legacy import
`boltz.model.model.Boltz1`; a newer upstream Boltz release may require API
adaptation. Pin the exact HighSS/Boltz commit in the archival record.

## Experiment tracking

Weights & Biases is disabled unless a `wandb` block is enabled. Authenticate via
the W&B CLI or environment; never store an API key in a configuration file.

## Reproducibility boundary

The configs use all parameters stated in the manuscript and retain unstated
values from the original sanitized training template. Exact validation splits
and the exact Boltz commit are still required for exact checkpoint reproduction.
See `docs/PARAMETER_SOURCES.md`.
