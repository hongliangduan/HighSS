# HighSS fine-tuning

This directory contains the reference Boltz-1 fine-tuning entry point,
portable monomer and complex configurations, data-processing utilities and
evaluation scripts associated with HighSS.

Associated manuscript: *Programmable disulfide topology enables de novo design
of multi-disulfide peptide binders*.

The public inference and design workflow is documented in the repository root.
Large checkpoints and the chemical component dictionary are distributed
separately and are intentionally not stored in GitHub.

## Contents

```text
training/
├── train.py
├── validate_config.py
├── requirements.txt
├── configs/
│   ├── highss_mon_ft.yaml
│   └── highss_com_ft.yaml
├── process/
├── eval/
├── data/README.md
├── checkpoints/README.md
├── outputs/README.md
├── PARAMETER_SOURCES.md
└── SECURITY.md
```

No training structures, MSAs, processed tensors, validation splits or model
weights are included.

## Training sequence

1. Use the Boltz-1-compatible source/runtime associated with the HighSS release.
2. Place the released base Boltz-1 checkpoint at
   `training/checkpoints/boltz1.ckpt`.
3. Prepare the monomer data and validation IDs as described in
   `training/data/README.md`.
4. Validate the monomer configuration and start monomer fine-tuning:

```bash
python training/validate_config.py training/configs/highss_mon_ft.yaml
python training/train.py training/configs/highss_mon_ft.yaml
```

The selected checkpoint is copied to `training/outputs/mon_ft/mon-FT.ckpt`.
Copy or link it to `training/checkpoints/mon-FT.ckpt` before complex
fine-tuning.

5. Prepare the complex data and validation IDs, then run:

```bash
python training/validate_config.py training/configs/highss_com_ft.yaml
python training/train.py training/configs/highss_com_ft.yaml
```

The selected checkpoint is copied to `training/outputs/com_ft/com-FT.ckpt`.

## Portable path overrides

Machine-specific paths should be supplied as OmegaConf dot-list overrides,
for example:

```bash
python training/train.py training/configs/highss_mon_ft.yaml \
  pretrained=/path/to/boltz1.ckpt \
  data.datasets.0.target_dir=/path/to/monomer/processed \
  data.datasets.0.msa_dir=/path/to/monomer/processed/msa \
  data.datasets.0.split=/path/to/monomer/validation_ids.txt \
  output=/path/to/output/mon_ft
```

## Software compatibility

The scripts use the legacy `boltz.model.model.Boltz1` training API. Pin the
Boltz source version used for the study; a newer upstream release may require
adaptation.

## Experiment tracking

Weights & Biases is disabled unless a `wandb` block is enabled. Authenticate
through the W&B CLI or environment variables. Do not store credentials in
configuration files.

## Reproducibility boundary

These are reference fine-tuning scripts and configurations. Exact checkpoint
reproduction additionally requires the original validation-ID files, processed
training inputs and the exact Boltz source revision. Values not explicitly
reported in the manuscript are identified in `training/PARAMETER_SOURCES.md`.
