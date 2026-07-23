## Fine-tuning

Reference Boltz-1 fine-tuning scripts and configurations for the
monomer-fine-tuned (`mon-FT`) and complex-fine-tuned (`com-FT`) models are
provided in [`training/`](training/README.md).

The released checkpoints and chemical component dictionary are distributed
separately through Figshare and are not stored in this Git repository.

The public configurations document the training workflow and parameters used
for the associated study. Exact checkpoint reproduction additionally requires
the original processed inputs, validation splits and study-compatible Boltz
source revision; these boundaries are documented in
[`training/PARAMETER_SOURCES.md`](training/PARAMETER_SOURCES.md).
