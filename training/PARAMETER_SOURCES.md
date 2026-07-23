# Parameter provenance

## Directly stated in the manuscript

- sequential training: released Boltz-1 weights → mon-FT → com-FT;
- mon-FT trained on 1,610 monomers; com-FT on 398 complexes;
- structure module only; confidence head inherited unchanged;
- best-validation epochs: 57 (mon-FT) and 68 (com-FT);
- Adam: beta1=0.9, beta2=0.95, epsilon=1e-8;
- AlphaFold3-style schedule, maximum learning rate 1.8e-3 after 1,000 warm-up steps;
- float32, one NVIDIA A800 80 GB GPU;
- gradient clipping 10.0;
- EMA decay 0.999;
- gradient accumulation 64;
- cluster-based sampling;
- maximum crop: 384 tokens / 3,456 atoms;
- MSA features used during fine-tuning;
- stock Boltz-1 structure objective: diffusion, distogram and smooth-LDDT losses;
- three recycling steps for reported predictions.

## Retained from the original sanitized training template / Boltz defaults

The manuscript does not enumerate every architecture and diffusion setting.
The released configs retain the values present in the original training YAML,
including 400 samples per epoch, batch size 1, seed 42, 200 diffusion sampling
steps during training/validation, and the remaining Boltz-1 architecture and
diffusion-process parameters.

## Explicit release assumptions

- `max_epochs: 58` and `69` are selected so zero-based Lightning epoch indices
  57 and 68 are reached. They should not be interpreted as proof that the
  original jobs stopped immediately after those epochs.
- Exact monomer and complex validation-ID files were not recoverable from the
  manuscript and are not fabricated here.
- The Boltz source commit used for the reported experiments must be pinned from
  the authors' original environment or from the vendored Boltz runtime in the
  HighSS release. Current upstream `main` may not be training-compatible with
  these legacy `boltz.model.model.Boltz1` imports.
