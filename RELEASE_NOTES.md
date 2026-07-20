# HighSS v0.1.1-rc3

RC3 fixes relative-path handling when `--work_dir` is different from the
directory where HighSS is launched.

## RC3 fixes

- Resolves `--target_pdb`, `--checkpoint`, `--ccd_path`, and `--work_dir`
  against the launch directory before changing the current working directory.
- Prevents a relative checkpoint such as `checkpoints/model.ckpt` from being
  incorrectly interpreted as `<work_dir>/checkpoints/model.ckpt`.
- Adds an automated regression test for this exact failure to `setup.sh`.
- Retains the complete dependency installation and CUDA 12.8 validation added
  in RC2.

## RC2 fixes

- Removed `--no-deps` from the canonical installer.
- Installs all dependencies declared by HighSS and the vendored Boltz runtime.
- Adds `pip check` and broad runtime import validation.
- Keeps the validated PyTorch 2.11.0 CUDA 12.8 build pinned.
- Makes `--checkpoint` and `--ccd_path` explicit required arguments.
- Retains automatic Holo/Apo prediction and target-only MSA behavior.

## Required external files

- `checkpoints/com-FT.ckpt`
- `checkpoints/ccd.pkl`

The source package does not contain these large files and does not require
`boltz1_conf.ckpt`.

## Validated workflows

- clean-environment installation: PASS
- no-MSA automatic design and Holo/Apo prediction: PASS
- target-only MSA automatic design and Holo/Apo prediction: PASS
- custom checkpoint and local CCD/cache propagation: PASS
- relative path handling outside `work_dir`: PASS
- no `boltz1_conf.ckpt` creation with a custom checkpoint: PASS
