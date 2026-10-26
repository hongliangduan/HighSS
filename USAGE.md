# HighSS usage notes

## Required inputs

Provide a local target PDB file, the target chain IDs, the binder length, the cysteine pairing, a model checkpoint and a local
chemical component dictionary. Residue positions in `--disulfide_pairs` refer to the designed binder, starting at 1.

The model checkpoint and dictionary are supplied separately:

- `checkpoints/com-FT.ckpt`
- `checkpoints/ccd.pkl`

## MSA

Target-only MSA is enabled by default (`--msa_mode target`). MSA is generated for target chains only; the binder remains
single-sequence. Use `--msa_mode none` to disable MSA generation. The target-MSA mode requires access to the MSA service.

## Disulfide loss

| Setting | Command option | Behavior |
|---|---|---|
| Enabled (default) | `--use_disulfide_loss true` | Enables the disulfide loss; the distogram term is backpropagated to the sequence logits |
| Disabled | `--use_disulfide_loss false` | Removes all explicit disulfide-loss terms and their backpropagation |

Omitting the option selects the enabled setting. The prescribed Cys positions and bond-input settings remain when the loss is
disabled. Use the master switch to disable the full loss; setting an individual loss weight to zero affects only that term.

## Example commands

- `bash examples/run_highss_disulfide_modes.sh on` runs an MDM2 design with disulfide loss and backpropagation enabled.
- `bash examples/run_highss_disulfide_modes.sh off` runs an MDM2 design without explicit disulfide loss.

Omitting the script argument selects `on`. Set `CHECKPOINT`, `CCD_PATH`, `GPU_ID`, `MSA_MODE`, `DESIGN_SAMPLES`, `SEED`, `WORK_DIR` or `SUFFIX`
to configure the checkpoint, dictionary, device, sample count and output location.

Additional examples are provided for [no-MSA design](examples/run_highss_single.sh) and
[target-only MSA design](examples/run_highss_target_msa.sh).

The default optimization settings are 30 pre-run, 80 soft, 45 temperature and 5 hard iterations, followed by 2 semi-greedy steps.
Parameter definitions are in `docs/PARAMETERS.md`.

## Logs and output files

The run header prints the effective settings. Epoch logs contain loss terms and sampled SG-SG distances. Output directories
contain sequence records and Holo/Apo prediction files.

`scripts/plot_loss_trajectory.py` reads a design log and writes PNG and PDF figures plus a CSV with per-step loss terms, weights,
totals and SG-SG distances. Example plotting inputs are in `examples/loss_trajectory/`.

## Installation and file paths

The setup script installs the dependencies declared by HighSS and the bundled Boltz runtime, including the pinned PyTorch
2.11.0 CUDA 12.8 build.

`--target_pdb`, `--checkpoint`, `--ccd_path` and `--work_dir` are resolved against the launch directory before HighSS changes its
working directory. Both `--checkpoint` and `--ccd_path` are required.

HighSS supports no-MSA and target-only MSA inputs, automatic Holo/Apo prediction, and local checkpoint and dictionary paths.
