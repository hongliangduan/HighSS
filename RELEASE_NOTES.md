# HighSS configuration notes

## Disulfide loss: enabled or disabled

Disulfide loss is enabled by default. Use one switch for the standard comparison:

| Setting | Command option | Behavior |
|---|---|---|
| Enabled (default) | `--use_disulfide_loss true` | Standard distogram and geometry objectives with backpropagation |
| Disabled | `--use_disulfide_loss false` | No explicit disulfide-loss terms or geometry gradient steps |

Omitting the option selects the enabled setting. The geometry gradient passes through the final denoising operation; earlier
sampling steps do not retain a gradient graph. There is no separate gradient option to add for a standard run.

## Standard objective

The standard disulfide objective includes the distogram pairing term and coordinate-based SG-SG distance, worst-pair, angle and
dihedral terms. The distogram loss is -log P(CB-CB distance in the bonding window), averaged over prescribed pairs, with weight
4.0 and a 3.3-4.8 A window. Geometry terms are evaluated during structure-prediction stages and use the scheduled weights in
`docs/PARAMETERS.md`.

The default input settings are `--loop_fix_cys true`, `--pre_drop_bonds true` and `--loop_profile token`. The optimization budget
is 30 pre-run, 80 soft, 45 temperature and 5 hard iterations, followed by 2 semi-greedy steps.

CB-CB guidance and optimization losses do not guarantee final SG-SG geometry. Screen generated Holo/Apo structures with
`scripts/check_disulfide_geometry.py`.

## Disabled-loss control

`--use_disulfide_loss false` sets `--disulfide_loss`, `--disulfide_loss_soft`, `--disulfide_loss_temp`, `--disulfide_loss_hard`,
`--disulfide_worst_pair_loss`, `--disulfide_angle_loss`, `--disulfide_dihedral_loss`, `--disulfide_distogram_loss`,
`--disulfide_grad_steps`, `--disulfide_sds_scale` and `--debug_sds_profile_every` to zero. An explicitly nonzero value for one of
these options together with the switch is an error. Setting only `--disulfide_loss 0` is insufficient because stage weights and
the distogram term may remain active.

Prescribed Cys, pairs and input settings remain. Compare the two settings using the same target, checkpoint, seed and
optimization budget to measure the contribution of the explicit loss terms. The pairing can still influence other objectives
through input conditioning in the disabled setting.

## Commands and diagnostics

- `bash examples/run_highss_disulfide_modes.sh on` runs one MDM2 example with disulfide loss enabled.
- `bash examples/run_highss_disulfide_modes.sh off` runs the same setup without explicit disulfide loss. Omitting the argument
  selects `on`. Checkpoint, CCD path, GPU, seed, sample count and output settings can be supplied through environment variables.
- The run header reports the loss and geometry-backpropagation status. `--debug_grad_norm` reports the combined gradient of all
  active disulfide terms; it does not isolate individual geometry components.
- Individual weights, diffusion-gradient depth, the optional input/loss preset and auxiliary denoiser-estimate terms are advanced
  settings in `docs/PARAMETERS.md`. Auxiliary terms and their profiling diagnostics are disabled by default.
- `--seed` controls random draws; without it, a run is unseeded. `--set_train` defaults to `true`.
- Epoch logs include `sg_sg:` distances, with `[seed]` and `[loop_seq]` entries.
- `scripts/plot_loss_trajectory.py` plots total loss, weighted components, raw terms, SG-SG distances and semi-greedy iPTM from a
  design log. Example logs and figures are in `examples/loss_trajectory/`.

## Installation and file paths

The setup script installs the dependencies declared by HighSS and the bundled Boltz runtime, including the pinned PyTorch
2.11.0 CUDA 12.8 build. It runs dependency and import checks and a relative-path regression check.

`--target_pdb`, `--checkpoint`, `--ccd_path` and `--work_dir` are resolved against the launch directory before the program changes
its working directory. A relative checkpoint path is therefore independent of the selected work directory.

The model checkpoint and chemical component dictionary are supplied separately:

- `checkpoints/com-FT.ckpt`
- `checkpoints/ccd.pkl`

`--checkpoint` and `--ccd_path` are required. A custom checkpoint does not require `boltz1_conf.ckpt`. Design supports no-MSA and
target-only MSA inputs, automatic Holo/Apo prediction, and local checkpoint and CCD/cache paths.
