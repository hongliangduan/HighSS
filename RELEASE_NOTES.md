# HighSS configuration notes

## Default disulfide guidance

Disulfide loss is enabled by default. The distogram pairing term supplies a gradient to the sequence logits; the coordinate-based
geometry terms are computed from sampled structures and enter the reported total loss. Direct geometry backpropagation is
controlled separately.

| Option | Default | Effect |
|---|---|---|
| `--use_disulfide_loss` | `true` | Enables explicit disulfide-loss terms |
| `--disulfide_distogram_loss` | `4.0` | Weight of the CB-CB pairing term |
| `--disulfide_distogram_form` | `nll` | Uses the negative log probability of the bonding window |
| `--disulfide_grad_steps` | `0` | No gradient through sampled coordinates |
| `--disulfide_sds_scale` | `0.0` | Denoiser-estimate geometry branch disabled |
| `--loop_fix_cys` | `true` | Fixes prescribed Cys in every loop forward pass |
| `--pre_drop_bonds` | `true` | Omits the bond feature during the pre-run only |
| `--loop_profile` | `token` | Uses a per-token binder profile |

The distogram term is -log P(CB-CB distance in the bonding window), averaged over the prescribed pairs. Its default window is
3.3-4.8 A. It is differentiable without structure sampling. CB-CB guidance does not guarantee SG-SG closure in the final prediction;
use `scripts/check_disulfide_geometry.py` to inspect generated structures.

The optimization defaults are 30 pre-run, 80 soft, 45 temperature and 5 hard iterations, followed by 2 semi-greedy steps.
Coordinate-based distance, worst-pair, angle and dihedral weights and their stage schedule are listed in `docs/PARAMETERS.md`.

## Geometry backpropagation

- `--disulfide_grad_steps K`: with K >= 1, the last K diffusion steps retain an autograd graph for the coordinate-based disulfide
  losses. Other coordinate-based terms use detached coordinates. Start with K = 1 when enabling this path.
- `--disulfide_sds_scale S`: with S > 0, applies geometry terms to denoiser estimates of the sampled structure at additional noise
  levels. This branch is disabled by default; details are in `docs/PARAMETERS.md`.
- `--disulfide_grad_steps 0` does not disable the distogram-loss gradient.

## Disabling explicit disulfide losses

`--use_disulfide_loss false` removes all disulfide-loss terms in every stage: the distogram term, coordinate-based distance,
worst-pair, angle and dihedral terms, and SDS terms. It also disables geometry gradient steps and SDS diagnostics. The switch
sets `--disulfide_loss`, `--disulfide_loss_soft`, `--disulfide_loss_temp`, `--disulfide_loss_hard`, `--disulfide_worst_pair_loss`,
`--disulfide_angle_loss`, `--disulfide_dihedral_loss`, `--disulfide_distogram_loss`, `--disulfide_grad_steps`,
`--disulfide_sds_scale` and `--debug_sds_profile_every` to 0. Explicitly requesting a nonzero value for one of these options
with the master switch off is an error.

Setting `--disulfide_loss 0` alone does not disable the whole objective: scheduled geometry weights and the distogram term may
remain active. Use the master switch for a control with no explicit disulfide loss.

The prescribed pairs still fix the Cys positions and supply the bond-input feature. `--loop_fix_cys`, `--pre_drop_bonds` and
`--loop_profile` keep their values because they control model inputs. The pairing can therefore still influence other objectives.
The run header prints `disulfide loss OFF` when the master switch is off.

## Optional input/loss preset

`--rc3_defaults true` selects the following preset. Explicit command-line options override preset values regardless of argument
order. Other options keep their values.

| Option | Preset value |
|---|---|
| `--disulfide_distogram_loss` | `0.0` |
| `--disulfide_distogram_form` | `prob` |
| `--loop_fix_cys` | `false` |
| `--pre_drop_bonds` | `false` |
| `--loop_profile` | `mean` |

This preset keeps the coordinate-based losses. With `--disulfide_grad_steps 0 --disulfide_sds_scale 0`, those terms enter the
reported total but contribute no sequence gradient. Use `--use_disulfide_loss false` to remove them as well.

## Commands, diagnostics and plots

- `examples/run_highss_disulfide_modes.sh` runs one MDM2 example in the selected mode: `none`, `default`, `geometry`, `both` or
  `preset`. The main README gives the complete options and environment overrides.
- `--debug_grad_norm` reports the combined gradient of active `disulfide*` terms. To compare K = 0 and K = 1 for the geometry path,
  set `--disulfide_distogram_loss 0 --disulfide_sds_scale 0` in both runs.
- `--loop_drop_bonds` and `--debug_sds_profile_every` support input and gradient diagnostics; both are disabled by default.
- `--seed` controls random draws; without it, a run is unseeded. `--set_train` defaults to `true`.
- The run header prints effective input/loss settings. Epoch logs include `sg_sg:` distances, with `[seed]` and `[loop_seq]` entries.
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
