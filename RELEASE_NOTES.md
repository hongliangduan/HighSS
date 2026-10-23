# HighSS v0.2.1

Adds a master switch for the disulfide loss and a script that draws the loss trajectory of a design. A default run is unchanged:
it uses the recipe of v0.2.0.

## New option

| Option | Default | Effect |
|---|---|---|
| `--use_disulfide_loss` | `true` | `false` = no disulfide loss at all |

- `--use_disulfide_loss true` (default) is the recipe of v0.2.0 described below: the distogram term, which acts on the
  sequence, and the structure terms, which are computed from sampled coordinates and, with `--disulfide_grad_steps 0` and
  `--disulfide_sds_scale 0`, are only logged.
- `--use_disulfide_loss false` removes the disulfide loss in every stage: no distogram term, no structure term (distance, worst
  pair, angle, dihedral), no gradient steps through the diffusion model, no SDS terms, and no disulfide entry in the loss line
  of an epoch. The switch sets `--disulfide_loss`, `--disulfide_loss_soft`, `--disulfide_loss_temp`, `--disulfide_loss_hard`, `--disulfide_worst_pair_loss`, `--disulfide_angle_loss`, `--disulfide_dihedral_loss`, `--disulfide_distogram_loss`, `--disulfide_grad_steps`, `--disulfide_sds_scale` and `--debug_sds_profile_every`
  to 0; a non-zero value given for one of them together with the switch is an error.
- The stage weights are in that list because, with `--disulfide_schedule true`, they and not `--disulfide_loss` set the weight
  of the structure terms in each stage. In RC3 and v0.2.0, `--disulfide_loss 0` alone therefore did not switch the structure
  terms off.
- Not touched by the switch: the prescribed pairs still fix the Cys positions and give the bond feature of the model input, and
  `--loop_fix_cys`, `--pre_drop_bonds` and `--loop_profile` keep their values (they are input settings, not losses). Add
  `--rc3_defaults true` to run them with their RC3 values as well.
- `--use_disulfide_loss false` is not the RC3 behaviour (`--rc3_defaults true`): RC3 had the structure terms, which were logged
  only; the switch has none.
- The run header (`loop guidance:` line) prints `disulfide loss OFF` when the switch is off.

## New script

- `scripts/plot_loss_trajectory.py` draws the loss trajectory of one design from the log of a run: total loss, weighted
  components, raw terms, SG-SG distances and semi-greedy iPTM. Example logs and figures: `examples/loss_trajectory/`.

# HighSS v0.2.0

The design loop now carries a gradient from the prescribed disulfide pairing to the sequence. In RC3 the disulfide terms were
computed from sampled structures and did not reach the sequence logits. v0.2.0 adds a distogram term that does, and switches on
the loop settings that the term works with (listed below).

## Changed defaults

The command line of v0.2.0 uses the five options below by default. `--rc3_defaults true` restores the RC3 values of all of them;
an option given explicitly still wins.

| Option | RC3 behaviour | v0.2.0 default |
|---|---|---|
| `--disulfide_distogram_loss` | `0.0` | `4.0` |
| `--disulfide_distogram_form` | `prob` | `nll` |
| `--loop_fix_cys` | `false` | `true` |
| `--pre_drop_bonds` | `false` | `true` |
| `--loop_profile` | `mean` | `token` |

- `--disulfide_distogram_loss 4.0` with `--disulfide_distogram_form nll`: the distogram term -log P(CB-CB of every prescribed pair
  in the bonding window), averaged over the pairs, is the path by which the pairing reaches the sequence logits. No structure
  sampling is needed.
- `--loop_fix_cys true`: the prescribed Cys are written into the features of every forward pass of the loop.
- `--pre_drop_bonds true`: the pre-run runs without the bond feature (the later stages and the final prediction keep it), so that
  the distogram term has something to optimize in the stage that sets the starting sequence of the later stages.
- `--loop_profile token`: every binder token gets its own residue as profile, as `boltz predict` does, instead of the averaged
  composition of the binder, so that the distogram optimized in the loop is the one the model gives in the final prediction.

Not changed: the structure-based disulfide terms and their schedule (`--disulfide_loss*`, `--disulfide_worst_pair_loss`, angle and
dihedral terms), `--disulfide_grad_steps 0`, `--disulfide_sds_scale 0`, the optimization schedule and `--semi_greedy_steps 2`.
Parameter details: `docs/PARAMETERS.md`, section 11.

## Notes

- Designs of v0.2.0 are not comparable with RC3 designs. Record the version, and add `--rc3_defaults true` to continue a campaign
  with the RC3 behaviour.
- The distogram term acts on CB-CB distances and does not guarantee SG-SG closure in the final prediction, so the final screening
  (`scripts/check_disulfide_geometry.py`) stays necessary.
- The run header prints the values in force (`loop guidance:` line).

## Other additions since RC3

Off by default, or diagnostic; the RC3 defaults are unchanged unless listed above.

- `--disulfide_grad_steps K` (default 0 = RC3 behaviour): with K >= 1 the disulfide geometry loss is backpropagated to the
  sequence logits through the last K diffusion steps.
- `--disulfide_sds_scale S` (default 0): disulfide terms on one-step denoiser estimates of the noised sampled structure, with
  autograd (`docs/PARAMETERS.md`, section 10).
- `--seed` (default: unseeded) and `--set_train` (default `true`, the original behaviour).
- `--loop_drop_bonds`, `--debug_grad_norm`, `--debug_sds_profile_every`: diagnostics.
- The `Epoch` lines of the log carry the SG-SG distances of the prescribed pairs (`sg_sg:`); the log also has `[seed]` and
  `[loop_seq]` lines.

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
