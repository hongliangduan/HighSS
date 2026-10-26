# HighSS parameter reference

This guide describes input, output, optimization and loss-weight options for running HighSS.

Disulfide loss is enabled by default. `--use_disulfide_loss false` removes every explicit disulfide-loss term (section 11).

The default values are defined in `highss/cli.py`.

## 1. Target input

### `--target_pdb`

- Type: path
- Required: yes
- Meaning: local PDB file containing the target structure.

Only the chains selected by `--target_chains` are used.

### `--target_chains`

- Type: comma-separated chain IDs
- Required: yes
- Examples: `A`, `A,B`

These are chain IDs in the input PDB. HighSS rewrites output chain assignments internally; the designed binder defaults to output chain `A`.

### `--target_name`

- Type: string
- Default: input PDB filename stem

Used in output names. It does not change the target structure.

### `--binder_chain`

- Type: chain ID
- Default: `A`

Output chain ID assigned to the designed binder. The target chains are remapped as needed so they do not collide with the binder chain.

Keep `A` unless a downstream workflow requires another chain ID. The binder is always the first chain of the input YAML and is
designed as such under any ID; the ID only names it in the YAML and in the predicted structures. The `config.yaml` saved with
the results lists the binder under the design engine's own key, `A`.

### `--contact_residues`

- Type: comma-separated PDB residue numbers
- Default: empty

Enables pocket conditioning toward specified residues. Give the residue numbers as they appear in the supplied PDB file (the
residue-number column of its ATOM records), with the insertion code appended where there is one, for example `52A`.

HighSS converts the numbers to 1-based positions in the sequence of the selected chain, writes those positions into the pocket
constraint and prints the conversion in the run header. A number that is not a standard residue of the chain stops the run before
the design starts.

Example (`examples/mdm2/mdm2_pro.pdb` numbers its residues 25–109):

```bash
--contact_residues "54,58,61,67"
```

This is a conditioning signal, not a guarantee that every listed residue forms a final contact.

### `--contact_target_chain`

- Type: input PDB chain ID
- Default: empty

Identifies the chain containing `--contact_residues`.

- Single-chain target: may be omitted.
- Multi-chain target: required when `--contact_residues` is used.

### `--msa_mode`

- Choices: `none`, `target`
- Default: `target`

`none`:
- binder MSA is empty;
- target chains are treated as single sequences.

`target`:
- binder MSA remains empty;
- MSA is generated only for target chains.

Target-only MSA is enabled when this option is omitted. The target-MSA mode requires access to the MSA service. Use `none` to disable MSA generation.

### `--msa_max_seqs`

- Type: integer
- Default: `4096`

Maximum number of MSA sequences passed to the model.

Lowering this value reduces memory and preprocessing cost. It may also reduce target evolutionary information. It has no practical effect in `--msa_mode none`.

## 2. Binder length and topology

### `--binder_length`

- Type: integer
- Required: yes
- Minimum accepted by CLI: `4`

The output binder has exactly this many residues.

HighSS sets both internal minimum and maximum length to this value; it does not sample a length range.

### `--disulfide_pairs`

- Type: comma-separated 1-based residue pairs
- Required: yes
- Example: `2-17,6-13`

The example fixes residues 2, 17, 6, and 13 to cysteine and requests two bonds:

```text
2 ↔ 17
6 ↔ 13
```

Rules:

- positions are 1-based;
- every position must lie between 1 and `binder_length`;
- a pair cannot connect a residue to itself;
- the same cysteine position cannot be reused in more than one pair.

Pairing order defines topology. For example, `2-17,6-13` and `2-13,6-17` use the same four cysteine positions but encode different two-disulfide topologies.

## 3. Paths, output, and execution

### `--checkpoint`

- Type: path
- Required: yes
- Recommended value: `checkpoints/com-FT.ckpt`

The source package does not include the model file. Provide the custom checkpoint explicitly.

The checkpoint must include the confidence head, because the design loss uses pLDDT, PAE and iPTM. HighSS stops with an error
if the loaded model has none.

### `--ccd_path`

- Type: path
- Required: yes
- Recommended value: `checkpoints/ccd.pkl`

The directory containing this file is automatically passed to final Boltz prediction as `--cache`.

### `--work_dir`

- Type: directory
- Default: current directory

HighSS creates `inputs/` and `outputs/` below this directory.

All path arguments are resolved from the launch directory before HighSS
changes into `work_dir`, so relative checkpoint, CCD, and target paths remain
valid. Absolute run directories are still recommended for production jobs.

### `--suffix`

- Type: string
- Default: `run`

Appended to the output directory name:

```text
highss_<target_name>_<suffix>
```

Use topology, length, MSA mode, and batch information in the suffix.

### `--gpu_id`

- Type: integer
- Default: `0`

HighSS sets:

```text
CUDA_VISIBLE_DEVICES=<gpu_id>
```

The selected physical GPU then appears to PyTorch as local `cuda:0`.

### `--design_samples`

- Type: integer
- Default: `1`

Number of independent design attempts.

This is one of the most important production parameters. A single sample only demonstrates that a run can complete. Multi-disulfide design should use a batch and screen final predictions.

Practical starting point:

```text
32 samples per topology
```

Each sample generates an independent design.

### `--redo_boltz_predict`

- Type: boolean
- Default: `true`

When true, HighSS automatically runs final independent Boltz predictions for:

- Holo target–binder complex;
- Apo binder alone.

Keep this enabled for normal use. Setting it to false leaves YAML files but skips final independent structure prediction.

Accepted boolean forms include `true/false`, `yes/no`, `1/0`, and `on/off`.

### `--show_animation`

- Type: boolean
- Default: `false`

Displays the optimization animation after a design. This is mainly useful in interactive environments and is not needed for batch jobs.

### `--save_trajectory`

- Type: boolean
- Default: `false`

Collects per-iteration structural trajectory information. This increases memory and output overhead.

Keep false for production unless trajectory inspection is specifically required.

## 4. Optimization schedule

HighSS uses a warm-up followed by differentiable sequence optimization and optional semi-greedy refinement.

### `--optimizer_type`

- Choices: `SGD`, `AdamW`
- Default: `SGD`

Controls optimization of binder residue logits.

Selects the optimizer used for sequence-logit updates.

### `--pre_iteration`

- Type: integer
- Default: `30`

Warm-up iterations before the main design stages. The warm-up evaluates the distogram head only (no structure sampling) and uses
`--learning_rate_pre`.

Reducing this saves time but may give unstable initialization. Increasing it adds cost and has not been established as universally better.

### `--soft_iteration`

- Type: integer
- Default: `80`

Number of soft-sequence iterations for `--design_algorithm 3stages`.

This parameter is not the soft-stage count used by `3stages_extra`; that mode uses internal `soft_iteration_1` and `soft_iteration_2` values from the configuration file.

### `--temp_iteration`

- Type: integer
- Default: `45`

Temperature-annealing iterations. The sequence distribution is progressively hardened toward discrete amino-acid choices.

### `--hard_iteration`

- Type: integer
- Default: `5`

Final hard-sequence optimization iterations.

### `--semi_greedy_steps`

- Type: integer
- Default: `2`

After differentiable optimization, each step evaluates ten single-position proposals and accepts the best proposal only if complex ipTM improves.

Fixed cysteine positions are never mutated.

Increasing this parameter adds approximately ten extra model evaluations per step and can be expensive.

### `--recycling_steps`

- Type: integer
- Default: `1`

Number of model recycling steps used during differentiable design.

Number of recycling steps used by the model.

### `--seed`

- Type: integer
- Default: unset (runs are not seeded)

Seeds Python, NumPy and PyTorch at the start of every design with `seed + <design index>`. The log shows a `[seed]` line for every design. The helix weight, which is drawn at random for every design (`--helix_loss_min`, `--helix_loss_max`), is fixed by the seed as well.

Seeding fixes the random draws of a run. GPU kernels and library versions can still change the numerical results.

### `--set_train`

- Type: boolean
- Default: `true`

Runs the forward passes of the design loop with the model in train mode (dropout active), which is the original behaviour. `false` uses eval mode, as `boltz predict` does.

### Production and smoke-test schedules

Default stage settings:

```text
pre_iteration       30
soft_iteration      80
temp_iteration      45
hard_iteration      5
semi_greedy_steps   2
recycling_steps     1
```

Software-only smoke test:

```text
pre_iteration       1
soft_iteration      1
temp_iteration      1
hard_iteration      1
semi_greedy_steps   0
```

The smoke-test schedule is not suitable for assessing sequence quality, interface quality, or disulfide success.

## 5. Sequence parameterization and learning rate

### `--design_algorithm`

- Choices: `3stages`, `3stages_extra`
- Default: `3stages`

`3stages`:
1. one soft stage using `soft_iteration` and `e_soft`;
2. one temperature-annealing stage;
3. one hard stage.

`3stages_extra`:
1. soft stage 1 using `e_soft_1`;
2. soft stage 2 using `e_soft_2`;
3. temperature-annealing stage;
4. hard stage.

The two soft-stage iteration counts for `3stages_extra` come from `highss/configs/default_protein_config.yaml` and are not exposed as CLI arguments.

### `--learning_rate_pre`

- Type: float
- Default: `1.0`

Learning rate used during warm-up.

### `--learning_rate`

- Type: float
- Default: `0.1`

Learning rate used during the main optimization stages.

The pre-run and subsequent stages use their respective learning-rate settings.

### `--e_soft`

- Type: float
- Default: `0.8`

Final value of the soft-mixing weight of the soft stage used by `3stages`.

Over the iterations of the soft stage the weight `w` rises linearly from 0 to this value, and the input residue distribution is
`w * softmax(2 * logits) + (1 - w) * logits`. It is an advanced parameter.

### `--e_soft_1`

- Type: float
- Default: `0.8`
- Active only for: `3stages_extra`

Controls the first soft stage.

### `--e_soft_2`

- Type: float
- Default: `1.0`
- Active only for: `3stages_extra`

Controls the second soft stage.

## 6. Contact and fold-shape objectives

### `--inter_chain_cutoff`

- Type: float, Å
- Default: `20.0`

Distance threshold used by the binder–target contact objective.

A smaller cutoff asks for tighter predicted contacts; a larger cutoff makes contact satisfaction easier but less spatially specific.

### `--intra_chain_cutoff`

- Type: float, Å
- Default: `14.0`

Distance threshold used by the binder intrachain contact objective.

This contributes to compact internal organization.

### `--num_inter_contacts`

- Type: integer
- Default: `2`

Number of favorable binder–target contacts considered by the interface-contact objective.

This is an objective-selection parameter rather than the exact number of physical contacts in the final structure.

### `--num_intra_contacts`

- Type: integer
- Default: `2`

Number of favorable binder intrachain contacts considered by the internal-contact objective.

### `--optimize_contact_per_binder_pos`

- Type: boolean
- Default: `true`

Uses the binder side as the position-wise basis when selecting interface contacts.

Enables contact optimization per binder position.

### `--mask_ligand`

- Type: boolean
- Default: `false`

During warm-up, masks non-binder tokens/atoms when enabled.

Set to false for protein targets.

### `--distogram_only`

- Type: boolean
- Default: `false`

When true, optimization uses distogram-only model output and omits confidence, PAE, and coordinate-derived losses during the main design call.

False uses structure prediction in addition to the distogram. True runs the distogram-only mode.

### `--helix_loss_min`, `--helix_loss_max`

- Type: float
- Defaults: `-0.3`, `0.0`

For every design sample, HighSS draws the helix-loss weight uniformly from this interval.

The helix term is the mean, over residue pairs (i, i+3) of the binder, of -log P(distance < 6 A) taken from the distogram. A negative
weight penalizes this contact pattern, a positive weight rewards it, and zero removes the term.

The interval specifies the range from which the helix weight is sampled for each design.

## 7. General loss weights

The total objective is a weighted sum. Increasing a weight makes that term more influential relative to the others; it does not guarantee the corresponding property.

### `--con_loss`

- Default: `1.0`
- Meaning: binder intrachain contact/compactness objective.

### `--i_con_loss`

- Default: `1.0`
- Meaning: binder–target interface contact objective.

### `--plddt_loss`

- Default: `0.1`
- Meaning: confidence objective based on predicted local structure quality.

### `--pae_loss`

- Default: `0.4`
- Meaning: binder intrachain predicted-aligned-error objective.

### `--i_pae_loss`

- Default: `0.1`
- Meaning: binder–target interface predicted-aligned-error objective.

### `--rg_loss`

- Default: `0.0`
- Meaning: radius-of-gyration objective.

At the default weight of zero, radius of gyration is reported but not weighted into optimization.


## 8. Disulfide distance objective

### `--disulfide_distance_mode`

- Choices: `hinge`, `mse`, `bounded`
- Default: `bounded`

`hinge`:
- penalizes SG–SG distance above a cutoff;
- useful for encouraging closure without penalizing shorter distances.

`mse`:
- penalizes squared deviation from `disulfide_target_dist`;
- strongly centers the predicted distance.

`bounded`:
- penalizes values outside `[lower_bound, upper_bound]`;
- may optionally include a center penalty;
- default distance mode.

### `--disulfide_target_dist`

- Type: float, Å
- Default: `2.05`
- Active mainly in: `mse`

Target SG–SG distance.

### `--disulfide_lower_bound`

- Type: float, Å
- Default: `0.0`
- Active in: `bounded`

Lower edge of the unpenalized interval.

A value of zero adds no lower-bound penalty for short distances.

### `--disulfide_upper_bound`

- Type: float, Å
- Default: `2.80`
- Active in: `bounded`

Upper edge of the unpenalized interval.

Lowering this value makes the closure objective stricter and may compete more strongly with interface/fold objectives.

### `--disulfide_center_weight`

- Type: float
- Default: `0.0`
- Active in: `bounded`

Optional weight for pulling in-bound distances toward `disulfide_target_dist`.

Zero disables the center-distance penalty inside the permitted interval.

### `--disulfide_hinge_cutoff`

- Type: float, Å
- Default: `2.3`
- Active in: unscheduled `hinge`, and as the general fallback cutoff

When scheduling is enabled, stage-specific values are derived from the schedule settings.

## 9. Disulfide scheduling and weights

### `--disulfide_schedule`

- Type: boolean
- Default: `true`

Enables stage-dependent disulfide strength and a relaxed-to-strict closure schedule.

True enables stage-specific distance-loss weights and cutoffs.

### `--disulfide_hinge_cutoff_start`

- Type: float, Å
- Default: `3.5`

Relaxed closure cutoff at the beginning of scheduled optimization.

### `--disulfide_hinge_cutoff_end`

- Type: float, Å
- Default: `2.3`

Stricter closure cutoff toward the end of scheduled optimization.

### `--disulfide_loss`

- Type: float
- Default: `0.008`

Fallback/base disulfide distance weight. It is also used when stage-specific weights are not supplied or scheduling is disabled.

### `--disulfide_loss_soft`

- Type: float
- Default: `0.004`

Distance weight in the soft stage.

### `--disulfide_loss_temp`

- Type: float
- Default: `0.008`

Distance weight in the temperature-annealing stage.

### `--disulfide_loss_hard`

- Type: float
- Default: `0.015`

Distance weight in the hard stage.

The increasing sequence:

```text
0.004 → 0.008 → 0.015
```

allows early sequence/interface exploration and applies stronger closure pressure after the sequence becomes more discrete.

These values multiply the distance-loss term in the respective stages.

### `--disulfide_worst_pair_loss`

- Type: float
- Default: `0.5`

Adds a term based on the worst requested pair.

This prevents the mean distance objective from appearing good when one bond is closed but another requested bond remains open. It is especially important for 2SS and 3SS designs.

## 10. Disulfide angle and dihedral guidance

### `--disulfide_angle_loss`

- Type: float
- Default: `0.0005`

Weight for the Cβ–Sγ–Sγ angle guidance term.

### `--disulfide_angle_target`

- Type: float, degrees
- Default: `103.0`

Target angle used by the geometry term.

### `--disulfide_angle_tolerance`

- Type: float, degrees
- Default: `25.0`

Deviation tolerated before angle penalty grows.

### `--disulfide_dihedral_loss`

- Type: float
- Default: `0.0002`

Weight for Cβ–Sγ–Sγ–Cβ dihedral guidance.

### `--disulfide_dihedral_target_abs`

- Type: float, degrees
- Default: `90.0`

Absolute target value. Both positive and negative handed dihedrals can satisfy the absolute target.

### `--disulfide_dihedral_tolerance`

- Type: float, degrees
- Default: `45.0`

Deviation tolerated around the absolute dihedral target.

Angle and dihedral targets and tolerances define the corresponding geometry-loss terms.

### `--debug_grad_norm`

- Type: boolean
- Default: `false`

Prints, in every epoch after the pre-run, the norm of the gradient of the disulfide terms and of all other terms with respect to the
sequence logits (`NONE` = the term has no autograd graph) and their ratio. Diagnostic; adds backward passes.

### `--disulfide_filter_cutoff`

- Type: float, Å
- Default: `2.8`
- Status: reserved/not active

This value is accepted by the CLI and passed through function calls, but it is not used to reject candidates or filter final structures.

Do not assume that setting this option performs automatic final screening. Use `scripts/check_disulfide_geometry.py` on generated CIF files.

## 11. Disulfide loss and sequence features

### `--use_disulfide_loss`

- Type: boolean
- Default: `true`

Master switch for the explicit disulfide objective.

| Setting | Command option | Behavior |
|---|---|---|
| Enabled (default) | `--use_disulfide_loss true` | Computes the distogram and geometry loss terms; the distogram term is backpropagated to the binder sequence logits |
| Disabled | `--use_disulfide_loss false` | Removes all explicit disulfide-loss terms and their backpropagation |

The enabled setting is used when the option is omitted. The disabled setting preserves the prescribed Cys positions and
bond-input settings. Nonzero overrides for disabled loss terms are rejected.

Use this switch to disable the full objective. `--disulfide_loss 0` alone does not switch the structure terms off when
stage-specific weights are active, and it does not disable the distogram term.

### Example commands

Run `bash examples/run_highss_disulfide_modes.sh on` with disulfide loss and backpropagation enabled, or
`bash examples/run_highss_disulfide_modes.sh off` without explicit disulfide loss. The script accepts checkpoint, dictionary,
GPU, seed, sample-count and output settings through environment variables; see the main README.

### Input settings

| Option | Default | Meaning |
|---|---|---|
| `--loop_fix_cys` | `true` | Keep prescribed Cys fixed in the loop sequence features |
| `--pre_drop_bonds` | `true` | Omit the bond feature during the pre-run only |
| `--loop_profile` | `token` | Use a per-token binder profile |

### `--disulfide_distogram_loss`

- Type: float
- Default: `4.0`

Weight of the distogram Cys-Cys term. For every prescribed pair the term takes the distance distribution that the model predicts
between the two CB atoms (the distogram), adds up the probability of the bins inside the bonding window
(`--disulfide_distogram_lo` to `--disulfide_distogram_hi`) and takes the negative logarithm of that probability
(`--disulfide_distogram_form nll`); the loss is the average over the pairs. It is differentiable at every stage, needs no structure
sampling and is logged every epoch as `disulfide_distogram_loss`.


### `--disulfide_distogram_form`

- Type: `nll` or `prob`
- Default: `nll`

`nll` uses `-log P(window)`. `prob` uses `1 - P(window)`.

### `--disulfide_distogram_lo`, `--disulfide_distogram_hi`

- Type: float, Angstrom
- Default: `3.3` and `4.8`

Bonding window of the CB-CB distance of a prescribed pair, used by the distogram term. These values describe disulfide geometry; do
not change them to tune a design.

### `--loop_fix_cys`

- Type: boolean
- Default: `true`

With true, the prescribed Cys residues are written into the sequence features of every forward pass and are fixed during
sequence optimization.

### `--pre_drop_bonds`

- Type: boolean
- Default: `true`

Removes the bond feature from the model input in the pre-run only. The soft, temperature and hard stages, semi-greedy updates
and final prediction retain the bond feature.

### `--loop_profile`

- Type: `token`, `token_pre` or `mean`
- Default: `token`

MSA profile feature of the binder tokens inside the design loop. `boltz predict` gives every token the profile of its own residue
(the mean over the MSA rows of that token; with `--msa_mode none` simply the residue itself). `mean` writes the averaged composition of
the whole binder into every binder token, so the loop profile differs from that of the final prediction. `token` writes the per-token
profile in every design stage, `token_pre` in the pre-run only. The profile is an input
feature; the sequence gradient still flows through the residue type only. With `--msa_mode target` the profile of a binder token is
the featurizer's own (the mean over all MSA rows of that token).

## 12. Example command

```bash
highss \
  --target_pdb target.pdb \
  --target_chains A \
  --target_name target \
  --binder_length 25 \
  --disulfide_pairs "2-25,6-21,10-16" \
  --msa_mode target \
  --msa_max_seqs 4096 \
  --checkpoint checkpoints/com-FT.ckpt \
  --ccd_path checkpoints/ccd.pkl \
  --work_dir runs/target_25aa_3ss \
  --suffix target_25aa_3ss \
  --gpu_id 0 \
  --design_samples 32 \
  --pre_iteration 30 \
  --soft_iteration 80 \
  --temp_iteration 45 \
  --hard_iteration 5 \
  --semi_greedy_steps 2 \
  --recycling_steps 1 \
  --optimizer_type SGD \
  --learning_rate_pre 1.0 \
  --learning_rate 0.1 \
  --distogram_only false \
  --disulfide_distance_mode bounded \
  --disulfide_lower_bound 0.0 \
  --disulfide_upper_bound 2.80 \
  --disulfide_center_weight 0.0 \
  --disulfide_schedule true \
  --disulfide_loss_soft 0.004 \
  --disulfide_loss_temp 0.008 \
  --disulfide_loss_hard 0.015 \
  --disulfide_worst_pair_loss 0.5 \
  --disulfide_hinge_cutoff_start 3.5 \
  --disulfide_hinge_cutoff_end 2.3 \
  --disulfide_angle_loss 0.0005 \
  --disulfide_angle_target 103.0 \
  --disulfide_angle_tolerance 25.0 \
  --disulfide_dihedral_loss 0.0002 \
  --disulfide_dihedral_target_abs 90.0 \
  --disulfide_dihedral_tolerance 45.0 \
  --redo_boltz_predict true \
  --show_animation false \
  --save_trajectory false
```

The command uses the default settings listed in this guide.

## 13. Additional command-line options

| Option | Default | Function |
|---|---|---|
| `--loop_drop_bonds` | `false` | Omit bond features in the sequence-optimization loop. |
| `--disulfide_grad_steps` | `0` | Number of final denoising steps retaining gradients for geometry loss; `0` computes and logs the geometry terms without a sequence gradient. |
| `--disulfide_sds_scale` | `0.0` | Weight multiplier for geometry terms evaluated on auxiliary denoiser estimates. |
| `--disulfide_sds_samples` | `2` | Number of noise levels sampled for auxiliary geometry terms. |
| `--disulfide_sds_sigma_lo` | `1.5` | Minimum auxiliary noise level, in angstroms. |
| `--disulfide_sds_sigma_hi` | `6.0` | Maximum auxiliary noise level, in angstroms. |
| `--disulfide_sds_clip` | `2.0` | Maximum SG-SG upper-bound violation, in angstroms, included in auxiliary distance terms. |
| `--debug_sds_profile_every` | `0` | Print an auxiliary-gradient profile at the specified epoch interval; zero disables it. |
| `--debug_sds_profile_t` | `1.5,3,6,12,24,48` | Noise levels used for the auxiliary-gradient profile. |
| `--rc3_defaults` | `false` | Apply the built-in input/loss configuration; explicit command-line values take precedence. |
| `--help` | — | Display command-line usage and exit. |
