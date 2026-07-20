# HighSS parameter reference

This document describes the command-line parameters exposed by HighSS v0.1.1 RC3.

The defaults listed here are the defaults in `highss/cli.py`. Unless a parameter is explicitly discussed below as a tuning variable, keep the validated default.

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

Keep `A` unless a downstream workflow requires another chain ID.

### `--contact_residues`

- Type: comma-separated target residue numbers
- Default: empty

Enables pocket conditioning toward specified residues in the supplied PDB numbering.

Example:

```bash
--contact_residues "24,27,31,54"
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
- Default: `none`

`none`:
- binder MSA is empty;
- target chains are treated as single sequences.

`target`:
- binder MSA remains empty;
- MSA is generated only for target chains.

Use `none` for the simplest and most reproducible local workflow. Use `target` when target evolutionary information is desired and network access to the MSA service is available.

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
- Recommended value: `checkpoints/best-model-epoch68-rmsd5.0984.ckpt`

The source package does not include the model file. Provide the custom checkpoint explicitly.

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

Appended to the output version name:

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

Larger batches improve exploration but scale runtime approximately linearly.

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

The validated baseline uses `SGD`. Changing optimizer changes optimization dynamics and should be treated as an experimental setting.

### `--pre_iteration`

- Type: integer
- Default: `30`

Warm-up iterations used to initialize a structural/sequence state before the main design stages.

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

The validated HighSS baseline is `1`. Larger values increase compute and memory. Do not assume that more recycling always improves design.

### Production and smoke-test schedules

Validated production baseline:

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

In RC3, the two soft-stage iteration counts for `3stages_extra` come from `highss/configs/default_protein_config.yaml` and are not exposed as CLI arguments. Use `3stages` unless deliberately benchmarking the extra mode.

### `--learning_rate_pre`

- Type: float
- Default: `1.0`

Learning rate used during warm-up.

### `--learning_rate`

- Type: float
- Default: `0.1`

Learning rate used during the main optimization stages.

Changing either learning rate can produce instability, early sequence collapse, or insufficient movement. Keep the defaults unless running controlled experiments.

### `--e_soft`

- Type: float
- Default: `0.8`

Soft-sequence exponent/temperature-control value used by `3stages`.

This changes how sharply residue logits are converted into a soft sequence representation. It is an advanced parameter.

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

This is the validated setting for HighSS and should normally remain true.

### `--mask_ligand`

- Type: boolean
- Default: `false`

During warm-up, masks non-binder tokens/atoms when enabled.

HighSS is scoped to protein targets; keep this false for the validated protein-binder workflow.

### `--distogram_only`

- Type: boolean
- Default: `false`

When true, optimization uses distogram-only model output and omits confidence, PAE, and coordinate-derived losses during the main design call.

Keep false for normal HighSS design. True is a lower-information experimental mode.

### `--helix_loss_min`, `--helix_loss_max`

- Type: float
- Defaults: `-0.3`, `0.0`

For every design sample, HighSS draws the helix-loss weight uniformly from this interval.

Negative values reward the helix-associated contact pattern used by the internal helix objective; values near zero reduce that bias.

For short disulfide-rich peptides, keep the validated interval unless deliberately studying secondary-structure bias.

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

At the RC3 default, radius of gyration is reported but not weighted into optimization. Positive nonzero values are experimental and can over-favor compactness.

Do not tune multiple loss weights simultaneously without recording a controlled comparison.

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
- validated HighSS mode.

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

The validated value of zero intentionally avoids adding a short-distance lower-bound penalty during optimization. Final stereochemical screening is still required.

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

The validated value zero treats the interval as a one-sided closure objective rather than a precise bond-length restraint.

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

Keep true for the validated baseline.

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

Raising these values can improve closure pressure but can also damage interface quality, produce optimization instability, or force unrealistic local geometry.

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

Angle and dihedral terms are weak guidance terms, not a substitute for final atomic-geometry validation.

### `--disulfide_filter_cutoff`

- Type: float, Å
- Default: `2.8`
- RC3 status: reserved/not active

In RC3 this value is accepted by the CLI and passed through function calls, but it is not used to reject candidates or filter final structures.

Do not assume that setting this option performs automatic final screening. Use `scripts/check_disulfide_geometry.py` on generated CIF files.

## 11. Recommended baseline command

```bash
highss \
  --target_pdb target.pdb \
  --target_chains A \
  --target_name target \
  --binder_length 25 \
  --disulfide_pairs "2-25,6-21,10-16" \
  --msa_mode target \
  --msa_max_seqs 4096 \
  --checkpoint checkpoints/best-model-epoch68-rmsd5.0984.ckpt \
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

## 12. Tuning order

When improving a production campaign, change parameters in this order:

1. increase `design_samples`;
2. compare topologies and cysteine placements;
3. inspect final Holo/Apo confidence and disulfide geometry;
4. only then consider modest changes to disulfide weights or cutoffs;
5. change learning rates, loss balance, or optimizer only in controlled benchmarks.

Do not judge a parameter change from one candidate. Compare distributions across a sufficiently large matched batch.
