# HighSS workflow

HighSS designs peptide binders for protein targets with a specified binder length and cysteine pairing.

## Topology representation

Specify each disulfide bond by the two residue positions within the designed binder. The first residue is numbered 1. For example:

```text
2-17,6-13
```

This specifies a pair between residues 2 and 17 and a pair between residues 6 and 13. These positions are assigned to cysteine.
The pairs are represented as intrabinder SG-SG bond constraints. With the default `--loop_fix_cys true`, the prescribed cysteines
are fixed in the sequence features during optimization.

## Design steps

1. Read the target structure and selected chains from a local PDB file.
2. Prepare the target and binder input features, including the requested cysteine pairs and target-only MSA (enabled by default).
3. Optimize the binder sequence through the pre-run, soft, temperature and hard stages.
4. Apply the configured semi-greedy sequence updates.
5. Generate the final Holo and Apo predictions when `--redo_boltz_predict true`.

## Geometry objective

The coordinate-based disulfide objective contains SG-SG distance, worst-pair, angle and dihedral terms. Distance modes are:

- `hinge`: penalize distances above a cutoff;
- `mse`: penalize deviation from a target distance;
- `bounded`: penalize values outside a lower/upper interval, with an optional center term.

These terms are evaluated on the coordinates sampled during the structure stages and enter the logged total. They carry a sequence
gradient only when `--disulfide_grad_steps` is greater than 0 (default 0).

Stage-specific weights, distance cutoffs, angle targets and dihedral targets are listed in `docs/PARAMETERS.md`.

## Distogram pairing term

The distogram term uses the predicted CB-CB distance distribution for each prescribed Cys pair. Its default form is the negative
log probability of the 3.3-4.8 A window, averaged over the pairs, with weight 4.0.

## Disulfide loss

| Setting | Option | Behavior |
|---|---|---|
| Enabled (default) | `--use_disulfide_loss true` | Enables the disulfide loss; the distogram term backpropagates to the binder sequence logits |
| Disabled | `--use_disulfide_loss false` | Removes all explicit disulfide-loss terms and their backpropagation |

The input pairing and prescribed Cys positions remain when the loss is disabled. No additional option is needed for the default
enabled setting. The main README and `examples/run_highss_disulfide_modes.sh` provide runnable commands.

## Output files

The output directory contains sequence and loss records, final prediction inputs, Holo predictions and Apo predictions. Use
`scripts/plot_loss_trajectory.py` to draw the logged optimization trajectory. Directory names are described in the main README.

## Final screening

Use `scripts/check_disulfide_geometry.py` to inspect the prescribed cysteine pairing in generated structures. The main README
provides the command syntax.
