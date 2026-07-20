# HighSS method

HighSS extends differentiable Boltz-based binder design with explicit control of disulfide-rich binder topology.

## Topology representation

A topology is supplied as 1-based residue pairs, for example:

```text
2-25,6-21,10-16
```

HighSS translates each pair into an intrabinder SG-SG bond constraint and fixes all involved residues to cysteine before schema parsing, during sequence hardening, during semi-greedy updates, and during final structure generation.

## Geometry objective

The available distance modes are:

- `hinge`: penalize SG-SG distances above a cutoff;
- `mse`: penalize deviation from a target distance;
- `bounded`: penalize values outside a lower/upper interval, with an optional weak center term.

A scheduled objective can use a relaxed closure cutoff in the soft stage, anneal the cutoff during the temperature stage, and apply a stricter cutoff during the hard stage. A worst-pair term prevents one open pair from being hidden by already satisfied pairs. Optional Cβ-Sγ-Sγ angle and Cβ-Sγ-Sγ-Cβ dihedral terms provide stereochemical guidance.

## Final screening

HighSS optimization and Boltz confidence scores should be followed by direct geometry checks. A useful screening order is:

1. requested pairing topology matches the closest SG-SG pairing;
2. expected SG-SG distances fall within the chosen acceptance interval;
3. no unintended Cys pair forms a competing short contact;
4. interface confidence and structural inspection support the intended target interaction.
