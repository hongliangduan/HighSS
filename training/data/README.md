# Training data are not included

This release contains code and portable configurations only. It does not contain
PDB-derived structures, MSAs, processed tensors or train/validation splits.

The manuscript reports 1,610 non-redundant monomer entries and 398 complex
entries. The curated entry list, sequences and disulfide connectivities are in
Supplementary Table 3. To reproduce fine-tuning, convert those entries to the
Boltz-1 processed training format and place them under:

```text
data/monomer/processed/
data/monomer/processed/msa/
data/monomer/validation_ids.txt
data/complex/processed/
data/complex/processed/msa/
data/complex/validation_ids.txt
```

The exact validation-ID files determine checkpoint selection. They cannot be
reconstructed from the manuscript text alone and must be supplied from the
original study records for bitwise reproduction.
