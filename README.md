# HighSS

**HighSS** is a research software package for de novo design of protein and peptide binders with:

- an exact user-defined binder length;
- fixed cysteine positions;
- a programmable intramolecular disulfide-pairing topology;
- optional target-only multiple-sequence alignments; and
- automatic complex (Holo) and binder-only (Apo) structure prediction.

## Associated manuscript

**Programmable disulfide topology enables de novo design of multi-disulfide peptide binders**



## Overview

HighSS accepts a protein target structure and a set of 1-based cysteine pairs. It constructs a binder of the requested length, fixes the specified positions to cysteine, applies geometry-aware disulfide objectives during sequence optimization, and automatically performs final Holo and Apo predictions.

For example:

```text
binder length:      18
disulfide topology: 2-17,6-13
```

HighSS interprets this as two intrabinder disulfide bonds:

```text
Cys2  — Cys17
Cys6  — Cys13
```

Different pairings of the same cysteine positions represent different topologies. With one disulfide bond, different designs represent different cysteine placements rather than alternative pairing topologies.

## Scope

HighSS currently supports:

- protein targets supplied as local PDB files;
- one designed binder chain;
- exact binder length;
- one or more user-specified disulfide pairs;
- no-MSA design;
- target-only MSA design;
- optional target-pocket conditioning;
- automatic Holo prediction;
- automatic Apo prediction;
- local model checkpoint and CCD cache files.

HighSS does not include ProteinMPNN, LigandMPNN, Rosetta, AlphaFold3, or Chai-based post-processing.

## Requirements

The provided setup script creates a Python 3.10 conda environment and installs the CUDA 12.8 PyTorch build used for the validated release.

Required external files are not included in the source archive:

```text
checkpoints/best-model-epoch68-rmsd5.0984.ckpt
checkpoints/ccd.pkl
```

`boltz1_conf.ckpt` is not required.

For target-only MSA mode, the machine must be able to access the ColabFold/MMseqs2 MSA service.

## Installation

From the HighSS source directory, run the canonical one-command installer:

```bash
chmod +x setup.sh
./setup.sh
conda activate highss
```

The installer pins the validated CUDA 12.8 PyTorch build and then installs
the declared dependencies of both HighSS and the vendored Boltz runtime.
A fresh installation downloads several large CUDA wheels and may take time.


Relative paths supplied to `--target_pdb`, `--checkpoint`, `--ccd_path`, and
`--work_dir` are resolved from the directory where `highss` is launched.
Changing into `--work_dir` does not change the meaning of the other paths.

Confirm the installation:

```bash
highss --help
```

## Quick start

### No-MSA design

```bash
highss \
  --target_pdb examples/mdm2/mdm2_pro.pdb \
  --target_chains A \
  --target_name mdm2 \
  --binder_length 18 \
  --disulfide_pairs "2-17,6-13" \
  --msa_mode none \
  --checkpoint checkpoints/best-model-epoch68-rmsd5.0984.ckpt \
  --ccd_path checkpoints/ccd.pkl \
  --gpu_id 0 \
  --design_samples 32 \
  --suffix mdm2_18aa_2ss
```

### Target-only MSA design

```bash
highss \
  --target_pdb examples/mdm2/mdm2_pro.pdb \
  --target_chains A \
  --target_name mdm2 \
  --binder_length 25 \
  --disulfide_pairs "2-25,6-21,10-16" \
  --msa_mode target \
  --msa_max_seqs 4096 \
  --checkpoint checkpoints/best-model-epoch68-rmsd5.0984.ckpt \
  --ccd_path checkpoints/ccd.pkl \
  --gpu_id 0 \
  --design_samples 32 \
  --suffix mdm2_25aa_3ss_target_msa
```

In target-only MSA mode, the designed binder remains single-sequence and only the target chain receives an MSA.

## Essential design parameters

The parameters most users should set are:

| Parameter | Meaning | Recommended use |
|---|---|---|
| `--target_pdb` | Local target PDB file | Required |
| `--target_chains` | Target chains to retain, such as `A` or `A,B` | Required |
| `--binder_length` | Exact number of residues in the designed binder | Required |
| `--disulfide_pairs` | 1-based cysteine pairing topology | Required |
| `--msa_mode` | `none` or target-only `target` MSA | Start with `none`; use `target` when target MSA information is desired |
| `--design_samples` | Number of independent designs | Use multiple samples; 32 is a practical initial batch |
| `--gpu_id` | Physical GPU selected through `CUDA_VISIBLE_DEVICES` | Select one available GPU |
| `--checkpoint` | HighSS model checkpoint | Point to the supplied custom checkpoint |
| `--ccd_path` | Local CCD dictionary | Point to `ccd.pkl` |
| `--suffix` | Name appended to the output run | Use a unique, informative label |
| `--redo_boltz_predict` | Automatically run final Holo and Apo predictions | Keep `true` for normal use |

The validated optimization schedule is:

```text
pre_iteration:       30
soft_iteration:      80
temp_iteration:      45
hard_iteration:      5
semi_greedy_steps:   2
recycling_steps:     1
optimizer_type:      SGD
learning_rate_pre:   1.0
learning_rate:       0.1
```

These defaults should be treated as the baseline. A `1/1/1/1` run with zero semi-greedy steps is only a software smoke test and is not a meaningful design run.

A complete explanation of every command-line parameter, default value, tuning direction, and RC3 limitation is provided in:

```text
docs/PARAMETERS.md
```

## Disulfide topology format

`--disulfide_pairs` uses **1-based binder residue positions**.

Examples:

```text
14 residues, one disulfide:
2-13

18 residues, nested two-disulfide topology:
2-17,6-13

18 residues, crossed two-disulfide topology:
2-13,6-17

25 residues, three-disulfide topology:
2-16,6-21,10-25

25 residues, nested three-disulfide topology:
2-25,6-21,10-16
```

HighSS validates that:

- every position lies within the requested binder length;
- every pair contains two different positions; and
- a cysteine position is not assigned to multiple pairs.

## Pocket conditioning

To bias design toward specified target residues:

```bash
highss \
  ... \
  --contact_residues "24,27,31,54" \
  --contact_target_chain A
```

For a single-chain target, `--contact_target_chain` may be omitted. For a multi-chain target, it is required whenever `--contact_residues` is used.

Target residue numbering is taken from the supplied PDB.

## Disulfide objective

The validated RC3 disulfide settings are:

```text
distance mode:                   bounded
lower bound:                     0.0 Å
upper bound:                     2.80 Å
center weight:                   0.0
schedule:                        true
soft/temp/hard weights:          0.004 / 0.008 / 0.015
worst-pair weight:               0.5
hinge cutoff start/end:          3.5 / 2.3 Å
angle target/tolerance:          103° / 25°
dihedral |target|/tolerance:     90° / 45°
angle/dihedral loss weights:     0.0005 / 0.0002
```

These terms guide optimization but do not guarantee that every final prediction contains the requested geometry. The final Holo and Apo structures must still be screened.

## Outputs

A run writes results under:

```text
<work_dir>/outputs/highss_<target_name>_<suffix>/
```

Important subdirectories include:

```text
results_yaml/        final Holo input YAML
results_yaml_apo/    final Apo input YAML
results_final/       automatic Holo prediction
results_final_apo/   automatic Apo prediction
loss/                optimization records
animation/           optional trajectory-related output
```

The Holo and Apo predictions automatically use the directory containing `ccd.pkl` as the Boltz cache and use the checkpoint supplied with `--checkpoint`.

## Final disulfide screening

Final predicted structures should be screened directly:

```bash
python scripts/check_disulfide_geometry.py \
  --structure path/to/model.cif \
  --binder_chain A \
  --binder_bonds "2_SG_17_SG,6_SG_13_SG" \
  --bond_index_base 1
```

A practical screening workflow should consider:

1. whether the requested cysteine pairing is present;
2. SG–SG distances for every requested pair;
3. competing short contacts between unintended cysteine pairs;
4. Holo interface confidence;
5. Apo structural consistency; and
6. experimental validation.

Generate multiple independent samples and rank or filter the resulting designs. A successful software run does not imply that every candidate passes disulfide-geometry or biochemical validation.

## Reproducibility

For each release, record:

- the HighSS version;
- the model-checkpoint SHA256;
- the `ccd.pkl` SHA256;
- the target PDB and selected chains;
- binder length and disulfide topology;
- MSA mode;
- random seeds, when applicable;
- optimization settings; and
- GPU, PyTorch, and CUDA versions.

## License and bundled runtime

The vendored `boltz/` runtime retains its upstream MIT license in:

```text
boltz/LICENSE
```

Additional third-party copyright and license notices embedded in individual source files must be retained.

A separate license for the HighSS-specific source should be added at the repository root before public redistribution.

## Disclaimer

HighSS is experimental research software. Computational confidence scores and predicted disulfide geometry do not establish binding, folding, stability, specificity, or biological activity. Experimental validation is required.
