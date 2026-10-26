# HighSS 666666

**HighSS** is a research software package for de novo design of protein and peptide binders with:

- an exact user-defined binder length;
- fixed cysteine positions;
- a programmable intramolecular disulfide-pairing topology;
- target-only multiple-sequence alignments enabled by default; and
- automatic complex (Holo) and binder-only (Apo) structure prediction.

## Overview

HighSS accepts a protein target structure and cysteine pairs specified by residue positions within the binder. It constructs a binder of the requested length, fixes the specified positions to cysteine, applies geometry-aware disulfide objectives during sequence optimization, and automatically performs final Holo and Apo predictions.

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

## Requirements

The provided setup script creates a Python 3.10 conda environment and installs the CUDA 12.8 PyTorch build.

Required external files are not included in the source archive. They are available through Figshare
(<https://doi.org/10.6084/m9.figshare.33069275>) and are placed as follows:

```text
checkpoints/com-FT.ckpt
checkpoints/ccd.pkl
```

`boltz1_conf.ckpt` is not required.

The default `--msa_mode target` requires access to the ColabFold/MMseqs2 MSA service. `--msa_mode none` runs without contacting that service.

## Installation

From the HighSS source directory, run the canonical one-command installer:

```bash
chmod +x setup.sh
./setup.sh
conda activate highss
```

The installer pins the CUDA 12.8 PyTorch build and then installs
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

Target-only MSA and disulfide-loss backpropagation are enabled by default:

```bash
highss \
  --target_pdb examples/mdm2/mdm2_pro.pdb \
  --target_chains A \
  --target_name mdm2 \
  --binder_length 18 \
  --disulfide_pairs "2-17,6-13" \
  --checkpoint checkpoints/com-FT.ckpt \
  --ccd_path checkpoints/ccd.pkl \
  --gpu_id 0 \
  --design_samples 32 \
  --suffix mdm2_18aa_2ss
```

MSA is generated only for the target chains; the designed binder remains single-sequence. `--msa_mode target` selects this
setting explicitly. Use `--msa_mode none` to disable MSA generation.

Runnable examples are provided for [no-MSA design](examples/run_highss_single.sh) and
[target-only MSA design](examples/run_highss_target_msa.sh).

## Essential design parameters

The parameters most users should set are:

| Parameter | Meaning | Recommended use |
|---|---|---|
| `--target_pdb` | Local target PDB file | Required |
| `--target_chains` | Target chains to retain, such as `A` or `A,B` | Required |
| `--binder_length` | Exact number of residues in the designed binder | Required |
| `--disulfide_pairs` | Paired Cys positions within the designed binder; the first residue is numbered 1 | Required |
| `--msa_mode` | Target-only MSA or no MSA | Default `target`; use `none` to disable MSA |
| `--design_samples` | Number of independent designs | Set the number of designs to generate |
| `--gpu_id` | Physical GPU selected through `CUDA_VISIBLE_DEVICES` | Select one available GPU |
| `--checkpoint` | HighSS model checkpoint | Point to the supplied custom checkpoint |
| `--ccd_path` | Local CCD dictionary | Point to `ccd.pkl` |
| `--suffix` | Name appended to the output run | Use a unique, informative label |
| `--redo_boltz_predict` | Automatically run final Holo and Apo predictions | Keep `true` for normal use |

The default optimization settings are:

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

Parameter descriptions and default values are provided in [USAGE.md](USAGE.md) and [docs/PARAMETERS.md](docs/PARAMETERS.md).

## Disulfide topology format

`--disulfide_pairs` specifies residue positions within the designed binder. **The first residue is numbered 1.**

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
  --contact_residues "54,58,61,67" \
  --contact_target_chain A
```

For a single-chain target, `--contact_target_chain` may be omitted. For a multi-chain target, it is required whenever `--contact_residues` is used.

Residues are given by the residue numbers of the supplied PDB file, not by their position in the chain: `examples/mdm2/mdm2_pro.pdb`
numbers its residues 25–109, so `54` is Leu54. An insertion code is appended to the number, for example `52A`. HighSS converts the
numbers to positions in the target sequence before it writes the constraint, prints the conversion in the run header, and stops with
an error if a number is not a standard residue of the chain.

## Disulfide objective

The default coordinate-based disulfide settings are:

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

Disulfide loss and its backpropagation are enabled by default. The loss includes a distogram pairing term, which is
backpropagated to the sequence logits, and coordinate-based SG-SG distance, worst-pair, angle and dihedral terms, which are
evaluated during stages that sample coordinates and enter the logged total; they carry a sequence gradient only when
`--disulfide_grad_steps` is greater than 0 (default 0).

The default input and distogram settings are `--disulfide_distogram_loss 4.0`, `--disulfide_distogram_form nll`,
`--loop_fix_cys true`, `--pre_drop_bonds true` and `--loop_profile token`. Their details are in `docs/PARAMETERS.md`.

### Enable or disable disulfide loss

`--use_disulfide_loss` (default `true`) is the master switch. Append one of the following settings to a normal `highss` command:

| Setting | Command option | Effect |
|---|---|---|
| Enabled (default) | `--use_disulfide_loss true` | Enables the disulfide loss; the distogram term is backpropagated to the sequence logits |
| Disabled | `--use_disulfide_loss false` | Removes every explicit disulfide-loss term and its backpropagation |

The enabled setting is used when this option is omitted.

Disabling the loss preserves the prescribed Cys positions and bond-input settings.

Setting `--disulfide_loss 0` alone is insufficient because scheduled stage weights and the distogram term may remain active.
The master switch removes the disulfide-loss terms. Conflicting nonzero loss settings are rejected.

For a runnable MDM2 example, select one setting per invocation:

```bash
bash examples/run_highss_disulfide_modes.sh on
bash examples/run_highss_disulfide_modes.sh off
```

Omitting the argument selects `on`. Each invocation generates one 18-residue MDM2 binder with pairs `2-17,6-13`, seed 1 and the
full default optimization budget, with target-only MSA. Set `CHECKPOINT`, `CCD_PATH`, `GPU_ID`, `MSA_MODE`, `DESIGN_SAMPLES`,
`SEED`, `WORK_DIR` or `SUFFIX` as needed:

```bash
GPU_ID=3 CHECKPOINT=/path/to/com-FT.ckpt CCD_PATH=/path/to/ccd.pkl \
  bash examples/run_highss_disulfide_modes.sh on
```

The run header reports the loss settings. `--debug_grad_norm true` prints gradient norms during optimization.

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

## Loss trajectory

`scripts/plot_loss_trajectory.py` draws the loss trajectory of one design from the log of a run: the total loss and its weighted components over the optimization stages, the raw contact, helix, confidence and distogram terms, the SG–SG distances of the prescribed pairs in structures sampled during optimization, and the iPTM of the semi-greedy candidates. It writes PNG and PDF figures and a CSV containing per-step loss terms, their weights, logged and reconstructed totals, and SG–SG distances, and prints a summary and a draft figure legend.

```bash
PYTHONUNBUFFERED=1 highss ... 2>&1 | tee run.txt
python scripts/plot_loss_trajectory.py run.txt --target MDM2
```

The script needs only Python, NumPy and Matplotlib. Example logs and figures are in [`examples/loss_trajectory/`](examples/loss_trajectory/README.md).

## Fine-tuning

Reference Boltz-1 fine-tuning scripts and configurations for the
monomer-fine-tuned (`mon-FT`) and complex-fine-tuned (`com-FT`) models are
provided in [`training/`](training/README.md).

The released checkpoints (`mon-FT.ckpt`, `com-FT.ckpt`) and the chemical component dictionary (`ccd.pkl`) are distributed
separately through Figshare (<https://doi.org/10.6084/m9.figshare.33069275>) and are not stored in this Git repository.

Training commands, required inputs and configuration paths are described in
[`training/README.md`](training/README.md).

## Tests

```bash
pip install -e ".[test]"
python -m pytest tests
```

The tests use toy tensors and stubbed models; they need neither a checkpoint nor a GPU.

## License and bundled runtime

The vendored `boltz/` runtime retains its upstream MIT license in:

```text
boltz/LICENSE
```

Additional third-party copyright and license notices embedded in individual source files must be retained.


## Disclaimer

HighSS is experimental research software. Computational confidence scores and predicted disulfide geometry do not establish binding, folding, stability, specificity, or biological activity. Experimental validation is required.
