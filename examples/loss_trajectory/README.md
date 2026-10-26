# Loss trajectory of a design

`scripts/plot_loss_trajectory.py` draws the optimization trajectory of one design from the log of a HighSS run. It needs only Python, NumPy and Matplotlib.

| File | Content |
|---|---|
| `mdm2_no_msa.txt` | Log of a run without MSA (`--msa_mode none`) |
| `mdm2_target_msa.txt` | Log of a run with a target-only MSA (`--msa_mode target`; the binder stays single-sequence) |
| `mdm2_no_msa_trajectory.png`, `.csv` | Figure and per-step loss table written by the script for the first log |
| `mdm2_target_msa_trajectory.png`, `.csv` | The same for the second log |

Both runs use MDM2 (`examples/mdm2/mdm2_pro.pdb`), an 18-residue binder, the pairs `2-17,6-13`, one design and seed 1. The bundled logs are example inputs for the plotting script and preserve their recorded run settings. The logs are cut to the part the script reads (the run header up to the final sequence); local paths are replaced by `/path/to/HighSS`.

## Run a design

```bash
PYTHONUNBUFFERED=1 highss \
  --target_pdb examples/mdm2/mdm2_pro.pdb \
  --target_chains A \
  --target_name mdm2 \
  --binder_length 18 \
  --disulfide_pairs "2-17,6-13" \
  --checkpoint checkpoints/com-FT.ckpt \
  --ccd_path checkpoints/ccd.pkl \
  --gpu_id 0 \
  --design_samples 1 \
  --seed 1 \
  --suffix traj 2>&1 | tee run.txt
```

Disulfide loss and backpropagation are enabled by default (`--use_disulfide_loss true`). Add `--use_disulfide_loss false` to disable all explicit disulfide losses. Target-only MSA is enabled by default. Use `--msa_mode none` to disable MSA generation.

## Draw

```bash
python scripts/plot_loss_trajectory.py run.txt --target MDM2
```

The script writes `run_trajectory.png`, `run_trajectory.pdf` and `run_trajectory.csv` next to the log, and prints a summary and a draft figure legend. With the example logs:

```bash
python scripts/plot_loss_trajectory.py examples/loss_trajectory/mdm2_no_msa.txt --target MDM2 --dpi 200
```

| Option | Meaning |
|---|---|
| `--design N` | Select the N-th design in a log that contains several designs (default 1); later designs use output suffix `_dN` |
| `--out PREFIX` | Prefix of the output files (default: `<log>_trajectory`) |
| `--weights a=1,b=2` | Loss weights, only if the run did not use the defaults |
| `--grad_steps N` | Number of denoising steps that carry the geometry gradient, for the draft legend. The log header records only whether the gradient was on (`--disulfide_grad_steps` greater than 0), so without this option the legend shows `[N]` for such a log |
| `--target NAME` | Target name for the draft legend |
| `--dpi N` | Resolution of the PNG (default 300) |
| `--font NAME` | Font family to try first (default: Arial, Helvetica, Liberation Sans, DejaVu Sans) |

## Figure

| Panel | Content |
|---|---|
| a | Total loss and the weighted contribution of each group of loss terms |
| b | Intra-chain contact, inter-chain contact and helix terms (raw values) |
| c | pLDDT and PAE terms (raw values; not part of the pre-run objective) |
| d | Distogram term of the prescribed Cys pairs |
| e | Sγ–Sγ distance of each prescribed pair in the structure sampled at each step |
| f | iPTM of the candidates tested in each semi-greedy step |

Panels that have no data in the log are left out. Shaded bands mark the stages (pre-run, soft, annealed, hard one-hot).

## Notes

- The log prints the raw value of every loss term and the total loss. The script multiplies the terms by the default weights (read from the `loop guidance` line of the header for the distogram term) and checks that the weighted sum reproduces the logged total (`check: ... -> ok`). The helix weight is drawn at random for every design and is fitted from the log. If a run used other weights, pass them with `--weights`.
- The Sγ–Sγ distances in panel e belong to structures sampled inside the optimization loop, not to the final prediction. Screen final structures with `scripts/check_disulfide_geometry.py`.
