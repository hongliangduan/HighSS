"""Synthetic highss logs in the print formats of design_engine.py and cli.py, for the tests of scripts/plot_loss_trajectory.py.

The numbers are made up (smooth curves + noise); the totals are the weighted sum of the printed terms, as in the engine.
"""
from __future__ import annotations

import numpy as np

W = {"con_loss": 1.0, "i_con_loss": 1.0, "plddt_loss": 0.1, "pae_loss": 0.4, "i_pae_loss": 0.1, "rg_loss": 0.0,
     "disulfide_distogram_loss": 4.0, "disulfide_worst_pair_loss": 0.5, "disulfide_angle_loss": 0.0005,
     "disulfide_dihedral_loss": 0.0002}
SS_SCHEDULE = {"soft": 0.004, "temp": 0.008, "hard": 0.015}
ALPHA = "ARNDCQEGHILKMFPSTWYV"


def _decay(i, n, a, b, k=3.0):
    t = i / max(n - 1, 1)
    return b + (a - b) * np.exp(-k * t)


def header(length=18, pairs="2-17,6-13", form="nll", weight=4.0, off=False):
    guide = ("disulfide loss OFF (no disulfide term in any stage)" if off else f"distogram term {form} x {weight:g}")
    return "\n".join([
        "HighSS run",
        "  target PDB:       /path/to/HighSS/examples/mdm2/mdm2_pro.pdb",
        "  input chains:     A",
        "  output chains:    A",
        "  binder chain:     A",
        f"  binder length:    {length}",
        f"  disulfide pairs:  {pairs}",
        "  binder bonds:     [((1, 16), (5, 12))]",
        "  MSA mode:         none",
        "  YAML:             /x/y.yaml",
        "  output version:   highss_mdm2_traj",
        f"  loop guidance:    {guide}; loop_fix_cys True; pre_drop_bonds True; loop_profile token",
    ]) + "\n"


def synth_design(rng, seed=1, length=18, n_pairs=2, pre=30, soft=80, temp=45, hard=5, semi=2, helix_w=-0.137,
                 off=False, with_final=True, noise=True, sg=True, fields=True, nll_w=4.0, nan_at=None):
    out = []
    if seed is not None:
        out.append(f"[seed] {seed}")
    out.append("pre-run warm up")
    n_all = pre + soft + temp + hard
    seq = "".join(rng.choice(list(ALPHA.replace("C", "")), size=length))
    for c in ([1, 16, 5, 12] if length == 18 else [0, length - 1]):      # the prescribed Cys (fixed by --loop_fix_cys)
        seq = seq[:c] + "C" + seq[c + 1:]
    k_global = 0

    def stage(name, n, pre_run, lr_of, soft_of, hard_of, temp_of, ss_stage):
        nonlocal k_global, seq
        for i in range(n):
            g = k_global
            t = g / max(n_all - 1, 1)
            v = {
                "con_loss": _decay(g, n_all, 4.9, 1.9) + rng.normal(0, 0.05),
                "i_con_loss": _decay(g, n_all, 3.9, 1.3) + rng.normal(0, 0.05),
                "helix_loss": 0.75 + 0.85 * (1 - np.exp(-3.5 * t)) + rng.normal(0, 0.04),
            }
            if not off:
                v["disulfide_distogram_loss"] = _decay(g, n_all, 3.6, 0.45, k=2.2) + rng.normal(0, 0.04)
            if not pre_run:
                v["plddt_loss"] = _decay(g - pre, n_all - pre, 0.55, 0.22, k=2.0) + rng.normal(0, 0.01)
                v["i_pae_loss"] = _decay(g - pre, n_all - pre, 0.70, 0.40, k=2.0) + rng.normal(0, 0.01)
                v["pae_loss"] = _decay(g - pre, n_all - pre, 0.45, 0.20, k=2.0) + rng.normal(0, 0.01)
                v["rg_loss"] = 0.30 + rng.normal(0, 0.01)
                if not off:
                    v["disulfide_loss"] = max(0.0, _decay(g - pre, n_all - pre, 3.0, 0.05) + rng.normal(0, 0.05))
                    v["disulfide_worst_pair_loss"] = max(0.0, _decay(g - pre, n_all - pre, 2.0, 0.08) + rng.normal(0, 0.03))
                    v["disulfide_angle_loss"] = max(0.0, _decay(g - pre, n_all - pre, 12.0, 3.0) + rng.normal(0, 0.3))
                    v["disulfide_dihedral_loss"] = max(0.0, _decay(g - pre, n_all - pre, 40.0, 8.0) + rng.normal(0, 1.0))
            w = dict(W)
            w["disulfide_distogram_loss"] = nll_w
            w["helix_loss"] = helix_w
            w["disulfide_loss"] = SS_SCHEDULE[ss_stage]
            if off:
                for k in list(w):
                    if k.startswith("disulfide"):
                        w[k] = 0.0
            total = sum(v[k] * w.get(k, 0.0) for k in v)
            items = [f"{k}:{x:.2f}" for k, x in v.items()]
            if nan_at is not None and g == nan_at:
                items[0] = "con_loss:nan"
            if not pre_run and sg and not off:
                d = [2.05 + (6.0 - 2.05 * 0.0) * np.exp(-0.06 * (g - pre)) * (0.8 + 0.4 * j) + rng.normal(0, 0.02) for j in range(n_pairs)]
                items.append("sg_sg:" + "/".join(f"{x:.2f}" for x in d))
            if i in (0, n - 1):
                if i == n - 1 and name != "hard":
                    for _ in range(3):      # a few residues change during a stage
                        pos = int(rng.integers(0, length))
                        if seq[pos] != "C":
                            seq = seq[:pos] + str(rng.choice(list(ALPHA.replace("C", "")))) + seq[pos + 1:]
                out.append(f"[loop_seq] pre_run={pre_run} stage={ss_stage} epoch={i} seq={seq}")
            head = f"Epoch {i}: lr: {lr_of(i):.3f}"
            if fields:
                head += f", soft: {soft_of(i):.2f}, hard: {hard_of(i):.2f}, temp: {temp_of(i):.2f}, total loss: {total:.2f}"
            out.append(f"{head}, {items}")
            if noise and i % 7 == 3:
                out.append("  0%|          | 0/150 [elapsed: 00:00 remaining: ?]\r  3%|▎         | 5/150 [elapsed: 00:01 remaining: 00:30]")
            k_global += 1

    stage("pre", pre, True, lambda i: 1.0, lambda i: 1.0, lambda i: 0.0, lambda i: 1.0, "soft")
    out.append("warm up done")
    out.append("-" * 100)
    out.append("logits to softmax(T=0.8)")
    out.append("-" * 100)
    stage("soft", soft, False, lambda i: 0.1, lambda i: 0.8 * i / soft, lambda i: 0.0, lambda i: 1.0, "soft")
    stage("temp", temp, False, lambda i: 0.1 * (0.01 + 0.99 * (1 - i / temp) ** 2), lambda i: 1.0, lambda i: 0.0,
          lambda i: 0.01 + 0.99 * (1 - i / temp) ** 2, "temp")
    stage("hard", hard, False, lambda i: 0.001, lambda i: 1.0, lambda i: 1.0, lambda i: 0.01, "hard")
    if semi:
        prev = 0.62 + rng.uniform(0, 0.1)
        out.append(f"best design iptm [{prev:.7f}]")
        out.append(f"Semi-greedy steps {semi}")
        for s in range(semi):
            vals = prev + rng.normal(0.01, 0.03, size=10)
            for t, x in enumerate(vals):
                out.append(f"Step {s}, Epoch {t}, iptm {x:.3f}")
            b = int(np.argmax(vals))
            if vals[b] > prev:
                out.append(f"Step {s}, Epoch {b}, Update sequence, iptm [{vals[b]:.7f}], previous iptm [{prev:.7f}]")
                out.append("Update sequence " + seq)
                prev = vals[b]
    if with_final:
        out += ["-" * 100, "Holo Protein PLDDT: 0.812", "Apo Protein PLDDT: 0.790", "-" * 100,
                "Holo Complex PLDDT: 0.801", "Apo Complex PLDDT: 0.700", "-" * 100, "-" * 100, "rmsd 3.2512345", "-" * 100,
                "best_sequence " + seq]
    return "\n".join(out) + "\n"


def synth_log(n_designs=1, seed=1, **kw):
    rng = np.random.default_rng(12345)
    hdr_kw = {k: kw.pop(k) for k in ("length", "pairs", "form", "weight") if k in kw}
    off = kw.get("off", False)
    if "nll_w" in kw and "weight" not in hdr_kw:
        hdr_kw["weight"] = kw["nll_w"]
    text = header(off=off, **hdr_kw)
    for k in range(n_designs):
        text += synth_design(rng, seed=None if seed is None else seed + k, helix_w=-0.05 - 0.07 * k, **kw)
    text += "HighSS design completed successfully.\n"
    return text
