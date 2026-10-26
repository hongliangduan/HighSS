#!/usr/bin/env python3
"""Loss trajectory of one design, drawn from the log of a HighSS run.

    python scripts/plot_loss_trajectory.py run.txt

Reads the 'Epoch ...' lines of the design loop (pre-run, soft, annealed, hard one-hot) and the semi-greedy lines, and writes
    <log>_trajectory.png / .pdf   the figure
    <log>_trajectory.csv          every number in it (raw terms, weighted terms, Sγ–Sγ distances)
and prints a summary and a draft figure legend.

Options
    --design K        the K-th design of a log that holds several (default 1)
    --out PREFIX      output files PREFIX.png/.pdf/.csv (default: next to the log)
    --weights a=1,b=2 loss weights, only if the run did not use the HighSS defaults (con_loss, i_con_loss, plddt_loss, pae_loss,
                      i_pae_loss, disulfide_distogram_loss, disulfide_loss [constant, no stage schedule], helix_loss, ...)
    --target NAME     target name for the draft legend
    --grad_steps N    number of denoising steps that carry the geometry gradient; the log header records only whether it was on,
                      so the legend shows [N] when the gradient was on and N is not given
    --dpi N           resolution of the PNG (default 300)
    --font NAME       font family to try first (default: Arial, Helvetica, Liberation Sans, DejaVu Sans)

The 'Epoch' lines print the raw value of every loss term and the total loss, which is the weighted sum of the terms. The weights are
the HighSS defaults, except the helix weight (drawn at random for every design, so it is fitted from the log) and the weight of the
distogram term (read from the 'loop guidance' line of the log header). The script checks that the weighted sum of the logged terms
reproduces the logged total and says so if it does not.

Needs only the Python standard library, numpy and matplotlib.
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------------------------------------------------------
# log parsing
# --------------------------------------------------------------------------------------------------------------------------
NUM = r"(?:[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|[-+]?nan|[-+]?inf)"
EPOCH_RE = re.compile(
    rf"Epoch\s+(\d+):\s*lr:\s*({NUM})"
    rf"(?:,\s*soft:\s*({NUM}),\s*hard:\s*({NUM}),\s*temp:\s*({NUM}),\s*total loss:\s*({NUM}))?"
    r"\s*,?\s*\[(.*)\]",
    re.I,
)
ITEM_RE = re.compile(r"'([A-Za-z_]\w*):([^']*)'")
LOOP_RE = re.compile(r"\[loop_seq\]\s+pre_run=(\w+)\s+stage=(\w+)\s+epoch=(\d+)\s+seq=(\S+)")
SEED_RE = re.compile(r"^\s*\[seed\]\s+(-?\d+)")
BEST_RE = re.compile(rf"best design iptm\s*\[?\s*({NUM})", re.I)
SEMI_RE = re.compile(r"Semi-greedy steps\s+(\d+)")
CAND_RE = re.compile(rf"Step\s+(\d+),\s*Epoch\s+(\d+),\s*iptm\s+\[?\s*({NUM})", re.I)
UPD_RE = re.compile(rf"Step\s+(\d+),\s*Epoch\s+(\d+),\s*Update sequence,\s*iptm\s+\[?\s*({NUM})\]?,\s*previous iptm\s+\[?\s*({NUM})", re.I)
UPDSEQ_RE = re.compile(r"^\s*Update sequence\s+([A-Za-z]+)\s*$")
FINAL_RES = {
    "holo_plddt": re.compile(rf"Holo Protein PLDDT:\s*({NUM})", re.I),
    "holo_complex_plddt": re.compile(rf"Holo Complex PLDDT:\s*({NUM})", re.I),
    "apo_holo_rmsd": re.compile(rf"^\s*rmsd\s+\[?\s*({NUM})", re.I),
}
BEST_SEQ_RE = re.compile(r"^\s*best_sequence\s+([A-Za-z]+)\s*$")
HDR_LENGTH = re.compile(r"binder length:\s+(\d+)")
HDR_PAIRS = re.compile(r"disulfide pairs:\s+(\S+)")
HDR_GUIDE = re.compile(rf"loop guidance:.*?distogram term (\w+) x ({NUM})", re.I)
HDR_OFF = re.compile(r"disulfide loss OFF")
HDR_GEOMETRY = re.compile(r"geometry backprop\s+(ON|OFF)", re.I)
HDR_DROP = re.compile(r"pre_drop_bonds\s+(True|False)", re.I)
HDR_MSA = re.compile(r"MSA mode:\s+(\w+)", re.I)

STAGE_ORDER = ("soft", "annealed", "hard one-hot")


def to_float(tok):
    try:
        return float(tok)
    except (TypeError, ValueError):
        return float("nan")


def finite(x) -> bool:
    return x is not None and math.isfinite(x)


@dataclass
class Row:
    step: int
    stage_id: int
    epoch: int
    lr: float
    soft: float | None
    hard: float | None
    temp: float | None
    total: float | None
    terms: dict
    sg: list
    stage: str = ""


@dataclass
class Design:
    index: int
    seed: int | None
    rows: list
    loops: list = field(default_factory=list)       # (row index it precedes, stage, epoch, sequence, pre_run)
    semi_steps: int | None = None
    semi_start: float | None = None
    candidates: list = field(default_factory=list)  # (step, trial, iptm)
    updates: list = field(default_factory=list)     # (step, trial, iptm, previous iptm)
    final: dict = field(default_factory=dict)


def read_lines(path) -> list:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    return text.replace("\r", "\n").split("\n")


def parse_header(lines) -> dict:
    """What the 'HighSS run' header says (all optional)."""
    out = {}
    for ln in lines:
        if "pre-run warm up" in ln:
            break
        m = HDR_LENGTH.search(ln)
        if m and "length" not in out:
            out["length"] = int(m.group(1))
        m = HDR_PAIRS.search(ln)
        if m and "pairs" not in out:
            out["pairs"] = m.group(1)
        m = HDR_GUIDE.search(ln)
        if m and "nll_weight" not in out:
            out["form"], out["nll_weight"] = m.group(1), to_float(m.group(2))
        m = HDR_GEOMETRY.search(ln)
        if m and "geometry_backprop" not in out:
            out["geometry_backprop"] = m.group(1).upper() == "ON"
        m = HDR_DROP.search(ln)
        if m and "pre_drop_bonds" not in out:
            out["pre_drop_bonds"] = m.group(1).lower() == "true"
        m = HDR_MSA.search(ln)
        if m and "msa_mode" not in out:
            out["msa_mode"] = m.group(1).lower()
        if HDR_OFF.search(ln):
            out["loss_off"] = True
    return out


def split_designs(lines) -> list:
    """[(seed, first line, end line)] - a design starts at its 'pre-run warm up' line (its '[seed] n' line is just before)."""
    starts = [i for i, ln in enumerate(lines) if "pre-run warm up" in ln]
    if not starts:
        return [(None, 0, len(lines), False)]
    out = []
    for k, s in enumerate(starts):
        seed = None
        for j in range(s - 1, max(s - 4, (starts[k - 1] if k else -1)), -1):
            m = SEED_RE.match(lines[j])
            if m:
                seed = int(m.group(1))
                break
        out.append((seed, s, starts[k + 1] if k + 1 < len(starts) else len(lines), True))
    return out


def parse_design(lines, index, seed, has_marker) -> Design:
    rows, loops = [], []
    d = Design(index=index, seed=seed, rows=rows, loops=loops)
    phase = "pre" if has_marker else "main"
    stage_phase, stage_id, prev_epoch = {}, -1, None
    for ln in lines:
        if "pre-run warm up" in ln:
            phase, prev_epoch = "pre", None
            continue
        if "warm up done" in ln:
            phase, prev_epoch = "main", None
            continue
        m = EPOCH_RE.search(ln)
        if m:
            epoch = int(m.group(1))
            if prev_epoch is None or epoch <= prev_epoch:        # the epoch counter restarts in every stage
                stage_id += 1
                stage_phase[stage_id] = phase
            prev_epoch = epoch
            soft, hard, temp, total = (None if m.group(k) is None else to_float(m.group(k)) for k in (3, 4, 5, 6))
            terms, sg = {}, []
            for name, val in ITEM_RE.findall(m.group(7)):
                if name == "sg_sg" or "/" in val:
                    sg = [to_float(x) for x in val.split("/")]
                else:
                    terms[name] = to_float(val)
            rows.append(Row(len(rows), stage_id, epoch, to_float(m.group(2)), soft, hard, temp, total, terms, sg))
            continue
        m = LOOP_RE.search(ln)
        if m:
            loops.append((len(rows), m.group(2), int(m.group(3)), m.group(4), m.group(1) == "True"))
            continue
        m = BEST_RE.search(ln)
        if m:
            d.semi_start = to_float(m.group(1))
            continue
        m = SEMI_RE.search(ln)
        if m:
            d.semi_steps = int(m.group(1))
            continue
        m = UPD_RE.search(ln)
        if m:
            d.updates.append((int(m.group(1)), int(m.group(2)), to_float(m.group(3)), to_float(m.group(4))))
            continue
        m = CAND_RE.search(ln)
        if m:
            d.candidates.append((int(m.group(1)), int(m.group(2)), to_float(m.group(3))))
            continue
        for key, rx in FINAL_RES.items():
            m = rx.search(ln)
            if m:
                d.final[key] = to_float(m.group(1))
        m = BEST_SEQ_RE.match(ln)
        if m:
            d.final["sequence"] = m.group(1)
    name_stages(rows, stage_phase, has_marker)
    return d


def _looks_like_pre_run(rs) -> bool:
    if all(r.soft is not None for r in rs):
        return all(r.soft == 1.0 and r.hard == 0.0 and r.temp == 1.0 for r in rs)
    lrs = sorted(r.lr for r in rs if finite(r.lr))
    return bool(lrs) and lrs[len(lrs) // 2] >= 0.5


def name_stages(rows, stage_phase, has_marker) -> None:
    ids = sorted({r.stage_id for r in rows})
    by = {k: [r for r in rows if r.stage_id == k] for k in ids}
    pos = 0
    for k in ids:
        rs = by[k]
        pre = stage_phase.get(k) == "pre" or (not has_marker and k == ids[0] and len(ids) > 1 and _looks_like_pre_run(rs))
        if pre:
            name = "pre-run"
        else:
            soft = [r.soft for r in rs if finite(r.soft)]
            hard = [r.hard for r in rs if finite(r.hard)]
            if soft and hard:
                hard_med = sorted(hard)[len(hard) // 2]
                name = "hard one-hot" if hard_med >= 0.99 else ("annealed" if min(soft) >= 0.995 else "soft")
            else:
                name = STAGE_ORDER[pos] if pos < len(STAGE_ORDER) else f"stage {pos + 1}"
            pos += 1
        for r in rs:
            r.stage = name


def bands(rows) -> list:
    """[(name, first step, last step)] with neighbouring stages of the same name merged."""
    out = []
    for r in rows:
        if out and out[-1][0] == r.stage:
            out[-1][2] = r.step
        else:
            out.append([r.stage, r.step, r.step])
    return [tuple(b) for b in out]


# --------------------------------------------------------------------------------------------------------------------------
# weights
# --------------------------------------------------------------------------------------------------------------------------
BASE_WEIGHTS = {
    "con_loss": 1.0, "i_con_loss": 1.0, "plddt_loss": 0.1, "pae_loss": 0.4, "i_pae_loss": 0.1, "rg_loss": 0.0,
    "disulfide_distogram_loss": 4.0, "disulfide_worst_pair_loss": 0.5, "disulfide_angle_loss": 0.0005,
    "disulfide_dihedral_loss": 0.0002,
}
STAGE_SS_WEIGHT = {"soft": 0.004, "annealed": 0.008, "hard one-hot": 0.015}   # disulfide_loss under --disulfide_schedule true


@dataclass
class Weights:
    base: dict
    schedule: dict | None
    helix: float | None = None
    helix_fitted: bool = False
    unknown: set = field(default_factory=set)

    def of(self, name, stage) -> float:
        if name == "helix_loss":
            return self.helix or 0.0
        if name == "disulfide_loss":
            return self.schedule.get(stage, 0.0) if self.schedule is not None else self.base.get(name, 0.0)
        if name not in self.base:
            self.unknown.add(name)
            return 0.0
        return self.base[name]


def parse_weight_overrides(text: str) -> dict:
    out = {}
    for part in [p for p in (text or "").split(",") if p.strip()]:
        key, sep, val = part.partition("=")
        if not sep or not key.strip():
            raise SystemExit(f"--weights wants name=value,name=value ... (got {part!r})")
        try:
            out[key.strip()] = float(val)
        except ValueError:
            raise SystemExit(f"--weights: {val!r} is not a number (in {part!r})")
    return out


def make_weights(header: dict, overrides: dict) -> Weights:
    base, schedule, helix = dict(BASE_WEIGHTS), dict(STAGE_SS_WEIGHT), None
    if header.get("loss_off"):
        base = {k: (0.0 if k.startswith("disulfide") else v) for k, v in base.items()}
        schedule = {k: 0.0 for k in schedule}
    elif finite(header.get("nll_weight")):
        base["disulfide_distogram_loss"] = header["nll_weight"]
    for key, val in overrides.items():
        if key == "helix_loss":
            helix = val
        elif key == "disulfide_loss":
            base[key], schedule = val, None
        else:
            base[key] = val
    return Weights(base=base, schedule=schedule, helix=helix)


def fit_helix_weight(rows, W: Weights) -> None:
    """The helix weight is drawn at random for every design and is not in the log: total = sum(w*term), solved for w by least squares."""
    if W.helix is not None:
        return
    num = den = 0.0
    for r in rows:
        h = r.terms.get("helix_loss")
        if not finite(h) or not finite(r.total):
            continue
        rest = sum(W.of(k, r.stage) * v for k, v in r.terms.items() if k != "helix_loss" and finite(v))
        num += (r.total - rest) * h
        den += h * h
    if den > 1e-9:
        W.helix, W.helix_fitted = num / den, True


def reconstruct(rows, W: Weights) -> dict:
    """Weighted sum of the logged terms against the logged total."""
    devs, excess = [], []
    for r in rows:
        if not finite(r.total):
            continue
        used = [(k, v) for k, v in r.terms.items() if finite(v)]
        rec = sum(W.of(k, r.stage) * v for k, v in used)
        tol = 0.005 * sum(abs(W.of(k, r.stage)) for k, _ in used) + 0.006       # every printed number is rounded to 0.01
        devs.append(r.total - rec)
        excess.append(abs(r.total - rec) - tol)
    if not devs:
        return {"n": 0}
    return {"n": len(devs), "max_dev": max(abs(x) for x in devs), "max_excess": max(excess)}


GROUPS = (      # label, colour, terms, dash (0 solid, 1 dashed, 2 dotted)
    ("contacts + helicity", "#2a78d6", ("con_loss", "i_con_loss", "helix_loss"), 0),
    ("confidence", "#eb6834", ("plddt_loss", "pae_loss", "i_pae_loss"), 0),
    ("distogram term", "#1baf7a", ("disulfide_distogram_loss",), 0),
    ("Sγ–Sγ geometry", "#1baf7a", ("disulfide_loss", "disulfide_worst_pair_loss", "disulfide_angle_loss", "disulfide_dihedral_loss"), 1),
)
OTHER_COLOR = "#52514e"


def group_contributions(rows, W: Weights) -> dict:
    """{group label: [weighted sum of the group's terms in every step, nan where none of them is in the objective]}"""
    known = {t for _, _, ts, _ in GROUPS for t in ts}
    labels = [g[0] for g in GROUPS]
    members = {g[0]: set(g[2]) for g in GROUPS}
    extra = {k for r in rows for k in r.terms if k not in known and W.of(k, r.stage) != 0.0}
    if extra:
        labels.append("other terms")
        members["other terms"] = extra
    out = {lab: [] for lab in labels}
    for r in rows:
        for lab in labels:
            vals = [W.of(k, r.stage) * v for k, v in r.terms.items()
                    if k in members[lab] and finite(v) and W.of(k, r.stage) != 0.0]
            out[lab].append(sum(vals) if vals else float("nan"))
    return out


# --------------------------------------------------------------------------------------------------------------------------
# csv, summary, legend
# --------------------------------------------------------------------------------------------------------------------------
def ordered_terms(rows) -> list:
    seen = []
    for r in rows:
        for k in r.terms:
            if k not in seen:
                seen.append(k)
    canon = [t for _, _, ts, _ in GROUPS for t in ts]
    return sorted(seen, key=lambda k: (canon.index(k) if k in canon else len(canon), seen.index(k)))


def write_csv(path, design: Design, W: Weights) -> None:
    rows = design.rows
    terms = ordered_terms(rows)
    n_sg = max((len(r.sg) for r in rows), default=0)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "stage", "epoch_in_stage", "lr", "soft", "hard", "temp", "total_loss_logged", "total_loss_weighted_sum"]
                   + terms + [f"w_{t}" for t in terms] + [f"sg_sg_{i + 1}" for i in range(n_sg)])
        for r in rows:
            rec = sum(W.of(k, r.stage) * v for k, v in r.terms.items() if finite(v))
            w.writerow([r.step, r.stage, r.epoch, r.lr, "" if r.soft is None else r.soft, "" if r.hard is None else r.hard,
                        "" if r.temp is None else r.temp, "" if r.total is None else r.total, f"{rec:.4f}"]
                       + [r.terms.get(t, "") for t in terms]
                       + [f"{W.of(t, r.stage):g}" if t in r.terms else "" for t in terms]
                       + [r.sg[i] if i < len(r.sg) else "" for i in range(n_sg)])


def pair_labels(header: dict, n: int) -> list:
    pairs = [p for p in (header.get("pairs") or "").split(",") if "-" in p]
    if len(pairs) == n:
        return ["Cys" + p.split("-")[0] + "–Cys" + p.split("-")[1] for p in pairs]
    return [f"pair {i + 1}" for i in range(n)]


def stage_counts(rows) -> list:
    return [(name, last - first + 1) for name, first, last in bands(rows)]


def median(values) -> float:
    v = sorted(x for x in values if finite(x))
    if not v:
        return float("nan")
    m = len(v) // 2
    return v[m] if len(v) % 2 else 0.5 * (v[m - 1] + v[m])


def distogram_trace(design: Design) -> dict:
    """The distogram term at the first pre-run step ('start'), its median over the first and over the last ten pre-run steps ('first',
    'last') and its largest value after the pre-run ('main_max'); a key is missing when the log has too few such steps."""
    key = "disulfide_distogram_loss"
    pre = [r.terms[key] for r in design.rows if r.stage == "pre-run" and finite(r.terms.get(key))]
    main = [r.terms[key] for r in design.rows if r.stage != "pre-run" and finite(r.terms.get(key))]
    out = {}
    if len(pre) >= 20:
        out["start"], out["first"], out["last"] = pre[0], median(pre[:10]), median(pre[-10:])
    if main:
        out["main_max"] = max(main)
    return out


def sg_within(design: Design, limit: float | None = None):
    """(steps at which every prescribed pair is within `limit` Å, steps with a Sγ–Sγ distance of every pair)."""
    limit = REF_LINES[1][0] if limit is None else limit
    rows = [r for r in design.rows if r.sg and all(finite(d) for d in r.sg)]
    return sum(all(d <= limit for d in r.sg) for r in rows), len(rows)


def peak_cut(arrays, factor: float = 1.6):
    """A spike - the highest step of all curves of panel (a) lies more than `factor` times above the second-highest step - would
    squeeze the rest of the trajectory; the axis is then cut just above the second-highest step.
    Returns (top of the axis, index of the spike, its value) or None."""
    n = max((len(a) for a in arrays), default=0)
    env = []
    for i in range(n):
        vals = [a[i] for a in arrays if i < len(a) and finite(a[i])]
        env.append(max(vals) if vals else float("-inf"))
    if n < 10:
        return None
    order = sorted(range(n), key=lambda i: env[i], reverse=True)
    i1, i2 = order[0], order[1]
    if not env[i2] > 0 or env[i1] <= factor * env[i2]:
        return None
    return 1.12 * env[i2], i1, env[i1]


def num2(v: float) -> str:
    """Two decimals; one significant digit for values below 0.01."""
    return "0" if v == 0 else f"{v:.2f}" if abs(v) >= 0.01 else f"{v:.1g}"


def panel_a_series(design: Design, W: Weights):
    """Curves of panel (a): the total loss, the group contributions, and the names of the groups that have data."""
    groups = group_contributions(design.rows, W)
    total = [r.total if finite(r.total) else float("nan") for r in design.rows]
    if not any(finite(v) for v in total):
        total = [sum(W.of(k, r.stage) * v for k, v in r.terms.items() if finite(v)) for r in design.rows]
    present = [lab for lab, vals in groups.items() if any(finite(v) for v in vals)]
    return total, groups, present


def panel_a_cut(design: Design, W: Weights):
    total, groups, present = panel_a_series(design, W)
    return peak_cut([total] + ([groups[lab] for lab in present] if len(present) > 1 else []))


LOOP_STAGE_NAME = {"soft": "soft", "temp": "annealed", "hard": "hard one-hot"}


def sequence_lines(design: Design) -> list:
    """The binder sequence at the first and the last epoch of every stage ('[loop_seq]' lines) and how many residues changed."""
    if not design.loops:
        return []
    out, prev = ["sequence at the start and the end of each stage:"], None
    for _, stage, epoch, seq, pre_run in design.loops:
        label = f"{'pre-run' if pre_run else LOOP_STAGE_NAME.get(stage, stage)} epoch {epoch}"
        changed = "" if prev is None or len(prev) != len(seq) else f"   ({sum(a != b for a, b in zip(prev, seq))} changed)"
        out.append(f"  {label:22s} {seq}{changed}")
        prev = seq
    return out


def summary_lines(design: Design, W: Weights, check: dict, header: dict, files: list, log_name: str) -> list:
    rows = design.rows
    L = [f"log: {log_name}   design {design.index}" + (f"   seed {design.seed}" if design.seed is not None else "")]
    L.append("steps: " + ", ".join(f"{n} {c}" for n, c in stage_counts(rows)) + f"   ({len(rows)} in total)")
    present = {k for r in rows for k in r.terms}
    parts = [f"{k} {v:g}" for k, v in W.base.items() if k in present and k != "disulfide_loss" and v != 0.0]
    if "disulfide_loss" in present:
        parts.append(("disulfide_loss " + "/".join(f"{W.schedule[s]:g}" for s in STAGE_SS_WEIGHT) + " (soft/annealed/hard)")
                     if W.schedule is not None else f"disulfide_loss {W.base.get('disulfide_loss', 0.0):g}")
    if "helix_loss" in present:
        parts.append("helix_loss " + ("-" if W.helix is None else f"{W.helix:.3f}" + (" (fitted from the log)" if W.helix_fitted else "")))
    L.append("weights: " + ", ".join(parts))
    if check.get("n"):
        verdict = "ok" if check["max_excess"] <= 0.02 else "DOES NOT MATCH - the run used other weights? give them with --weights"
        L.append(f"check: weighted sum of the logged terms vs the logged total, largest difference {check['max_dev']:.3f} "
                 f"({check['n']} steps) -> {verdict}")
    else:
        L.append("check: the log has no 'total loss' values; the total is the weighted sum of the logged terms")
    if W.unknown:
        L.append("terms without a weight (counted as 0): " + ", ".join(sorted(W.unknown)))
    for name, first, last in bands(rows):
        seg = rows[first:last + 1]
        a, b = seg[0], seg[-1]
        ta = "nan" if not finite(a.total) else f"{a.total:.2f}"
        tb = "nan" if not finite(b.total) else f"{b.total:.2f}"
        tot = [r.total for r in seg if finite(r.total)]
        spread = f" (median {median(tot):.2f}, range {min(tot):.2f}-{max(tot):.2f})" if tot else ""
        extra = ""
        nll = [r.terms["disulfide_distogram_loss"] for r in seg if finite(r.terms.get("disulfide_distogram_loss"))]
        if nll:
            extra = f"   distogram NLL median {median(nll):.2f}, max {max(nll):.2f}"
        L.append(f"  {name:13s} steps {first:3d}-{last:3d}   total loss {ta} -> {tb}{spread}{extra}")
    trace = distogram_trace(design)
    if "first" in trace:
        L.append(f"distogram NLL in the pre-run: {trace['start']:.2f} at the first step; median {trace['first']:.2f} over the first ten steps, "
                 f"{trace['last']:.2f} over the last ten")
    ok, n_sg = sg_within(design)
    if n_sg:
        L.append(f"Sγ–Sγ: all prescribed pairs within {REF_LINES[1][0]:g} Å in {ok} of {n_sg} sampled steps ({100.0 * ok / n_sg:.0f}%)")
    L += sequence_lines(design)
    last = rows[-1]
    if last.sg:
        labs = pair_labels(header, len(last.sg))
        L.append("Sγ–Sγ at the last step: " + ", ".join(f"{lab} {d:.2f} Å" for lab, d in zip(labs, last.sg)))
    if design.candidates:
        acc = {(s, t) for s, t, _, _ in design.updates}
        L.append(f"semi-greedy: {len({s for s, _, _ in design.candidates})} steps, {len(design.candidates)} candidates, "
                 f"{len(acc)} accepted; iPTM {design.semi_start if design.semi_start is not None else float('nan'):.3f} before"
                 + (f" -> {design.updates[-1][2]:.3f} after" if design.updates else ", no candidate was accepted"))
    if design.final:
        names = {"holo_plddt": "holo binder pLDDT", "holo_complex_plddt": "holo complex pLDDT", "apo_holo_rmsd": "apo-holo RMSD"}
        L.append("final: " + ", ".join(f"{names.get(k, k)} {v:.3f}" for k, v in design.final.items() if k != "sequence")
                 + (f"   sequence {design.final['sequence']}" if "sequence" in design.final else ""))
    L += ["files:"] + [f"  {p}" for p in files]
    return L


def entry_jump(design: Design, W: Weights):
    """Text for the rise of the total loss where the first main-run step starts after the pre-run, if the terms that enter there explain it."""
    rows = design.rows
    pre = [r for r in rows if r.stage == "pre-run"]
    main = [r for r in rows if r.stage != "pre-run"]
    if not pre or not main or not (finite(pre[-1].total) and finite(main[0].total)):
        return None
    jump = main[0].total - pre[-1].total
    new_terms = [k for k, v in main[0].terms.items() if k not in pre[-1].terms and finite(v)]
    new_sum = sum(W.of(k, main[0].stage) * main[0].terms[k] for k in new_terms)
    if jump <= 0.3 or new_sum < 0.5 * jump:
        return None
    what = []
    if any(k in new_terms for k in ("plddt_loss", "pae_loss", "i_pae_loss")):
        what.append("confidence")
    if any(k.startswith("disulfide") for k in new_terms):
        what.append("Sγ–Sγ geometry")
    return ("The rise of the total loss at the first soft step is the entry of terms that are not part of the pre-run objective ("
            + " and ".join(what or ["further"]) + " terms).")


def draft_legend(design: Design, header: dict, panels: list, target: str, W: Weights | None = None, grad_steps: int | None = None) -> str:
    counts = dict(stage_counts(design.rows))
    bits = [f"target: {target}"]
    if header.get("length"):
        bits.append(f"{header['length']}-residue binder")
    if header.get("pairs"):
        bits.append("prescribed disulfide pairs " + ", ".join(pair_labels(header, len(header["pairs"].split(",")))))
    if header.get("msa_mode"):
        bits.append("target MSA, single-sequence binder" if header["msa_mode"] == "target" else "no MSA")
    bits.append("logged settings")
    if design.seed is not None:
        bits.append(f"seed {design.seed}")
    names_in_log = {k for r in design.rows for k in r.terms}
    geometry = any(k.startswith("disulfide_") and k != "disulfide_distogram_loss" for k in names_in_log)
    both_parts = geometry and "disulfide_distogram_loss" in names_in_log
    parts = {
        "a": "Total loss (black) and the weighted contribution of each group of loss terms"
             + (" (the distogram term and the Sγ–Sγ geometry terms are the two parts of the disulfide objective)" if both_parts else ""),
        "b": "intra-chain contact, inter-chain contact and helix terms (raw values)",
        "c": "pLDDT and PAE terms (raw values; not part of the pre-run objective)",
        "d": ("one minus the probability" if header.get("form") == "prob" else "negative log-probability")
             + ", under the distogram head, that the Cβ–Cβ distance of a prescribed Cys pair lies in "
             "the 3.3–4.8 Å window (default; mean over the pairs), the term through which the prescribed connectivity reaches the sequence logits",
        "e": "Sγ–Sγ distance of each prescribed pair in the structure sampled at that step (horizontal lines: 2.05 Å, the disulfide "
             "bond length, and 2.8 Å, the upper edge of the distance interval that the structure loss leaves unpenalized; "
             "no structure is sampled in the pre-run)",
        "f": "iPTM of the ten candidates tested at each semi-greedy step",
    }
    letters = [p for p in "abcdef" if p in panels]
    pan = "; ".join(f"({p}) {parts[p]}" for p in letters)
    st = ", ".join(f"{n} ({c} steps)" for n, c in counts.items())
    jump = entry_jump(design, W) if W is not None else None
    cut = panel_a_cut(design, W) if W is not None and "a" in panels else None
    if cut is not None:
        top, k, peak = cut
        total = panel_a_series(design, W)[0]
        what = "the total loss" if finite(total[k]) and abs(total[k] - peak) < 1e-9 else "the highest curve"
        cut_text = (f"The vertical axis of (a) is cut at {top:.3g}; the arrowhead marks {what}, {peak:.1f}, "
                    f"at step {k} ({design.rows[k].stage}).")
    else:
        cut_text = None
    trace = distogram_trace(design)
    if "d" in panels and header.get("pre_drop_bonds") and {"start", "last", "main_max"} <= set(trace):
        d_text = (f"The pre-run runs without the bond feature; the distogram term was {num2(trace['start'])} at the first step and "
                  f"its median over the last ten pre-run steps was {num2(trace['last'])}. From the soft stage on the bond feature is "
                  f"supplied and the term did not exceed {num2(trace['main_max'])}.")
    else:
        d_text = None
    ok, n_sg = sg_within(design)
    sg_text = (f"Over the {n_sg} steps in which a structure was sampled, all prescribed pairs were within {REF_LINES[1][0]:g} Å "
               f"in {100.0 * ok / n_sg:.0f}% of the steps." if "e" in panels and n_sg else None)
    # The run header says only whether the geometry gradient was on, not through how many denoising steps: the number comes from
    # --grad_steps, and the legend leaves [N] to fill in when it is not given.
    if grad_steps is None and header.get("geometry_backprop") is False:
        grad_steps = 0
    depth_unknown = grad_steps is None and header.get("geometry_backprop") is True
    if both_parts and (grad_steps is not None or depth_unknown):
        grad = ("The Sγ–Sγ geometry terms are evaluated on sampled coordinates and enter the logged total without giving a gradient to "
                "the sequence logits; the prescribed pairing reaches the sequence through the distogram term." if grad_steps == 0 else
                f"The Sγ–Sγ geometry terms are backpropagated through the last {'[N]' if depth_unknown else grad_steps} "
                "denoising step(s) of the structure module.")
    else:
        grad = None
    tail = [t for t in (jump, cut_text, d_text, sg_text, grad) if t]
    return (f"Optimization trajectory of one design run ({'; '.join(bits)}). {pan}. "
            f"Shaded bands mark the stages: {st}. The total loss is the weighted sum of the loss terms."
            + "".join(" " + t for t in tail))


# --------------------------------------------------------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------------------------------------------------------
INK, INK2, MUTED, GRID, AXIS, BAND = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#f3f2ee"
C_CONTACT, C_CONF, C_SS = GROUPS[0][1], GROUPS[1][1], GROUPS[2][1]      # the geometry group has the colour of the distogram term
DASH = {0: "-", 1: (0, (4, 2)), 2: (0, (1, 1.6))}
TERM_STYLE = {      # raw term -> (legend label, colour, line style); the colour is the colour of its group, the dash tells the terms apart
    "con_loss": ("intra-chain contact", C_CONTACT, DASH[0]),
    "i_con_loss": ("inter-chain contact", C_CONTACT, DASH[1]),
    "helix_loss": ("helix", C_CONTACT, DASH[2]),
    "plddt_loss": ("pLDDT loss", C_CONF, DASH[0]),
    "pae_loss": ("PAE (intra-chain)", C_CONF, DASH[1]),
    "i_pae_loss": ("PAE (inter-chain)", C_CONF, DASH[2]),
    "disulfide_distogram_loss": ("distogram NLL of the prescribed pairs", C_SS, DASH[0]),
}
TERM_PANELS = (     # letter, title, terms, y label
    ("b", "Contact and helix terms", ("con_loss", "i_con_loss", "helix_loss"), "raw loss"),
    ("c", "Confidence terms", ("plddt_loss", "pae_loss", "i_pae_loss"), "raw loss"),
    ("d", "Distogram term of the prescribed Cys pairs: −log P(Cβ–Cβ distance in the window)", ("disulfide_distogram_loss",), "−log P"),
)
# figure geometry in inches
FIG_W, LM, RM = 7.2, 0.72, 0.72
TOP, STRIP_H, STRIP_GAP, GAP, XLAB, TITLE_ROW, BOTTOM = 0.08, 0.26, 0.58, 0.38, 0.50, 0.30, 0.06
PANEL_H = {"a": 1.35, "b": 0.78, "c": 0.78, "d": 0.78, "e": 0.88, "f": 0.85}
REF_LINES = ((2.05, "2.05 Å", "top"), (2.8, "2.8 Å", "bottom"))     # S–S bond length; --disulfide_upper_bound (bounded mode)


def figure_panels(design: Design) -> list:
    """Letters of the panels that have data."""
    rows = design.rows
    have = lambda names: any(finite(r.terms.get(k)) for r in rows for k in names)
    out = ["a"] + [letter for letter, _, names, _ in TERM_PANELS if have(names)]
    if any(r.sg for r in rows):
        out.append("e")
    if design.candidates:
        out.append("f")
    return out


def x_ticks(n: int) -> list:
    step = 20 if n > 100 else 10 if n > 40 else 5 if n > 15 else 1
    ticks = list(range(0, n, step))
    if (n - 1) - ticks[-1] > 0.3 * step:
        ticks.append(n - 1)
    return ticks


def draw(design: Design, W: Weights, header: dict, png: Path, pdf: Path, dpi: int = 300, font: str = "") -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ([font] if font else []) + ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": 7.5, "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "axes.unicode_minus": True,
    })

    rows = design.rows
    n = len(rows)
    x = np.arange(n)
    letters = figure_panels(design)
    shared = [p for p in letters if p != "f"]
    has_f = "f" in letters
    titles = {"a": "Total loss and weighted contributions", "e": "Sγ–Sγ distance of the prescribed pairs",
              "f": "Semi-greedy refinement: iPTM of the candidate sequences"}
    titles.update({p[0]: p[1] for p in TERM_PANELS})

    PW = FIG_W - LM - RM
    H = (TOP + STRIP_H + STRIP_GAP + sum(PANEL_H[p] for p in shared) + GAP * (len(shared) - 1) + XLAB
         + ((TITLE_ROW + PANEL_H["f"] + XLAB) if has_f else 0.0) + BOTTOM)
    fig = plt.figure(figsize=(FIG_W, H), facecolor="white")

    def add_axes(bottom, h):
        return fig.add_axes([LM / FIG_W, bottom / H, PW / FIG_W, h / H])

    cursor = H - TOP
    strip = add_axes(cursor - STRIP_H, STRIP_H)
    cursor -= STRIP_H + STRIP_GAP
    axes = {}
    for p in letters:
        bottom = cursor - PANEL_H[p]
        axes[p] = add_axes(bottom, PANEL_H[p])
        cursor = bottom - (GAP if p not in (shared[-1], "f") else XLAB + TITLE_ROW)
    xlim = (-0.5, n - 0.5)

    def style(ax):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelcolor=INK2, length=2.5, width=0.5, labelsize=7)
        ax.yaxis.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)

    def series(ax, y, color, ls, lw=1.25, z=3):
        y = np.asarray(y, dtype=float)
        if np.isfinite(y).any():
            ax.plot(x, y, color=color, ls=ls, lw=lw, solid_capstyle="round", dash_capstyle="round", zorder=z)

    def header_row(ax, letter, title, handles, stacked=False, height=1.0):
        """Letter and title above the panel, the legend beside the title - or, with many entries, under it."""
        y = 1.06 + (0.30 / height if stacked else 0.0)
        ax.text(-LM / PW + 0.01, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", color=INK, va="bottom", ha="left")
        ax.text(0.0, y, title, transform=ax.transAxes, fontsize=7.8, color=INK, va="bottom", ha="left")
        if len(handles) > 1:
            kw = dict(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3) if stacked else \
                dict(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=len(handles) if len(handles) <= 3 else 2)
            ax.legend(handles=handles, frameon=False, fontsize=6.6, handlelength=2.6, columnspacing=1.1, labelspacing=0.25,
                      borderaxespad=0.1, labelcolor=INK2, **kw)

    # stage bands behind the panels, and the stage strip on top
    bnds = bands(rows)
    for p in shared:
        ax = axes[p]
        for j, (name, first, last) in enumerate(bnds):
            if j % 2 == 0:
                ax.axvspan(first - 0.5, last + 0.5, color=BAND, lw=0, zorder=0)
            if j:
                ax.axvline(first - 0.5, color=AXIS, lw=0.6, zorder=1)
        ax.set_xlim(*xlim)
    strip.set_xlim(*xlim)
    strip.set_ylim(0, 1)
    strip.axis("off")
    for j, (name, first, last) in enumerate(bnds):
        cnt = last - first + 1
        strip.add_patch(plt.Rectangle((first - 0.5, 0.0), cnt, 1.0, facecolor=BAND if j % 2 == 0 else "white", edgecolor=AXIS, lw=0.6))
        narrow = cnt / n < 0.10
        label = f"{name.replace(' one-hot', '')} ({cnt})" if narrow else f"{name} ({cnt})"
        if narrow and last == n - 1:        # a band too narrow for its name: the name goes into the right margin
            strip.annotate(label, xy=(last + 0.5, 0.5), xytext=(4, 0), textcoords="offset points", ha="left", va="center",
                           fontsize=6.8, color=INK2, annotation_clip=False)
        else:
            strip.text((first + last) / 2, 0.5, label, ha="center", va="center", fontsize=6.4 if narrow else 6.8, color=INK2)

    # panels without data in the first band (the pre-run) say so
    def note_pre_run(ax, empty):
        if not bnds or bnds[0][0] != "pre-run" or not empty:
            return
        _, first, last = bnds[0]
        if (last - first + 1) / n * PW < 0.85:         # the band is too narrow for the note
            return
        ax.text((first + last) / 2, 0.5, "not computed\nin the pre-run", transform=ax.get_xaxis_transform(), ha="center", va="center",
                fontsize=6.2, color=INK2, linespacing=1.2, zorder=1)

    first_rows = rows[bnds[0][1]:bnds[0][2] + 1] if bnds else []

    # (a) total loss and the weighted contributions of the groups of terms
    total_l, groups, present = panel_a_series(design, W)
    total = np.array(total_l, dtype=float)
    ax = axes["a"]
    style(ax)
    handles = [Line2D([0], [0], color=INK, lw=1.9, label="total loss")]
    series(ax, total, INK, "-", lw=1.9, z=5)
    colors = {g[0]: g[1] for g in GROUPS}
    dashes = {g[0]: DASH[g[3]] for g in GROUPS}
    if len(present) > 1:            # with one group alone the group is the total
        for lab in present:
            series(ax, groups[lab], colors.get(lab, OTHER_COLOR), dashes.get(lab, "-"), lw=1.1)
            handles.append(Line2D([0], [0], color=colors.get(lab, OTHER_COLOR), ls=dashes.get(lab, "-"), lw=1.4, label=lab))
    lo, hi = ax.get_ylim()
    cut = panel_a_cut(design, W)
    if cut is not None:             # one spike would squeeze the rest of the trajectory: cut the axis, mark the spike
        hi, k, peak = cut
        tr = ax.get_xaxis_transform()
        ax.plot([k], [0.94], marker="^", ms=3.6, mfc=INK, mec="white", mew=0.5, ls="none", transform=tr, clip_on=False, zorder=6)
        left = k < 0.85 * n
        ax.text(k + (1.6 if left else -1.6), 0.94, f"{peak:.1f}", transform=tr, ha="left" if left else "right", va="center",
                fontsize=6.3, color=INK2, zorder=6)
    ax.set_ylim(min(lo, 0.0), hi)
    if min(lo, 0.0) < 0 < hi:
        ax.axhline(0, color=AXIS, lw=0.6, zorder=1)
    ax.set_ylabel("weighted loss", color=INK2, fontsize=7.5)
    header_row(ax, "a", titles["a"], handles, stacked=len(handles) > 4, height=PANEL_H["a"])

    # (b)-(d) raw terms
    for letter, title, names, ylabel in TERM_PANELS:
        if letter not in axes:
            continue
        if letter == "d" and header.get("form") == "prob":      # --disulfide_distogram_form prob
            title, ylabel = title.replace("−log P", "1 − P"), "1 − P"
        ax = axes[letter]
        style(ax)
        handles = []
        for name in names:
            y = [r.terms.get(name, np.nan) for r in rows]
            if np.isfinite(np.asarray(y, dtype=float)).any():
                lab, col, ls = TERM_STYLE[name]
                series(ax, y, col, ls)
                handles.append(Line2D([0], [0], color=col, ls=ls, lw=1.4, label=lab))
        note_pre_run(ax, not any(finite(r.terms.get(nm)) for r in first_rows for nm in names))
        ax.margins(y=0.10)
        ax.set_ylabel(ylabel, color=INK2, fontsize=7.5)
        header_row(ax, letter, title, handles)

    # (e) Sγ–Sγ distances
    if "e" in axes:
        ax = axes["e"]
        style(ax)
        n_sg = max(len(r.sg) for r in rows)
        labs = pair_labels(header, n_sg)
        handles = []
        for i in range(n_sg):
            series(ax, [r.sg[i] if i < len(r.sg) else np.nan for r in rows], C_SS, DASH[i % 3])
            handles.append(Line2D([0], [0], color=C_SS, ls=DASH[i % 3], lw=1.4, label=labs[i]))
        note_pre_run(ax, not any(r.sg for r in first_rows))
        ax.margins(y=0.10)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(min(ymin, 1.6), max(ymax, 3.2))
        x_first = next((r.step for r in rows if r.sg), 0) - 0.5        # the reference lines start where a structure is sampled
        for yref, txt, va in REF_LINES:
            ax.hlines(yref, x_first, n - 0.5, colors=MUTED, linewidth=0.6, zorder=2)
            ax.annotate(txt, xy=(1.0, yref), xycoords=("axes fraction", "data"), xytext=(3, 0), textcoords="offset points",
                        ha="left", va=va, fontsize=6.3, color=INK2, annotation_clip=False)
        ax.set_ylabel("distance (Å)", color=INK2, fontsize=7.5)
        header_row(ax, "e", titles["e"], handles)

    # x axis of the shared panels
    for p in shared:
        axes[p].tick_params(labelbottom=(p == shared[-1]))
    xa = axes[shared[-1]]
    xa.set_xticks(x_ticks(n))
    stages = ", ".join(dict.fromkeys(b[0] for b in bnds))
    xa.set_xlabel(f"optimization step ({stages})", color=INK2, fontsize=7.5, labelpad=3)

    # (f) semi-greedy refinement
    if has_f:
        ax = axes["f"]
        style(ax)
        steps = sorted({s for s, _, _ in design.candidates})
        per = max(sum(1 for s2, _, _ in design.candidates if s2 == s) for s in steps)
        pitch = per + 1.5
        accepted = {s: (t, v) for s, t, v, _ in design.updates}
        best = design.semi_start if finite(design.semi_start) else (design.updates[0][3] if design.updates else float("nan"))
        for j, s in enumerate(steps):
            x0 = j * pitch
            if j % 2 == 0:
                ax.axvspan(x0 - 0.9, x0 + per - 0.1, color=BAND, lw=0, zorder=0)
            if finite(best):
                ax.plot([x0 - 0.5, x0 + per - 0.5], [best, best], color=MUTED, lw=0.9, zorder=2)
            for s2, t, v in design.candidates:
                if s2 == s:
                    ok = accepted.get(s, (None, None))[0] == t
                    ax.plot([x0 + t], [v], marker="o", ms=4.4 if ok else 3.8, mfc=INK if ok else "white", mec=INK if ok else INK2,
                            mew=0.8, ls="none", zorder=4)
            if s in accepted:
                best = accepted[s][1]
        ax.set_xlim(-0.9, (len(steps) - 1) * pitch + per - 0.1)
        ax.set_xticks([j * pitch + (per - 1) / 2 for j in range(len(steps))])
        ax.set_xticklabels([f"step {s + 1}" for s in steps])
        ax.margins(y=0.15)
        ax.set_ylabel("iPTM", color=INK2, fontsize=7.5)
        ax.set_xlabel("semi-greedy step (candidates in the order tested)", color=INK2, fontsize=7.5, labelpad=3)
        handles = [Line2D([0], [0], marker="o", ls="none", mfc="white", mec=INK2, ms=4, label="candidate"),
                   Line2D([0], [0], marker="o", ls="none", mfc=INK, mec=INK, ms=4.4, label="accepted"),
                   Line2D([0], [0], color=MUTED, lw=1.0, label="best so far")]
        header_row(ax, "f", titles["f"], handles)

    fig.savefig(png, dpi=dpi, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    return letters


# --------------------------------------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Loss trajectory of one design, drawn from the log of a HighSS run.")
    ap.add_argument("log", help="the log of the run (the output of highss, e.g. run.txt)")
    ap.add_argument("--design", type=int, default=1, help="K-th design of a log with several designs (default 1)")
    ap.add_argument("--out", default=None, help="output prefix (default: <log>_trajectory next to the log)")
    ap.add_argument("--weights", default="", help="loss weights if the run did not use the defaults, e.g. plddt_loss=0.2,pae_loss=0.4")
    ap.add_argument("--target", default="[target]", help="target name for the draft legend")
    ap.add_argument("--grad_steps", type=int, default=None,
                    help="Number of denoising steps that carry the geometry gradient (the log header records only whether it was on)")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--font", default="", help="font family to try first (default: Arial, Helvetica, Liberation Sans, DejaVu Sans)")
    args = ap.parse_args(argv)

    path = Path(args.log)
    if not path.is_file():
        print(f"no such file: {path}", file=sys.stderr)
        return 2
    lines = read_lines(path)
    header = parse_header(lines)
    segs = split_designs(lines)
    if not 1 <= args.design <= len(segs):
        print(f"--design {args.design}: the log holds {len(segs)} design(s)", file=sys.stderr)
        return 2
    seed, a, b, marker = segs[args.design - 1]
    design = parse_design(lines[a:b], args.design, seed, marker)
    if not design.rows:
        print("no 'Epoch ...' lines in this design - is this the log of a highss run? "
              "(the run must have printed at least the first epochs)", file=sys.stderr)
        return 2

    W = make_weights(header, parse_weight_overrides(args.weights))
    fit_helix_weight(design.rows, W)
    check = reconstruct(design.rows, W)

    prefix = Path(args.out) if args.out else path.with_name(path.stem + "_trajectory" + (f"_d{args.design}" if args.design > 1 else ""))
    png, pdf, csv_path = (prefix.with_name(prefix.name + s) for s in (".png", ".pdf", ".csv"))
    write_csv(csv_path, design, W)
    letters = draw(design, W, header, png, pdf, args.dpi, args.font)
    print("\n".join(summary_lines(design, W, check, header, [str(png), str(pdf), str(csv_path)], path.name)))
    if header.get("geometry_backprop") and args.grad_steps is None:
        print("\nnote: the log header says the geometry gradient was on but not through how many denoising steps; "
              "pass --grad_steps N to put N in the legend.")
    print("\nDraft legend:\n" + draft_legend(design, header, letters, args.target, W, args.grad_steps))
    return 0


if __name__ == "__main__":
    sys.exit(main())
