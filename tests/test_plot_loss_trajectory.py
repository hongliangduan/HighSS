"""Tests of scripts/plot_loss_trajectory.py: parsing of the log of the design loop, stage detection, weights, the figure and the CSV.

The 'engine contract' tests take the print statements out of the engine source (design_engine.py, cli.py of the tree under test)
and feed their output to the parser, so that a change of the log format in the engine is noticed here.
The example logs of examples/loss_trajectory are read as well.
"""
from __future__ import annotations

import ast
import csv
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(HERE))
import plot_loss_trajectory as P            # noqa: E402
from synthetic_log import synth_log         # noqa: E402

TREE = ROOT
EXAMPLES = ROOT / "examples" / "loss_trajectory"

# the epoch lines of a design loop that prints no header (only the loop itself)
PLAIN_LOG = """Starting Boltz design step...
pre-run warm up
set in train mode
Epoch 0: lr: 1.000, soft: 1.00, hard: 0.00, temp: 1.00, total loss: 8.82, ['con_loss:4.91', 'i_con_loss:3.91', 'helix_loss:0.75']
Epoch 1: lr: 1.000, soft: 1.00, hard: 0.00, temp: 1.00, total loss: 8.46, ['con_loss:4.89', 'i_con_loss:3.58', 'helix_loss:0.86']
Epoch 2: lr: 1.000, soft: 1.00, hard: 0.00, temp: 1.00, total loss: 8.07, ['con_loss:4.74', 'i_con_loss:3.33', 'helix_loss:1.26']
Epoch 3: lr: 1.000, soft: 1.00, hard: 0.00, temp: 1.00, total loss: 7.64, ['con_loss:4.77', 'i_con_loss:2.87', 'helix_loss:1.46']
Epoch 4: lr: 1.000, soft: 1.00, hard: 0.00, temp: 1.00, total loss: 7.31, ['con_loss:4.44', 'i_con_loss:2.88', 'helix_loss:1.58']
"""


def parse(text, design=1):
    lines = text.replace("\r", "\n").split("\n")
    segs = P.split_designs(lines)
    seed, a, b, marker = segs[design - 1]
    return P.parse_design(lines[a:b], design, seed, marker), P.parse_header(lines)


def run(tmp_path, text, *args, name="run.txt"):
    log = tmp_path / name
    log.write_text(text, encoding="utf-8")
    rc = P.main([str(log), *args])
    return rc, log


# ----------------------------------------------------------------------------------------------------------------------
# parsing
# ----------------------------------------------------------------------------------------------------------------------
def test_a_log_without_the_highss_header_is_read():
    d, h = parse(PLAIN_LOG)
    assert len(d.rows) == 5
    assert [r.stage for r in d.rows] == ["pre-run"] * 5
    assert d.rows[0].total == 8.82 and d.rows[4].total == 7.31
    assert d.rows[0].terms == {"con_loss": 4.91, "i_con_loss": 3.91, "helix_loss": 0.75}
    assert d.rows[3].soft == 1.0 and d.rows[3].hard == 0.0 and d.rows[3].temp == 1.0 and d.rows[3].lr == 1.0
    assert d.seed is None and h == {}


def test_a_highss_log_has_the_four_stages_with_their_step_counts():
    d, h = parse(synth_log())
    assert P.stage_counts(d.rows) == [("pre-run", 30), ("soft", 80), ("annealed", 45), ("hard one-hot", 5)]
    assert [r.step for r in d.rows] == list(range(160))
    assert d.seed == 1
    assert h["length"] == 18 and h["pairs"] == "2-17,6-13" and h["form"] == "nll" and h["nll_weight"] == 4.0


def test_stage_lengths_other_than_the_defaults():
    d, _ = parse(synth_log(pre=10, soft=7, temp=6, hard=3))
    assert P.stage_counts(d.rows) == [("pre-run", 10), ("soft", 7), ("annealed", 6), ("hard one-hot", 3)]


def test_a_stage_of_one_epoch_is_still_a_stage():
    d, _ = parse(synth_log(pre=3, soft=4, temp=3, hard=1))
    assert P.stage_counts(d.rows) == [("pre-run", 3), ("soft", 4), ("annealed", 3), ("hard one-hot", 1)]


def test_stages_are_found_by_the_restart_of_the_epoch_counter_when_the_lines_have_no_soft_hard_temp_fields():
    d, _ = parse(synth_log(fields=False))
    assert P.stage_counts(d.rows) == [("pre-run", 30), ("soft", 80), ("annealed", 45), ("hard one-hot", 5)]
    assert all(r.total is None for r in d.rows)


def test_a_log_without_the_warm_up_markers_still_has_its_pre_run_found():
    text = "\n".join(l for l in synth_log().split("\n") if "pre-run warm up" not in l and "warm up done" not in l)
    d, _ = parse(text)
    assert P.stage_counts(d.rows) == [("pre-run", 30), ("soft", 80), ("annealed", 45), ("hard one-hot", 5)]


def test_several_designs_are_split_and_each_has_its_seed():
    text = synth_log(n_designs=3)
    segs = P.split_designs(text.split("\n"))
    assert [s[0] for s in segs] == [1, 2, 3]
    for k in (1, 2, 3):
        d, _ = parse(text, k)
        assert len(d.rows) == 160 and d.seed == k
        assert len(d.candidates) == 20          # the semi-greedy lines of THIS design only
        assert d.final["sequence"] and d.final["holo_plddt"] == pytest.approx(0.812)


def test_progress_bars_glued_to_an_epoch_line_do_not_hide_it():
    text = synth_log(pre=4, soft=2, temp=2, hard=1, noise=False)
    text = text.replace("Epoch 2: ", "  5%|▌   | 3/60 [00:01<00:20]\rEpoch 2: ", 1)
    d, _ = parse(text)
    assert len(d.rows) == 9 and d.rows[2].epoch == 2


def test_nan_values_do_not_break_the_parse_or_the_figure(tmp_path):
    text = synth_log(nan_at=5)
    d, _ = parse(text)
    assert math.isnan(d.rows[5].terms["con_loss"])
    rc, log = run(tmp_path, text)
    assert rc == 0 and (tmp_path / "run_trajectory.png").stat().st_size > 10_000


def test_a_single_sg_sg_value_is_read_as_a_list_of_one():
    text = synth_log(n_pairs=1)
    d, _ = parse(text)
    assert all(len(r.sg) == 1 for r in d.rows[30:])
    assert all(r.sg == [] for r in d.rows[:30])


def test_the_semi_greedy_lines_are_read():
    d, _ = parse(synth_log())
    assert d.semi_steps == 2 and len(d.candidates) == 20 and 0.5 < d.semi_start < 0.9
    assert {s for s, _, _ in d.candidates} == {0, 1}
    assert all(0 <= t < 10 for _, t, _ in d.candidates)
    for s, t, new, prev in d.updates:
        assert new > prev and any(c == (s, t, new) or (c[0] == s and c[1] == t) for c in d.candidates)


def test_a_log_that_is_still_running_is_read_up_to_where_it_got():
    text = "\n".join(synth_log().split("\n")[:40])
    d, _ = parse(text)
    assert 10 < len(d.rows) < 40 and P.stage_counts(d.rows) == [("pre-run", len(d.rows))]


def test_the_loop_seq_lines_are_attached_to_the_epoch_they_precede():
    d, _ = parse(synth_log())
    first = d.loops[0]
    assert first[0] == 0 and first[1] == "soft" and first[2] == 0 and first[4] is True
    stages = [(l[4], l[1], l[2]) for l in d.loops]
    assert stages == [(True, "soft", 0), (True, "soft", 29), (False, "soft", 0), (False, "soft", 79),
                      (False, "temp", 0), (False, "temp", 44), (False, "hard", 0), (False, "hard", 4)]


# ----------------------------------------------------------------------------------------------------------------------
# the format of the engine
# ----------------------------------------------------------------------------------------------------------------------
def engine_text():
    return (TREE / "highss" / "design_engine.py").read_text(encoding="utf-8")


def fstring_of(src, start):
    """The f-string literal of the first print(...) that starts with `start` (source text, evaluated with eval below)."""
    m = re.search(r"print\((f\"" + re.escape(start) + r".*?\")\)\s*$", src, re.M)
    assert m, f"print of {start!r} not found in the engine"
    return m.group(1)


@pytest.mark.skipif(not (TREE / "highss" / "design_engine.py").exists(), reason="engine source not found")
def test_the_epoch_line_the_engine_prints_is_parsed():
    f = fstring_of(engine_text(), "Epoch {i}:")
    class T:
        def item(self): return 7.25
    ns = dict(i=3, current_lr=0.1, opt={"soft": 0.4, "hard": 0.0, "temp": 1.0}, total_loss=T(),
              loss_str=["con_loss:3.10", "i_con_loss:2.20", "disulfide_distogram_loss:1.05", "sg_sg:2.31/2.05"])
    line = eval(f, {}, ns)
    d, _ = parse("pre-run warm up\n" + line + "\n")
    r = d.rows[0]
    assert (r.epoch, r.lr, r.soft, r.hard, r.temp, r.total) == (3, 0.1, 0.4, 0.0, 1.0, 7.25)
    assert r.terms == {"con_loss": 3.1, "i_con_loss": 2.2, "disulfide_distogram_loss": 1.05} and r.sg == [2.31, 2.05]


@pytest.mark.skipif(not (TREE / "highss" / "design_engine.py").exists(), reason="engine source not found")
def test_the_sg_sg_item_of_the_engine_is_what_the_parser_expects():
    src = engine_text()
    m = re.search(r"def format_sg_distances\(distances\):.*?return (.*?)\n", src, re.S)
    assert m
    ns = {}
    exec("def format_sg_distances(distances):\n    return " + m.group(1), ns)
    item = ns["format_sg_distances"]([2.3141, 2.0499, 5.0])
    assert item == "sg_sg:2.31/2.05/5.00"
    d, _ = parse(f"Epoch 0: lr: 0.100, soft: 1.00, hard: 0.00, temp: 1.00, total loss: 1.00, ['con_loss:1.00', '{item}']\n")
    assert d.rows[0].sg == [2.31, 2.05, 5.0]


@pytest.mark.skipif(not (TREE / "highss" / "design_engine.py").exists(), reason="engine source not found")
def test_the_loop_seq_and_semi_greedy_prints_of_the_engine_are_parsed():
    src = engine_text()
    out = [eval(fstring_of(src, "[loop_seq]"), {}, dict(pre_run=True, disulfide_stage="soft", i=0, current_sequence="ACDE"))]
    prev = np.array([0.7123456], dtype=np.float32)
    best = np.array([0.7421387], dtype=np.float32)
    out.append("best design iptm " + str(prev))
    out.append(eval(fstring_of(src, "Step {step}, Epoch {t}, iptm"), {}, dict(step=1, t=4, iptm=np.array([0.7013], dtype=np.float32))))
    out.append(eval(fstring_of(src, "Step {step}, Epoch {best_id}, Update sequence"), {},
                    dict(step=1, best_id=4, best_iptm=best, prev_iptm=prev)))
    out.append(eval(fstring_of(src, "Update sequence {best_seq}"), {}, dict(best_seq="ACDF")))
    # the statements themselves, as they are written in the engine
    assert 'print("best design iptm", prev_iptm)' in src and 'print("Semi-greedy steps", semi_greedy_steps)' in src
    assert 'print("best_sequence", best_sequence)' in src and 'print("rmsd", rmsd)' in src
    out += ["Semi-greedy steps 2", "Holo Protein PLDDT: 0.700", "Holo Complex PLDDT: 0.650", "rmsd 3.5", "best_sequence ACDF"]
    d, _ = parse("pre-run warm up\n" + "\n".join(out) + "\n")
    assert d.loops == [(0, "soft", 0, "ACDE", True)]
    assert d.semi_start == pytest.approx(0.7123456, abs=1e-6) and d.semi_steps == 2
    assert d.candidates == [(1, 4, 0.701)]
    assert d.updates == [(1, 4, pytest.approx(0.7421387, abs=1e-6), pytest.approx(0.7123456, abs=1e-6))]
    assert d.final == {"holo_plddt": 0.7, "holo_complex_plddt": 0.65, "apo_holo_rmsd": 3.5, "sequence": "ACDF"}


@pytest.mark.skipif(not (TREE / "highss" / "cli.py").exists(), reason="cli source not found")
def test_the_header_lines_of_the_cli_are_parsed():
    src = (TREE / "highss" / "cli.py").read_text(encoding="utf-8")
    for needle in ('print(f"  binder length:    {args.binder_length}")', 'print(f"  disulfide pairs:  {args.disulfide_pairs}")',
                   'print(f"  loop guidance:    {term}; "', "distogram term {args.disulfide_distogram_form} x {args.disulfide_distogram_loss:g}",
                   "disulfide loss OFF (no disulfide term in any stage)", "pre_drop_bonds {args.pre_drop_bonds}",
                   'print(f"  MSA mode:         {args.msa_mode}")'):
        assert needle in src, needle
    h = P.parse_header(["  binder length:    18", "  disulfide pairs:  2-17,6-13", "  MSA mode:         target",
                        "  loop guidance:    distogram term nll x 4; loop_fix_cys True; pre_drop_bonds True; loop_profile token"])
    assert h == {"length": 18, "pairs": "2-17,6-13", "form": "nll", "nll_weight": 4.0, "pre_drop_bonds": True, "msa_mode": "target"}
    h = P.parse_header(["  loop guidance:    distogram term nll x 4; loop_fix_cys True; pre_drop_bonds False; loop_profile token"])
    assert h["pre_drop_bonds"] is False
    h = P.parse_header(["  loop guidance:    disulfide loss OFF (no disulfide term in any stage); loop_fix_cys True"])
    assert h == {"loss_off": True}


@pytest.mark.skipif(not (TREE / "highss" / "design_engine.py").exists(), reason="engine source not found")
def test_the_weights_of_the_script_are_the_defaults_of_the_cli_and_the_schedule_of_the_engine(monkeypatch):
    pytest.importorskip("torch", reason="highss.cli needs the HighSS dependencies; the plotting script itself does not")
    monkeypatch.syspath_prepend(str(TREE))
    monkeypatch.syspath_prepend(str(TREE / "boltz" / "src"))
    monkeypatch.setattr(sys, "argv", ["highss", "--target_pdb", "x", "--target_chains", "A", "--binder_length", "18",
                                      "--disulfide_pairs", "2-17,6-13", "--checkpoint", "c", "--ccd_path", "d"])
    from highss import cli
    a = cli.parse_arguments()
    expected = {"con_loss": a.con_loss, "i_con_loss": a.i_con_loss, "plddt_loss": a.plddt_loss, "pae_loss": a.pae_loss,
                "i_pae_loss": a.i_pae_loss, "rg_loss": a.rg_loss, "disulfide_distogram_loss": a.disulfide_distogram_loss,
                "disulfide_worst_pair_loss": a.disulfide_worst_pair_loss, "disulfide_angle_loss": a.disulfide_angle_loss,
                "disulfide_dihedral_loss": a.disulfide_dihedral_loss}
    assert P.BASE_WEIGHTS == expected
    assert a.disulfide_schedule is True
    assert P.STAGE_SS_WEIGHT == {"soft": a.disulfide_loss_soft, "annealed": a.disulfide_loss_temp, "hard one-hot": a.disulfide_loss_hard}
    assert (a.helix_loss_min, a.helix_loss_max) == (-0.3, 0.0)


# ----------------------------------------------------------------------------------------------------------------------
# weights and the check
# ----------------------------------------------------------------------------------------------------------------------
def fit(text, design=1, overrides=""):
    d, h = parse(text, design)
    W = P.make_weights(h, P.parse_weight_overrides(overrides))
    P.fit_helix_weight(d.rows, W)
    return d, h, W, P.reconstruct(d.rows, W)


def test_the_helix_weight_is_fitted_per_design():
    text = synth_log(n_designs=3)
    for k, true in ((1, -0.05), (2, -0.12), (3, -0.19)):
        _, _, W, chk = fit(text, k)
        assert W.helix == pytest.approx(true, abs=0.006) and W.helix_fitted
        assert chk["n"] == 160 and chk["max_excess"] <= 0.02


def test_such_a_log_gives_a_small_helix_weight_and_a_matching_total():
    _, _, W, chk = fit(PLAIN_LOG)
    assert -0.02 < W.helix <= 0.0 and chk["max_excess"] <= 0.02


def test_wrong_weights_are_noticed():
    _, _, _, chk = fit(synth_log(), overrides="pae_loss=2.0")
    assert chk["max_excess"] > 0.02
    _, _, _, chk = fit(synth_log(), overrides="disulfide_loss=0.5")           # constant instead of the schedule
    assert chk["max_excess"] > 0.02


def test_the_distogram_weight_comes_from_the_header():
    d, h, W, chk = fit(synth_log(nll_w=2.0))
    assert W.base["disulfide_distogram_loss"] == 2.0 and chk["max_excess"] <= 0.02
    d, h, W, chk = fit(synth_log(nll_w=2.0).replace("distogram term nll x 2", "distogram term nll x 4"))
    assert chk["max_excess"] > 0.02


def test_a_run_with_the_switch_off_has_no_disulfide_weights_and_no_distogram_term():
    d, h, W, chk = fit(synth_log(off=True))
    assert h == {"length": 18, "pairs": "2-17,6-13", "loss_off": True, "pre_drop_bonds": True, "msa_mode": "none"}
    assert all(not k.startswith("disulfide") for r in d.rows for k in r.terms)
    assert all(v == 0.0 for k, v in W.base.items() if k.startswith("disulfide")) and chk["max_excess"] <= 0.02
    assert P.figure_panels(d) == ["a", "b", "c", "f"]


def test_a_given_helix_weight_is_not_fitted():
    _, _, W, _ = fit(synth_log(), overrides="helix_loss=-0.2")
    assert W.helix == -0.2 and not W.helix_fitted


def test_unknown_terms_are_reported_and_counted_as_zero():
    text = synth_log(pre=3, soft=2, temp=2, hard=1, noise=False).replace("'con_loss:", "'mystery_loss:5.00', 'con_loss:", 1)
    d, h, W, chk = fit(text)
    assert "mystery_loss" in W.unknown


def test_bad_weight_options_are_refused():
    with pytest.raises(SystemExit):
        P.parse_weight_overrides("pae_loss")
    with pytest.raises(SystemExit):
        P.parse_weight_overrides("pae_loss=abc")
    assert P.parse_weight_overrides("") == {} and P.parse_weight_overrides("a=1, b=2e-3") == {"a": 1.0, "b": 0.002}


def test_group_contributions_add_up_to_the_total():
    d, h, W, _ = fit(synth_log())
    groups = P.group_contributions(d.rows, W)
    for i, r in enumerate(d.rows):
        s = sum(v[i] for v in groups.values() if math.isfinite(v[i]))
        assert s == pytest.approx(r.total, abs=0.05)
    assert math.isnan(groups["confidence"][0]) and math.isfinite(groups["confidence"][40])


# ----------------------------------------------------------------------------------------------------------------------
# outputs
# ----------------------------------------------------------------------------------------------------------------------
def test_files_are_written_next_to_the_log(tmp_path, capsys):
    rc, log = run(tmp_path, synth_log(), "--target", "MDM2")
    assert rc == 0
    png, pdf, csv_path = (tmp_path / f"run_trajectory{s}" for s in (".png", ".pdf", ".csv"))
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and pdf.read_bytes()[:5] == b"%PDF-" and csv_path.stat().st_size > 1000
    out = capsys.readouterr().out
    assert "steps: pre-run 30, soft 80, annealed 45, hard one-hot 5   (160 in total)" in out
    assert "-> ok" in out and "Cys2–Cys17" in out and "target: MDM2" in out and "seed 1" in out
    assert "semi-greedy: 2 steps, 20 candidates" in out


def test_the_figure_has_the_width_of_one_page_and_all_panels(tmp_path):
    from matplotlib import image
    rc, _ = run(tmp_path, synth_log(), "--dpi", "100")
    img = image.imread(tmp_path / "run_trajectory.png")
    h, w = img.shape[:2]
    assert w == 720 and 800 < h < 1000


def test_the_panels_depend_on_what_the_log_has():
    d, _ = parse(synth_log())
    assert P.figure_panels(d) == list("abcdef")
    d, _ = parse(synth_log(semi=0))
    assert P.figure_panels(d) == list("abcde")
    d, _ = parse(synth_log(sg=False))
    assert P.figure_panels(d) == list("abcdf")
    d, _ = parse(PLAIN_LOG)
    assert P.figure_panels(d) == ["a", "b"]


@pytest.mark.parametrize("kw", [dict(), dict(semi=0), dict(sg=False), dict(off=True), dict(fields=False), dict(n_pairs=3),
                                dict(pre=4, soft=3, temp=3, hard=1), dict(semi=1), dict(pre=30, soft=0, temp=0, hard=0, semi=0)])
def test_the_figure_is_drawn_for_every_kind_of_log(tmp_path, kw):
    rc, _ = run(tmp_path, synth_log(**kw))
    assert rc == 0 and (tmp_path / "run_trajectory.png").stat().st_size > 5_000


def test_the_figure_of_such_a_log_is_drawn(tmp_path):
    rc, _ = run(tmp_path, PLAIN_LOG)
    assert rc == 0 and (tmp_path / "run_trajectory.png").stat().st_size > 5_000


def test_the_csv_holds_every_logged_number(tmp_path):
    rc, _ = run(tmp_path, synth_log())
    rows = list(csv.DictReader(open(tmp_path / "run_trajectory.csv", encoding="utf-8")))
    d, _ = parse(synth_log())
    assert len(rows) == 160
    r0, r50, r120 = rows[0], rows[50], rows[120]
    assert r0["stage"] == "pre-run" and r50["stage"] == "soft" and r120["stage"] == "annealed"
    assert float(r0["total_loss_logged"]) == d.rows[0].total
    assert float(r120["disulfide_distogram_loss"]) == d.rows[120].terms["disulfide_distogram_loss"]
    assert abs(float(r120["total_loss_weighted_sum"]) - float(r120["total_loss_logged"])) < 0.05
    assert r0["plddt_loss"] == "" and r0["sg_sg_1"] == "" and float(r120["sg_sg_2"]) == d.rows[120].sg[1]
    assert float(r50["w_disulfide_loss"]) == 0.004 and float(r120["w_disulfide_loss"]) == 0.008 and float(rows[157]["w_disulfide_loss"]) == 0.015
    assert float(r120["w_con_loss"]) == 1.0 and float(r0["w_helix_loss"]) == pytest.approx(-0.05, abs=0.006)


def test_design_numbers_and_file_names(tmp_path, capsys):
    text = synth_log(n_designs=3)
    rc, log = run(tmp_path, text, "--design", "2")
    assert rc == 0 and (tmp_path / "run_trajectory_d2.png").exists()
    assert "design 2   seed 2" in capsys.readouterr().out
    assert run(tmp_path, text, "--design", "4")[0] == 2 and run(tmp_path, text, "--design", "0")[0] == 2


def test_a_custom_output_prefix(tmp_path):
    rc, _ = run(tmp_path, synth_log(), "--out", str(tmp_path / "loss_figure"))
    assert rc == 0 and (tmp_path / "loss_figure.png").exists() and (tmp_path / "loss_figure.csv").exists()


def test_missing_and_empty_logs_are_reported_not_crashed(tmp_path, capsys):
    assert P.main([str(tmp_path / "nope.txt")]) == 2
    rc, _ = run(tmp_path, "just some text\nwithout epochs\n")
    assert rc == 2 and "no 'Epoch" in capsys.readouterr().err


def test_the_draft_legend_has_the_numbers_of_the_run(capsys, tmp_path):
    run(tmp_path, synth_log(), "--target", "MDM2")
    out = capsys.readouterr().out
    leg = out.split("Draft legend")[1]
    for needle in ("18-residue binder", "Cys2–Cys17, Cys6–Cys13", "seed 1", "pre-run (30 steps)", "soft (80 steps)", "annealed (45 steps)",
                   "hard one-hot (5 steps)", "(f) iPTM", "The rise of the total loss at the first soft step"):
        assert needle in leg, needle
    # panels that are not drawn are not described
    run(tmp_path, synth_log(semi=0, sg=False, off=True), "--out", str(tmp_path / "x"))
    leg = capsys.readouterr().out.split("Draft legend")[1]
    assert "(f)" not in leg and "(e)" not in leg and "(d)" not in leg


def test_the_summary_says_when_the_weights_do_not_fit(capsys, tmp_path):
    run(tmp_path, synth_log(), "--weights", "pae_loss=2")
    assert "DOES NOT MATCH" in capsys.readouterr().out


def test_x_ticks():
    assert P.x_ticks(160) == [0, 20, 40, 60, 80, 100, 120, 140, 159]
    assert P.x_ticks(5) == [0, 1, 2, 3, 4]
    assert P.x_ticks(30) == [0, 5, 10, 15, 20, 25, 29]
    assert P.x_ticks(1) == [0]


def test_the_sequence_lines_count_the_changed_residues():
    d, _ = parse(synth_log())
    lines = P.sequence_lines(d)
    assert lines[0].startswith("sequence at the start") and len(lines) == 9
    assert "pre-run epoch 0" in lines[1] and "annealed epoch 0" in lines[5] and "hard one-hot epoch 4" in lines[8]


# ----------------------------------------------------------------------------------------------------------------------
# groups, the gradient statement, the form of the distogram term
# ----------------------------------------------------------------------------------------------------------------------
def test_the_disulfide_objective_is_shown_as_its_two_parts():
    d, h, W, _ = fit(synth_log())
    groups = P.group_contributions(d.rows, W)
    assert list(groups) == ["contacts + helicity", "confidence", "distogram term", "Sγ–Sγ geometry"]
    for i, r in enumerate(d.rows):
        assert sum(v[i] for v in groups.values() if math.isfinite(v[i])) == pytest.approx(r.total, abs=0.05)
    assert math.isfinite(groups["distogram term"][0]) and math.isnan(groups["Sγ–Sγ geometry"][0])
    assert groups["distogram term"][5] == pytest.approx(4.0 * d.rows[5].terms["disulfide_distogram_loss"])
    assert math.isfinite(groups["Sγ–Sγ geometry"][40])


def test_the_legend_says_which_terms_carry_a_gradient(capsys, tmp_path):
    run(tmp_path, synth_log(), "--grad_steps", "0")
    leg = capsys.readouterr().out.split("Draft legend")[1]
    assert "without giving a gradient to the sequence logits" in leg and "through the distogram term" in leg
    run(tmp_path, synth_log(), "--grad_steps", "2")
    leg = capsys.readouterr().out.split("Draft legend")[1]
    assert "backpropagated through the last 2 denoising step(s)" in leg and "without giving a gradient" not in leg
    run(tmp_path, synth_log(off=True))
    leg = capsys.readouterr().out.split("Draft legend")[1]
    assert "gradient" not in leg and "distogram" not in leg


def test_the_prob_form_of_the_distogram_term_is_named_as_such(capsys, tmp_path):
    run(tmp_path, synth_log(form="prob"))
    leg = capsys.readouterr().out.split("Draft legend")[1]
    assert "one minus the probability" in leg
    run(tmp_path, synth_log())
    assert "negative log-probability" in capsys.readouterr().out.split("Draft legend")[1]


# ----------------------------------------------------------------------------------------------------------------------
# layout: no label over another label, nothing outside the figure - with the fonts of a server too
# ----------------------------------------------------------------------------------------------------------------------
def drawn_texts(fig):
    out = []
    for ax in fig.axes:
        out += list(ax.texts)
        leg = ax.get_legend()
        if leg:
            out += list(leg.get_texts())
        out += [ax.xaxis.label, ax.yaxis.label]
        for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            lo, hi = sorted(lim)
            for tick, loc in zip(axis.get_major_ticks(), axis.get_majorticklocs()):
                if lo - 1e-9 <= loc <= hi + 1e-9 and tick.label1.get_visible() and tick.label1.get_text():
                    out.append(tick.label1)
    return [t for t in out if t.get_visible() and t.get_text().strip()]


def inspect_figure(monkeypatch, tmp_path, text, *args):
    """Run the script on `text`; returns what the figure shows: the texts, the pairs of texts that overlap, the texts outside the figure."""
    import matplotlib.figure as mf
    found = {}
    with monkeypatch.context() as m:
        orig = mf.Figure.savefig

        def spy(self, *a, **k):
            if "bad" not in found:
                self.canvas.draw()
                r = self.canvas.get_renderer()
                items = [(t.get_text(), t.get_window_extent(r)) for t in drawn_texts(self)]
                W_, H_ = self.bbox.width, self.bbox.height
                found["bad"] = [(a_[0], b_[0]) for i, a_ in enumerate(items) for b_ in items[i + 1:]
                                if min(a_[1].x1, b_[1].x1) - max(a_[1].x0, b_[1].x0) > 0.5
                                and min(a_[1].y1, b_[1].y1) - max(a_[1].y0, b_[1].y0) > 0.5]
                found["outside"] = [t for t, bb in items if bb.x0 < -0.5 or bb.y0 < -0.5 or bb.x1 > W_ + 0.5 or bb.y1 > H_ + 0.5]
                found["texts"] = [t for t, _ in items]
            return orig(self, *a, **k)

        m.setattr(mf.Figure, "savefig", spy)
        rc, _ = run(tmp_path, text, *args)
    assert rc == 0
    return found


@pytest.mark.parametrize("font", ["", "DejaVu Sans"])
@pytest.mark.parametrize("kw", [dict(), dict(semi=0), dict(n_pairs=3), dict(off=True), dict(sg=False, semi=1)])
def test_no_label_covers_another_and_none_leaves_the_figure(tmp_path, monkeypatch, font, kw):
    found = inspect_figure(monkeypatch, tmp_path, synth_log(**kw), *(["--font", font] if font else []))
    assert len(found["texts"]) > 30
    assert found["bad"] == [] and found["outside"] == []


# ----------------------------------------------------------------------------------------------------------------------
# the example run without MSA (MDM2, 18 residues, default settings, seed 1): what the script reads, draws and says about it
# ----------------------------------------------------------------------------------------------------------------------
REAL = EXAMPLES / "mdm2_no_msa.txt"
NOTE = "not computed\nin the pre-run"


def real():
    return fit(REAL.read_text(encoding="utf-8"))


def test_the_example_run_is_read_completely():
    d, h, W, chk = real()
    assert P.stage_counts(d.rows) == [("pre-run", 30), ("soft", 80), ("annealed", 45), ("hard one-hot", 5)]
    assert d.seed == 1 and h["pairs"] == "2-17,6-13" and h["pre_drop_bonds"] is True and h["nll_weight"] == 4.0
    assert W.helix == pytest.approx(-0.128, abs=0.002) and chk["n"] == 160 and chk["max_excess"] <= 0.02
    assert d.rows[2].total == 26.64 and d.rows[2].terms["disulfide_distogram_loss"] == 4.69
    assert d.rows[0].sg == [] and d.rows[30].sg == [1.18, 2.77] and d.rows[159].sg == [1.85, 1.53]
    assert d.semi_start == pytest.approx(0.97176397, abs=1e-7) and d.semi_steps == 2 and len(d.candidates) == 20
    assert d.updates == [(0, 5, pytest.approx(0.9753826, abs=1e-7), pytest.approx(0.97176397, abs=1e-7))]
    assert d.final == {"holo_plddt": 0.985, "holo_complex_plddt": 0.985, "apo_holo_rmsd": pytest.approx(0.7623493, abs=1e-6),
                       "sequence": "FCWLWCELMGIDCDSYCG"}
    assert [(s, st, ep) for _, st, ep, _, s in d.loops] == [(True, "soft", 0), (True, "soft", 29), (False, "soft", 0), (False, "soft", 79),
                                                             (False, "temp", 0), (False, "temp", 44), (False, "hard", 0), (False, "hard", 4)]


def test_one_spike_cuts_the_axis_of_panel_a_in_the_example_run():
    d, h, W, _ = real()
    top, k, peak = P.panel_a_cut(d, W)
    assert k == 2 and peak == 26.64 and top == pytest.approx(1.12 * 13.10, abs=0.01)
    assert P.entry_jump(d, W) is None            # the total falls where the main run starts (the distogram term drops to zero)


def test_peak_cut_rules():
    flat = [5.0] * 30
    assert P.peak_cut([flat]) is None
    one = list(flat)
    one[7] = 40.0
    assert P.peak_cut([one]) == (pytest.approx(1.12 * 5.0), 7, 40.0)
    two = list(one)
    two[9] = 38.0                                # two spikes of the same height: nothing is cut
    assert P.peak_cut([two]) is None
    assert P.peak_cut([one[:8]]) is None         # too few steps
    nan = list(one)
    nan[3] = float("nan")
    assert P.peak_cut([nan])[1] == 7
    other = [0.0] * 30
    other[11] = 90.0                             # the spike can lie in another curve than the first one
    assert P.peak_cut([flat, other])[1] == 11
    assert P.peak_cut([[0.0] * 30]) is None      # a zero baseline is not scaled
    assert P.peak_cut([]) is None


def test_median_and_number_format():
    assert P.median([3, 1, 2]) == 2 and P.median([4, 1, 3, 2]) == 2.5 and math.isnan(P.median([])) and P.median([1, float("nan")]) == 1
    assert [P.num2(v) for v in (0.0, 0.004, 0.03, 0.465, 0.1)] == ["0", "0.004", "0.03", "0.47", "0.10"]


def test_the_distogram_trace_and_the_sg_counts_of_the_example_run():
    d, h, W, _ = real()
    tr = P.distogram_trace(d)
    assert tr["start"] == 0.74 and tr["first"] == pytest.approx(0.47, abs=0.006) and tr["last"] == pytest.approx(0.10, abs=0.006)
    assert tr["main_max"] == 0.03
    assert P.sg_within(d) == (109, 130)
    d, h, W, _ = fit(synth_log(pre=12, soft=5, temp=5, hard=2, semi=0))
    assert "first" not in P.distogram_trace(d)                                  # fewer than twenty pre-run steps
    d, h, W, _ = fit(synth_log(sg=False))
    assert P.sg_within(d) == (0, 0)


def test_the_summary_and_the_legend_of_the_example_run_carry_its_numbers(capsys, tmp_path):
    rc, _ = run(tmp_path, REAL.read_text(encoding="utf-8"), "--target", "MDM2")
    out = capsys.readouterr().out
    summary, leg = out.split("Draft legend")
    assert "distogram NLL in the pre-run: 0.74 at the first step; median 0.47 over the first ten steps, 0.10 over the last ten" in summary
    assert "all prescribed pairs within 2.8 Å in 109 of 130 sampled steps (84%)" in summary
    assert "total loss 9.75 -> 13.10 (median 6.49, range 5.64-26.64)" in summary
    for needle in ("The vertical axis of (a) is cut at 14.7; the arrowhead marks the total loss, 26.6, at step 2 (pre-run).",
                   "The pre-run runs without the bond feature; the distogram term was 0.74 at the first step and its median over the "
                   "last ten pre-run steps was 0.10. From the soft stage on the bond feature is supplied and the term did not exceed 0.03.",
                   "prescribed disulfide pairs Cys2–Cys17, Cys6–Cys13; no MSA; logged settings; seed 1)",
                   "Over the 130 steps in which a structure was sampled, all prescribed pairs were within 2.8 Å in 84% of the steps.",
                   "no structure is sampled in the pre-run"):
        assert needle in leg, needle


def test_the_sentences_about_the_bond_feature_and_the_cut_are_left_out_when_they_do_not_apply(capsys, tmp_path):
    run(tmp_path, synth_log())                                                  # no spike in this log
    leg = capsys.readouterr().out.split("Draft legend")[1]
    assert "vertical axis of (a) is cut" not in leg and "The pre-run runs without the bond feature" in leg
    run(tmp_path, synth_log().replace("pre_drop_bonds True", "pre_drop_bonds False"))
    assert "without the bond feature" not in capsys.readouterr().out.split("Draft legend")[1]
    run(tmp_path, synth_log(off=True))
    assert "bond feature" not in capsys.readouterr().out.split("Draft legend")[1]


def with_total(text, k, value):
    """The log with the total loss of the k-th Epoch line (0-based over the whole log) replaced by `value`."""
    lines, seen = text.split("\n"), -1
    for j, ln in enumerate(lines):
        if ln.startswith("Epoch "):
            seen += 1
            if seen == k:
                lines[j] = re.sub(r"total loss: [-+0-9.eE]+", f"total loss: {value:.2f}", ln)
                break
    return "\n".join(lines)


@pytest.mark.parametrize("k, label_side", [(0, "right of the arrowhead"), (80, "right of the arrowhead"), (158, "left of the arrowhead")])
def test_a_spike_in_the_total_is_cut_and_marked_in_the_figure(tmp_path, monkeypatch, k, label_side):
    text = with_total(synth_log(), k, 500.0)
    d, h = parse(text)
    W = P.make_weights(h, {})
    P.fit_helix_weight(d.rows, W)
    top, kk, peak = P.panel_a_cut(d, W)
    assert kk == k and peak == 500.0 and top < 60
    found = inspect_figure(monkeypatch, tmp_path, text)
    assert "500.0" in found["texts"] and found["bad"] == [] and found["outside"] == []


def test_panels_without_data_in_the_pre_run_say_so(tmp_path, monkeypatch):
    assert inspect_figure(monkeypatch, tmp_path, synth_log())["texts"].count(NOTE) == 2          # confidence terms and Sγ–Sγ distances
    assert inspect_figure(monkeypatch, tmp_path, synth_log(sg=False))["texts"].count(NOTE) == 1
    assert inspect_figure(monkeypatch, tmp_path, synth_log(pre=2, soft=100, temp=40, hard=5))["texts"].count(NOTE) == 0   # band too narrow
    assert inspect_figure(monkeypatch, tmp_path, REAL.read_text(encoding="utf-8"))["texts"].count(NOTE) == 2


# the example run with a target MSA (same target, binder length, pairs and seed)
REAL_MSA = EXAMPLES / "mdm2_target_msa.txt"


def real_msa():
    return fit(REAL_MSA.read_text(encoding="utf-8"))


def test_the_target_msa_run_is_read_completely():
    d, h, W, chk = real_msa()
    assert h["msa_mode"] == "target" and h["pairs"] == "2-17,6-13" and h["pre_drop_bonds"] is True and d.seed == 1
    assert P.stage_counts(d.rows) == [("pre-run", 30), ("soft", 80), ("annealed", 45), ("hard one-hot", 5)]
    assert W.helix == pytest.approx(-0.128, abs=0.002) and chk["n"] == 160 and chk["max_excess"] <= 0.02     # the same seed draws the same helix weight
    assert d.rows[0].total == 9.84 and d.rows[0].terms["disulfide_distogram_loss"] == 0.78
    assert d.rows[30].sg and d.rows[159].sg == [2.44, 2.63]
    assert d.semi_start == pytest.approx(0.9304317, abs=1e-7) and len(d.candidates) == 20
    assert d.updates == [(0, 1, pytest.approx(0.9437075, abs=1e-7), pytest.approx(0.9304317, abs=1e-7))]
    assert d.final["sequence"] == "WCEIKCGKKTKGCEWVCT" and d.final["holo_plddt"] == 0.97
    assert P.panel_a_cut(d, W) is None                          # no spike: the axis of panel (a) is not cut
    assert P.sg_within(d) == (130, 130)
    tr = P.distogram_trace(d)
    assert tr["start"] == 0.78 and tr["last"] == pytest.approx(0.06, abs=0.006) and tr["main_max"] == 0.01


def test_the_legend_of_the_target_msa_run_names_the_msa_condition_and_has_no_cut(capsys, tmp_path):
    rc, _ = run(tmp_path, REAL_MSA.read_text(encoding="utf-8"), "--target", "MDM2")
    leg = capsys.readouterr().out.split("Draft legend")[1]
    for needle in ("prescribed disulfide pairs Cys2–Cys17, Cys6–Cys13; target MSA, single-sequence binder; logged settings; seed 1)",
                   "the distogram term was 0.78 at the first step and its median over the last ten pre-run steps was 0.06",
                   "the term did not exceed 0.01", "all prescribed pairs were within 2.8 Å in 100% of the steps"):
        assert needle in leg, needle
    assert "vertical axis of (a) is cut" not in leg


@pytest.mark.parametrize("font", ["", "DejaVu Sans"])
def test_no_label_covers_another_in_the_figure_of_the_target_msa_run(tmp_path, monkeypatch, font):
    found = inspect_figure(monkeypatch, tmp_path, REAL_MSA.read_text(encoding="utf-8"), *(["--font", font] if font else []))
    assert found["texts"].count(NOTE) == 2 and len(found["texts"]) > 30
    assert found["bad"] == [] and found["outside"] == []


def test_the_msa_condition_is_left_out_when_the_log_does_not_say(capsys, tmp_path):
    run(tmp_path, synth_log().replace("  MSA mode:         none\n", ""))
    leg = capsys.readouterr().out.split("Draft legend")[1]
    assert "MSA" not in leg
    run(tmp_path, synth_log())
    assert "no MSA; logged settings" in capsys.readouterr().out.split("Draft legend")[1]


@pytest.mark.parametrize("font", ["", "DejaVu Sans"])
def test_no_label_covers_another_in_the_figure_of_the_example_run(tmp_path, monkeypatch, font):
    found = inspect_figure(monkeypatch, tmp_path, REAL.read_text(encoding="utf-8"), *(["--font", font] if font else []))
    assert "26.6" in found["texts"] and len(found["texts"]) > 30
    assert found["bad"] == [] and found["outside"] == []


# ----------------------------------------------------------------------------------------------------------------------
def test_the_script_needs_only_the_standard_library_numpy_and_matplotlib():
    tree = ast.parse((ROOT / "scripts" / "plot_loss_trajectory.py").read_text(encoding="utf-8"))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            mods.add(node.module.split(".")[0])
    assert mods <= set(sys.stdlib_module_names) | {"numpy", "matplotlib"}, mods


@pytest.mark.parametrize("status", ["ON", "OFF"])
def test_the_header_records_distogram_weight_and_geometry_status(status):
    header = P.parse_header([f"  loop guidance:    disulfide loss ON; distogram term nll x 2.5; geometry backprop {status}; loop_fix_cys True"])
    assert header["nll_weight"] == 2.5 and header["form"] == "nll"
    assert header["geometry_backprop"] is (status == "ON")
    design, _, weights, _ = fit(synth_log())
    legend = P.draft_legend(design, header, list("abcdef"), "MDM2", weights)
    if status == "ON":
        # the header does not say through how many denoising steps: the legend must not make a number up
        assert "backpropagated through the last [N] denoising step(s)" in legend and "last 1 " not in legend
        assert "backpropagated through the last 3 denoising step(s)" in P.draft_legend(design, header, list("abcdef"), "MDM2",
                                                                                      weights, 3)
    else:
        assert "without giving a gradient to the sequence logits" in legend


def test_a_log_with_the_geometry_gradient_on_leaves_the_step_count_to_the_user(capsys, tmp_path):
    text = synth_log().replace("distogram term nll x 4;", "distogram term nll x 4; geometry backprop ON;", 1)
    assert "geometry backprop ON;" in text
    assert run(tmp_path, text)[0] == 0
    out = capsys.readouterr().out
    before, legend = out.split("Draft legend")
    assert "backpropagated through the last [N] denoising step(s)" in legend and "last 1 " not in legend
    assert "--grad_steps N" in before
    assert run(tmp_path, text, "--grad_steps", "3")[0] == 0
    out = capsys.readouterr().out
    before, legend = out.split("Draft legend")
    assert "backpropagated through the last 3 denoising step(s)" in legend and "[N]" not in legend
    assert "--grad_steps" not in before


def test_gradient_status_is_not_inferred_from_loss_values():
    design, header, weights, _ = fit(synth_log())
    legend = P.draft_legend(design, header, list("abcdef"), "MDM2", weights)
    assert "without giving a gradient to the sequence logits" not in legend
    assert "backpropagated through the last" not in legend
