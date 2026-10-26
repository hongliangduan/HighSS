"""Tests for --use_disulfide_loss, the master switch of the disulfide loss.

true (default) = the standard distogram and geometry objective. false = no disulfide loss at all: every disulfide loss weight and both gradient
routes are 0 (cli.LOSS_OFF), a non-zero value given for one of them together with the switch is refused, and the engine, which
computes no term whose weight is 0, has no disulfide term in its objective in any stage.  The engine tests run the real code of
boltz_hallucination(): the nested get_model_loss() and the stage-schedule block of design(), both extracted from the source, with a toy
model and spies for the three disulfide terms.
"""
import ast
import random
import re
import shutil
import sys
import textwrap
from pathlib import Path

import pytest
import torch

import highss.design_engine as de
from highss import cli

ROOT = Path(__file__).resolve().parents[1]
BASE = ["highss", "--target_pdb", "t.pdb", "--target_chains", "A", "--binder_length", "18",
        "--disulfide_pairs", "2-17,6-13", "--checkpoint", "m.ckpt", "--ccd_path", "ccd.pkl"]
OFF = ["--use_disulfide_loss", "false"]
STAGES = ("soft", "temp", "hard")
STRUCTURE_WEIGHTS = {"soft": 0.004, "temp": 0.008, "hard": 0.015}      # --disulfide_loss_soft / _temp / _hard


@pytest.fixture(autouse=True)
def _seeded():
    """The engine tests run get_model_loss() on random toy tensors; the same tensors every time."""
    torch.manual_seed(1234)
    random.seed(1234)


def parse(monkeypatch, *extra):
    monkeypatch.setattr(sys, "argv", BASE + list(extra))
    return vars(cli.parse_arguments())


# ------------------------------------------------------------------------------------------------ the command line
def test_the_switch_is_on_by_default(monkeypatch):
    got = parse(monkeypatch)
    assert got["use_disulfide_loss"] is True
    assert got["disulfide_grad_steps"] == 0 and got["disulfide_distogram_loss"] > 0


@pytest.mark.parametrize("value", ["false", "no", "0", "off", "False"])
def test_every_false_spelling_switches_the_loss_off(monkeypatch, value):
    assert parse(monkeypatch, "--use_disulfide_loss", value)["use_disulfide_loss"] is False


def test_a_bad_value_is_refused(monkeypatch):
    monkeypatch.setattr(sys, "argv", BASE + ["--use_disulfide_loss", "maybe"])
    with pytest.raises(SystemExit):
        cli.parse_arguments()


def test_off_sets_exactly_the_table_to_zero_and_changes_nothing_else(monkeypatch):
    on, off = parse(monkeypatch), parse(monkeypatch, *OFF)
    assert off.pop("use_disulfide_loss") is False and on.pop("use_disulfide_loss") is True
    assert all(off[key] == value for key, value in cli.LOSS_OFF.items())
    changed = {key for key in on if on[key] != off[key]}
    # the weights that are not 0 by default: the structure terms with their three stage weights, and the distogram term
    assert changed == {"disulfide_loss", "disulfide_loss_soft", "disulfide_loss_temp", "disulfide_loss_hard",
                       "disulfide_worst_pair_loss", "disulfide_angle_loss", "disulfide_dihedral_loss", "disulfide_distogram_loss"}
    assert on["disulfide_distogram_loss"] == 4.0 and on["disulfide_loss_soft"] == 0.004


def test_off_keeps_the_input_settings_and_the_pairs(monkeypatch):
    off = parse(monkeypatch, *OFF)
    assert off["loop_fix_cys"] is True and off["pre_drop_bonds"] is True and off["loop_profile"] == "token"
    assert off["disulfide_pairs"] == "2-17,6-13" and off["disulfide_schedule"] is True and off["disulfide_distogram_form"] == "nll"


def test_the_table_covers_every_disulfide_loss_option(monkeypatch):
    names = set(parse(monkeypatch))
    weights = {n for n in names if n.startswith("disulfide_") and (n.endswith("_loss") or re.search(r"_loss_(soft|temp|hard)$", n))}
    routes = {"disulfide_grad_steps", "disulfide_sds_scale", "debug_sds_profile_every"}
    assert weights | routes == set(cli.LOSS_OFF), "a disulfide loss option is missing from LOSS_OFF (or LOSS_OFF has an extra one)"


@pytest.mark.parametrize("name", sorted(cli.LOSS_OFF))
def test_off_refuses_a_non_zero_value_of_every_option_of_the_table(monkeypatch, capsys, name):
    monkeypatch.setattr(sys, "argv", BASE + OFF + ["--" + name, "1"])
    with pytest.raises(SystemExit) as exc:
        cli.parse_arguments()
    assert exc.value.code == 2
    err = " ".join(capsys.readouterr().err.split())
    assert f"--{name} 1" in err and "no disulfide loss at all" in err and "keep --use_disulfide_loss true" in err


def test_off_accepts_explicit_zeros(monkeypatch):
    plain = parse(monkeypatch, *OFF)
    assert parse(monkeypatch, *OFF, "--disulfide_loss", "0", "--disulfide_grad_steps", "0", "--disulfide_distogram_loss", "0.0") == plain


def test_off_names_every_offender(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", BASE + OFF + ["--disulfide_loss", "0.5", "--disulfide_sds_scale", "2"])
    with pytest.raises(SystemExit):
        cli.parse_arguments()
    err = " ".join(capsys.readouterr().err.split())
    assert "--disulfide_loss 0.5" in err and "--disulfide_sds_scale 2" in err


def test_on_accepts_every_option_of_the_table(monkeypatch):
    got = parse(monkeypatch, "--use_disulfide_loss", "true", "--disulfide_distogram_loss", "2", "--disulfide_loss", "0.01",
                "--disulfide_grad_steps", "1", "--disulfide_sds_scale", "0.5")
    assert got["disulfide_distogram_loss"] == 2.0 and got["disulfide_loss"] == 0.01
    assert got["disulfide_grad_steps"] == 1 and got["disulfide_sds_scale"] == 0.5


def test_the_position_of_the_switch_does_not_matter(monkeypatch):
    first = parse(monkeypatch, *OFF, "--seed", "3")
    monkeypatch.setattr(sys, "argv", ["highss", "--use_disulfide_loss=false"] + BASE[1:] + ["--seed", "3"])
    assert vars(cli.parse_arguments()) == first
    assert first["seed"] == 3 and first["disulfide_distogram_loss"] == 0.0


def test_off_with_rc3_defaults_is_rc3_input_settings_and_no_loss(monkeypatch):
    rc3 = parse(monkeypatch, "--rc3_defaults", "true")
    both = parse(monkeypatch, *OFF, "--rc3_defaults", "true")
    assert both == {**rc3, **cli.LOSS_OFF, "use_disulfide_loss": False}
    assert both["loop_fix_cys"] is False and both["pre_drop_bonds"] is False and both["loop_profile"] == "mean"
    assert both["disulfide_distogram_form"] == "prob"
    assert parse(monkeypatch, "--rc3_defaults", "true", *OFF) == both                  # the order of the two switches does not matter


def test_rc3_defaults_alone_keeps_the_structure_terms(monkeypatch):
    rc3 = parse(monkeypatch, "--rc3_defaults", "true")
    assert rc3["use_disulfide_loss"] is True and rc3["disulfide_loss"] == 0.008 and rc3["disulfide_loss_hard"] == 0.015
    assert rc3["disulfide_distogram_loss"] == 0.0                   # the RC3 behaviour: structure terms (logged only), no distogram term


def test_the_help_text_describes_the_switch(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["highss", "--help"])
    with pytest.raises(SystemExit) as exc:
        cli.parse_arguments()
    assert exc.value.code == 0
    text = re.sub(r"\s+", " ", capsys.readouterr().out)
    m = re.search(r"(?<=\s)--use_disulfide_loss USE_DISULFIDE_LOSS (.*?)\(default: ([^)]*)\)", text)
    assert m and m.group(2) == "True"
    for key in cli.LOSS_OFF:
        assert f"--{key}" in m.group(1), key
    assert "--rc3_defaults true" in m.group(1) and "no disulfide loss at all" in m.group(1)


# ------------------------------------------------------------------------------------------------ main() with the engine stubbed
def _run_main(monkeypatch, tmp_path, capsys, *extra):
    import highss.design_engine as engine
    pdb = tmp_path / "t.pdb"
    shutil.copy(ROOT / "examples" / "mdm2" / "mdm2_pro.pdb", pdb)
    (tmp_path / "m.ckpt").write_bytes(b"x")
    (tmp_path / "ccd.pkl").write_bytes(b"x")
    calls = {}

    class FakeModel:
        def train(self):
            return self

    monkeypatch.setattr(engine, "get_boltz_model", lambda *a, **k: FakeModel())
    monkeypatch.setattr(engine, "run_highss_design", lambda **kw: calls.update(kw))
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/boltz")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setattr(sys, "argv", ["highss", "--target_pdb", str(pdb), "--target_chains", "A", "--binder_length", "18",
                                      "--disulfide_pairs", "2-17,6-13", "--checkpoint", str(tmp_path / "m.ckpt"),
                                      "--ccd_path", str(tmp_path / "ccd.pkl"), "--work_dir", str(tmp_path / "run"),
                                      "--msa_mode", "none", *extra])
    cli.main()
    return calls, capsys.readouterr().out


def test_main_with_the_switch_on_hands_the_recipe_to_the_engine(monkeypatch, tmp_path, capsys):
    calls, out = _run_main(monkeypatch, tmp_path, capsys)
    assert calls["loss_scales"]["disulfide_distogram_loss"] == 4.0 and calls["loss_scales"]["disulfide_loss"] == 0.008
    assert (calls["disulfide_loss_soft"], calls["disulfide_loss_temp"], calls["disulfide_loss_hard"]) == (0.004, 0.008, 0.015)
    assert calls["disulfide_grad_steps"] == 0
    line = [l for l in out.splitlines() if l.startswith("  loop guidance:")][0]
    assert "disulfide loss ON; distogram term nll x 4;" in line and "OFF" not in line and "geometry backprop" not in line


def test_main_with_the_switch_off_hands_no_disulfide_loss_to_the_engine(monkeypatch, tmp_path, capsys):
    calls, out = _run_main(monkeypatch, tmp_path, capsys, *OFF)
    scales = calls["loss_scales"]
    assert {k for k in scales if "disulfide" in k} == {"disulfide_loss", "disulfide_worst_pair_loss", "disulfide_distogram_loss",
                                                       "disulfide_angle_loss", "disulfide_dihedral_loss"}
    assert all(scales[k] == 0 for k in scales if "disulfide" in k)
    assert (calls["disulfide_loss_soft"], calls["disulfide_loss_temp"], calls["disulfide_loss_hard"]) == (0.0, 0.0, 0.0)
    assert calls["disulfide_grad_steps"] == 0 and calls["disulfide_sds_scale"] == 0 and calls["disulfide_sds_profile_every"] == 0
    # the pairing itself is still handed over: the Cys positions and the bond feature are not losses
    assert calls["binder_bonds"] == "2_SG_17_SG,6_SG_13_SG" and calls["fix_residues"] == "2_C,6_C,13_C,17_C"
    cfg = calls["config"]
    assert cfg["loop_fix_cys"] is True and cfg["pre_drop_bonds"] is True and cfg["loop_profile"] == "token"
    line = [l for l in out.splitlines() if l.startswith("  loop guidance:")][0]
    assert "disulfide loss OFF (no disulfide term in any stage);" in line and "distogram term" not in line
    assert "loop_fix_cys True" in line and "pre_drop_bonds True" in line


def test_main_refuses_a_contradiction_before_loading_anything(monkeypatch, tmp_path, capsys):
    with pytest.raises(SystemExit):
        _run_main(monkeypatch, tmp_path, capsys, *OFF, "--disulfide_loss", "0.01")


def test_main_with_the_switch_on_passes_the_gradient_routes_on(monkeypatch, tmp_path, capsys):
    # Explicit advanced gradient and profiling settings reach the engine when the loss is enabled.
    calls, _ = _run_main(monkeypatch, tmp_path, capsys, "--disulfide_grad_steps", "1", "--disulfide_sds_scale", "0.5",
                         "--debug_sds_profile_every", "5")
    assert calls["disulfide_grad_steps"] == 1 and calls["disulfide_sds_scale"] == 0.5 and calls["disulfide_sds_profile_every"] == 5
    assert calls["loss_scales"]["disulfide_loss"] == 0.008 and calls["loss_scales"]["disulfide_distogram_loss"] == 4.0


# ------------------------------------------------------------------------------------------------ the engine: what an epoch computes
ENGINE_SRC = Path(de.__file__).read_text(encoding="utf-8")
ENGINE_TREE = ast.parse(ENGINE_SRC)


def _function(name):
    nodes = [n for n in ast.walk(ENGINE_TREE) if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(nodes) == 1, name
    return nodes[0]


def _lines(first, last):
    return textwrap.dedent("".join(ENGINE_SRC.splitlines(keepends=True)[first - 1:last]))


def _schedule_source():
    """The three statements of design() that turn the stage into the loss weights of the epoch (current_loss_scales)."""
    def walk(body):
        for i, st in enumerate(body):
            if isinstance(st, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "current_loss_scales" for t in st.targets):
                return body, i
            for attr in ("body", "orelse", "finalbody"):
                sub = getattr(st, attr, None)
                if isinstance(sub, list) and (found := walk(sub)):
                    return found
        return None
    body, i = walk(_function("design").body)
    assert isinstance(body[i + 1], ast.Assign) and isinstance(body[i + 2], ast.If)
    return _lines(body[i].lineno, body[i + 2].end_lineno)


SCHEDULE_SRC = _schedule_source()
MODEL_LOSS_SRC = _lines(_function("get_model_loss").lineno, _function("get_model_loss").end_lineno)


def _epoch_weights(kw, stage, i=3, iters=10):
    """current_loss_scales that design() builds for epoch i of a stage, from what main() handed to the engine."""
    ns = {"loss_scales": kw["loss_scales"], "binder_bonds": kw["binder_bonds"], "iters": iters, "i": i, "disulfide_stage": stage,
          "_scheduled_disulfide_params": de._scheduled_disulfide_params}
    for key in ("disulfide_schedule", "disulfide_hinge_cutoff", "disulfide_hinge_cutoff_start", "disulfide_hinge_cutoff_end",
                "disulfide_loss_soft", "disulfide_loss_temp", "disulfide_loss_hard", "disulfide_sds_scale"):
        ns[key] = kw[key]
    exec(SCHEDULE_SRC, ns)
    return ns["current_loss_scales"]


class World:
    """The real get_model_loss() on toy tensors; the three disulfide terms are spies that record that they were asked for."""

    N_B, N_T = 8, 3

    def __init__(self, **closure):
        n = self.N_B + self.N_T
        self.n = n
        self.calls = []
        self.model_kwargs = []
        chain_mask = torch.zeros(1, n)
        chain_mask[:, :self.N_B] = 1.0
        world = self

        class Model:
            def _out(self, full):
                out = {"pdistogram": torch.randn(1, n, n, 64, requires_grad=True)}
                if full:
                    out.update(plddt=torch.rand(1, n), pae=torch.rand(1, n, n) * 31, sample_atom_coords=torch.randn(1, 4 * n, 3))
                return out

            def get_distogram_confidence(self, batch, **kw):
                world.model_kwargs.append(kw)
                return self._out(True)

            def get_distogram(self, batch):
                return self._out(False), None, None, None

        def spy(name, result):
            def fn(*a, **k):
                world.calls.append(name)
                return result
            return fn

        scalar = lambda v: torch.tensor(v, requires_grad=True)
        ns = {
            "torch": torch, "random": random, "device": torch.device("cpu"), "chain_to_number": de.chain_to_number,
            "get_mid_points": de.get_mid_points, "get_con_loss": de.get_con_loss, "_get_helix_loss": de._get_helix_loss,
            "get_plddt_loss": de.get_plddt_loss, "get_pae_loss": de.get_pae_loss, "format_sg_distances": de.format_sg_distances,
            "_sds_profile_ts": de._sds_profile_ts, "sds_profile_report": spy("sds_profile_report", None),
            "mid_points": torch.linspace(2, 22, 64), "chain_mask": chain_mask, "length": self.N_B,
            "boltz_model": Model(), "binder_bonds": "2_SG_17_SG,6_SG_13_SG", "debug_grad_norm": False,
            "disconnect_feats": True, "disconnect_pairformer": False, "residue_index_base": 1, "disulfide_stage": "soft",
            "disulfide_grad_steps": 0, "disulfide_sds_scale": 0.0, "disulfide_sds_samples": 2, "disulfide_sds_sigma_lo": 1.5,
            "disulfide_sds_sigma_hi": 6.0, "disulfide_sds_clip": 2.0, "disulfide_sds_profile_every": 0,
            "disulfide_sds_profile_t": (1.5, 3.0), "disulfide_distogram_lo": 3.3, "disulfide_distogram_hi": 4.8,
            "disulfide_distogram_form": "nll", "disulfide_hinge_cutoff": 2.3, "disulfide_target_dist": 2.05,
            "disulfide_distance_mode": "bounded", "disulfide_lower_bound": 0.0, "disulfide_upper_bound": 2.8,
            "disulfide_center_weight": 0.0, "disulfide_angle_target": 103.0, "disulfide_angle_tolerance": 25.0,
            "disulfide_dihedral_target_abs": 90.0, "disulfide_dihedral_tolerance": 45.0,
            "add_rg_loss": spy("rg", (scalar(0.1), scalar(1.0))),
            "add_disulfide_distogram_loss": spy("distogram_term", (scalar(0.7), [0.5, 0.6])),
            "add_disulfide_distance_loss": spy("structure_terms", (scalar(1.0), scalar(1.5), scalar(2.0), scalar(3.0), [2.1, 2.2])),
            "disulfide_sds_losses": spy("sds_terms", (scalar(1.0), scalar(1.0), scalar(1.0), scalar(1.0))),
        }
        ns.update(closure)
        exec(MODEL_LOSS_SRC, ns)
        self.get_model_loss = ns["get_model_loss"]

    def run(self, scales, pre_run=False, distogram_only=False):
        n = self.n
        batch = {"res_type": torch.zeros(1, n, 33), "entity_id": torch.zeros(1, n, dtype=torch.long)}
        predict_args = {"recycling_steps": 1, "sampling_steps": 200, "diffusion_samples": 1}
        out = self.get_model_loss(batch, [], [], [], [], [], [], [], pre_run=pre_run, distogram_only=distogram_only,
                                  predict_args=predict_args, loss_scales=dict(scales))
        return out[0], out[8]                                                          # total loss, the list of "name:value" strings


def _weights_of(monkeypatch, tmp_path, capsys, *extra):
    calls, _ = _run_main(monkeypatch, tmp_path, capsys, *extra)
    calls["loss_scales"]["helix_loss"] = -0.1                                          # run_highss_design draws it per design
    return calls


@pytest.mark.parametrize("stage", STAGES)
def test_with_the_switch_off_no_epoch_has_a_disulfide_term(monkeypatch, tmp_path, capsys, stage):
    kw = _weights_of(monkeypatch, tmp_path, capsys, *OFF)
    scales = _epoch_weights(kw, stage)
    assert all(v == 0 for k, v in scales.items() if "disulfide" in k), scales
    world = World()
    total, loss_str = world.run(scales)
    assert world.calls == ["rg"], world.calls                                          # no distogram term, no structure term, no SDS
    assert not any("disulfide" in s or s.startswith("sg_sg") for s in loss_str), loss_str
    assert world.model_kwargs[0]["structure_grad_steps"] == 0 and world.model_kwargs[0]["structure_sds_samples"] == 0
    assert torch.isfinite(total)


def test_with_the_switch_off_the_pre_run_has_no_disulfide_term(monkeypatch, tmp_path, capsys):
    kw = _weights_of(monkeypatch, tmp_path, capsys, *OFF)
    world = World()
    _total, loss_str = world.run(_epoch_weights(kw, "soft"), pre_run=True)
    assert world.calls == [] and not any("disulfide" in s for s in loss_str), (world.calls, loss_str)


@pytest.mark.parametrize("stage", STAGES)
def test_zero_weights_keep_the_gradient_routes_closed_even_if_they_are_asked_for(monkeypatch, tmp_path, capsys, stage):
    # the engine's own second line of defence: a route is only opened for a term that has a weight
    kw = _weights_of(monkeypatch, tmp_path, capsys, *OFF)
    world = World(disulfide_grad_steps=3, disulfide_sds_scale=1.0)
    world.run(_epoch_weights(kw, stage))
    assert world.model_kwargs[0]["structure_grad_steps"] == 0 and world.model_kwargs[0]["structure_sds_samples"] == 0
    assert world.calls == ["rg"]


@pytest.mark.parametrize("stage", STAGES)
def test_with_the_switch_on_the_epoch_has_the_standard_terms(monkeypatch, tmp_path, capsys, stage):
    kw = _weights_of(monkeypatch, tmp_path, capsys)
    scales = _epoch_weights(kw, stage)
    assert scales["disulfide_loss"] == STRUCTURE_WEIGHTS[stage] and scales["disulfide_distogram_loss"] == 4.0
    world = World()
    _total, loss_str = world.run(scales)
    assert sorted(world.calls) == ["distogram_term", "rg", "structure_terms"], world.calls
    names = [s.split(":")[0] for s in loss_str]
    assert "disulfide_distogram_loss" in names and "disulfide_loss" in names and any(s.startswith("sg_sg") for s in loss_str)
    assert world.model_kwargs[0]["structure_grad_steps"] == 0 and world.model_kwargs[0]["structure_sds_samples"] == 0


@pytest.mark.parametrize("stage", STAGES)
def test_with_the_switch_on_the_geometry_route_opens_only_when_asked_for(monkeypatch, tmp_path, capsys, stage):
    kw = _weights_of(monkeypatch, tmp_path, capsys, "--disulfide_grad_steps", "1")
    world = World(disulfide_grad_steps=1)
    world.run(_epoch_weights(kw, stage))
    assert world.model_kwargs[0]["structure_grad_steps"] == 1 and world.model_kwargs[0]["structure_sds_samples"] == 0


def test_main_names_the_geometry_route_in_the_header_when_it_is_opened(monkeypatch, tmp_path, capsys):
    calls, out = _run_main(monkeypatch, tmp_path, capsys, "--disulfide_grad_steps", "1")
    line = [l for l in out.splitlines() if l.startswith("  loop guidance:")][0]
    assert calls["disulfide_grad_steps"] == 1 and "distogram term nll x 4; geometry backprop ON;" in line


@pytest.mark.parametrize("term", ["worst_pair", "angle", "dihedral"])
def test_a_structure_term_whose_own_weight_is_0_is_not_evaluated(monkeypatch, tmp_path, capsys, term):
    # the same rule one level down: --disulfide_<term>_loss 0 removes that term from the epoch, the others stay
    kw = _weights_of(monkeypatch, tmp_path, capsys, f"--disulfide_{term}_loss", "0")
    world = World()
    _total, loss_str = world.run(_epoch_weights(kw, "hard"))
    names = [s.split(":")[0] for s in loss_str]
    others = {f"disulfide_{t}_loss" for t in ("worst_pair", "angle", "dihedral")} - {f"disulfide_{term}_loss"}
    assert f"disulfide_{term}_loss" not in names and "disulfide_loss" in names and others <= set(names), names


def test_with_the_switch_on_the_pre_run_has_only_the_distogram_term(monkeypatch, tmp_path, capsys):
    kw = _weights_of(monkeypatch, tmp_path, capsys)
    world = World()
    _total, loss_str = world.run(_epoch_weights(kw, "soft"), pre_run=True)
    assert world.calls == ["distogram_term"] and [s.split(":")[0] for s in loss_str].count("disulfide_distogram_loss") == 1


def test_a_zero_base_weight_alone_does_not_switch_the_structure_terms_off(monkeypatch, tmp_path, capsys):
    # why the switch zeroes the stage weights as well (documented in PARAMETERS.md): with the schedule on they decide the weight
    kw = _weights_of(monkeypatch, tmp_path, capsys, "--disulfide_loss", "0", "--disulfide_distogram_loss", "0")
    for stage in STAGES:
        scales = _epoch_weights(kw, stage)
        assert scales["disulfide_loss"] == STRUCTURE_WEIGHTS[stage]
        world = World()
        world.run(scales)
        assert "structure_terms" in world.calls
    kw = _weights_of(monkeypatch, tmp_path, capsys, "--disulfide_loss", "0", "--disulfide_schedule", "false",
                     "--disulfide_distogram_loss", "0")
    assert _epoch_weights(kw, "hard")["disulfide_loss"] == 0


def test_the_engine_reads_no_disulfide_weight_that_the_switch_does_not_zero():
    read = set(re.findall(r"""\.get\(\s*['"](disulfide_\w+)['"]""", MODEL_LOSS_SRC + SCHEDULE_SRC))
    read |= set(re.findall(r"""loss_scales\[\s*['"](disulfide_\w+)['"]\s*\]""", MODEL_LOSS_SRC + SCHEDULE_SRC))
    derived = {"disulfide_sds_loss"}               # ss_weight x disulfide_sds_scale, built by the schedule block
    assert read - derived <= set(cli.LOSS_OFF) | {"disulfide_stage"}, read - derived - set(cli.LOSS_OFF)
    assert "disulfide_sds_loss" in read


def test_design_hands_the_scheduled_weights_to_get_model_loss():
    assert len(re.findall(r"predict_args, current_loss_scales, binder_chain", ENGINE_SRC)) == 1


def test_the_scheduled_weights_of_all_zero_options_are_zero():
    for stage in STAGES + ("none",):
        for frac in (0.0, 0.5, 1.0):
            weight, _cutoff = de._scheduled_disulfide_params(stage, frac, 0.0, True, 2.3, 3.5, 2.3, 0.0, 0.0, 0.0)
            assert weight == 0.0, (stage, frac)


# ------------------------------------------------------------------------------------------------ documentation
def test_the_documentation_names_the_switch_and_what_it_zeroes():
    params = (ROOT / "docs" / "PARAMETERS.md").read_text(encoding="utf-8")
    m = re.search(r"^### `--use_disulfide_loss`\n(.*?)(?=^#{2,3} )", params, re.S | re.M)
    assert m, "section of --use_disulfide_loss"
    section = " ".join(m.group(1).split())
    assert "- Default: `true`" in m.group(1)
    assert "does not switch the structure terms off" in section
    notes = (ROOT / "USAGE.md").read_text(encoding="utf-8")
    for key in cli.LOSS_OFF:
        if key.endswith("_loss") or re.search(r"_loss_(soft|temp|hard)$", key):
            assert f"`--{key}`" in params, key
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "`--use_disulfide_loss` (default `true`) is the master switch" in readme
    for text in (notes, readme, (ROOT / "docs" / "METHOD.md").read_text(encoding="utf-8")):
        assert "--use_disulfide_loss true" in text and "--use_disulfide_loss false" in text
