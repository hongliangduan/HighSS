"""Tests for the command-line defaults and the optional input/loss preset.

The default objective backpropagates the distogram term; the coordinate-based geometry terms are logged. Explicit options override the preset.
The master switch is enabled by default; its disabled behavior is tested in test_switch.py.
"""
import inspect
import re
import shutil
import sys
from pathlib import Path

import pytest

import highss
from highss import cli
from highss import design_engine as de

ROOT = Path(__file__).resolve().parents[1]
VERSION = "0.2.1"

BASE = ["highss", "--target_pdb", "t.pdb", "--target_chains", "A", "--binder_length", "18",
        "--disulfide_pairs", "2-17,6-13", "--checkpoint", "m.ckpt", "--ccd_path", "ccd.pkl"]

# Snapshot of the optional preset (the required options are supplied by BASE).
RC3_SNAPSHOT = {
    'binder_chain': 'A',
    'binder_length': 18,
    'ccd_path': 'ccd.pkl',
    'checkpoint': 'm.ckpt',
    'con_loss': 1.0,
    'contact_residues': '',
    'contact_target_chain': '',
    'debug_grad_norm': False,
    'debug_sds_profile_every': 0,
    'debug_sds_profile_t': '1.5,3,6,12,24,48',
    'design_algorithm': '3stages',
    'design_samples': 1,
    'distogram_only': False,
    'disulfide_angle_loss': 0.0005,
    'disulfide_angle_target': 103.0,
    'disulfide_angle_tolerance': 25.0,
    'disulfide_center_weight': 0.0,
    'disulfide_dihedral_loss': 0.0002,
    'disulfide_dihedral_target_abs': 90.0,
    'disulfide_dihedral_tolerance': 45.0,
    'disulfide_distance_mode': 'bounded',
    'disulfide_distogram_form': 'prob',
    'disulfide_distogram_hi': 4.8,
    'disulfide_distogram_lo': 3.3,
    'disulfide_distogram_loss': 0.0,
    'disulfide_filter_cutoff': 2.8,
    'disulfide_grad_steps': 0,
    'disulfide_hinge_cutoff': 2.3,
    'disulfide_hinge_cutoff_end': 2.3,
    'disulfide_hinge_cutoff_start': 3.5,
    'disulfide_loss': 0.008,
    'disulfide_loss_hard': 0.015,
    'disulfide_loss_soft': 0.004,
    'disulfide_loss_temp': 0.008,
    'disulfide_lower_bound': 0.0,
    'disulfide_pairs': '2-17,6-13',
    'disulfide_schedule': True,
    'disulfide_sds_clip': 2.0,
    'disulfide_sds_samples': 2,
    'disulfide_sds_scale': 0.0,
    'disulfide_sds_sigma_hi': 6.0,
    'disulfide_sds_sigma_lo': 1.5,
    'disulfide_target_dist': 2.05,
    'disulfide_upper_bound': 2.8,
    'disulfide_worst_pair_loss': 0.5,
    'e_soft': 0.8,
    'e_soft_1': 0.8,
    'e_soft_2': 1.0,
    'gpu_id': 0,
    'hard_iteration': 5,
    'helix_loss_max': 0.0,
    'helix_loss_min': -0.3,
    'i_con_loss': 1.0,
    'i_pae_loss': 0.1,
    'inter_chain_cutoff': 20.0,
    'intra_chain_cutoff': 14.0,
    'learning_rate': 0.1,
    'learning_rate_pre': 1.0,
    'loop_drop_bonds': False,
    'loop_fix_cys': False,
    'mask_ligand': False,
    'msa_max_seqs': 4096,
    'msa_mode': 'target',
    'num_inter_contacts': 2,
    'num_intra_contacts': 2,
    'optimize_contact_per_binder_pos': True,
    'optimizer_type': 'SGD',
    'pae_loss': 0.4,
    'plddt_loss': 0.1,
    'pre_drop_bonds': False,
    'pre_iteration': 30,
    'recycling_steps': 1,
    'redo_boltz_predict': True,
    'rg_loss': 0.0,
    'save_trajectory': False,
    'seed': None,
    'semi_greedy_steps': 2,
    'set_train': True,
    'show_animation': False,
    'soft_iteration': 80,
    'suffix': 'run',
    'target_chains': 'A',
    'target_name': '',
    'target_pdb': 't.pdb',
    'temp_iteration': 45,
    'work_dir': '.',
}

CHANGED = {"loop_fix_cys": True, "pre_drop_bonds": True, "disulfide_distogram_loss": 4.0, "disulfide_distogram_form": "nll",
           "loop_profile": "token"}
RESTORED = {"loop_fix_cys": False, "pre_drop_bonds": False, "disulfide_distogram_loss": 0.0, "disulfide_distogram_form": "prob",
            "loop_profile": "mean"}
NEW_DEFAULTS = {**RC3_SNAPSHOT, **CHANGED, "rc3_defaults": False, "use_disulfide_loss": True}
RC3_RUN = {**RC3_SNAPSHOT, **RESTORED, "rc3_defaults": True, "use_disulfide_loss": True}


def _text(value):
    return str(value).lower() if isinstance(value, bool) else str(value)


def parse(monkeypatch, *extra):
    monkeypatch.setattr(sys, "argv", BASE + list(extra))
    return vars(cli.parse_arguments())


# ------------------------------------------------------------------------------------------------ the defaults
def test_the_defaults_enable_the_standard_objective(monkeypatch):
    assert parse(monkeypatch) == NEW_DEFAULTS


def test_rc3_defaults_gives_exactly_the_old_values_back(monkeypatch):
    got = parse(monkeypatch, "--rc3_defaults", "true")
    assert got == RC3_RUN
    # Every snapshot value remains available through the optional preset.
    assert {k: v for k, v in got.items() if k in RC3_SNAPSHOT} == RC3_SNAPSHOT


def test_the_table_of_restored_values_is_the_one_in_the_code():
    assert cli.RC3_DEFAULTS == RESTORED


@pytest.mark.parametrize("value", ["false", "no", "0"])
def test_rc3_defaults_false_changes_nothing(monkeypatch, value):
    assert parse(monkeypatch, "--rc3_defaults", value) == NEW_DEFAULTS


@pytest.mark.parametrize("name", sorted(RESTORED))
def test_an_explicit_option_wins_over_rc3_defaults(monkeypatch, name):
    got = parse(monkeypatch, "--rc3_defaults", "true", "--" + name, _text(CHANGED[name]))
    assert got == {**RC3_RUN, name: CHANGED[name]}


@pytest.mark.parametrize("name", sorted(RESTORED))
def test_an_explicit_rc3_value_works_without_the_switch(monkeypatch, name):
    got = parse(monkeypatch, "--" + name, _text(RESTORED[name]))
    assert got == {**NEW_DEFAULTS, name: RESTORED[name]}


def test_the_position_of_the_switch_does_not_matter(monkeypatch):
    first = parse(monkeypatch, "--rc3_defaults", "true", "--seed", "3", "--disulfide_distogram_loss", "2")
    monkeypatch.setattr(sys, "argv", ["highss", "--rc3_defaults=true"] + BASE[1:] + ["--seed", "3", "--disulfide_distogram_loss=2"])
    second = vars(cli.parse_arguments())
    assert first == second == {**RC3_RUN, "seed": 3, "disulfide_distogram_loss": 2.0}


def test_a_bad_value_is_refused(monkeypatch):
    monkeypatch.setattr(sys, "argv", BASE + ["--rc3_defaults", "maybe"])
    with pytest.raises(SystemExit):
        cli.parse_arguments()


def test_the_required_options_are_still_required(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["highss", "--rc3_defaults", "true"])
    with pytest.raises(SystemExit):
        cli.parse_arguments()


def test_the_config_reaches_the_engine_signature(monkeypatch):
    params = inspect.signature(de.boltz_hallucination).parameters
    run_params = inspect.signature(de.run_highss_design).parameters
    monkeypatch.setattr(sys, "argv", BASE)
    cfg = cli.load_config(cli.parse_arguments())
    for key in ("loop_fix_cys", "pre_drop_bonds", "loop_profile"):
        assert key in cfg and key in params
    assert cfg["loop_fix_cys"] is True and cfg["pre_drop_bonds"] is True and cfg["loop_profile"] == "token"
    for key in ("disulfide_distogram_lo", "disulfide_distogram_hi", "disulfide_distogram_form"):
        assert key in run_params


# ------------------------------------------------------------------------------------------------ --help
def _help(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["highss", "--help"])
    with pytest.raises(SystemExit) as exc:
        cli.parse_arguments()
    assert exc.value.code == 0
    return re.sub(r"\s+", " ", capsys.readouterr().out)


def _entry(text, name):
    """(help text, default) of the entry '--name METAVAR help (default: x)' of the option list. METAVAR is an upper-case name or
    {choices}, so a mention of the option inside another option's help text ('--rc3_defaults true') is not an entry."""
    m = re.search(rf"(?<=\s)--{name} (?:[A-Z][A-Z0-9_]*|\{{[^}}]*\}}) (.*?)\(default: ([^)]*)\)", text)
    assert m, name
    return m.group(1), m.group(2)


def test_help_shows_the_new_defaults(monkeypatch, capsys):
    text = _help(monkeypatch, capsys)
    for name, value in CHANGED.items():
        assert _entry(text, name)[1] == str(value), name
    assert _entry(text, "rc3_defaults")[1] == "False"
    assert _entry(text, "use_disulfide_loss")[1] == "True"


def test_help_has_no_stale_claims_about_the_defaults(monkeypatch, capsys):
    text = _help(monkeypatch, capsys)
    for name in list(CHANGED) + ["rc3_defaults", "use_disulfide_loss"]:
        help_text = _entry(text, name)[0]
        assert not re.search(r"default[s]? (is |= )?(false|off|0|prob|mean)\b|\(default,", help_text, re.I), (name, help_text)
        assert not re.search(r"\bDefault (false|true)\b", help_text), (name, help_text)
    # the switch names the options and values it restores
    help_text = _entry(text, "rc3_defaults")[0]
    for name, value in RESTORED.items():
        assert f"--{name} {_text(value)}" in help_text, name


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


def test_main_hands_the_recipe_to_the_engine(monkeypatch, tmp_path, capsys):
    calls, out = _run_main(monkeypatch, tmp_path, capsys)
    assert calls["loss_scales"]["disulfide_distogram_loss"] == 4.0
    assert calls["disulfide_distogram_form"] == "nll"
    cfg = calls["config"]
    assert cfg["loop_fix_cys"] is True and cfg["pre_drop_bonds"] is True and cfg["loop_drop_bonds"] is False
    assert cfg["loop_profile"] == "token"
    assert calls["disulfide_grad_steps"] == 0 and calls["disulfide_sds_scale"] == 0.0
    assert calls["binder_bonds"] == "2_SG_17_SG,6_SG_13_SG" and calls["fix_residues"] == "2_C,6_C,13_C,17_C"
    line = [l for l in out.splitlines() if l.startswith("  loop guidance:")]
    assert len(line) == 1
    assert "distogram term nll x 4;" in line[0] and "loop_fix_cys True" in line[0] and "pre_drop_bonds True" in line[0]
    assert "loop_profile token" in line[0] and "rc3_defaults" not in line[0] and "geometry backprop" not in line[0]


def test_main_with_rc3_defaults_hands_the_old_values_to_the_engine(monkeypatch, tmp_path, capsys):
    calls, out = _run_main(monkeypatch, tmp_path, capsys, "--rc3_defaults", "true")
    assert calls["loss_scales"]["disulfide_distogram_loss"] == 0.0
    assert calls["disulfide_distogram_form"] == "prob"
    cfg = calls["config"]
    assert cfg["loop_fix_cys"] is False and cfg["pre_drop_bonds"] is False and cfg["loop_profile"] == "mean"
    line = [l for l in out.splitlines() if l.startswith("  loop guidance:")][0]
    assert "distogram term prob x 0;" in line and "loop_fix_cys False" in line and "(--rc3_defaults:" in line


def test_main_with_an_explicit_option_next_to_rc3_defaults(monkeypatch, tmp_path, capsys):
    calls, _ = _run_main(monkeypatch, tmp_path, capsys, "--rc3_defaults", "true", "--disulfide_distogram_loss", "2",
                         "--loop_fix_cys", "true")
    assert calls["loss_scales"]["disulfide_distogram_loss"] == 2.0 and calls["config"]["loop_fix_cys"] is True
    assert calls["disulfide_distogram_form"] == "prob" and calls["config"]["pre_drop_bonds"] is False


# ------------------------------------------------------------------------------------------------ version and documentation
def test_the_version_is_the_same_everywhere():
    assert (ROOT / "VERSION").read_text(encoding="utf-8").strip() == VERSION == highss.__version__
    assert re.search(rf'^version = "{re.escape(VERSION)}"$', (ROOT / "pyproject.toml").read_text(encoding="utf-8"), re.M)


def _section(text, name):
    m = re.search(rf"^###[^\n]*`--{name}`[^\n]*\n(.*?)(?=^#{{2,3}} )", text, re.S | re.M)
    return m.group(1) if m else None


def test_the_parameter_reference_matches_the_defaults(monkeypatch):
    text = (ROOT / "docs" / "PARAMETERS.md").read_text(encoding="utf-8")
    got = parse(monkeypatch)
    names = ["msa_mode", "disulfide_distogram_loss", "disulfide_distogram_form", "disulfide_distogram_lo", "disulfide_distogram_hi",
             "loop_fix_cys", "pre_drop_bonds", "loop_profile", "use_disulfide_loss"]
    for name in names:
        section = _section(text, name)
        assert section is not None, name
        default_line = re.search(r"^- Default: (.+)$", section, re.M)
        assert default_line, name
        assert f"`{_text(got[name])}`" in default_line.group(1), (name, default_line.group(1), got[name])
    assert "loop_profile" in text


def test_the_usage_and_readme_name_the_switch_and_the_options():
    notes = (ROOT / "USAGE.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    method = (ROOT / "docs" / "METHOD.md").read_text(encoding="utf-8")
    params = (ROOT / "docs" / "PARAMETERS.md").read_text(encoding="utf-8")
    for text in (notes, readme, method):
        assert "--use_disulfide_loss true" in text and "--use_disulfide_loss false" in text
        assert not re.search(r"\bv0\.[012]\.", text)
    for name in CHANGED:
        assert f"`--{name}`" in params
    assert "backpropagation" in readme and "backpropagates" in method
    assert method.index("## Geometry objective") < method.index("## Distogram pairing term") < method.index("## Final screening")
    for text in (notes, params):
        assert not re.search(r"@@[A-Z_]+@@", text) and not re.search(r"\{[A-Za-z_']", text)
