"""Input side of a run: pocket residue numbers, the binder chain ID, the checkpoint check and the files git tracks.

``--contact_residues`` takes the residue numbers of the PDB file; the pocket constraint of the input YAML counts positions along the
chain sequence. The generated YAML lists the binder first, so the binder is entity 0 whatever chain ID it is given.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from highss import cli
from highss import design_engine as de
from highss import input_utils as iu

ROOT = Path(__file__).resolve().parents[1]
MDM2 = ROOT / "examples" / "mdm2" / "mdm2_pro.pdb"      # chain A, residues 25-109


def _pdb_chains(tmp_path, residues):
    """A minimal PDB: one CA atom for each (chain, number, insertion code, residue name)."""
    lines = []
    for serial, (chain, number, icode, name) in enumerate(residues, start=1):
        lines.append(f"ATOM  {serial:5d}  CA  {name:>3s} {chain}{number:4d}{icode or ' '}   "
                     f"{0.0:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C")
    path = tmp_path / "toy.pdb"
    path.write_text("\n".join(lines) + "\nEND\n")
    return path


def _pdb(tmp_path, residues, chain="A"):
    """A minimal single-chain PDB from (number, insertion code, residue name)."""
    return _pdb_chains(tmp_path, [(chain, number, icode, name) for number, icode, name in residues])


# ---------------------------------------------------------------------------------------------- PDB numbers -> sequence positions
def test_the_residue_labels_follow_the_order_of_the_sequence():
    labels = iu.get_chain_residue_labels(MDM2)["A"]
    seq = iu.get_chains_sequence(MDM2)["A"]
    assert len(labels) == len(seq) == 85 and labels[0] == "25" and labels[-1] == "109"
    assert [seq[labels.index(n)] for n in ("54", "58", "61", "67")] == list("LGIY")      # Leu54, Gly58, Ile61, Tyr67


def test_pdb_numbers_become_positions_in_the_chain_sequence():
    labels = iu.get_chain_residue_labels(MDM2)["A"]
    seq = iu.get_chains_sequence(MDM2)["A"]
    positions = iu.resolve_contact_residues("54, 58,61,67", labels, "A")
    assert positions == [30, 34, 37, 43]
    assert [seq[p - 1] for p in positions] == list("LGIY")


def test_a_chain_numbered_from_1_without_gaps_keeps_its_numbers(tmp_path):
    pdb = _pdb(tmp_path, [(n, "", "ALA") for n in range(1, 21)])
    assert iu.resolve_contact_residues("3,7,20", iu.get_chain_residue_labels(pdb)["A"], "A") == [3, 7, 20]


def test_gaps_insertion_codes_and_negative_numbers(tmp_path):
    pdb = _pdb(tmp_path, [(-2, "", "GLY"), (-1, "", "ALA"), (1, "", "SER"), (2, "", "THR"), (5, "", "VAL"), (6, "", "LEU"),
                          (6, "A", "ILE"), (7, "", "MET")])
    labels = iu.get_chain_residue_labels(pdb)["A"]
    assert labels == ["-2", "-1", "1", "2", "5", "6", "6A", "7"]
    assert iu.get_chains_sequence(pdb)["A"] == "GASTVLIM"
    assert iu.resolve_contact_residues("-1,5,6A,7", labels, "A") == [2, 5, 7, 8]
    assert iu.resolve_contact_residues("05, 6a", labels, "A") == [5, 7]          # leading zero, lower-case insertion code


def test_each_chain_has_its_own_numbering(tmp_path):
    rows = [f"ATOM  {i:5d}  CA  ALA {c}{n:4d}    {0.0:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C"
            for i, (c, n) in enumerate([("A", 10), ("A", 11), ("B", 10), ("B", 11), ("B", 12)], start=1)]
    pdb = tmp_path / "two.pdb"
    pdb.write_text("\n".join(rows) + "\n")
    labels = iu.get_chain_residue_labels(pdb)
    assert iu.resolve_contact_residues("12", labels["B"], "B") == [3]
    with pytest.raises(ValueError, match="not a standard residue of chain A"):
        iu.resolve_contact_residues("12", labels["A"], "A")


@pytest.mark.parametrize("spec, message", [
    ("24", r"Contact residue 24 is not a standard residue of chain A in the PDB \(residues 25 to 109\)"),
    ("110", "not a standard residue of chain A"),
    ("54B", "not a standard residue of chain A"),
    ("5x4", "Invalid contact residue '5x4'"),
    ("A:54", "Invalid contact residue 'A:54'"),
    (",", "lists no residue numbers"),
])
def test_a_residue_that_is_not_in_the_chain_is_refused(spec, message):
    with pytest.raises(ValueError, match=message):
        iu.resolve_contact_residues(spec, iu.get_chain_residue_labels(MDM2)["A"], "A")


def test_the_mdm2_target_sequence_is_read_as_before():
    seqs = iu.get_chains_sequence(MDM2)
    assert list(seqs) == ["A"] and seqs["A"].startswith("ETLVRPKPLLLKLL") and seqs["A"].endswith("YRNLVV")


# ----------------------------------------------------------------------------------------------- main() with the engine stubbed
def _run_main(monkeypatch, tmp_path, capsys, *extra, source=MDM2, chains="A", confidence=None):
    import highss.design_engine as engine
    pdb = tmp_path / "t.pdb"
    shutil.copy(source, pdb)
    (tmp_path / "m.ckpt").write_bytes(b"x")
    (tmp_path / "ccd.pkl").write_bytes(b"x")
    calls = {}

    class FakeModel:
        def train(self):
            return self

    if confidence is not None:
        FakeModel.confidence_prediction = confidence       # what a loaded Boltz1 reports

    monkeypatch.setattr(engine, "get_boltz_model", lambda *a, **k: FakeModel())
    monkeypatch.setattr(engine, "run_highss_design", lambda **kw: calls.update(kw))
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/boltz")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setattr(sys, "argv", ["highss", "--target_pdb", str(pdb), "--target_chains", chains, "--binder_length", "18",
                                      "--disulfide_pairs", "2-17,6-13", "--checkpoint", str(tmp_path / "m.ckpt"),
                                      "--ccd_path", str(tmp_path / "ccd.pkl"), "--work_dir", str(tmp_path / "run"),
                                      "--msa_mode", "none", *extra])
    cli.main()
    return calls, capsys.readouterr().out


def _yaml_of(calls):
    return yaml.safe_load((Path(calls["yaml_dir"]) / "t.yaml").read_text())


def test_main_writes_the_pocket_constraint_with_sequence_positions(monkeypatch, tmp_path, capsys):
    calls, out = _run_main(monkeypatch, tmp_path, capsys, "--contact_residues", "54,58,61,67", "--contact_target_chain", "A")
    data = _yaml_of(calls)
    pocket = [c["pocket"] for c in data["constraints"] if "pocket" in c]
    assert len(pocket) == 1 and pocket[0]["binder"] == "A"
    assert pocket[0]["contacts"] == [["B", 30], ["B", 34], ["B", 37], ["B", 43]]
    target = data["sequences"][1]["protein"]
    assert target["id"] == ["B"] and [target["sequence"][p - 1] for _, p in pocket[0]["contacts"]] == list("LGIY")
    assert calls["config"]["pocket_conditioning"] is True
    line = [l for l in out.splitlines() if l.startswith("  contact residues:")]
    assert line and "chain A PDB 54,58,61,67 -> sequence positions 30,34,37,43" in line[0]


def test_main_counts_from_the_start_of_the_chosen_chain_of_a_multi_chain_target(monkeypatch, tmp_path, capsys):
    rows = [("A", n, "", name) for n, name in [(10, "ALA"), (11, "GLY"), (12, "SER"), (13, "THR"), (14, "VAL")]]
    rows += [("B", n, "", name) for n, name in [(100, "LEU"), (101, "ILE"), (103, "MET"), (104, "PHE"), (105, "TRP")]]   # no residue 102
    source = _pdb_chains(tmp_path, rows)
    calls, out = _run_main(monkeypatch, tmp_path, capsys, "--contact_residues", "103,105", "--contact_target_chain", "B",
                           source=source, chains="A,B")
    data = _yaml_of(calls)
    targets = {entry["protein"]["id"][0]: entry["protein"]["sequence"] for entry in data["sequences"][1:]}
    assert targets == {"B": "AGSTV", "C": "LIMFW"}                  # PDB chains A and B are written as chains B and C of the YAML
    pocket = [c["pocket"] for c in data["constraints"] if "pocket" in c]
    assert pocket[0]["contacts"] == [["C", 3], ["C", 5]]
    assert [targets["C"][p - 1] for _, p in pocket[0]["contacts"]] == ["M", "W"]
    assert "chain B PDB 103,105 -> sequence positions 3,5" in out
    with pytest.raises(ValueError, match="Contact residue 103 is not a standard residue of chain A"):
        _run_main(monkeypatch, tmp_path, capsys, "--contact_residues", "103", "--contact_target_chain", "A",
                  source=source, chains="A,B")


def test_main_needs_no_contact_chain_for_a_single_chain_target(monkeypatch, tmp_path, capsys):
    calls, _ = _run_main(monkeypatch, tmp_path, capsys, "--contact_residues", "54")
    assert [c["pocket"]["contacts"] for c in _yaml_of(calls)["constraints"] if "pocket" in c] == [[["B", 30]]]


def test_main_without_contact_residues_writes_no_pocket_and_no_header_line(monkeypatch, tmp_path, capsys):
    calls, out = _run_main(monkeypatch, tmp_path, capsys)
    assert not any("pocket" in c for c in _yaml_of(calls)["constraints"])
    assert calls["config"]["pocket_conditioning"] is False and "contact residues" not in out


def test_a_missing_residue_stops_main_before_anything_is_written(monkeypatch, tmp_path, capsys):
    with pytest.raises(ValueError, match="Contact residue 24 is not a standard residue of chain A"):
        _run_main(monkeypatch, tmp_path, capsys, "--contact_residues", "24,54")
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("binder_chain", ["A", "B", "C", "Z"])
def test_the_binder_is_entity_0_whatever_its_chain_id(monkeypatch, tmp_path, capsys, binder_chain):
    calls, out = _run_main(monkeypatch, tmp_path, capsys, "--binder_chain", binder_chain)
    data = _yaml_of(calls)
    # the YAML lists the binder first and Boltz numbers the entities in YAML order
    first = data["sequences"][0]["protein"]
    assert first["id"] == [binder_chain] and set(first["sequence"]) == {"X", "C"}
    assert all(entry["protein"]["id"] != [binder_chain] for entry in data["sequences"][1:])
    # ... so the engine has to be asked for entity 0 and not for the table entry of the letter
    assert de.chain_to_number[calls["config"]["binder_chain"]] == 0
    bonds = [c["bond"] for c in data["constraints"] if "bond" in c]
    assert len(bonds) == 2 and all(b["atom1"][0] == binder_chain == b["atom2"][0] for b in bonds)
    assert f"  binder chain:     {binder_chain}" in out


def test_the_engine_is_never_asked_for_the_target_entity():
    assert cli.ENGINE_BINDER_KEY == "A" and de.chain_to_number[cli.ENGINE_BINDER_KEY] == 0
    assert de.chain_to_number["B"] == 1       # the table entry of the first target chain when the binder is chain A


# ------------------------------------------------------------------------------------------------------------- the checkpoint
def test_a_model_without_the_confidence_head_stops_the_run_before_the_design(monkeypatch, tmp_path, capsys):
    with pytest.raises(ValueError, match="no confidence head.*com-FT.ckpt"):
        _run_main(monkeypatch, tmp_path, capsys, confidence=False)
    assert not (tmp_path / "run" / "outputs").exists()


def test_a_model_with_the_confidence_head_runs(monkeypatch, tmp_path, capsys):
    calls, _ = _run_main(monkeypatch, tmp_path, capsys, confidence=True)
    assert calls["config"]["binder_chain"] == cli.ENGINE_BINDER_KEY and (tmp_path / "run" / "outputs").is_dir()


# ----------------------------------------------------------------------------------------------------------- what git tracks
def test_the_training_output_readme_is_not_swallowed_by_the_ignore_rules(tmp_path):
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is not installed")
    (tmp_path / "training" / "outputs").mkdir(parents=True)
    shutil.copy(ROOT / ".gitignore", tmp_path / ".gitignore")
    shutil.copy(ROOT / "training" / ".gitignore", tmp_path / "training" / ".gitignore")
    (tmp_path / "training" / "outputs" / "README.md").write_text("x\n")
    (tmp_path / "training" / "outputs" / "model.ckpt").write_text("x\n")
    (tmp_path / "outputs" / "run").mkdir(parents=True)
    (tmp_path / "outputs" / "run" / "a.txt").write_text("x\n")
    subprocess.run([git, "init", "-q"], cwd=tmp_path, check=True)

    def ignored(path):
        return subprocess.run([git, "check-ignore", "-q", path], cwd=tmp_path).returncode == 0

    assert not ignored("training/outputs/README.md")
    assert ignored("training/outputs/model.ckpt") and ignored("outputs/run/a.txt")
