from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable

import yaml
from boltz.data.msa.mmseqs2 import run_mmseqs2
from boltz.data.parse.a3m import parse_a3m

logger = logging.getLogger(__name__)


class InputConfig:
    """Input directories for a HighSS run."""

    def __init__(self, main_dir: str | Path):
        self.MAIN_DIR = Path(main_dir)
        self.PDB_DIR = self.MAIN_DIR / "PDB"
        self.MSA_DIR = self.MAIN_DIR / "MSA"
        self.YAML_DIR = self.MAIN_DIR / "yaml"

    def setup_directories(self) -> None:
        for directory in (self.MAIN_DIR, self.PDB_DIR, self.MSA_DIR, self.YAML_DIR):
            directory.mkdir(parents=True, exist_ok=True)


_AA3_TO_AA1 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}


def _residue_label(number: str, insertion_code: str = "") -> str:
    """PDB residue label such as ``54``, ``-3`` or ``52A``."""
    try:
        number = str(int(number))
    except ValueError:
        number = number.upper()
    return number + insertion_code.upper()


def _read_chain_residues(pdb_path: str | Path) -> dict[str, list[tuple[str, str]]]:
    """Read the standard amino-acid residues of every chain of a PDB file.

    Returns, for each chain ID, ``(label, one-letter code)`` pairs in file order. The label is the residue number of the PDB file
    followed by its insertion code. Only ATOM records of the 20 standard amino acids are read, and each residue is listed once.
    """
    residues: dict[str, list[tuple[str, str]]] = {}
    seen: dict[str, set[tuple[str, str]]] = {}
    with Path(pdb_path).open() as handle:
        for line in handle:
            if not line.startswith("ATOM") or len(line) < 27:
                continue
            chain = line[21].strip() or "_"
            residue = line[17:20].strip().upper()
            residue_number = line[22:26].strip()
            insertion_code = line[26].strip()
            aa = _AA3_TO_AA1.get(residue)
            if aa is None:
                continue
            key = (residue_number, insertion_code)
            residues.setdefault(chain, [])
            seen.setdefault(chain, set())
            if key not in seen[chain]:
                seen[chain].add(key)
                residues[chain].append((_residue_label(residue_number, insertion_code), aa))
    return residues


def get_chains_sequence(pdb_path: str | Path) -> dict[str, str]:
    """Extract standard amino-acid sequences from a PDB file."""
    return {chain: "".join(aa for _, aa in residues) for chain, residues in _read_chain_residues(pdb_path).items()}


def get_chain_residue_labels(pdb_path: str | Path) -> dict[str, list[str]]:
    """PDB residue labels (``54``, ``52A``) of each chain, in the order of the sequences from :func:`get_chains_sequence`."""
    return {chain: [label for label, _ in residues] for chain, residues in _read_chain_residues(pdb_path).items()}


_RESIDUE_TOKEN = re.compile(r"^(-?\d+)([A-Za-z]?)$")


def resolve_contact_residues(spec: str, labels: list[str], chain: str = "") -> list[int]:
    """Convert PDB residue numbers of one chain into 1-based positions in the chain sequence.

    ``spec`` is the comma-separated ``--contact_residues`` text with residue numbers as they appear in the PDB file; an insertion
    code may follow the number (``52A``). ``labels`` are the residue labels of that chain from :func:`get_chain_residue_labels`.
    The pocket constraint of the input YAML counts residues from 1 along the chain sequence that HighSS passes to the predictor,
    so the position of a residue is its place in that sequence.
    """
    where = {label: position for position, label in enumerate(labels, start=1)}
    positions: list[int] = []
    for raw in spec.split(","):
        token = raw.strip()
        if not token:
            continue
        match = _RESIDUE_TOKEN.match(token)
        if match is None:
            raise ValueError(f"Invalid contact residue '{token}'. Use PDB residue numbers such as 54 or 52A.")
        label = _residue_label(match.group(1), match.group(2))
        if label not in where:
            span = f"{labels[0]} to {labels[-1]}" if labels else "none"
            raise ValueError(
                f"Contact residue {token} is not a standard residue of chain {chain or '?'} in the PDB (residues {span})."
            )
        positions.append(where[label])
    if not positions:
        raise ValueError("--contact_residues lists no residue numbers.")
    return positions


def parse_disulfide_pairs(spec: str, binder_length: int) -> list[tuple[int, int]]:
    """Parse 1-based pairs such as ``2-17,6-13`` and validate the topology."""
    if not spec.strip():
        raise ValueError("At least one disulfide pair is required, e.g. 2-17.")
    pairs: list[tuple[int, int]] = []
    used: set[int] = set()
    canonical: set[tuple[int, int]] = set()
    for raw in spec.split(","):
        fields = raw.strip().replace(":", "-").split("-")
        if len(fields) != 2:
            raise ValueError(f"Invalid disulfide pair '{raw}'. Use 1-based POS-POS format.")
        try:
            left, right = (int(fields[0]), int(fields[1]))
        except ValueError as exc:
            raise ValueError(f"Invalid disulfide pair '{raw}'. Positions must be integers.") from exc
        if left == right:
            raise ValueError(f"Invalid disulfide pair '{raw}': both positions are identical.")
        if not (1 <= left <= binder_length and 1 <= right <= binder_length):
            raise ValueError(
                f"Disulfide pair '{raw}' is outside binder length {binder_length}."
            )
        pair = tuple(sorted((left, right)))
        if pair in canonical:
            raise ValueError(f"Duplicate disulfide pair '{raw}'.")
        overlap = {left, right} & used
        if overlap:
            raise ValueError(
                f"Cys position(s) {sorted(overlap)} occur in more than one disulfide pair."
            )
        canonical.add(pair)
        used.update((left, right))
        pairs.append((left, right))
    return pairs


def disulfide_cli_fields(pairs: Iterable[tuple[int, int]]) -> tuple[str, str]:
    """Return internal SG-SG bond and fixed-Cys strings using 1-based indices."""
    pairs = list(pairs)
    bonds = ",".join(f"{left}_SG_{right}_SG" for left, right in pairs)
    positions = sorted({position for pair in pairs for position in pair})
    fixed = ",".join(f"{position}_C" for position in positions)
    return bonds, fixed


def build_initial_binder_sequence(length: int, pairs: Iterable[tuple[int, int]]) -> str:
    """Build an X sequence with cysteines pre-positioned for SG atom parsing."""
    seq = ["X"] * length
    for left, right in pairs:
        seq[left - 1] = "C"
        seq[right - 1] = "C"
    return "".join(seq)


def assign_target_chain_ids(count: int, binder_chain: str) -> list[str]:
    available = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c != binder_chain]
    if count > len(available):
        raise ValueError("Too many target chains for one-letter chain identifiers.")
    return available[:count]


def build_constraints(
    binder_chain: str,
    pairs: Iterable[tuple[int, int]],
    target_output_chain: str | None = None,
    contact_residues: str = "",
) -> list[dict]:
    constraints: list[dict] = []
    if contact_residues:
        if not target_output_chain:
            raise ValueError("A contact target chain is required when contact residues are used.")
        residues = [int(x.strip()) for x in contact_residues.split(",") if x.strip()]
        constraints.append({
            "pocket": {
                "binder": binder_chain,
                "contacts": [[target_output_chain, residue] for residue in residues],
            }
        })
    for left, right in pairs:
        constraints.append({
            "bond": {
                "atom1": [binder_chain, left, "SG"],
                "atom2": [binder_chain, right, "SG"],
            }
        })
    return constraints


def process_msa(
    chain_id: str,
    sequence: str,
    target_name: str,
    config: InputConfig,
    max_seqs: int,
) -> Path:
    """Generate and parse an unpaired target MSA through the ColabFold server."""
    msa_chain_dir = config.MSA_DIR / f"{target_name}_{chain_id}"
    env_dir = msa_chain_dir.with_name(f"{msa_chain_dir.name}_env")
    env_dir.mkdir(parents=True, exist_ok=True)
    msa_a3m_path = env_dir / "msa.a3m"
    msa_npz_path = env_dir / "msa.npz"

    if not msa_a3m_path.exists():
        unpaired_msa = run_mmseqs2(
            [sequence],
            str(msa_chain_dir),
            use_env=True,
            use_pairing=False,
            host_url="https://api.colabfold.com",
            pairing_strategy="greedy",
        )
        msa_a3m_path.write_text(unpaired_msa[0])

    if not msa_npz_path.exists():
        msa = parse_a3m(msa_a3m_path, taxonomy=None, max_seqs=max_seqs)
        msa.dump(msa_npz_path)
    logger.info("Processed MSA for %s chain %s", target_name, chain_id)
    return msa_npz_path


def generate_protein_binder_yaml(
    *,
    target_name: str,
    target_sequences: list[str],
    binder_chain: str,
    binder_length: int,
    disulfide_pairs: list[tuple[int, int]],
    config: InputConfig,
    msa_mode: str,
    msa_max_seqs: int,
    contact_target_index: int | None = None,
    contact_residues: str = "",
) -> tuple[dict, Path, list[str]]:
    """Write a HighSS protein-target YAML with an empty binder MSA.

    The binder is always the first sequence of the YAML. ``contact_residues`` are 1-based positions in the sequence of the
    contact target chain; the command line converts PDB residue numbers to these positions first.
    """
    target_output_chains = assign_target_chain_ids(len(target_sequences), binder_chain)
    sequences = [{
        "protein": {
            "id": [binder_chain],
            "sequence": build_initial_binder_sequence(binder_length, disulfide_pairs),
            "msa": "empty",
        }
    }]

    for chain_id, sequence in zip(target_output_chains, target_sequences):
        msa: str = "empty"
        if msa_mode == "target":
            msa = str(process_msa(chain_id, sequence, target_name, config, msa_max_seqs))
        sequences.append({
            "protein": {
                "id": [chain_id],
                "sequence": sequence,
                "msa": msa,
            }
        })

    contact_chain = None
    if contact_target_index is not None:
        contact_chain = target_output_chains[contact_target_index]
    constraints = build_constraints(
        binder_chain,
        disulfide_pairs,
        target_output_chain=contact_chain,
        contact_residues=contact_residues,
    )
    content = {"version": 1, "sequences": sequences, "constraints": constraints}
    output_path = config.YAML_DIR / f"{target_name}.yaml"
    output_path.write_text(yaml.safe_dump(content, sort_keys=False))
    return content, output_path, target_output_chains
