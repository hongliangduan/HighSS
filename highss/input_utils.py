from __future__ import annotations

import logging
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


def get_chains_sequence(pdb_path: str | Path) -> dict[str, str]:
    """Extract standard amino-acid sequences from a PDB file."""
    aa3_to_aa1 = {
        "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
        "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
        "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
        "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    }
    sequences: dict[str, list[str]] = {}
    seen: dict[str, set[tuple[str, str]]] = {}
    with Path(pdb_path).open() as handle:
        for line in handle:
            if not line.startswith("ATOM") or len(line) < 27:
                continue
            chain = line[21].strip() or "_"
            residue = line[17:20].strip().upper()
            residue_number = line[22:26].strip()
            insertion_code = line[26].strip()
            aa = aa3_to_aa1.get(residue)
            if aa is None:
                continue
            key = (residue_number, insertion_code)
            sequences.setdefault(chain, [])
            seen.setdefault(chain, set())
            if key not in seen[chain]:
                seen[chain].add(key)
                sequences[chain].append(aa)
    return {chain: "".join(seq) for chain, seq in sequences.items()}


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
    """Write a HighSS protein-target YAML with an empty binder MSA."""
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
