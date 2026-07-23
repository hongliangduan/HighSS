from __future__ import annotations

import sys
from pathlib import Path

from omegaconf import OmegaConf


def main(path: str) -> int:
    cfg = OmegaConf.load(path)
    problems: list[str] = []
    for key in ("pretrained", "resume"):
        value = cfg.get(key)
        if value and not Path(str(value)).exists():
            problems.append(f"missing {key}: {value}")
    for i, dataset in enumerate(cfg.data.datasets):
        for key in ("target_dir", "msa_dir", "split"):
            value = dataset.get(key)
            if value and not Path(str(value)).exists():
                problems.append(f"dataset {i} missing {key}: {value}")
    if problems:
        print("Configuration is syntactically valid, but required local files are missing:")
        for item in problems:
            print(f"- {item}")
        return 1
    print("Configuration and required paths are valid.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/train/validate_config.py <config.yaml>")
    raise SystemExit(main(sys.argv[1]))
