#!/usr/bin/env python3
"""Check SG-SG distances for designed disulfide peptide binders.

Example:
python scripts/check_disulfide_geometry.py \
  --structure outputs/.../result.cif \
  --binder_chain A \
  --binder_bonds 0_SG_24_SG,3_SG_15_SG,9_SG_21_SG \
  --bond_index_base 0 \
  --strict_cutoff 2.2
"""
import argparse
import csv
from pathlib import Path
import gemmi


def parse_bonds(s: str, index_base: int):
    pairs = []
    for item in (s or '').split(','):
        item = item.strip()
        if not item:
            continue
        r1, a1, r2, a2 = item.split('_')
        if a1.upper() != 'SG' or a2.upper() != 'SG':
            continue
        pairs.append((int(r1) - index_base, int(r2) - index_base))
    return pairs


def get_polymer_residues(chain):
    residues = []
    for res in chain:
        # skip waters and obvious non-polymer records when possible
        if res.name in {'HOH', 'WAT'}:
            continue
        if len(res) == 0:
            continue
        residues.append(res)
    return residues


def atom_pos(residue, atom_name='SG'):
    for atom in residue:
        if atom.name.strip() == atom_name:
            return atom.pos
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--structure', required=True, help='Predicted .cif or .pdb file')
    ap.add_argument('--binder_chain', default='A')
    ap.add_argument('--binder_bonds', required=True)
    ap.add_argument('--bond_index_base', type=int, choices=[0, 1], default=0)
    ap.add_argument('--strict_cutoff', type=float, default=2.2)
    ap.add_argument('--loose_cutoff', type=float, default=2.5)
    ap.add_argument('--csv_out', default='')
    args = ap.parse_args()

    structure = gemmi.read_structure(args.structure)
    model = structure[0]
    chain = model[args.binder_chain]
    residues = get_polymer_residues(chain)
    rows = []
    ok_strict = True
    ok_loose = True
    for i, j in parse_bonds(args.binder_bonds, args.bond_index_base):
        if i < 0 or j < 0 or i >= len(residues) or j >= len(residues):
            rows.append({'pair': f'{i}-{j}', 'distance': 'NA', 'strict_pass': False, 'loose_pass': False, 'note': 'residue index out of range'})
            ok_strict = ok_loose = False
            continue
        p1 = atom_pos(residues[i], 'SG')
        p2 = atom_pos(residues[j], 'SG')
        if p1 is None or p2 is None:
            rows.append({'pair': f'{i}-{j}', 'distance': 'NA', 'strict_pass': False, 'loose_pass': False, 'note': 'missing SG atom'})
            ok_strict = ok_loose = False
            continue
        d = p1.dist(p2)
        strict = d <= args.strict_cutoff
        loose = d <= args.loose_cutoff
        ok_strict = ok_strict and strict
        ok_loose = ok_loose and loose
        rows.append({'pair': f'{i}-{j}', 'distance': f'{d:.3f}', 'strict_pass': strict, 'loose_pass': loose, 'note': ''})

    print(f'structure: {args.structure}')
    print(f'binder_chain: {args.binder_chain}, residues_seen: {len(residues)}')
    for r in rows:
        print(f"pair {r['pair']}: distance={r['distance']} strict={r['strict_pass']} loose={r['loose_pass']} {r['note']}")
    print(f'ALL_STRICT_PASS={ok_strict}')
    print(f'ALL_LOOSE_PASS={ok_loose}')

    if args.csv_out:
        out = Path(args.csv_out)
        with out.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['structure', 'pair', 'distance', 'strict_pass', 'loose_pass', 'note'])
            w.writeheader()
            for r in rows:
                rr = {'structure': args.structure, **r}
                w.writerow(rr)
        print(f'wrote {out}')

if __name__ == '__main__':
    main()
