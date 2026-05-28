#!/usr/bin/env python3
"""
get_atom_indices.py
===================
Helper utility: extract atom indices for the ligand and binding-site residues
from a GROMACS .tpr file, so you can paste them into plumed_residence.dat.

Usage
-----
    python3 get_atom_indices.py --tpr topol.tpr --ligand LIG --site 45,67,89,120
    python3 get_atom_indices.py --tpr topol.tpr --ligand ATP

    # Or from a pre-existing GROMACS dump:
    gmx dump -s topol.tpr > tpr_dump.txt
    python3 get_atom_indices.py --dump tpr_dump.txt --ligand LIG

Requirements
------------
    MDAnalysis   (pip install MDAnalysis)

Output
------
    Prints atom index ranges ready to paste into PLUMED input files.
"""

import argparse
import sys
import subprocess
import os
import tempfile

def parse_args():
    p = argparse.ArgumentParser(
        description="Get atom indices from a GROMACS TPR for PLUMED input.")
    p.add_argument("--tpr",     help="Path to GROMACS .tpr file")
    p.add_argument("--gro",     help="Path to .gro or .pdb structure file (alternative to tpr)")
    p.add_argument("--ligand",  required=True,
                   help="Residue name of the ligand (e.g. LIG, ATP, NAD)")
    p.add_argument("--site",    default=None,
                   help="Comma-separated residue IDs of binding-site residues "
                        "(e.g. 45,67,89). If omitted, only ligand info is shown.")
    p.add_argument("--heavy",   action="store_true", default=True,
                   help="Report heavy atoms only (no hydrogen). Default: True")
    return p.parse_args()


def get_indices_mda(struct_file, ligand_resname, site_resids, heavy_only):
    """Use MDAnalysis to extract atom indices."""
    try:
        import MDAnalysis as mda
    except ImportError:
        sys.exit("[ERROR] MDAnalysis not installed. Run:  pip install MDAnalysis")

    print(f"[INFO] Loading structure: {struct_file}")
    u = mda.Universe(struct_file)

    # Ligand
    lig_sel = f"resname {ligand_resname}"
    if heavy_only:
        lig_sel += " and not name H*"
    lig = u.select_atoms(lig_sel)

    if len(lig) == 0:
        sys.exit(f"[ERROR] No atoms found for selection '{lig_sel}'. "
                 f"Available residue names: {set(u.residues.resnames)}")

    # PLUMED uses 1-based indexing
    lig_indices = lig.indices + 1
    lig_range = f"{lig_indices.min()}-{lig_indices.max()}"

    print(f"\n── Ligand ({ligand_resname}) ────────────────────────────────────────")
    print(f"  Total atoms (incl. H) : {len(u.select_atoms('resname ' + ligand_resname))}")
    print(f"  Heavy atoms           : {len(lig)}")
    print(f"  PLUMED indices (1-based): {lig_indices.tolist()}")
    print(f"  Compact range          : {lig_range}")
    print(f"\n  ➜  In plumed_residence.dat:")
    print(f"       coord: COORDINATION GROUPA={lig_range} ...")

    # Binding site
    if site_resids:
        resid_list = [int(r.strip()) for r in site_resids.split(",")]
        sel_str = "resid " + " ".join(str(r) for r in resid_list)
        if heavy_only:
            sel_str += " and not name H*"
        site_atoms = u.select_atoms(sel_str)

        if len(site_atoms) == 0:
            print(f"\n[WARN] No atoms found for site residues: {resid_list}")
        else:
            site_indices = site_atoms.indices + 1
            # Build compact ranges
            ranges = []
            start = prev = site_indices[0]
            for idx in site_indices[1:]:
                if idx == prev + 1:
                    prev = idx
                else:
                    ranges.append(f"{start}-{prev}" if start != prev else str(start))
                    start = prev = idx
            ranges.append(f"{start}-{prev}" if start != prev else str(start))
            compact = ",".join(ranges)

            print(f"\n── Binding site (residues {site_resids}) ────────────────────────")
            print(f"  Heavy atoms           : {len(site_atoms)}")
            print(f"  Compact range          : {compact}")
            print(f"\n  ➜  In plumed_residence.dat:")
            print(f"       coord: COORDINATION ... GROUPB={compact} ...")

    # Also suggest CENTER atoms for binding site
    print("\n── Suggested plumed_residence.dat snippet ──────────────────────────")
    print(f"""
MOLINFO STRUCTURE=topology.pdb

lig:  CENTER ATOMS={lig_range}
site: CENTER ATOMS={compact if site_resids else 'REPLACE_WITH_SITE_INDICES'}

d:    DISTANCE ATOMS=lig,site

coord: COORDINATION \\
  GROUPA={lig_range} \\
  GROUPB={compact if site_resids else 'REPLACE_WITH_SITE_INDICES'} \\
  R_0=0.35 NN=6 MM=12

PRINT ARG=d,coord FILE=colvar.dat STRIDE=1
""")


def main():
    args = parse_args()

    struct_file = args.tpr or args.gro
    if struct_file is None:
        sys.exit("[ERROR] Provide either --tpr or --gro.")
    if not os.path.isfile(struct_file):
        sys.exit(f"[ERROR] File not found: {struct_file}")

    # If a .tpr was given, convert to PDB first via gmx editconf
    if struct_file.endswith(".tpr"):
        print("[INFO] Converting .tpr → .pdb via gmx editconf ...")
        tmp_pdb = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False)
        tmp_pdb.close()
        ret = subprocess.run(
            ["gmx", "editconf", "-f", struct_file, "-o", tmp_pdb.name, "-quiet"],
            capture_output=True)
        if ret.returncode != 0:
            print("[WARN] gmx editconf failed — trying MDAnalysis directly on TPR ...")
            get_indices_mda(struct_file, args.ligand, args.site, args.heavy)
        else:
            get_indices_mda(tmp_pdb.name, args.ligand, args.site, args.heavy)
            os.unlink(tmp_pdb.name)
    else:
        get_indices_mda(struct_file, args.ligand, args.site, args.heavy)


if __name__ == "__main__":
    main()
