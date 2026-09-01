#!/usr/bin/env python
"""
Build tests/data/sample_rxns_100.csv for cuik-molmaker by sampling 100 real
reactions from the RDB7 dataset (CC BY 4.0).

RDB7 source (CC BY 4.0):
  Spiekermann, K. A.; Pattanaik, L.; Green, W. H. "High Accuracy Barrier
  Heights, Enthalpies, and Rate Coefficients for Chemical Reactions."
  Sci. Data 2022, 9, 417.  Zenodo: https://doi.org/10.5281/zenodo.13328872

Run in cuikmm env:  python gen_rdb7_sample.py
Deterministic (seed=7) -> reproduces the committed CSV.
"""
import csv
import os
import random

from rdkit import Chem
from rdkit import RDLogger

import cuik_molmaker

RDLogger.DisableLog("rdApp.*")
random.seed(7)

RDB7_CSV = os.environ.get(
    "RDB7_CSV", "/home/akshatz/bond_order_free/barriers_rdb7/dataset/rdb7_data.csv")
CUIK_DATA = os.environ.get(
    "CUIK_DATA_DIR", "/data3/akshatz/cuik-molmaker/tests/data")
N_RDB7 = 100


def valid(rxn):
    """parse+sanitize both sides; no atom at map 0; cuik featurizes OK."""
    try:
        r, p = rxn.split(">>")
        pr = Chem.SmilesParserParams(); pr.removeHs = False
        mr = Chem.MolFromSmiles(r, pr); mp = Chem.MolFromSmiles(p, pr)
        if mr is None or mp is None:
            return False
        if any(a.GetAtomMapNum() == 0 for a in mr.GetAtoms()):
            return False
        if any(a.GetAtomMapNum() == 0 for a in mp.GetAtoms()):
            return False
        ao = cuik_molmaker.atom_onehot_feature_names_to_array(
            ["atomic-number-common", "total-degree", "formal-charge",
             "chirality", "num-hydrogens", "hybridization-expanded"])
        af = cuik_molmaker.atom_float_feature_names_to_array(["aromatic", "mass"])
        bf = cuik_molmaker.bond_feature_names_to_array(
            ["is-null", "bond-type-onehot", "conjugated", "in-ring", "stereo"])
        mi = cuik_molmaker.reaction_mode_to_int("REAC_DIFF")
        cuik_molmaker.batch_reaction_featurizer([r], [p], ao, af, bf,
                                                True, False, False, mi)
        return True
    except Exception:
        return False


def main():
    rxns = [row["smiles"].strip() for row in csv.DictReader(open(RDB7_CSV))]
    print(f"RDB7 total: {len(rxns)}")
    random.shuffle(rxns)

    picked = []
    for s in rxns:
        if len(picked) >= N_RDB7:
            break
        if valid(s):
            picked.append(s)
    print(f"picked {len(picked)} valid RDB7 reactions")

    out = os.path.join(CUIK_DATA, "sample_rxns_100.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rxn_smiles"])
        for s in picked:
            w.writerow([s])
    print(f"wrote {out}: {len(picked)} RDB7 reactions")


if __name__ == "__main__":
    main()
