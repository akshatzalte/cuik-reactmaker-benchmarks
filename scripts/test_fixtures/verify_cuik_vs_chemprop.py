#!/usr/bin/env python
"""
Re-verify cuik C++ batch_reaction_featurizer == chemprop Python
CondensedGraphOfReactionFeaturizer on the NEW synthetic dataset.
Per-reaction positional comparison, all 4 versions x 6 modes.
"""
import csv
import os
import sys
import numpy as np
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

import cuik_molmaker
from chemprop.featurizers.molgraph.reaction import CondensedGraphOfReactionFeaturizer, RxnMode
from chemprop.featurizers.atom import MultiHotAtomFeaturizer, RIGRAtomFeaturizer
from chemprop.featurizers.bond import MultiHotBondFeaturizer, RIGRBondFeaturizer
from chemprop.utils import make_mol

CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.environ.get("CUIK_DATA_DIR", "/data3/akshatz/cuik-molmaker/tests/data"),
    "sample_rxns_100.csv")

MODES = ["REAC_DIFF", "REAC_PROD", "PROD_DIFF",
         "REAC_DIFF_BALANCE", "REAC_PROD_BALANCE", "PROD_DIFF_BALANCE"]

# cuik feature-name configs (mirror gen_reaction_refs.py)
CUIK_CFG = {
    "V1":   (["atomic-number","total-degree","formal-charge","chirality","num-hydrogens","hybridization"],
             ["aromatic","mass"], ["is-null","bond-type-onehot","conjugated","in-ring","stereo"]),
    "V2":   (["atomic-number-common","total-degree","formal-charge","chirality","num-hydrogens","hybridization-expanded"],
             ["aromatic","mass"], ["is-null","bond-type-onehot","conjugated","in-ring","stereo"]),
    "ORGANIC": (["atomic-number-organic","total-degree","formal-charge","chirality","num-hydrogens","hybridization-organic"],
             ["aromatic","mass"], ["is-null","bond-type-onehot","conjugated","in-ring","stereo"]),
    "RIGR": (["atomic-number-common","total-degree","num-hydrogens"],
             ["mass"], ["is-null","in-ring"]),
}

def chemprop_featurizers(version):
    if version == "V1":   af = MultiHotAtomFeaturizer.v1()
    elif version == "V2": af = MultiHotAtomFeaturizer.v2()
    elif version == "ORGANIC": af = MultiHotAtomFeaturizer.organic()
    elif version == "RIGR":    af = RIGRAtomFeaturizer()
    bf = RIGRBondFeaturizer() if version == "RIGR" else MultiHotBondFeaturizer()
    return af, bf

def main():
    rows = [r["rxn_smiles"].strip() for r in csv.DictReader(open(CSV))]
    reac = [s.split(">>")[0] for s in rows]
    prod = [s.split(">>")[1] for s in rows]
    print(f"{len(rows)} reactions from {CSV}\n")

    grand_maxV = grand_maxE = 0.0
    ei_mismatch = rei_mismatch = 0
    errors = []

    for version, (ao_n, af_n, bf_n) in CUIK_CFG.items():
        ao = cuik_molmaker.atom_onehot_feature_names_to_array(ao_n)
        afl = cuik_molmaker.atom_float_feature_names_to_array(af_n)
        bfa = cuik_molmaker.bond_feature_names_to_array(bf_n)
        cp_af, cp_bf = chemprop_featurizers(version)

        for mode in MODES:
            mode_int = cuik_molmaker.reaction_mode_to_int(mode)
            cgr = CondensedGraphOfReactionFeaturizer(
                atom_featurizer=cp_af, bond_featurizer=cp_bf, mode_=RxnMode[mode])
            vmax = emax = 0.0
            for i, (rs, ps) in enumerate(zip(reac, prod)):
                try:
                    rct = make_mol(rs, keep_h=True, add_h=False)
                    pdt = make_mol(ps, keep_h=True, add_h=False)
                    mg = cgr((rct, pdt), None, None)
                    Vc, Ec, eic, reic, _ = cuik_molmaker.batch_reaction_featurizer(
                        [rs], [ps], ao, afl, bfa, True, False, False, mode_int)
                    if Vc.shape != mg.V.shape:
                        errors.append(f"{version}/{mode} rxn{i}: V shape {Vc.shape} vs {mg.V.shape}")
                        continue
                    if Ec.shape != mg.E.shape:
                        errors.append(f"{version}/{mode} rxn{i}: E shape {Ec.shape} vs {mg.E.shape}")
                        continue
                    vmax = max(vmax, float(np.abs(Vc - mg.V).max()) if Vc.size else 0.0)
                    emax = max(emax, float(np.abs(Ec - mg.E).max()) if Ec.size else 0.0)
                    if not np.array_equal(eic, mg.edge_index):
                        globals_ei = ei_mismatch + 1
                        errors.append(f"{version}/{mode} rxn{i}: edge_index mismatch")
                    if not np.array_equal(reic.ravel(), np.asarray(mg.rev_edge_index).ravel()):
                        errors.append(f"{version}/{mode} rxn{i}: rev_edge_index mismatch")
                except Exception as e:
                    errors.append(f"{version}/{mode} rxn{i}: EXC {type(e).__name__}: {e}")
            grand_maxV = max(grand_maxV, vmax); grand_maxE = max(grand_maxE, emax)
            print(f"{version:8s} {mode:18s}  maxdiff V={vmax:.2e}  E={emax:.2e}")

    print(f"\nGRAND max_diff: V={grand_maxV:.3e}  E={grand_maxE:.3e}")
    print(f"errors/mismatches: {len(errors)}")
    for e in errors[:30]:
        print("  ", e)

if __name__ == "__main__":
    main()
