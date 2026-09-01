#!/usr/bin/env python
"""
Generate ~90 chemically reasonable, license-free synthetic E2/SN2 reactions
(atom-mapped, explicit mapped H, balanced) + keep 10 hand-crafted reactions.

Same *flavor* as QMrxn20 E2/SN2 (halide leaving groups; H-/halide/OH/CN
nucleophiles; CH3/NH2/NO2/CN/F/Cl substituents on an ethane-ish skeleton)
but independently constructed -- no rows copied from the licensed dataset.

Every candidate is gated on: RDKit parse(removeHs=False) + sanitize on BOTH
sides, balanced atom-map sets (no atom left at map 0), and a successful
cuik_molmaker.batch_reaction_featurizer call.
"""
import csv
import itertools
import os
import random

from rdkit import Chem
from rdkit import RDLogger

import cuik_molmaker

RDLogger.DisableLog("rdApp.*")
random.seed(7)

# --- map counter ------------------------------------------------------------
class Maps:
    def __init__(self):
        self.n = 0
    def __call__(self):
        self.n += 1
        return self.n

# --- substituent fragments (attachment atom first) --------------------------
# each returns a SMILES string with all atoms mapped; sanitize-tested forms.
def frag(kind, m):
    if kind == "H":
        return f"[H:{m()}]"
    if kind == "CH3":
        c = m(); return f"[C:{c}]([H:{m()}])([H:{m()}])[H:{m()}]"
    if kind == "NH2":
        n = m(); return f"[N:{n}]([H:{m()}])[H:{m()}]"
    if kind == "OH":
        o = m(); return f"[O:{o}][H:{m()}]"
    if kind == "F":
        return f"[F:{m()}]"
    if kind == "Cl":
        return f"[Cl:{m()}]"
    if kind == "CN":
        c = m(); return f"[C:{c}]#[N:{m()}]"
    if kind == "NO2":
        n = m(); return f"[N+:{n}](=[O:{m()}])[O-:{m()}]"
    raise ValueError(kind)

# 10 hand-crafted, license-free reactions (original toy reactions, not from any
# external dataset). Sole coverage for num_only / BALANCE-mode divergence.
HANDCRAFTED = [
    '[CH4:1]>>[CH3:1][OH:2]',
    '[CH3:1][Br:2]>>[CH4:1]',
    '[CH3:2][Cl:1].[OH2:6]>>[CH3:2][OH:6].[Cl:1][H:7]',
    '[H-:1].[C:2]([H:3])([H:4])([H:5])[Br:6]>>[H:1][C:2]([H:3])([H:4])[H:5].[Br-:6]',
    '[C:1]([H:2])([H:3])([H:4])[OH:5].[F-:6]>>[C:1]([H:3])([H:4])=[O:5].[F:6][H:2]',
    '[CH3:1][CH3:2]>>[CH3:1][CH3:2].[Na+:88].[Cl-:89]',
    '[Na+:1].[CH3:2][OH:3]>>[Na:1][OH:3].[CH3:2][H:4]',
    '[C:1](=[O:2])[OH:3].[H:4][OH:5]>>[C:1]([H:4])([OH:3])[OH:5]',
    '[C:1]([H:2])([H:3])=[C:4]([H:5])[H:6]>>[C:1]([H:2])([H:3])([H:5])[C:4]([H:6])[OH:9]',
    '[CH3:1][C:2](=[O:3])[OH:4].[H:5][OH:6]>>[CH3:1][C:2](=[O:3])[OH:6].[H:5][OH:4]',
]

SUBS = ["H", "CH3", "NH2", "F", "Cl", "CN", "NO2", "OH"]
LG = ["F", "Cl", "Br"]
# nucleophile: (reactant_anion, product_prefix_builder)
def nuc(kind, m):
    if kind == "H":
        a = m(); return f"[H-:{a}]", f"[H:{a}]"
    if kind in ("F", "Cl", "Br"):
        a = m(); return f"[{kind}-:{a}]", f"[{kind}:{a}]"
    if kind == "OH":
        o = m(); h = m(); return f"[O-:{o}][H:{h}]", f"[O:{o}]([H:{h}])"
    if kind == "CN":
        c = m(); n = m(); return f"[C-:{c}]#[N:{n}]", f"[C:{c}](#[N:{n}])"
    raise ValueError(kind)

NUCS = ["H", "F", "Cl", "Br", "OH", "CN"]


def sn2(lg, nu, s1, s2, s3, chiral=False):
    m = Maps()
    c = m()
    lg_m = m()
    f1, f2, f3 = frag(s1, m), frag(s2, m), frag(s3, m)
    nu_r, nu_p = nuc(nu, m)
    cc = "[C@:%d]" % c if chiral else "[C:%d]" % c
    cp = "[C@@:%d]" % c if chiral else "[C:%d]" % c  # inversion
    reac = f"[{lg}:{lg_m}]{cc}({f1})({f2}){f3}.{nu_r}"
    prod = f"[{lg}-:{lg_m}].{nu_p}{cp}({f1})({f2}){f3}"
    return reac + ">>" + prod


def e2(lg, base, a1, a2, b1, b2, stereo=False):
    m = Maps()
    ca = m(); cb = m()
    lg_m = m(); hb = m()
    fa1, fa2 = frag(a1, m), frag(a2, m)
    fb1, fb2 = frag(b1, m), frag(b2, m)
    if base in ("F", "Cl", "Br"):
        b_m = m()
        base_r = f"[{base}-:{b_m}]"
        base_p = f"[{base}:{b_m}][H:{hb}]"
    elif base == "OH":
        o = m(); bh = m()
        base_r = f"[O-:{o}][H:{bh}]"
        base_p = f"[O:{o}]([H:{bh}])[H:{hb}]"
    reac = (f"[{lg}:{lg_m}][C:{ca}]({fa1})({fa2})"
            f"[C:{cb}]([H:{hb}])({fb1}){fb2}.{base_r}")
    if stereo:
        prod = (f"[{lg}-:{lg_m}].[C:{ca}]({fa1})(/{fa2})"
                f"=[C:{cb}](\\{fb1}){fb2}.{base_p}")
    else:
        prod = (f"[{lg}-:{lg_m}].[C:{ca}]({fa1})({fa2})"
                f"=[C:{cb}]({fb1}){fb2}.{base_p}")
    return reac + ">>" + prod


def valid(rxn):
    """parse+sanitize both sides; balanced maps; no atom at map 0; cuik OK."""
    try:
        r, p = rxn.split(">>")
        pr = Chem.SmilesParserParams(); pr.removeHs = False
        mr = Chem.MolFromSmiles(r, pr); mp = Chem.MolFromSmiles(p, pr)
        if mr is None or mp is None:
            return False, "parse"
        rm = sorted(a.GetAtomMapNum() for a in mr.GetAtoms())
        pm = sorted(a.GetAtomMapNum() for a in mp.GetAtoms())
        if 0 in rm or 0 in pm:
            return False, "unmapped atom"
        if set(rm) != set(pm):
            return False, "map imbalance"
        # featurize smoke test (V2 / REAC_DIFF)
        ao = cuik_molmaker.atom_onehot_feature_names_to_array(
            ["atomic-number-common", "total-degree", "formal-charge",
             "chirality", "num-hydrogens", "hybridization-expanded"])
        af = cuik_molmaker.atom_float_feature_names_to_array(["aromatic", "mass"])
        bf = cuik_molmaker.bond_feature_names_to_array(
            ["is-null", "bond-type-onehot", "conjugated", "in-ring", "stereo"])
        mi = cuik_molmaker.reaction_mode_to_int("REAC_DIFF")
        cuik_molmaker.batch_reaction_featurizer([r], [p], ao, af, bf,
                                                True, False, False, mi)
        return True, "ok"
    except Exception as e:
        return False, f"exc:{e}"


def main():
    pool = []
    # SN2 sweep
    for lg, nu in itertools.product(LG, NUCS):
        for s1, s2 in itertools.combinations_with_replacement(SUBS, 2):
            pool.append(("sn2", sn2(lg, nu, s1, s2, "H")))
    # a few chiral SN2 (4 distinct groups for genuine stereocenter)
    for lg, nu, s1, s2, s3 in [
        ("Cl", "H", "CH3", "NH2", "F"), ("Br", "F", "CH3", "Cl", "NH2"),
        ("Cl", "CN", "CH3", "F", "NH2"), ("F", "OH", "CH3", "Cl", "CN"),
        ("Br", "Cl", "CH3", "NH2", "CN"),
    ]:
        pool.append(("sn2c", sn2(lg, nu, s1, s2, s3, chiral=True)))
    # E2 sweep
    for lg, base in itertools.product(LG, ["F", "Cl", "Br", "OH"]):
        for a1, b1 in [("H", "H"), ("CH3", "H"), ("H", "CH3"),
                       ("NH2", "H"), ("F", "H"), ("CH3", "CH3"), ("Cl", "H")]:
            pool.append(("e2", e2(lg, base, a1, "H", b1, "H")))
    # a few stereo E2 products (cis/trans markers)
    for lg, base, a1, b1 in [
        ("Br", "Cl", "CH3", "CH3"), ("Cl", "F", "CH3", "NH2"),
        ("Br", "OH", "CH3", "Cl"), ("Cl", "Br", "NH2", "CH3"),
    ]:
        pool.append(("e2s", e2(lg, base, a1, "H", b1, "H", stereo=True)))

    # dedup, validate
    seen = set(); valids = []
    for tag, rxn in pool:
        if rxn in seen:
            continue
        seen.add(rxn)
        ok, why = valid(rxn)
        if ok:
            valids.append((tag, rxn))

    # split by family; FORCE-include stereo cases so chirality/bond-stereo
    # features are exercised (otherwise the test goes blind to them).
    sn2_stereo = [r for t, r in valids if t == "sn2c"]
    e2_stereo = [r for t, r in valids if t == "e2s"]
    sn2_plain = [r for t, r in valids if t == "sn2"]
    e2_plain = [r for t, r in valids if t == "e2"]
    print(f"SN2 stereo:{len(sn2_stereo)} plain:{len(sn2_plain)}  "
          f"E2 stereo:{len(e2_stereo)} plain:{len(e2_plain)}")

    random.shuffle(sn2_plain); random.shuffle(e2_plain)
    sn2_pick = sn2_stereo + sn2_plain[:50 - len(sn2_stereo)]
    e2_pick = e2_stereo + e2_plain[:40 - len(e2_stereo)]
    pick = sn2_pick + e2_pick
    random.shuffle(pick)
    nchiral = sum('@' in s for s in pick)
    nbond = sum(('/' in s or '\\' in s) for s in pick)
    print(f"picked: {len(pick)} synthetic (chiral={nchiral}, bond-stereo={nbond})")

    final = pick + HANDCRAFTED
    print(f"final total: {len(final)} (= {len(pick)} synthetic + {len(HANDCRAFTED)} handcrafted)")

    cuik_data = os.environ.get(
        "CUIK_DATA_DIR", "/data3/akshatz/cuik-molmaker/tests/data")
    out = os.path.join(cuik_data, "sample_rxns_100.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rxn_smiles"])
        for s in final:
            w.writerow([s])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
