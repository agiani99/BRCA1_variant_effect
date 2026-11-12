"""Phase 3: structures, per-residue SASA/pLDDT and interface residues.

Downloads (data/raw/structures/): AlphaFold model of P38398 and PDB 1JM7, 1T15, 1JNX
Outputs (data/processed/): residue_features.csv, interface_residues.csv
"""
import warnings

import pandas as pd
from Bio.Data.PDBData import protein_letters_3to1_extended as three_to_one
from Bio.PDB import MMCIFParser, NeighborSearch, PDBParser, ShrakeRupley
from Bio.PDB.PDBExceptions import PDBConstructionWarning

from common import RAW, PROCESSED, UNIPROT_ID, session

warnings.simplefilter("ignore", PDBConstructionWarning)
STRUCT = RAW / "structures"
STRUCT.mkdir(exist_ok=True)

# Max accessible surface area per residue (Tien et al. 2013, theoretical), A^2
MAX_ASA = {"A": 129, "R": 274, "N": 195, "D": 193, "C": 167, "E": 223, "Q": 225,
           "G": 104, "H": 224, "I": 197, "L": 201, "K": 236, "M": 224, "F": 240,
           "P": 159, "S": 155, "T": 172, "W": 285, "Y": 263, "V": 174}

# complex -> (BRCA1 chain, partner chain, partner name, BRCA1 residue offset)
# Chain IDs are checked against the downloaded files at runtime (see describe()).
COMPLEXES = {
    "1JM7": ("A", "B", "BARD1", 0),     # BRCA1 RING / BARD1 RING
    "1T15": ("A", "B", "BRIP1/BACH1 pS990 phosphopeptide", 0),  # BRCT + phosphopeptide
}
# NOTE: 1JNX contains only the BRCT domain (no peptide), so it is not a complex.
CONTACT_CUTOFF = 5.0  # A, any heavy atom pair


def fetch_structures() -> dict:
    s = session()
    paths = {}
    meta = s.get(f"https://alphafold.ebi.ac.uk/api/prediction/{UNIPROT_ID}", timeout=60).json()[0]
    af = STRUCT / f"AF-{UNIPROT_ID}.pdb"
    if not af.exists():
        af.write_bytes(s.get(meta["pdbUrl"], timeout=120).content)
    paths["AF"] = af
    for pdb in COMPLEXES:
        p = STRUCT / f"{pdb}.cif"
        if not p.exists():
            p.write_bytes(s.get(f"https://files.rcsb.org/download/{pdb}.cif", timeout=60).content)
        paths[pdb] = p
    return paths


def load(path):
    parser = PDBParser(QUIET=True) if path.suffix == ".pdb" else MMCIFParser(QUIET=True)
    model = parser.get_structure(path.stem, str(path))[0]
    for ch in model:  # drop crystallographic waters
        for r in [r for r in ch if r.id[0] == "W"]:
            ch.detach_child(r.id)
    return model


def aa(res):
    return three_to_one.get(res.get_resname().upper(), "X") if res.id[0] == " " else None


def describe(model):
    return {ch.id: (len(ch), next(iter(ch)).id[1], list(ch)[-1].id[1]) for ch in model}


def sasa_per_residue(model, chain_ids=None):
    """Relative SASA per residue (0-1+) computed on the given model."""
    ShrakeRupley().compute(model, level="R")
    out = {}
    for ch in model:
        if chain_ids and ch.id not in chain_ids:
            continue
        for r in ch:
            a = aa(r)
            if a and a in MAX_ASA:
                out[(ch.id, r.id[1])] = (a, r.sasa / MAX_ASA[a])
    return out


def alphafold_features(path) -> pd.DataFrame:
    model = load(path)
    rsa = sasa_per_residue(model)
    rows = []
    for (ch, pos), (a, rel) in rsa.items():
        res = model[ch][pos]
        plddt = res["CA"].get_bfactor() if "CA" in res else float("nan")
        rows.append({"pos": pos, "wt": a, "plddt": plddt, "rsa_af": rel})
    return pd.DataFrame(rows).sort_values("pos")


def interface_residues(pdb_id: str, path) -> pd.DataFrame:
    brca1_ch, partner_ch, partner, offset = COMPLEXES[pdb_id]
    model = load(path)
    atoms = [a for a in model.get_atoms() if a.element != "H"]
    ns = NeighborSearch(atoms)
    # RSA in the complex vs. BRCA1 alone -> buried-by-partner
    rsa_complex = sasa_per_residue(model, {brca1_ch})
    for ch in list(model):
        if ch.id != brca1_ch:
            model.detach_child(ch.id)
    rsa_alone = sasa_per_residue(model, {brca1_ch})

    full = load(path)
    contacts = set()
    for r in full[brca1_ch]:
        if aa(r) is None:
            continue
        for a in r:
            for near in ns.search(a.coord, CONTACT_CUTOFF, level="R"):
                # partner residues incl. modified ones such as phosphoserine (H_SEP)
                if near.get_parent().id == partner_ch and not near.id[0].startswith("H_") or \
                        (near.get_parent().id == partner_ch and near.get_resname() == "SEP"):
                    contacts.add(r.id[1])
    rows = []
    for (ch, pos), (a, rel_c) in rsa_complex.items():
        rel_a = rsa_alone[(ch, pos)][1]
        rows.append({"pdb": pdb_id, "partner": partner, "pos": pos + offset, "wt": a,
                     "in_contact": pos in contacts,
                     "rsa_complex": rel_c, "rsa_alone": rel_a,
                     "delta_rsa": rel_a - rel_c})
    df = pd.DataFrame(rows)
    # interface = contact OR loses >= 10% relative SASA on binding
    df["interface"] = df["in_contact"] | (df["delta_rsa"] >= 0.10)
    return df


def main():
    paths = fetch_structures()
    for k, p in paths.items():
        print(k, p.name, describe(load(p)))

    feats = alphafold_features(paths["AF"])
    feats["confident"] = feats["plddt"] >= 70
    feats.to_csv(PROCESSED / "residue_features.csv", index=False)
    print(f"AlphaFold: {len(feats)} residues, {feats['confident'].sum()} with pLDDT>=70")

    ifc = pd.concat([interface_residues(k, paths[k]) for k in COMPLEXES], ignore_index=True)
    ifc.to_csv(PROCESSED / "interface_residues.csv", index=False)
    print(ifc[ifc["interface"]].groupby(["pdb", "partner"])["pos"].agg(["count", "min", "max"]))


if __name__ == "__main__":
    main()
