"""Phase 5: PPI-modulation features per variant and partner.

No FoldX / ColabFold complexes are available, so binding effects are proxied from
experimental complex structures:
  * RING          -> BARD1                      (PDB 1JM7)
  * BRCT groove   -> BRIP1, ABRAXAS1, RBBP8     (PDB 1T15, shared pSer-binding site)
  * coiled-coil   -> PALB2                      (region only, no structure yet)
Per-partner weights come from STRING (experimental + database evidence).

Output: data/processed/ppi_features.csv (one row per variant)
        data/processed/ppi_variant_partner.csv (long format)
"""
import numpy as np
import pandas as pd

from common import PROCESSED

# partner -> (source structure id or region label)
PARTNER_SOURCE = {
    "BARD1": "1JM7",
    "BRIP1": "1T15",
    "ABRAXAS1": "1T15",
    "RBBP8": "1T15",
}
REGION_ONLY = {"PALB2": (1391, 1424)}
# Well-established phospho-peptide recognition residues in the BRCT groove
BRCT_HOTSPOTS = {1655, 1656, 1699, 1702, 1836, 1837, 1838}


def partner_weights() -> dict:
    s = pd.read_csv(PROCESSED / "string_partners.csv").set_index("partner")
    # evidence-based weight in [0,1]: mean of experimental and database channels
    return ((s["experimental"] + s["database"]) / 2).to_dict()


def main():
    stab = pd.read_csv(PROCESSED / "stability_features.csv")
    ifc = pd.read_csv(PROCESSED / "interface_residues.csv")
    w = partner_weights()
    ifc = ifc.set_index(["pdb", "pos"])

    change = (stab["d_vol"].abs() / 100 + stab["d_hydro"].abs() / 9
              + (4 - stab["blosum62"]).clip(lower=0) / 8)
    stab = stab.assign(change=change)

    rows = []
    for partner, pdb in PARTNER_SOURCE.items():
        for r in stab.itertuples():
            key = (pdb, r.pos)
            if key not in ifc.index:
                continue
            i = ifc.loc[key]
            if not i["interface"]:
                continue
            burial_by_partner = max(i["delta_rsa"], 0.0)
            hotspot = r.pos in BRCT_HOTSPOTS and pdb == "1T15"
            score = (0.5 + burial_by_partner + (1.0 if hotspot else 0.0)) * r.change
            rows.append({"variant": r.variant, "partner": partner, "pos": r.pos,
                         "delta_rsa": i["delta_rsa"], "hotspot": hotspot,
                         "partner_weight": w.get(partner, 0.0), "ppi_risk": score})
    for partner, (lo, hi) in REGION_ONLY.items():
        for r in stab.itertuples():
            if lo <= r.pos <= hi:
                rows.append({"variant": r.variant, "partner": partner, "pos": r.pos,
                             "delta_rsa": np.nan, "hotspot": False,
                             "partner_weight": w.get(partner, 0.0),
                             "ppi_risk": 0.75 * r.change})  # region proxy, lower confidence
    long = pd.DataFrame(rows)
    long.to_csv(PROCESSED / "ppi_variant_partner.csv", index=False)

    agg = long.assign(wr=long["ppi_risk"] * long["partner_weight"]).groupby("variant").agg(
        ppi_risk_weighted=("wr", "sum"), ppi_risk_max=("ppi_risk", "max"),
        n_partners_hit=("partner", "nunique"),
        partners_hit=("partner", lambda x: ",".join(sorted(set(x)))),
        any_hotspot=("hotspot", "any")).reset_index()
    out = stab.drop(columns=["change"]).merge(agg, on="variant", how="left")
    out["n_partners_hit"] = out["n_partners_hit"].fillna(0).astype(int)
    out["ppi_risk_weighted"] = out["ppi_risk_weighted"].fillna(0.0)
    out["ppi_risk_max"] = out["ppi_risk_max"].fillna(0.0)
    out["partners_hit"] = out["partners_hit"].fillna("")
    out["any_hotspot"] = out["any_hotspot"].fillna(False)
    out.to_csv(PROCESSED / "ppi_features.csv", index=False)

    print(f"{(out['n_partners_hit'] > 0).sum()} of {len(out)} variants hit >=1 partner interface")
    print(out.groupby(["n_partners_hit", "label"]).size().unstack(fill_value=0))

    from sklearn.metrics import roc_auc_score
    lab = out[out["label"].isin(["LoF", "neutral"])]
    y = (lab["label"] == "LoF").astype(int)
    iface = lab[lab["n_partners_hit"] > 0]
    yi = (iface["label"] == "LoF").astype(int)
    print(f"AUROC ppi_risk_max, all labelled: {roc_auc_score(y, lab['ppi_risk_max']):.3f}")
    print(f"Interface variants only (n={len(iface)}, LoF={yi.sum()}): "
          f"ppi_risk_max {roc_auc_score(yi, iface['ppi_risk_max']):.3f}, "
          f"esm_llr {roc_auc_score(yi, -iface['esm_llr']):.3f}")


if __name__ == "__main__":
    main()
