"""Phase 6b: validate / retrain with Findlay 2018 saturation genome editing (SGE).

Input : data/raw/41586_2018_461_MOESM3_ESM.xlsx (Supplementary Table 1)
Output: data/processed/sge_variants.csv, sge_metrics.csv, predictions_sge.csv
SGE covers only RING (exons 2-5) and BRCT (exons 15-23) -> no information for disordered regions.

Labels: function score is log2-scaled, more negative = loss of function.
func.class: LOF / INT / FUNC. We use LOF vs FUNC (INT = intermediate, kept as 'hypomorphic').
Missense only; M1 (start-loss) is dropped because it is trivially LoF.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from classify import FEATURE_SETS, model, rule_score
from common import PROCESSED, RAW

SGE_FILE = RAW / "41586_2018_461_MOESM3_ESM.xlsx"


def load_sge() -> pd.DataFrame:
    d = pd.read_excel(SGE_FILE, header=2)
    d = d[(d["consequence"] == "Missense") & d["aa_pos"].notna()].copy()
    d["pos"] = d["aa_pos"].astype(int)
    d = d[d["pos"] > 1]
    d["variant"] = d["aa_ref"] + d["pos"].astype(str) + d["aa_alt"]
    # several SNVs can encode the same amino-acid change -> average score, majority class
    g = d.groupby("variant").agg(
        pos=("pos", "first"), sge_score=("function.score.mean", "mean"),
        sge_class=("func.class", lambda x: x.value_counts().idxmax()),
        n_snv=("variant", "size")).reset_index()
    g.to_csv(PROCESSED / "sge_variants.csv", index=False)
    return g


def auc(y, s):
    return roc_auc_score(y, s) if len(set(y)) == 2 else np.nan


def main():
    sge = load_sge()
    print(f"SGE missense amino-acid variants: {len(sge)}  "
          f"{sge['sge_class'].value_counts().to_dict()}")

    pred = pd.read_csv(PROCESSED / "predictions.csv")
    pred["any_hotspot"] = pred["any_hotspot"].astype(float)
    pred["rule_score"] = rule_score(pred)
    df = pred.merge(sge[["variant", "sge_score", "sge_class"]], on="variant", how="inner")
    print(f"{len(df)} of {len(sge)} SGE variants present in our table")

    # agreement of SGE with ClinVar on shared labelled variants
    cv = df[df["label"].isin(["LoF", "neutral"]) & df["sge_class"].isin(["LOF", "FUNC"])]
    print(f"\nSGE vs ClinVar concordance (n={len(cv)}):")
    print(pd.crosstab(cv["label"], cv["sge_class"]))

    # ---- (1) external validation of the ClinVar-trained model ----
    # p_lof is out-of-fold for ClinVar-labelled variants and from a model that never saw the
    # labels of VUS/unknown variants, so SGE is independent evidence for the latter.
    ev = df[df["sge_class"].isin(["LOF", "FUNC"])].copy()
    ev["y"] = (ev["sge_class"] == "LOF").astype(int)
    ev["clinvar_labelled"] = ev["label"].isin(["LoF", "neutral"])
    rows = []
    for subset, g in [("all SGE", ev), ("not ClinVar-labelled (independent)", ev[~ev["clinvar_labelled"]])]:
        for dom in ["all", "RING", "BRCT"]:
            s = g if dom == "all" else g[g["domain"] == dom]
            if len(s) < 20:
                continue
            for name, col, sign in [("p_lof (all features)", "p_lof", 1), ("esm_llr", "esm_llr", -1),
                                    ("rule_score", "rule_score", 1),
                                    ("ppi_risk_max", "ppi_risk_max", 1)]:
                rows.append({"set": subset, "domain": dom, "score": name, "n": len(s),
                             "n_LOF": int(s["y"].sum()), "AUROC": auc(s["y"], sign * s[col]),
                             "AUPRC": average_precision_score(s["y"], sign * s[col])})
    ext = pd.DataFrame(rows)
    print("\n(1) ClinVar-trained scores evaluated against SGE:")
    print(ext.round(3).to_string(index=False))

    # ---- (2) retrain on SGE labels, blocked CV ----
    y = ev["y"].reset_index(drop=True)
    X = ev.reset_index(drop=True)
    groups = X["pos"] // 25
    rows2 = []
    for name, cols in FEATURE_SETS.items():
        oof = np.zeros(len(X))
        for tr, te in GroupKFold(5).split(X, y, groups):
            oof[te] = model().fit(X.loc[tr, cols], y[tr]).predict_proba(X.loc[te, cols])[:, 1]
        X[f"oof_{name}"] = oof
        rows2.append({"trained_on": "SGE", "features": name, "n": len(X),
                      "AUROC": auc(y, oof), "AUPRC": average_precision_score(y, oof)})
        for dom in ("RING", "BRCT"):
            m = (X["domain"] == dom).values
            rows2.append({"trained_on": "SGE", "features": f"{name} [{dom}]", "n": int(m.sum()),
                          "AUROC": auc(y[m], oof[m]), "AUPRC": average_precision_score(y[m], oof[m])})
    cvm = pd.DataFrame(rows2)
    print("\n(2) Models retrained on SGE labels (blocked CV, 25-aa blocks):")
    print(cvm.round(3).to_string(index=False))

    pd.concat([ext.assign(kind="external"), cvm.assign(kind="sge_cv")]).to_csv(
        PROCESSED / "sge_metrics.csv", index=False)
    X.to_csv(PROCESSED / "predictions_sge.csv", index=False)

    # how many VUS can SGE now resolve?
    vus = df[df["label"] == "VUS"]
    print(f"\nClinVar VUS with SGE evidence: {len(vus)} -> "
          f"{vus['sge_class'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
