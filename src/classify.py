"""Phase 6: classify variants as LoF / hypomorphic-uncertain / neutral.

Training labels: ClinVar LoF (pathogenic/likely pathogenic) vs neutral (benign/likely benign).
Validation: GroupKFold with groups = 50-residue position blocks, so neighbouring
variants never straddle train and test (avoids positional leakage).
Compared feature sets: ESM only, stability, PPI, all, plus a transparent rule score.

Output: data/processed/predictions.csv, data/processed/cv_metrics.csv
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from common import PROCESSED

FEATURE_SETS = {
    "esm_only": ["esm_llr"],
    "stability": ["esm_llr", "blosum62", "d_hydro", "d_vol", "plddt", "rsa_af", "stability_risk"],
    "ppi": ["ppi_risk_max", "ppi_risk_weighted", "n_partners_hit", "any_hotspot"],
    "all": ["esm_llr", "blosum62", "d_hydro", "d_vol", "plddt", "rsa_af", "stability_risk",
            "ppi_risk_max", "ppi_risk_weighted", "n_partners_hit", "any_hotspot"],
}
P_LOF, P_NEUTRAL = 0.70, 0.30  # class thresholds on calibrated-ish probability


def model():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                          class_weight="balanced", random_state=0)


def rule_score(df: pd.DataFrame) -> pd.Series:
    """Transparent score: ESM conservation + structural/PPI evidence."""
    s = -df["esm_llr"] / 5.0                        # ~0-2: evolutionary intolerance
    s += df["stability_risk"].fillna(0) * 0.5       # buried + large change (structured only)
    s += df["ppi_risk_max"] * 0.5                   # interface contact
    s += df["any_hotspot"].astype(float) * 1.0      # pSer-binding hotspot
    return s


def cv_predict(X, y, groups, n_splits=5):
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits).split(X, y, groups):
        oof[te] = model().fit(X.iloc[tr], y.iloc[tr]).predict_proba(X.iloc[te])[:, 1]
    return oof


def assign_class(p, flag_stabilising=None):
    cls = np.where(p >= P_LOF, "LoF", np.where(p <= P_NEUTRAL, "neutral", "hypomorphic/uncertain"))
    return cls


def main():
    df = pd.read_csv(PROCESSED / "ppi_features.csv")
    df["any_hotspot"] = df["any_hotspot"].astype(float)
    df["rule_score"] = rule_score(df)

    lab = df[df["label"].isin(["LoF", "neutral"])].reset_index(drop=True)
    y = (lab["label"] == "LoF").astype(int)
    groups = lab["pos"] // 50
    print(f"labelled: {len(lab)} (LoF {y.sum()}, neutral {(1 - y).sum()}); {groups.nunique()} position blocks")

    metrics = []
    oof_all = None
    for name, cols in FEATURE_SETS.items():
        oof = cv_predict(lab[cols], y, groups)
        metrics.append({"model": name, "AUROC": roc_auc_score(y, oof),
                        "AUPRC": average_precision_score(y, oof), "Brier": brier_score_loss(y, oof)})
        if name == "all":
            oof_all = oof
    metrics.append({"model": "rule_score", "AUROC": roc_auc_score(y, lab["rule_score"]),
                    "AUPRC": average_precision_score(y, lab["rule_score"]), "Brier": np.nan})
    m = pd.DataFrame(metrics)
    m.to_csv(PROCESSED / "cv_metrics.csv", index=False)
    print(m.round(3).to_string(index=False))

    # per-domain performance of the full model (out-of-fold)
    lab["oof"] = oof_all
    for dom, g in lab.groupby("domain"):
        if g["label"].nunique() == 2:
            print(f"  {dom:20s} n={len(g):4d} LoF={int((g['label']=='LoF').sum()):3d} "
                  f"AUROC {roc_auc_score((g['label']=='LoF').astype(int), g['oof']):.3f}")

    # final model on all labelled data -> predict everything; use OOF for labelled variants
    cols = FEATURE_SETS["all"]
    final = model().fit(lab[cols], y)
    df["p_lof"] = final.predict_proba(df[cols])[:, 1]
    df.loc[df["variant"].isin(lab["variant"]), "p_lof"] = df["variant"].map(
        dict(zip(lab["variant"], oof_all))).dropna()
    df["pred_class"] = assign_class(df["p_lof"].values)
    # "stabilising-binding" flag is not available without binding ddG (see README)
    df.to_csv(PROCESSED / "predictions.csv", index=False)

    print("\nPredicted class by ClinVar label:")
    print(pd.crosstab(df["label"], df["pred_class"]))

    imp = permutation_importance(final, lab[cols], y, scoring="roc_auc", n_repeats=10, random_state=0)
    print("\nPermutation importance (AUROC drop):")
    print(pd.Series(imp.importances_mean, index=cols).sort_values(ascending=False).round(3).to_string())


if __name__ == "__main__":
    main()
