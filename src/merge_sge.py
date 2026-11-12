"""Add SGE (Findlay 2018) evidence to predictions.csv. Idempotent; rerun after classify.py.
Adds: sge_score, sge_class, final_call, call_source (SGE measurement beats model prediction)."""
import pandas as pd

from common import PROCESSED

SGE_TO_CALL = {"LOF": "LoF", "FUNC": "neutral", "INT": "hypomorphic/uncertain"}


def main():
    p = pd.read_csv(PROCESSED / "predictions.csv")
    p = p.drop(columns=[c for c in ("sge_score", "sge_class", "final_call", "call_source") if c in p])
    s = pd.read_csv(PROCESSED / "sge_variants.csv")[["variant", "sge_score", "sge_class"]]
    p = p.merge(s, on="variant", how="left")
    has = p["sge_class"].notna()
    p["final_call"] = p["pred_class"]
    p.loc[has, "final_call"] = p.loc[has, "sge_class"].map(SGE_TO_CALL)
    p["call_source"] = has.map({True: "SGE", False: "model"})
    p.to_csv(PROCESSED / "predictions.csv", index=False)
    print(f"{has.sum()} of {len(p)} variants have SGE measurements")
    print(pd.crosstab(p["call_source"], p["final_call"]))
    vus = p[(p["label"] == "VUS") & has]
    print(f"\nClinVar VUS resolved by SGE: {len(vus)}")
    print(vus["final_call"].value_counts().to_string())
    dis = p[(p["pred_class"] != p["final_call"]) & has]
    print(f"\nSGE overrides model call in {len(dis)} variants")

if __name__ == "__main__":
    main()
