"""Phase 7b: Streamlit explorer.  Run: streamlit run app.py"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from common import DOMAINS, PROCESSED, domain_of  # noqa: E402

st.set_page_config(page_title="BRCA1 variant explorer", layout="wide")
st.title("BRCA1 missense variant explorer")
st.caption("Predicted from ESM-2 and AlphaFold structural context; where SGE (Findlay 2018) measured the variant, that call is used. "
           "Research use only; not a clinical classification.")


@st.cache_data
def load():
    return pd.read_csv(PROCESSED / "predictions.csv").fillna({"partners_hit": ""})


df = load()
q = st.text_input("Look up a variant (e.g. C61G, M1775R)", "").strip().upper()
if q:
    hit = df[df["variant"] == q]
    if hit.empty:
        st.warning("Variant not in the table (only ClinVar/UniProt missense variants are included).")
    else:
        r = hit.iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Final call", r["final_call"], help=f"Source: {r['call_source']}; model class: {r['pred_class']}")
        c2.metric("P(LoF)", f"{r['p_lof']:.2f}")
        c3.metric("ClinVar", r["label"])
        if r["call_source"] == "SGE":
            st.info(f"Measured by saturation genome editing: {r['sge_class']} (score {r['sge_score']:.2f}).")
        if r["domain"] == "disordered/other":
            st.warning("Disordered region: structure/PPI modelling does not apply and the model "
                       "performed poorly here. Treat this call as low confidence.")
        st.write({"domain": r["domain"], "ESM log-likelihood ratio": round(r["esm_llr"], 2),
                  "pLDDT": round(r["plddt"], 1), "relative SASA": round(r["rsa_af"], 2),
                  "partner interfaces hit": r["partners_hit"] or "none",
                  "phospho-binding hotspot": bool(r["any_hotspot"])})

st.subheader("All variants")
c1, c2, c3 = st.columns(3)
classes = c1.multiselect("Final call", sorted(df["final_call"].unique()),
                         default=sorted(df["final_call"].unique()))
doms = c2.multiselect("Domain", sorted(df["domain"].unique()), default=sorted(df["domain"].unique()))
labels = c3.multiselect("ClinVar label", sorted(df["label"].unique()), default=sorted(df["label"].unique()))
view = df[df["final_call"].isin(classes) & df["domain"].isin(doms) & df["label"].isin(labels)]
st.write(f"{len(view)} variants")
st.scatter_chart(view, x="pos", y="p_lof", color="final_call", height=350)
st.dataframe(view[["variant", "domain", "label", "final_call", "call_source", "pred_class", "p_lof", "esm_llr", "plddt",
                   "rsa_af", "partners_hit"]].sort_values("p_lof", ascending=False),
             use_container_width=True, hide_index=True)
st.download_button("Download CSV", view.to_csv(index=False), "brca1_predictions.csv")
