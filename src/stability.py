"""Phase 4: stability / variant-effect features per variant.

FoldX/Rosetta are not available here, so stability is approximated by
  * ESM-2 wild-type-marginal log-likelihood ratio (sequence-based, whole protein)
  * BLOSUM62, hydrophobicity and volume change
  * structural context: relative SASA and pLDDT from the AlphaFold model
and a simple combined `stability_risk` heuristic (to be replaced by FoldX /
ThermoMPNN ddG in structured regions when available).

Usage: python src/stability.py [hf_model_name]
Output: data/processed/stability_features.csv
"""
import json
import sys

import numpy as np
import pandas as pd
import torch
from Bio.Align import substitution_matrices
from transformers import AutoTokenizer, EsmForMaskedLM

from common import PROCESSED, RAW, UNIPROT_ID

MODEL = sys.argv[1] if len(sys.argv) > 1 else "facebook/esm2_t12_35M_UR50D"
WINDOW, STRIDE = 1020, 510  # residues per window (ESM max 1022 tokens incl. cls/eos)

KD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5, "G": -0.4,
      "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8,
      "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2}  # Kyte-Doolittle
VOL = {"A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5, "Q": 143.8, "E": 138.4,
       "G": 60.1, "H": 153.2, "I": 166.7, "L": 166.7, "K": 168.6, "M": 162.9, "F": 189.9,
       "P": 112.7, "S": 89.0, "T": 116.1, "W": 227.8, "Y": 193.6, "V": 140.0}  # A^3


def protein_sequence() -> str:
    return json.loads((RAW / f"{UNIPROT_ID}.json").read_text(encoding="utf-8"))["sequence"]["value"]


@torch.no_grad()
def esm_log_probs(seq: str) -> np.ndarray:
    """(len(seq), 20-ish vocab) log-probs, each position taken from the window
    in which it is most central (wild-type-marginal scoring)."""
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = EsmForMaskedLM.from_pretrained(MODEL).eval()
    n = len(seq)
    starts = sorted(set(list(range(0, max(n - WINDOW, 0) + 1, STRIDE)) + [max(n - WINDOW, 0)]))
    out = np.full((n, model.config.vocab_size), np.nan, dtype=np.float32)
    best_dist = np.full(n, np.inf)
    for s in starts:
        e = min(s + WINDOW, n)
        enc = tok(seq[s:e], return_tensors="pt")
        lp = torch.log_softmax(model(**enc).logits[0], dim=-1)[1:-1].numpy()  # drop cls/eos
        centre = (s + e) / 2
        for i in range(s, e):
            d = abs(i - centre)
            if d < best_dist[i]:
                best_dist[i], out[i] = d, lp[i - s]
        print(f"window {s}-{e} done", flush=True)
    esm_log_probs.vocab = tok.get_vocab()
    return out


def main():
    seq = protein_sequence()
    variants = pd.read_csv(PROCESSED / "variants.csv")
    feats = pd.read_csv(PROCESSED / "residue_features.csv")[["pos", "plddt", "rsa_af"]]
    lp = esm_log_probs(seq)
    vocab = esm_log_probs.vocab
    blosum = substitution_matrices.load("BLOSUM62")

    bad = variants[[seq[p - 1] != w for p, w in zip(variants["pos"], variants["wt"])]]
    if len(bad):
        print(f"WARNING: {len(bad)} variants with wt mismatch dropped")
        variants = variants.drop(bad.index)

    df = variants[["variant", "pos", "wt", "mut", "domain", "label"]].copy()
    df["esm_llr"] = [lp[p - 1, vocab[m]] - lp[p - 1, vocab[w]]
                     for p, w, m in zip(df["pos"], df["wt"], df["mut"])]
    df["blosum62"] = [blosum[w][m] for w, m in zip(df["wt"], df["mut"])]
    df["d_hydro"] = [KD[m] - KD[w] for w, m in zip(df["wt"], df["mut"])]
    df["d_vol"] = [VOL[m] - VOL[w] for w, m in zip(df["wt"], df["mut"])]
    df = df.merge(feats, on="pos", how="left")

    # crude stability-risk heuristic, only meaningful where the model is confident
    # (pLDDT>=70) and the residue is buried: buried + large physicochemical change
    burial = (1 - df["rsa_af"].clip(0, 1))
    change = df["d_vol"].abs() / 100 + df["d_hydro"].abs() / 9 + (4 - df["blosum62"]).clip(lower=0) / 8
    df["stability_risk"] = np.where(df["plddt"] >= 70, burial * change, np.nan)
    df["structured"] = df["plddt"] >= 70

    df.to_csv(PROCESSED / "stability_features.csv", index=False)
    print(df.describe(include="all").T[["count", "mean"]].head(12))

    lab = df[df["label"].isin(["LoF", "neutral"])]
    try:
        from sklearn.metrics import roc_auc_score
        y = (lab["label"] == "LoF").astype(int)
        print(f"AUROC (ClinVar LoF vs neutral) esm_llr: {roc_auc_score(y, -lab['esm_llr']):.3f}")
        for dom, g in lab.groupby("domain"):
            if g["label"].nunique() == 2:
                print(f"  {dom:20s} n={len(g):4d}  AUROC {roc_auc_score((g['label']=='LoF').astype(int), -g['esm_llr']):.3f}")
        s = lab[lab["structured"]]
        print(f"AUROC structured stability_risk: {roc_auc_score((s['label']=='LoF').astype(int), s['stability_risk']):.3f}")
    except Exception as exc:  # pragma: no cover
        print("AUROC skipped:", exc)


if __name__ == "__main__":
    main()
