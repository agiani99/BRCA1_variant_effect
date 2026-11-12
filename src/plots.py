"""Phase 7a: static figures. Output: results/lollipop.png, results/cv_metrics.png"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import DOMAINS, PROCESSED, ROOT

CALL = "final_call"  # SGE measurement where available, else model class
COLORS = {"LoF": "#d62728", "hypomorphic/uncertain": "#ff9f1c", "neutral": "#2a9d8f"}


def lollipop(df: pd.DataFrame, out):
    fig, ax = plt.subplots(figsize=(14, 4.5))
    for name, (lo, hi) in DOMAINS.items():
        ax.axvspan(lo, hi, color="grey", alpha=0.15)
        ax.text((lo + hi) / 2, 1.04, name.replace("_", " "), ha="center", fontsize=8)
    for cls, c in COLORS.items():
        s = df[df[CALL] == cls]
        ax.vlines(s["pos"], 0, s["p_lof"], color=c, alpha=0.25, lw=0.6)
        sg = s[s.get("call_source", "model") == "SGE"] if "call_source" in s else s.iloc[0:0]
        ax.scatter(s["pos"], s["p_lof"], s=8, color=c, label=f"{cls} (n={len(s)}, {len(sg)} SGE-measured)")
        ax.scatter(sg["pos"], sg["p_lof"], s=22, facecolors="none", edgecolors="k", lw=0.4)
    ax.axhline(0.7, ls="--", c="k", lw=0.5)
    ax.axhline(0.3, ls="--", c="k", lw=0.5)
    ax.set_xlim(0, 1863)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("BRCA1 residue")
    ax.set_ylabel("P(loss of function)")
    ax.set_title("BRCA1 missense variants: final call (circled = SGE-measured, else model prediction; "
                 "grey = RING/coiled-coil/BRCT; elsewhere low confidence)", fontsize=9)
    ax.legend(loc="upper center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.15))
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def metrics_bar(m: pd.DataFrame, out):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(m["model"], m["AUROC"], color="#457b9d")
    ax.set_ylim(0.5, 1)
    ax.set_ylabel("AUROC (blocked CV)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main():
    res = ROOT / "results"
    res.mkdir(exist_ok=True)
    lollipop(pd.read_csv(PROCESSED / "predictions.csv"), res / "lollipop.png")
    metrics_bar(pd.read_csv(PROCESSED / "cv_metrics.csv"), res / "cv_metrics.png")
    print("wrote", [p.name for p in res.iterdir()])


if __name__ == "__main__":
    main()
