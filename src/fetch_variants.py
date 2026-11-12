"""Phase 1: collect BRCA1 missense variants from UniProt and ClinVar.

Outputs (data/processed/):
  uniprot_variants.csv, clinvar_variants.csv, variants.csv (merged)
"""
import re

import pandas as pd

from common import (DOMAINS, GENE, PROCESSED, RAW, THREE_TO_ONE, UNIPROT_ID,
                    domain_of, session)

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def fetch_uniprot() -> pd.DataFrame:
    url = f"https://rest.uniprot.org/uniprotkb/{UNIPROT_ID}.json"
    data = session().get(url, timeout=60).json()
    (RAW / f"{UNIPROT_ID}.json").write_text(
        __import__("json").dumps(data), encoding="utf-8")

    rows = []
    for f in data.get("features", []):
        if f["type"] != "Natural variant":
            continue
        loc = f["location"]
        start, end = loc["start"]["value"], loc["end"]["value"]
        alt = f.get("alternativeSequence", {})
        wt = alt.get("originalSequence", "")
        muts = alt.get("alternativeSequences", [])
        if start != end or len(wt) != 1 or not muts:
            continue  # keep single-residue substitutions only
        for m in muts:
            if len(m) != 1:
                continue
            rows.append({
                "variant": f"{wt}{start}{m}",
                "pos": start, "wt": wt, "mut": m,
                "uniprot_id": f.get("featureId", ""),
                "uniprot_desc": f.get("description", ""),
                "source": "uniprot",
            })
    return pd.DataFrame(rows)


def fetch_clinvar(max_records: int = 20000) -> pd.DataFrame:
    s = session()
    term = f'{GENE}[gene] AND "missense variant"[Molecular consequence]'
    r = s.get(f"{EUTILS}/esearch.fcgi", params={
        "db": "clinvar", "term": term, "retmax": max_records, "retmode": "json"},
        timeout=60).json()
    ids = r["esearchresult"]["idlist"]
    print(f"ClinVar: {len(ids)} records")

    pat = re.compile(r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})")
    rows = []
    for i in range(0, len(ids), 200):
        chunk = ids[i:i + 200]
        res = s.get(f"{EUTILS}/esummary.fcgi", params={
            "db": "clinvar", "id": ",".join(chunk), "retmode": "json"},
            timeout=120).json()["result"]
        for uid in res.get("uids", []):
            rec = res[uid]
            m = pat.search(rec.get("title", ""))
            if not m or m.group(1) not in THREE_TO_ONE or m.group(3) not in THREE_TO_ONE:
                continue
            wt, pos, mut = THREE_TO_ONE[m.group(1)], int(m.group(2)), THREE_TO_ONE[m.group(3)]
            sig = (rec.get("germline_classification") or rec.get("clinical_significance") or {})
            rows.append({
                "variant": f"{wt}{pos}{mut}", "pos": pos, "wt": wt, "mut": mut,
                "clinvar_id": uid,
                "clinvar_sig": sig.get("description", ""),
                "clinvar_review": sig.get("review_status", ""),
                "source": "clinvar",
            })
    return pd.DataFrame(rows)


def simple_label(sig: str) -> str:
    s = (sig or "").lower()
    if "conflicting" in s or "uncertain" in s:
        return "VUS"
    if "pathogenic" in s:
        return "LoF"
    if "benign" in s:
        return "neutral"
    return "unknown"


def main():
    up = fetch_uniprot()
    cv = fetch_clinvar()
    up.to_csv(PROCESSED / "uniprot_variants.csv", index=False)
    cv.to_csv(PROCESSED / "clinvar_variants.csv", index=False)

    cv["label"] = cv["clinvar_sig"].map(simple_label)
    # one row per variant; keep the strongest review status if duplicated
    cv = cv.sort_values("clinvar_review").drop_duplicates("variant", keep="last")
    merged = cv.merge(up[["variant", "uniprot_id", "uniprot_desc"]], on="variant", how="outer")
    parts = merged["variant"].str.extract(r"^(?P<wt>[A-Z])(?P<pos>\d+)(?P<mut>[A-Z])$")
    for col in ("pos", "wt", "mut"):
        merged[col] = merged[col].fillna(parts[col])
    merged["pos"] = merged["pos"].astype(int)
    merged["domain"] = merged["pos"].map(domain_of)
    merged["label"] = merged["label"].fillna("unknown")
    merged.drop(columns=["source"], errors="ignore").to_csv(PROCESSED / "variants.csv", index=False)

    print(f"UniProt: {len(up)}  ClinVar: {len(cv)}  merged: {len(merged)}")
    print(merged.groupby(["domain", "label"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
