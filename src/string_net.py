"""Phase 2: BRCA1 primary interactors from STRING (human, 9606).

Outputs (data/processed/): string_partners.csv, string_network.gml
Usage: python src/string_net.py [min_score=700] [limit=50]
"""
import sys

import networkx as nx
import pandas as pd

from common import GENE, PROCESSED, TAXON, session

API = "https://string-db.org/api/json"

# Known interface regions on BRCA1 (manual curation; extend after structure work)
KNOWN_INTERFACES = {
    "BARD1": "RING (1-109)",
    "PALB2": "coiled-coil (1391-1424)",
    "BRIP1": "BRCT (1646-1859), phospho-S990",
    "RBBP8": "BRCT (1646-1859), phospho-S327",
    "ABRAXAS1": "BRCT (1646-1859), phospho-S406",
}


def partners(min_score: int = 700, limit: int = 50) -> pd.DataFrame:
    r = session().get(f"{API}/interaction_partners", params={
        "identifiers": GENE, "species": TAXON, "required_score": min_score,
        "limit": limit, "caller_identity": "brca1-variant-workflow"}, timeout=60)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    # STRING returns scores on 0-1 in JSON
    df = df.rename(columns={"preferredName_B": "partner", "score": "combined",
                            "escore": "experimental", "dscore": "database"})
    keep = ["partner", "stringId_B", "combined", "experimental", "database", "tscore"]
    df = df[[c for c in keep if c in df.columns]]
    # keep only partners with experimental or curated-database evidence
    df = df[(df["experimental"] > 0) | (df["database"] > 0)]
    df["known_interface"] = df["partner"].map(KNOWN_INTERFACES).fillna("")
    return df.sort_values("combined", ascending=False).reset_index(drop=True)


def build_graph(df: pd.DataFrame) -> nx.Graph:
    g = nx.Graph()
    g.add_node(GENE)
    for _, r in df.iterrows():
        g.add_edge(GENE, r["partner"], weight=float(r["combined"]))
    return g


def main():
    min_score = int(sys.argv[1]) if len(sys.argv) > 1 else 700  # STRING scale 0-1000
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    df = partners(min_score, limit)
    df.to_csv(PROCESSED / "string_partners.csv", index=False)
    # GML avoids the numpy-2 incompatibility in some networkx graphml writers
    nx.write_gml(build_graph(df), PROCESSED / "string_network.gml")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
