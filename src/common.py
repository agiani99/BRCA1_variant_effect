"""Shared helpers: paths, HTTP session, amino-acid code conversion."""
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
for _d in (RAW, PROCESSED):
    _d.mkdir(parents=True, exist_ok=True)

UNIPROT_ID = "P38398"  # BRCA1_HUMAN
GENE = "BRCA1"
TAXON = 9606

THREE_TO_ONE = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
}

# Structured BRCA1 regions where stability / interface modelling is meaningful
DOMAINS = {
    "RING": (1, 109),
    "coiled_coil_PALB2": (1391, 1424),
    "BRCT": (1646, 1859),
}


def domain_of(pos: int) -> str:
    for name, (lo, hi) in DOMAINS.items():
        if lo <= pos <= hi:
            return name
    return "disordered/other"


def session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=5, backoff_factor=1.0, status_forcelist=(429, 500, 502, 503, 504))
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers["User-Agent"] = "brca1-variant-workflow/0.1"
    return s
