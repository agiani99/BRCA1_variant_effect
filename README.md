# BRCA1 missense variant effect workflow

Classify BRCA1 (UniProt P38398) missense variants as LoF / hypomorphic / neutral from
(a) predicted stability (ESM-2 + AlphaFold/PDB structural context) and (b) PPI-interface
context with primary STRING partners. Where saturation genome editing (SGE, Findlay 2018)
measured a variant, that measurement overrides the model.

## Setup
```bash
conda env create -f environment.yml
conda activate brca1
```

## Pipeline (run in order)
```bash
python src/fetch_variants.py   # UniProt + ClinVar         -> data/processed/variants.csv
python src/string_net.py       # STRING partners           -> string_partners.csv, string_network.gml
python src/structures.py       # AlphaFold, 1JM7, 1T15     -> residue_features.csv, interface_residues.csv
python src/stability.py        # ESM-2 + structural feats  -> stability_features.csv
python src/ppi.py              # PPI proxy features        -> ppi_features.csv
python src/classify.py         # blocked-CV classifier     -> predictions.csv, cv_metrics.csv
python src/sge.py              # SGE validation (needs data/raw/41586_2018_461_MOESM3_ESM.xlsx)
python src/merge_sge.py        # adds sge_*, final_call, call_source to predictions.csv
python src/plots.py            # results/lollipop.png, results/cv_metrics.png
streamlit run app.py           # interactive explorer
```
`classify.py` rewrites `predictions.csv`, so rerun `merge_sge.py` after it.

Optional (not used by the classifier): `python src/ppi_ddg.py run --workers 6` then
`python src/ppi_ddg.py merge` computes OpenMM binding ΔΔG for interface variants.

## Results and limits
- **Validation against SGE** (LoF vs functional, AUROC): combined model 0.86, ESM alone 0.78.
  Structure-aware features add about 0.1 over ESM alone. Slightly optimistic because the model
  was trained on ClinVar labels of neighbouring residues.
- **Coverage:** SGE covers only RING (exons 2-5) and BRCT (exons 15-23): 1,831 missense
  variants, which override the model in `final_call` and resolve 653 ClinVar VUS
  (487 functional, 90 LoF, 76 intermediate). Everything else, notably the ~85% disordered
  part and the PALB2 coiled-coil, is a low-confidence model prior.
- **PPI modelling has no usable signal.** The interface proxy scores about 0.54 AUROC against
  SGE. OpenMM/CHARMM36/GBn2 binding ΔΔG (344 variants, RING-BARD1 and BRCT-phosphopeptide)
  scores 0.58 and the monomer energy change 0.48, so neither was added to the classifier.
  Most LoF variants disrupt the fold rather than the interface.
- **No "activating" call is made.** Nothing here can reliably identify gain-of-function.
- ClinVar labels are region-biased; the "all features" CV AUROC of 0.93 is inflated by a
  pLDDT shortcut (within-domain 0.85-0.89).
- Research use only; not a clinical classification.
