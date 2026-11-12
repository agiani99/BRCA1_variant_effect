https://www.uniprot.org/uniprotkb/P38398/entry#disease_variants

using the variants here summarized I wonder whether it could be possible to setup a workflow
to predict whther BRCA1 human mutations on this protein can be activating, neutral or deactivating (loof of function)
depending on the primary PPI interactions with other proteins taken from STRINGS.
The predictions should be based on the a) predicted stability of BRCA1 and b) on the possibility that PPI interactions are modulated by single point mutations.
Any idea for me?

# Plan: BRCA1 variant effect workflow (Python)

I read `BRCA1_mutants.md`. It asks for a workflow that classifies human BRCA1 (UniProt P38398) missense variants as **activating, neutral or loss-of-function**. The call should rest on (a) predicted stability of BRCA1 and (b) whether the variant disrupts PPIs with its main STRING partners.

## Caveats
- **"Activating" is rare for BRCA1.** It is a tumour suppressor, so almost all functional variants are neutral or LoF. I suggest a 3-class scheme of **LoF / hypomorphic-uncertain / neutral**, with "activating" kept only as a flag for variants that stabilise or strengthen an interaction (a computed ΔΔG, not a validated gain of function).
- **BRCA1 is about 85% disordered.** Stability and interface modelling are only reliable in three regions:
  - RING domain (aa 1–109), which binds BARD1.
  - Coiled-coil (aa ~1391–1424), which binds PALB2.
  - BRCT tandem (aa 1646–1859), which binds phosphopeptides (ABRAXAS1, BRIP1, CtIP).
  
  Variants elsewhere need sequence-based predictors only.
- **Ground truth already exists.** Use the saturation genome editing (SGE) data (Findlay 2018, which covers RING and BRCT exons) and ClinVar for training and validation. Without them the score is unvalidated.

## Phases

**1. Variant collection** (`requests`, `pandas`)
- Pull the UniProt P38398 variant features through the REST API.
- Add ClinVar (via the NCBI E-utilities or the ClinVar VCF) and gnomAD allele frequencies.
- Add SGE functional scores as labels.
- Normalise everything to HGVS protein notation and one table: `variant, pos, wt, mut, label, source`.

**2. Interactome from STRING** (`requests` to the STRING API, `networkx`)
- Query `interaction_partners` for BRCA1 (9606) with a combined score ≥ 700, experimental and database evidence only.
- Expected top partners: BARD1, PALB2, BRIP1, RBBP8, ABRAXAS1, UBE2D3, NBN, TOPBP1, ATM, CHEK2, TP53.
- For each partner, find the interface region from PDB or UniProt binding annotations.

**3. Structures** (`biopython`, `freesasa`, `prody`)
- Use the AlphaFold model of P38398, filtered to pLDDT > 70.
- Use experimental complexes where available: RING–BARD1 (1JM7), BRCT with phosphopeptide (1T15, 1JNX).
- For the other partners, predict complexes with ColabFold (AlphaFold-Multimer), run separately on a GPU.
- Compute per-residue relative solvent accessibility and interface residues (distance cutoff 5 Å).

**4. Component (a): stability ΔΔG**
- Run FoldX `BuildModel` or Rosetta `cartesian_ddg`, driven from Python with `subprocess`. FoldX needs an academic licence.
- As fast ML alternatives, use ThermoMPNN or DDGun (both pip-installable).
- Add ESM-1v and ESM-1b log-likelihood ratios (`fair-esm`, `torch`) as a sequence-based fallback for disordered regions.
- Optionally add AlphaMissense and EVE scores as features.

**5. Component (b): PPI modulation**
- For variants at interface residues, compute FoldX `AnalyseComplex` ΔΔG_binding per partner complex.
- Add interface features: burial, change in hydrophobicity, charge and size, and phospho-peptide contact loss for BRCT.
- Add a weight per partner (the STRING score, plus a gene-level importance such as BARD1 as essential for the RING function), then a network-aware score.
- Compute a weighted sum of ΔΔG_binding over partners, and the number of partners likely disrupted.

**6. Classifier** (`scikit-learn`, `xgboost`, `shap`)
- Start with a transparent rule-based score:
  - LoF if ΔΔG_fold > ~2 kcal/mol, or ΔΔG_bind > ~1.5 at a high-confidence partner.
  - Neutral if both are small.
  - Hypomorphic in between.
  - "Activating" flag if the variant stabilises binding.
- Then train gradient boosting on the SGE and ClinVar labels.
- Cross-validate by domain or position, not by random split, to avoid leakage.
- Report AUROC, AUPRC and calibration, and explain features with SHAP.
- Benchmark against AlphaMissense, REVEL and CADD to show the added value of the PPI component.

**7. Outputs** (`matplotlib`, `plotly`, optional `streamlit`)
- A per-variant table, a BRCA1 lollipop plot coloured by class, and a 3D view of the interfaces with predicted disruptions.
- Optionally a small Streamlit app that takes a variant and returns the class and its evidence.

## Layout and tools
```
brca1/
  data/ (raw, processed)   src/{fetch,string_net,structure,stability,ppi,model}.py
  notebooks/   environment.yml   Snakefile (or Makefile)
```
Environment: conda with Python 3.11, `pandas`, `biopython`, `networkx`, `scikit-learn`, `xgboost`, `torch`, `fair-esm`, `shap`, `freesasa`. External tools: FoldX or Rosetta, ColabFold. Use Snakemake to keep it reproducible.

## Suggested order
1. Data and STRING (phases 1–2): about 1–2 days.
2. ESM and structure-based ΔΔG for the RING and BRCT domains (phases 3–4).
3. PPI ΔΔG for BARD1 and the BRCT phosphopeptides, where experimental structures exist (phase 5).
4. Rule-based score, validated against SGE (phase 6).
5. Extend to the other partners and to ML.

*Now a test run.*

The first form works (6,506 records). It needs quoted phrase and capital "Molecular".

I scaffolded the BRCA1 project in the `BRCA1` folder and ran phases 1–2 end to end. Both scripts work.

**Files**
- [environment.yml](../environment.yml): conda environment (Python 3.11, pandas, biopython, networkx, scikit-learn, xgboost, ESM, and so on).
- [README.md](../README.md): setup, how to run, and the roadmap.
- [src/common.py](common.py): shared paths, the HTTP session with retries, amino-acid code conversion, and the domain boundaries (RING, coiled-coil, BRCT).
- [src/fetch_variants.py](fetch_variants.py): pulls variants from UniProt and ClinVar and writes `data/processed/variants.csv`.
- [src/string_net.py](string_net.py): pulls BRCA1 partners from STRING and writes `string_partners.csv` and `string_network.gml`.

**Results of the test run**

| Source | Count |
|---|---|
| UniProt single-residue variants | 166 |
| ClinVar missense variants | 6,022 |
| Merged variants | 6,030 |

- **Labels:** pathogenic and likely pathogenic count as LoF, benign and likely benign as neutral, and uncertain or conflicting as VUS.
- **Training labels:** the usable labelled set is small. It has about 220 LoF variants and about 460 neutral ones, and most of the rest are VUS. This is why the SGE functional data matters.
- **BRCT domain:** it has 556 variants marked "unknown", meaning ClinVar gave no usable classification. I haven't looked at what those entries say.
- **STRING partners:** 50 partners came back, including all the expected ones: BARD1, PALB2, BRIP1, RBBP8, ABRAXAS1, BRCA2 and RAD51.
  - **Score ranking:** at the 700 cutoff, nearly all scores are about 0.999, so STRING's combined score does not rank them. For weighting partners in phase 5, use the evidence channels and the known-interface list instead.

**Fixes made along the way**
- The ClinVar query needed the phrase `"missense variant"[Molecular consequence]` in quotes and with a capital M. My first version returned zero records.
- networkx's GraphML writer fails with NumPy 2 in your Anaconda install, so the network is saved as GML.

The next step is phase 3: downloading the structures, the AlphaFold model of P38398 and the PDB complexes (1JM7, 1T15, 1JNX), and computing solvent accessibility and interface residues.

Phase 3 is done. I added [src/structures.py](structures.py), which downloads the structures, computes per-residue features and finds the interface residues. It ran end to end.

**What it downloads (into `data/raw/structures/`)**
- The AlphaFold model of BRCA1 (P38398).
- PDB 1JM7: the BRCA1 RING domain bound to BARD1.
- PDB 1T15: the BRCT domain bound to a BRIP1/BACH1 phosphopeptide.

**What it writes (into `data/processed/`)**
- `residue_features.csv`: for all 1,863 residues, the AlphaFold confidence (pLDDT) and relative solvent accessibility. Only 324 residues have pLDDT ≥ 70. The RING domain averages about 81 and the BRCT domain about 91, which matches the plan to model only those regions.
- `interface_residues.csv`: for each complex, which BRCA1 residues contact the partner (within 5 Å) or lose at least 10% of their solvent accessibility on binding.

**Checks**
- **BRCT/phosphopeptide interface:** 22 residues, including the known phosphate-binding residues S1655, R1699 and K1702.
- **RING/BARD1 interface:** 39 residues, mostly in the two helical regions at about 3–42 and 72–97.
- **ClinVar variants at these interfaces:**

| Interface | LoF | VUS | Neutral | Unknown |
|---|---|---|---|---|
| BARD1 | 20 | 92 | 11 | 108 |
| BRIP1 phosphopeptide | 17 | 43 | 1 | 58 |

  The few neutral variants relative to LoF fits interface residues being functionally important.

**Corrections to earlier plan**
- **1JNX:** this entry holds only the BRCT domain, with no peptide, so it isn't a complex. I removed it from the interface step.
- **Phosphoserine:** my first version ignored it, which would have missed the key phosphate contacts. That is fixed.
- **Numbering:** residue numbering in 1JM7 and 1T15 matches UniProt, so no offset is needed.

**Still missing**
- PALB2 (coiled-coil) and all the other STRING partners have no interface data yet. They need ColabFold complex predictions, which is the GPU step in phase 5.

Phase 4 is done. I added [src/stability.py](stability.py), and it writes `data/processed/stability_features.csv` for all 6,030 variants.

**What each variant gets**
- **ESM-2 score:** a log-likelihood ratio from ESM-2 35M, which you chose. Long sequences are handled in three overlapping windows, since the model takes at most about 1,000 residues.
- **Substitution features:** BLOSUM62 score and the changes in hydrophobicity and volume.
- **Structural context:** relative solvent accessibility and pLDDT from the AlphaFold model.
- **`stability_risk`:** a crude heuristic, defined only where pLDDT ≥ 70. It is burial multiplied by the size of the physicochemical change.

**How well it separates ClinVar LoF from neutral variants (AUROC)**

| Subset | AUROC | Variants |
|---|---|---|
| All | 0.85 | 671 |
| RING | 0.95 | 104 |
| BRCT | 0.81 | 213 |
| `stability_risk`, structured regions only | 0.82 | not counted |
| Coiled-coil | 0.78 | 6 (too few to trust) |
| Disordered/other | 0.42 | 354, only 8 of them LoF |

- The disordered/other result is not reliable, because there are almost no labelled LoF variants there.
- **Circularity:** ClinVar classifications partly use computational predictors, so these numbers are probably optimistic. The SGE functional data would be a cleaner test.

**Limitations**
- **No real ΔΔG:** `stability_risk` is a heuristic, not a free-energy change. FoldX needs a licence and isn't installed. ThermoMPNN is the free option for the RING and BRCT domains.
- **Model size:** ESM-2 35M is the small model. ESM-1v 650M would likely score better, and it is feasible on CPU with this windowing.

Phase 5 is done. I added [src/ppi.py](ppi.py), which writes `data/processed/ppi_features.csv` (one row per variant) and `ppi_variant_partner.csv` (one row per variant and partner).

**How the PPI effect is estimated**
- **BARD1:** from the RING structure (1JM7).
- **BRIP1, ABRAXAS1, RBBP8:** from the BRCT phosphopeptide structure (1T15), since all three bind the same groove.
- **PALB2:** from the coiled-coil region only (aa 1391–1424), with no structure, so it gets a lower-confidence weight.
- **Score:** each interface variant gets a risk from how much the partner buries the residue, a bonus for known phosphate-binding residues (S1655, R1699, K1702 and similar), and the size of the physicochemical change. Partners are weighted by their STRING evidence.
- **Not covered:** the other ~45 STRING partners have no structural information, so they contribute nothing.

**Results**
- 435 of 6,030 variants sit at a partner interface: 59 at BARD1 and 376 at the BRCT groove or PALB2 region. Of the 55 with a ClinVar label, 40 are LoF.
- Across all labelled variants the PPI score alone is weak (AUROC 0.58).
- Among the 55 labelled interface variants, ESM scores 0.92 and the PPI score 0.82. So the PPI proxy does not yet add information beyond ESM, but the sample is small.

**Limits**
- **No binding ΔΔG yet:** the score shows where a variant touches an interface and how big the change is. It cannot tell whether binding is weakened or strengthened, so there is no "activating" flag.
- **Next for PPI:** FoldX `AnalyseComplex`, or an AlphaFold-Multimer run for PALB2 and the other partners on a GPU, would replace this proxy.

Phase 6 is done, but the headline number needs a caveat. I added [src/classify.py](classify.py), which trains a gradient-boosting classifier on 677 ClinVar-labelled variants (219 LoF, 458 neutral). It validates with position-blocked cross-validation, so neighbouring variants never land in both train and test. It writes `predictions.csv` and `cv_metrics.csv`.

| Feature set | AUROC |
|---|---|
| ESM only | 0.82 |
| Stability features | 0.93 |
| PPI features | 0.58 |
| All features | 0.93 |
| Hand-weighted rule score (not trained) | 0.95 |

**Why 0.93 is misleading**
- **Label bias:** the ClinVar labels are strongly tied to region. The structured RING and BRCT domains are mostly LoF, and the 354 labelled disordered variants are almost all neutral, with only 8 LoF.
- **pLDDT as a shortcut:** the model uses pLDDT (the second most important feature) to tell "structured" from "disordered", which flatters the score.
- **Within-domain AUROC:** RING 0.89 and BRCT 0.85 are the honest numbers. Disordered/other is 0.28, which is worse than chance, so the model cannot score variants there.
- **PPI features:** they add essentially nothing on top of the others.
- **Rule score:** its weights were set by hand, but I had already seen earlier AUROCs when choosing them, so don't treat 0.95 as unbiased.

**Predicted classes** use thresholds of p ≥ 0.7 for LoF and ≤ 0.3 for neutral, with the rest "hypomorphic/uncertain".
- Of the 219 ClinVar LoF variants, 172 are called LoF and 28 neutral.
- Of the 458 neutral variants, 399 are called neutral and 21 LoF.
- Of the 4,471 VUS, 261 are called LoF, 245 uncertain and 3,965 neutral. The VUS calls in disordered regions are essentially a prior, not evidence.

**Missing from the plan**
- **Activating flag:** not implemented, since there is no binding ΔΔG.
- **SGE data:** the Findlay saturation genome editing data would give independent, less circular labels and would fix the disordered-region problem. I haven't added it yet.

Phase 7 is built, and all seven phases of the plan now have code. The Streamlit app has not been run yet, only compiled.

**What I added**
- [src/plots.py](src/plots.py) writes two figures to `results/`:
  - `lollipop.png` shows P(LoF) along the protein, coloured by predicted class, with the RING, coiled-coil and BRCT domains shaded.
  - `cv_metrics.png` compares the models by AUROC.
- [app.py](app.py) is a Streamlit explorer. Run it with `streamlit run app.py`. It has these features:
  - You can look up a variant (for example C61G) and see its class, P(LoF), ESM score, pLDDT, solvent accessibility and partner interfaces hit.
  - You can filter the full table by class, domain and ClinVar label, and download it as CSV.
  - Variants in disordered regions get a low-confidence warning.

**What the lollipop plot shows**
- Almost all predicted LoF (718 variants) and uncertain (624) calls fall inside the RING, coiled-coil and BRCT domains.
- Outside them the model calls essentially everything neutral (4,688 variants), with only a few exceptions near residues 785 and 1060. This is mostly the pLDDT shortcut from phase 6, not evidence of neutrality, so the plot is not a trustworthy map of the disordered 85% of BRCA1.

**What the workflow can and can't do now**
- **Reasonable:** ranking LoF versus neutral inside the RING and BRCT domains (blocked-CV AUROC about 0.85–0.89).
- **Not reliable:** calls in disordered regions, and any "activating" call, since there is no binding ΔΔG.

**Suggested improvements**
- **SGE data:** adding the Findlay saturation genome editing data would give independent labels and let the model learn disordered regions properly.
- **PPI structures:** FoldX or ThermoMPNN ΔΔG, and AlphaFold-Multimer complexes for PALB2 and the other partners, would replace the PPI proxy.
- **ESM model:** ESM-1v 650M is probably a better scorer than the 35M model I used.

I couldn't get the SGE data, so nothing has changed in the project. I also need to correct something I told you earlier.

**What went wrong**
- The Findlay 2018 SGE database at [sge.gs.washington.edu/BRCA1](https://sge.gs.washington.edu/BRCA1/) is an interactive app. From this PC it returned a server error (500) and showed no download link.
- The fetch agent also failed on DNS resolution for the same host.
- I don't know the exact file name or URL for the Nature supplementary table, and I didn't want to guess a download.

**Correction to what I said before**
- I said SGE would fix the disordered regions. It won't. The Findlay data covers only the RING and BRCT exons (exons 2–5 and 15–23), about 3,893 SNVs.
- SGE would still be useful as independent, less circular labels for those two domains, which is where the model already works best.
- Disordered regions stay unlabelled, and predictions there remain priors.

**What I need from you**
- Download Supplementary Table 1 from the Nature paper ([Findlay et al. 2018, PubMed 30209399](https://pubmed.ncbi.nlm.nih.gov/30209399/); the paper's data-availability section points to the table and the SGE site).
- Save it in `data/raw/`.
- Check the University of Washington's usage terms on the SGE site first, since the database says the data is their copyright.
- Once it's there, I'll write the loader and map SGE scores to protein variants (they are given as nucleotide changes, so I'll convert to amino-acid changes). Then I'll re-validate with it as the test set.

Sources: [Findlay et al. 2018 (PubMed)](https://pubmed.ncbi.nlm.nih.gov/30209399/), [SGE BRCA1 database](https://sge.gs.washington.edu/BRCA1/).