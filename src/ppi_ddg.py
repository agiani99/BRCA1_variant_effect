"""Phase 5b: physics-based binding ddG with OpenMM (CHARMM36 + GBn2 implicit solvent).

Protocol per variant (identical for the wild-type control of each site):
  1. PDBFixer: mutate the residue, rebuild side chain, drop waters/ions (phosphoserine is kept).
  2. Add hydrogens (pH 7; phosphoserine hydrogens from a small custom definition).
  3. Local minimisation on OpenCL: all heavy atoms are harmonically restrained except side chains
     of residues within 7 A of the mutated residue (and hydrogens), so only the neighbourhood relaxes.
  4. Energies (restraint term excluded): E_complex, E_BRCA1 (alone), E_partner (alone), evaluated on the
     minimised complex geometry (no re-relaxation of the unbound partners).
  5. dG_bind = E_complex - E_BRCA1 - E_partner ;  ddG_bind = dG_bind(mutant) - dG_bind(wild type)
     dE_mono = E_BRCA1(mutant) - E_BRCA1(wild type) is a crude monomer-stability proxy.
Zinc-coordinating residues of the RING are flagged and not modelled (zinc is removed).

Usage (long job, resumable - results are appended to data/processed/ddg/results.csv):
  python src/ppi_ddg.py run --workers 4                # all interface variants (default)
  python src/ppi_ddg.py run --workers 4 --all-structured   # every variant in the modelled residues
  python src/ppi_ddg.py run --limit 6 --workers 1     # quick test
  python src/ppi_ddg.py merge                         # aggregate + evaluate against SGE
"""
import argparse
import csv
import multiprocessing as mp
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd

from common import PROCESSED, RAW, THREE_TO_ONE

STRUCT = RAW / "structures"
OUT = PROCESSED / "ddg"
OUT.mkdir(exist_ok=True)
RESULTS = OUT / "results.csv"

ONE_TO_THREE = {v: k.upper() for k, v in THREE_TO_ONE.items()}
STANDARD = set(ONE_TO_THREE.values())
# pdb -> (BRCA1 chain, partner chain, partner label)
COMPLEXES = {"1JM7": ("A", "B", "BARD1"), "1T15": ("A", "B", "BRCT-pSer-peptide")}
SEP_H = """<Hydrogens><Residue name="SEP"><H name="HN" parent="N"/><H name="HA" parent="CA"/>
<H name="HB1" parent="CB"/><H name="HB2" parent="CB"/><H name="H3T" parent="O3P"/></Residue></Hydrogens>"""
NEAR = 7.0          # A, residues whose side chains may relax
K_RESTRAINT = 1000.0  # kJ/mol/nm^2
K_BACKBONE_NEAR = 250.0
FIELDS = ["key", "pdb", "pos", "variant", "wt", "mut", "status", "E_complex", "E_brca1", "E_partner",
          "dG_bind", "minutes"]

_ff = None


def init_worker():
    """Load the force field once per process."""
    global _ff
    import openmm.app as app
    hfile = OUT / "sep_hydrogens.xml"
    hfile.write_text(SEP_H)
    app.Modeller.loadHydrogenDefinitions(str(hfile))
    _ff = app.ForceField("charmm36_2024.xml", "implicit/gbn2.xml")


def zinc_ligands(pdb: str) -> set:
    """BRCA1/BARD1 residues with an atom within 3 A of a zinc ion."""
    from Bio.PDB import MMCIFParser, NeighborSearch
    model = MMCIFParser(QUIET=True).get_structure(pdb, str(STRUCT / f"{pdb}.cif"))[0]
    zn = [a for a in model.get_atoms() if a.element == "ZN"]
    atoms = [a for a in model.get_atoms() if a.element != "ZN"]
    if not zn:
        return set()
    ns = NeighborSearch(atoms)
    out = set()
    for z in zn:
        for a in ns.search(z.coord, 3.0):
            r = a.get_parent()
            out.add((r.get_parent().id, r.id[1]))
    return out


def clean_pdb(pdb: str):
    """mmCIF -> PDB copy (some mmCIF files carry '?' unit cells that PDBFixer cannot parse)."""
    out = STRUCT / f"{pdb}_clean.pdb"
    if not out.exists():
        from Bio.PDB import MMCIFParser, PDBIO
        s = MMCIFParser(QUIET=True).get_structure(pdb, str(STRUCT / f"{pdb}.cif"))
        io = PDBIO()
        io.set_structure(s)
        io.save(str(out) + f".{os.getpid()}.tmp")
        os.replace(str(out) + f".{os.getpid()}.tmp", out)  # atomic: safe with parallel workers
    return out


def connect_peptide_bonds(top, positions):
    """Add C(i)-N(i+1) bonds around modified residues (SEP), which PDB readers leave unconnected."""
    # intra-residue bonds of SEP from the force-field template (HETATM records carry no connectivity)
    tpl = _ff._templates["SEP"]
    names = [a.name for a in tpl.atoms]
    for r in top.residues():
        if r.name != "SEP":
            continue
        have = {a.name: a for a in r.atoms()}
        existing = {frozenset((b.atom1.index, b.atom2.index)) for b in top.bonds()}
        for i, j in tpl.bonds:
            a1, a2 = have.get(names[i]), have.get(names[j])
            if a1 is not None and a2 is not None and frozenset((a1.index, a2.index)) not in existing:
                top.addBond(a1, a2)
    for chain in top.chains():
        res = list(chain.residues())
        for r1, r2 in zip(res, res[1:]):
            if "SEP" not in (r1.name, r2.name):
                continue
            c = next((a for a in r1.atoms() if a.name == "C"), None)
            n = next((a for a in r2.atoms() if a.name == "N"), None)
            if c is None or n is None:
                continue
            dx = [positions[c.index][k] - positions[n.index][k] for k in range(3)]
            dist = sum(v.value_in_unit(v.unit) ** 2 for v in dx) ** 0.5  # nm
            already = any({b.atom1.index, b.atom2.index} == {c.index, n.index} for b in top.bonds())
            if dist < 0.2 and not already:
                top.addBond(c, n)


def build(pdb, chain, pos, mut):
    """Return a Modeller with hydrogens for (optionally mutated) complex."""
    import openmm.app as app
    from pdbfixer import PDBFixer
    fx = PDBFixer(filename=str(clean_pdb(pdb)))
    # Drop waters/ions/ligands first (keep phosphoserine) so the last protein residue of each chain
    # is recognised as terminal and gets its OXT; otherwise the CHARMM template cannot match.
    m0 = app.Modeller(fx.topology, fx.positions)
    m0.delete([r for r in m0.topology.residues() if r.name not in STANDARD and r.name != "SEP"])
    connect_peptide_bonds(m0.topology, m0.positions)  # HETATM phosphoserine gets no standard bonds
    fx.topology, fx.positions = m0.topology, m0.positions
    fx.findMissingResidues()
    fx.missingResidues = {}
    fx.findNonstandardResidues()
    fx.nonstandardResidues = []  # keep SEP
    if mut:
        wt3 = None
        for r in fx.topology.residues():
            if r.chain.id == chain and int(r.id) == pos:
                wt3 = r.name
        fx.applyMutations([f"{wt3}-{pos}-{ONE_TO_THREE[mut]}"], chain)
    fx.findMissingAtoms()
    fx.addMissingAtoms()
    mod = app.Modeller(fx.topology, fx.positions)
    mod.delete([r for r in mod.topology.residues() if r.name not in STANDARD and r.name != "SEP"])
    mod.addHydrogens(_ff, pH=7.0)
    return mod


def energy_group0(top, positions, platform, ff):
    import openmm as mm
    import openmm.app as app
    import openmm.unit as u
    system = ff.createSystem(top, nonbondedMethod=app.NoCutoff, constraints=None)
    ctx = mm.Context(system, mm.VerletIntegrator(0.001), platform)
    ctx.setPositions(positions)
    return ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)


def run_job(job):
    """job = (pdb, chain, pos, mut or None). Returns result dict."""
    import openmm as mm
    import openmm.app as app
    import openmm.unit as u
    pdb, chain, pos, mut = job
    brca1, partner, _ = COMPLEXES[pdb]
    t0 = time.time()
    res = {"key": f"{pdb}:{pos}:{mut or 'WT'}", "pdb": pdb, "pos": pos,
           "variant": "", "wt": "", "mut": mut or "WT", "status": "ok"}
    try:
        platform = mm.Platform.getPlatformByName(os.environ.get("DDG_PLATFORM", "OpenCL"))
        mod = build(pdb, chain, pos, mut)
        top, positions = mod.topology, mod.positions
        res["wt"] = next((THREE_TO_ONE.get(r.name.capitalize(), "?") for r in top.residues()
                          if r.chain.id == chain and int(r.id) == pos), "?") if not mut else ""

        # --- restraints: everything frozen except side chains near the mutated residue ---
        pos_nm = np.array([[p.x, p.y, p.z] for p in positions])
        resatoms = [(r, [a.index for a in r.atoms()]) for r in top.residues()]
        site = [i for r, idx in resatoms if r.chain.id == chain and int(r.id) == pos for i in idx]
        heavy = np.array([a.element is not None and a.element.symbol != "H" for a in top.atoms()])
        d = np.min(np.linalg.norm(pos_nm[:, None, :] - pos_nm[site][None, :, :], axis=2), axis=1)
        near_res = {r.index for r, idx in resatoms if np.min(d[idx]) < NEAR / 10}  # nm
        restr = mm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
        for name in ("k", "x0", "y0", "z0"):
            restr.addPerParticleParameter(name)
        backbone = {"N", "CA", "C", "O"}
        for a in top.atoms():
            if not heavy[a.index]:
                continue
            if a.residue.index in near_res:
                if a.name not in backbone:
                    continue
                k = K_BACKBONE_NEAR
            else:
                k = K_RESTRAINT
            restr.addParticle(a.index, [k, *pos_nm[a.index]])
        restr.setForceGroup(1)

        system = _ff.createSystem(top, nonbondedMethod=app.NoCutoff, constraints=None)
        system.addForce(restr)
        integ = mm.LangevinMiddleIntegrator(300 * u.kelvin, 1 / u.picosecond, 0.002 * u.picoseconds)
        sim = app.Simulation(top, system, integ, platform)
        sim.context.setPositions(positions)
        sim.minimizeEnergy(tolerance=2.0 * u.kilojoule_per_mole / u.nanometer, maxIterations=2000)
        st = sim.context.getState(getEnergy=True, getPositions=True, groups={0})
        e_complex = st.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
        minpos = st.getPositions()

        # --- unbound energies on the complex geometry ---
        energies = {}
        for label, drop in (("brca1", partner), ("partner", brca1)):
            m = app.Modeller(top, minpos)
            m.delete([r for r in m.topology.residues() if r.chain.id == drop])
            energies[label] = energy_group0(m.topology, m.positions, platform, _ff)
        res.update(E_complex=e_complex, E_brca1=energies["brca1"], E_partner=energies["partner"],
                   dG_bind=e_complex - energies["brca1"] - energies["partner"])
    except Exception as exc:  # keep long runs alive
        res["status"] = f"error: {str(exc)[:120]}"
        traceback.print_exc(file=sys.stderr)
    res["minutes"] = round((time.time() - t0) / 60, 2)
    return res


def make_jobs(all_structured: bool, limit: int | None):
    stab = pd.read_csv(PROCESSED / "stability_features.csv")
    ifc = pd.read_csv(PROCESSED / "interface_residues.csv")
    jobs, wt_sites = [], set()
    flagged = []
    for pdb, (chain, _, _) in COMPLEXES.items():
        zn = {p for c, p in zinc_ligands(pdb) if c == chain}
        sub = ifc[ifc["pdb"] == pdb]
        present = dict(zip(sub["pos"], sub["wt"]))
        if all_structured:
            positions = set(present)
        else:
            positions = set(sub[sub["interface"]]["pos"])
        v = stab[stab["pos"].isin(positions)]
        for r in v.itertuples():
            if present.get(r.pos) != r.wt:
                continue
            if r.pos in zn:
                flagged.append((pdb, r.pos, r.variant))
                continue
            jobs.append((pdb, chain, int(r.pos), r.mut))
            wt_sites.add((pdb, chain, int(r.pos)))
    muts = sorted(set(jobs), key=str)
    if limit:  # quick test: first `limit` mutants (spread over both complexes) + their WT controls
        muts = muts[::max(len(muts) // limit, 1)][:limit]
        wt_sites = {(p, c, pos) for p, c, pos, _ in muts}
    jobs = [(p, c, pos, None) for p, c, pos in sorted(wt_sites)] + muts
    if flagged:
        pd.DataFrame(flagged, columns=["pdb", "pos", "variant"]).to_csv(OUT / "zinc_ligand_variants.csv", index=False)
    return jobs


def cmd_run(args):
    jobs = make_jobs(args.all_structured, args.limit)
    done = set()
    if RESULTS.exists() and RESULTS.stat().st_size > 0:
        try:
            prev = pd.read_csv(RESULTS)
            done = set(prev.loc[prev["status"] == "ok", "key"])
        except (pd.errors.EmptyDataError, KeyError):
            print("results.csv unreadable (interrupted run?) - starting it afresh")
            RESULTS.unlink()
    elif RESULTS.exists():
        RESULTS.unlink()  # empty file left by an interrupted run
    todo = [j for j in jobs if f"{j[0]}:{j[2]}:{j[3] or 'WT'}" not in done]
    print(f"{len(jobs)} jobs, {len(done)} already done, {len(todo)} to run, {args.workers} workers", flush=True)
    new = not RESULTS.exists()
    t0 = time.time()
    with open(RESULTS, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
            fh.flush()
        with mp.get_context("spawn").Pool(args.workers, initializer=init_worker) as pool:
            for i, r in enumerate(pool.imap_unordered(run_job, todo), 1):
                w.writerow(r)
                fh.flush()
                el = (time.time() - t0) / 60
                print(f"[{i}/{len(todo)}] {r['key']} {r['status']} dG={r.get('dG_bind', float('nan')):.1f} "
                      f"({r['minutes']} min; elapsed {el:.0f} min, ETA {el / i * (len(todo) - i):.0f} min)", flush=True)


def cmd_merge(_):
    from sklearn.metrics import roc_auc_score
    r = pd.read_csv(RESULTS)
    r = r[r["status"] == "ok"]
    wt = r[r["mut"] == "WT"].set_index(["pdb", "pos"])[["dG_bind", "E_brca1"]]
    m = r[r["mut"] != "WT"].copy()
    m = m.join(wt, on=["pdb", "pos"], rsuffix="_wt")
    m["ddG_bind"] = m["dG_bind"] - m["dG_bind_wt"]
    m["dE_mono"] = m["E_brca1"] - m["E_brca1_wt"]
    ref = pd.read_csv(PROCESSED / "stability_features.csv")[["variant", "pos", "wt", "mut", "label"]]
    m = m.drop(columns=["variant", "wt"]).merge(ref, on=["pos", "mut"], how="left")
    m["ddG_bind_kcal"] = m["ddG_bind"] / 4.184
    m.to_csv(OUT / "ddg_variants.csv", index=False)
    print(f"{len(m)} variants with ddG; median |ddG_bind| = {m['ddG_bind_kcal'].abs().median():.2f} kcal/mol")

    sge_f = PROCESSED / "sge_variants.csv"
    if sge_f.exists():
        sge = pd.read_csv(sge_f)
        e = m.merge(sge[["variant", "sge_class"]], on="variant")
        e = e[e["sge_class"].isin(["LOF", "FUNC"])]
        y = (e["sge_class"] == "LOF").astype(int)
        print(f"Against SGE (n={len(e)}, LoF={int(y.sum())}):")
        for col in ("ddG_bind", "dE_mono"):
            clip = e[col].clip(-200, 200)  # damp clash outliers
            print(f"  {col:9s} AUROC {roc_auc_score(y, clip):.3f}")
            for pdb, g in e.groupby("pdb"):
                yy = (g["sge_class"] == "LOF").astype(int)
                if yy.nunique() == 2:
                    print(f"    {pdb}: n={len(g)} AUROC {roc_auc_score(yy, g[col].clip(-200, 200)):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--all-structured", action="store_true")
    p.set_defaults(fn=cmd_run)
    p2 = sub.add_parser("merge")
    p2.set_defaults(fn=cmd_merge)
    a = ap.parse_args()
    a.fn(a)
