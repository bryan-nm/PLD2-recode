"""The curated caption swaps: (protein, current caption, target caption) triples.

    python -m src.swaps build            # write data/swaps.jsonl from the SwissProt CSV
    python -m src.swaps check            # re-derive every target and verify it, no writes
    python -m src.swaps show sho1_sh3_to_pdz

EVERY TARGET IS GENERATED, NEVER TYPED. A swap is declared as an ordered list of literal
substitutions with an expected hit count, and the target caption is the result of applying them to
the real caption. `check` re-derives each target from the stored current caption and compares. A
hand-written target would be a second, silently divergent copy of a 1,500-character string, and the
one thing this whole experiment cannot afford is an uncontrolled difference between the two
captions: the contrast IS the experiment.

MINIMAL PAIRS. Everything outside the declared substitutions is byte-identical between current and
target. If the target were a freshly written caption, any measured movement could be a response to
phrasing, length, or vocabulary rather than to the property we meant to change.

THE TARGET PHRASING IS BORROWED FROM REAL ENTRIES, not invented. The FILIP text encoder saw
SwissProt's own conventions; a plausible-sounding string that SwissProt never writes is out of
distribution, and guidance toward it would measure the encoder's extrapolation rather than its
knowledge. Each swap records the accession(s) whose phrasing it copies. Two consequences worth
stating: there is no "PDZ domain" UniProt keyword (0 of 451,436 captions contain one), so the SH3
swap REMOVES the keyword rather than replacing it; and a catalytically dead peptidase S1 keeps its
GENE ONTOLOGY terms in real entries, because GO is assigned by homology, so the trypsin swap leaves
them alone.

EACH SWAP CARRIES ITS OWN ORACLE. Optimising a FILIP score and then measuring the FILIP score is
circular, so every swap names an INDEPENDENT sequence-level consequence that a tool knowing nothing
about FILIP can check. That is the column that decides whether an edit worked.

CONTROLS, generated for every swap:
  <id>__null    target == current. Any movement here is the machinery perturbing the protein for
                its own reasons, and sets the floor everything else must clear.
  <id>__decoy   target is a real caption from an unrelated protein. Separates "moved toward THIS
                caption" from "moved toward any caption that is not the current one".
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import sys

from config import SWISSPROT_COLS, SWISSPROT_CSV

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SWAPS_JSONL = os.path.join(DATA, "swaps.jsonl")

# A real, well-annotated protein with nothing in common with any of the five: an African swine
# fever virus membrane protein. Used as the decoy target.
DECOY_ACC = "P0C9F0"

# --------------------------------------------------------------------------------------
# The swaps. `edits` is [(literal to find, replacement, expected hit count)], applied in order.
# --------------------------------------------------------------------------------------
SWAPS = [
    dict(
        swap_id="sho1_sh3_to_pdz",
        acc="P40073",
        summary="Sho1 with a PDZ domain instead of an SH3 domain",
        # Sho1's SH3 domain is what recruits Pbs2; PDZ is a different peptide-binding fold with a
        # different ligand preference (C-terminal motifs rather than PxxP). The caption names the
        # domain in three places and the swap changes all three.
        phrasing_from=["O43464 / B3LVG7 (htra2: 'family names are pdz_6')"],
        oracle="Pfam/InterProScan on the edited sequence: does the SH3 match go and a PDZ match "
               "appear? SH3 and PDZ are both small beta-rich domains with strong HMM signatures, "
               "so this is a clean yes/no that knows nothing about FILIP.",
        edits=[
            ("family names are sh3_1", "family names are pdz_6", 1),
            # No UniProt keyword exists for PDZ (0 of 451,436 captions have one), so this removes
            # rather than replaces -- which is what a real PDZ-containing entry looks like.
            (", sh3 domain,", ",", 1),
            ("(via the sh3 domain)", "(via the pdz domain)", 1),
        ],
    ),
    dict(
        swap_id="lactamase_periplasm_to_cytosol",
        acc="P26918",
        summary="Beta-lactamase that lives in the cytosol instead of the periplasm",
        phrasing_from=["P0A9Q3 (arca: 'SUBCELLULAR LOCATION: cytoplasm', GO 'cytosol')"],
        oracle="SignalP (or an N-terminal hydrophobicity scan): a periplasmic beta-lactamase is "
               "exported by a cleaved N-terminal signal peptide and a cytosolic one has none. The "
               "'signal' keyword comes out of the caption, so the edited sequence should lose its "
               "predicted signal peptide. This is the most direct sequence consequence of the five.",
        edits=[
            ("SUBCELLULAR LOCATION: periplasm", "SUBCELLULAR LOCATION: cytoplasm", 1),
            ("GENE ONTOLOGY: periplasmic space,", "GENE ONTOLOGY: cytosol,", 1),
            # Keywords are alphabetical, so the insert and the deletions are separate edits that
            # each preserve the ordering.
            ("antibiotic resistance, hydrolase", "antibiotic resistance, cytoplasm, hydrolase", 1),
            ("metal-binding, periplasm, signal, zinc", "metal-binding, zinc", 1),
        ],
    ),
    dict(
        swap_id="hox2_dimer_to_monomer",
        acc="Q84U86",
        summary="HD-Zip transcription factor made a monomer by removing its leucine-zipper "
                "dimerization region",
        # The leucine zipper immediately C-terminal to the homeodomain IS this protein's
        # dimerization module -- a well-defined region, which is what makes it a fair test.
        phrasing_from=["generic homeobox entries ('belongs to the homeobox family.')",
                       "P26918 ('SUBUNIT: monomer.')"],
        oracle="Coiled-coil prediction (COILS/DeepCoil) or a leucine-heptad scan: the zipper is a "
               "heptad-repeat coiled coil, so removing it should remove the predicted coiled-coil "
               "segment while the homeodomain is left intact. Fold the edit and check the "
               "homeodomain still superposes on the original (TM to the N-terminal half).",
        edits=[
            ("PROTEIN NAME: homeobox-leucine zipper protein hox2",
             "PROTEIN NAME: homeobox protein hox2", 1),
            ("family names are halz, homeodomain", "family names are homeodomain", 1),
            ("SUBUNIT: homodimer. may form a heterodimer with hox1, hox3 or hox7.",
             "SUBUNIT: monomer.", 1),
            ("belongs to the hd-zip homeobox family. class ii subfamily.",
             "belongs to the homeobox family.", 1),
        ],
    ),
    dict(
        swap_id="arca_cytosol_to_membrane",
        acc="P0A9Q3",
        summary="Cytosolic response regulator ArcA made a membrane-anchored protein",
        # ArcA's own partner ArcB is an inner-membrane sensor kinase, so a membrane-anchored ArcA is
        # a construct a biologist would find plausible rather than absurd.
        phrasing_from=["A9IQI5 (atp synthase subunit b: 'cell inner membrane; single-pass "
                       "membrane protein', GO 'plasma membrane', the transmembrane keywords)"],
        oracle="Transmembrane-helix prediction (DeepTMHMM/Phobius, or a Kyte-Doolittle 19-residue "
               "window): a single-pass anchor is ~20 consecutive strongly hydrophobic residues, "
               "which is the most unambiguous sequence signature of the five. The original has "
               "none; the edit should acquire exactly one.",
        edits=[
            ("SUBCELLULAR LOCATION: cytoplasm",
             "SUBCELLULAR LOCATION: cell inner membrane; single-pass membrane protein", 1),
            ("GENE ONTOLOGY: cytosol,", "GENE ONTOLOGY: plasma membrane,", 1),
            ("KEYWORDS: activator, cytoplasm, dna-binding,",
             "KEYWORDS: activator, cell inner membrane, cell membrane, dna-binding,", 1),
            ("dna-binding, phosphoprotein", "dna-binding, membrane, phosphoprotein", 1),
            ("transcription regulation, two-component",
             "transcription regulation, transmembrane, transmembrane helix, two-component", 1),
        ],
    ),
    dict(
        swap_id="trypsin_catalytic_to_dead",
        acc="P00760",
        summary="Trypsin made non-catalytic, i.e. a serine protease homolog",
        # Modelled on a real catalytically dead member of the same family rather than invented:
        # SwissProt's convention is to drop ENZYME CLASS and CATALYTIC ACTIVITY, say so in
        # FUNCTION, and use the 'serine protease homolog' keyword -- while KEEPING the GO terms,
        # which are assigned by homology and survive in the real entries.
        phrasing_from=["A0A1I9KNP0 (vaa serine proteinase homolog 1: 'lacks enzymatic activity.', "
                       "keyword 'serine protease homolog', no ENZYME CLASS/CATALYTIC ACTIVITY)"],
        oracle="The peptidase S1 catalytic machinery is a motif scan away: the nucleophile sits in "
               "the near-invariant GDSGGP box, and the triad is His/Asp/Ser. Ask whether GDSGGP "
               "survives the edit, and whether the folded structure still places the three triad "
               "residues within catalytic distance. Both are computable and neither involves FILIP.",
        edits=[
            ("ENZYME CLASS: acting on peptide bonds (peptidases) ", "", 1),
            ("GENE: prss1 CATALYTIC ACTIVITY: preferential cleavage: arg-|-xaa, lys-|-xaa. ",
             "GENE: prss1 FUNCTION: lacks enzymatic activity. ", 1),
            ("disulfide bond, hydrolase, metal-binding, protease, reference proteome, secreted, "
             "serine protease, signal, zymogen",
             "disulfide bond, metal-binding, reference proteome, secreted, "
             "serine protease homolog, signal, zymogen", 1),
        ],
    ),
]


def apply_edits(caption: str, edits):
    """-> (edited caption, [(find, n_applied)]). Raises on a count that does not match."""
    out, log = caption, []
    for find, repl, want in edits:
        got = out.count(find)
        if got != want:
            raise SystemExit(
                f"edit {find!r} matched {got} time(s), expected {want}.\n"
                f"The caption has changed under this swap -- the substitution is anchored to the "
                f"exact SwissProt string, so a CSV update invalidates it rather than silently "
                f"editing the wrong span. Re-read the caption and re-anchor.")
        out = out.replace(find, repl)
        log.append([find, got])
    return out, log


def read_csv_rows(accs, csv_path=None):
    """-> {accession: {row, seq, caption}}. `row` is the CSV row index, which is the FILIP cache
    row: mini-embed's precompute writes pair_ids.json one accession per CSV row, in order."""
    csv_path = csv_path or SWISSPROT_CSV
    if not os.path.exists(csv_path):
        raise SystemExit(f"no SwissProt CSV at {csv_path} (config.SWISSPROT_CSV; "
                         f"override with PLD2_SWISSPROT_CSV)")
    id_col, prot_col, text_col = SWISSPROT_COLS
    want, out = set(accs), {}
    csv.field_size_limit(10 ** 9)
    with open(csv_path, newline="") as f:
        for i, r in enumerate(csv.DictReader(f)):
            a = r[id_col]
            if a in want:
                want.discard(a)
                out[a] = {"row": i, "seq": r[prot_col].strip().upper(),
                          "caption": r[text_col]}
                if not want:
                    break
    if want:
        raise SystemExit(f"not in {csv_path}: {sorted(want)}")
    return out


def build(csv_path=None, decoy=DECOY_ACC):
    accs = [s["acc"] for s in SWAPS] + [decoy]
    rows = read_csv_rows(accs, csv_path)
    dec = rows[decoy]
    out = []
    for spec in SWAPS:
        r = rows[spec["acc"]]
        target, log = apply_edits(r["caption"], spec["edits"])
        if target == r["caption"]:
            raise SystemExit(f"{spec['swap_id']}: the edits changed nothing")
        base = dict(accession=spec["acc"], cache_row=r["row"], length=len(r["seq"]),
                    sequence=r["seq"], caption_current=r["caption"])
        out.append(dict(base, swap_id=spec["swap_id"], kind="swap",
                        summary=spec["summary"], oracle=spec["oracle"],
                        phrasing_from=spec["phrasing_from"],
                        edits=[[f, t] for f, t, _ in spec["edits"]], edit_log=log,
                        caption_target=target))
        # The floor: if editing toward an IDENTICAL caption still moves the protein, the movement
        # is the machinery, not the caption.
        out.append(dict(base, swap_id=spec["swap_id"] + "__null", kind="control_null",
                        summary="CONTROL: target caption is identical to the current one",
                        oracle="No edit should be made, and the oracle should not move.",
                        phrasing_from=[], edits=[], edit_log=[],
                        caption_target=r["caption"]))
        # Separates "toward THIS caption" from "away from the current one".
        out.append(dict(base, swap_id=spec["swap_id"] + "__decoy", kind="control_decoy",
                        summary=f"CONTROL: target is an unrelated real caption ({decoy})",
                        oracle="Movement here is generic, not specific. The swap row must beat it.",
                        phrasing_from=[decoy], edits=[], edit_log=[],
                        caption_target=dec["caption"], decoy_accession=decoy,
                        decoy_cache_row=dec["row"]))
    return out


def write(records, path=SWAPS_JSONL):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def read(path=SWAPS_JSONL):
    if not os.path.exists(path):
        raise SystemExit(f"no {path}. Run `python -m src.swaps build` first.")
    return [json.loads(l) for l in open(path) if l.strip()]


FIELD_LABELS = ("PROTEIN NAME:", "ENZYME CLASS:", "GENE:", "FUNCTION:", "CATALYTIC ACTIVITY:",
                "COFACTOR:", "SUBCELLULAR LOCATION:", "GENE ONTOLOGY:", "LINEAGE:",
                "FAMILY NAMES:", "KEYWORDS:", "SUBUNIT:", "SIMILARITY:", "DOMAIN:", "PTM:",
                "PATHWAY:", "TISSUE SPECIFICITY:", "MISCELLANEOUS:", "INDUCTION:",
                "ACTIVITY REGULATION:", "DEVELOPMENTAL STAGE:", "BIOTECHNOLOGY:", "DISEASE:",
                "ALLERGEN:", "TOXIC DOSE:", "POLYMORPHISM:", "CAUTION:", "ORGANISM:",
                "BIOPHYSICOCHEMICAL PROPERTIES:")


def labels_in(caption):
    """The caption-schema field labels present, in the order they appear. These are registered as
    single special tokens by the text encoder (mini-embed DataCfg.caption_field_labels), so a
    malformed label would tokenise differently from every cached caption."""
    hits = [(caption.index(l), l) for l in FIELD_LABELS if l in caption]
    return [l for _, l in sorted(hits)]


def check(path=SWAPS_JSONL, csv_path=None):
    recs = read(path)
    rows = read_csv_rows({r["accession"] for r in recs}, csv_path)
    n_bad = 0

    def bad(rid, msg):
        nonlocal n_bad
        n_bad += 1
        print(f"  FAIL {rid}: {msg}")

    by_id = {s["swap_id"]: s for s in SWAPS}
    for r in recs:
        rid, src = r["swap_id"], rows[r["accession"]]
        if r["sequence"] != src["seq"]:
            bad(rid, "sequence does not match the CSV")
        if r["caption_current"] != src["caption"]:
            bad(rid, "current caption does not match the CSV")
        if r["cache_row"] != src["row"]:
            bad(rid, f"cache_row {r['cache_row']} != CSV row {src['row']}")
        if r["kind"] == "swap":
            # RE-DERIVE, do not trust the stored string.
            want, _ = apply_edits(r["caption_current"], by_id[rid]["edits"])
            if want != r["caption_target"]:
                bad(rid, "stored target is not what the declared edits produce")
            if labels_in(r["caption_current"]) != labels_in(r["caption_target"]) and \
                    rid != "trypsin_catalytic_to_dead":
                bad(rid, f"field labels changed: {labels_in(r['caption_current'])} -> "
                         f"{labels_in(r['caption_target'])}")
        elif r["kind"] == "control_null" and r["caption_target"] != r["caption_current"]:
            bad(rid, "null control target differs from current")
        elif r["kind"] == "control_decoy" and r["caption_target"] == r["caption_current"]:
            bad(rid, "decoy control target equals current")
    print(f"\n{len(recs)} records, {n_bad} problem(s)")
    return n_bad


def _diff(a, b, pad=8):
    """The spans that differ, as (context, removed, added, context) in WORDS.

    Word-level, and context taken from the matching run that precedes the change rather than from
    one string's offsets applied to the other -- a character-level diff of two captions realigns
    across unrelated words and renders an edit as gibberish, which is worse than useless in the
    view a human audits these by.
    """
    import difflib
    aw, bw = a.split(), b.split()
    sm = difflib.SequenceMatcher(None, aw, bw)
    ops = sm.get_opcodes()
    out = []
    for k, (tag, i1, i2, j1, j2) in enumerate(ops):
        if tag == "equal":
            continue
        pre = " ".join(aw[max(0, i1 - pad):i1])
        post = " ".join(aw[i2:i2 + pad])
        out.append((pre, " ".join(aw[i1:i2]), " ".join(bw[j1:j2]), post))
    return out


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("build", "check", "show", "list"))
    ap.add_argument("swap_id", nargs="?")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--out", default=SWAPS_JSONL)
    a = ap.parse_args()

    if a.cmd == "build":
        recs = build(a.csv)
        write(recs, a.out)
        print(f"[swaps] wrote {len(recs)} records to {a.out}")
        for r in recs:
            if r["kind"] == "swap":
                print(f"  {r['swap_id']:<32} {r['accession']}  {r['length']:>3} aa  "
                      f"{len(r['edits'])} edit(s)")
        raise SystemExit(check(a.out, a.csv))

    if a.cmd == "check":
        raise SystemExit(check(a.out, a.csv))

    if a.cmd == "list":
        for r in read(a.out):
            print(f"{r['swap_id']:<40} {r['kind']:<14} {r['accession']} {r['length']:>4} aa")
        return

    r = next((x for x in read(a.out) if x["swap_id"] == a.swap_id), None)
    if r is None:
        raise SystemExit(f"no swap {a.swap_id!r}; try `python -m src.swaps list`")
    print(f"{r['swap_id']}  ({r['kind']})\n{r['summary']}\n")
    print(f"accession {r['accession']}  cache row {r['cache_row']}  {r['length']} aa")
    if r.get("phrasing_from"):
        print("phrasing copied from: " + "; ".join(r["phrasing_from"]))
    print(f"\nORACLE: {r['oracle']}\n")
    print("EDITS (current -> target):")
    for before, old, new, after in _diff(r["caption_current"], r["caption_target"]):
        print(f"  ... {before}")
        print(f"    - {old or '(nothing)'}")
        print(f"    + {new or '(nothing)'}")
        print(f"  ... {after}\n")


if __name__ == "__main__":
    main()
