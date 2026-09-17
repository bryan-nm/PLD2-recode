# PLD2-recode

**Caption-directed protein editing.** Given a protein, its current caption, and a *target* caption,
make edits that move the protein toward the target — using PLD2's diffusion sampler with FILIP
guidance, and stopping when the match to the target is about as good as the original protein's
match to its own caption.

Forked from **PLD2 @ 1ee3833** (recorded in `.pld2-source-commit`). PLD2 continues on the
cold-start alignment path; the two do not interfere.

## Why editing, and why now

Round 1 of the alignment work measured PLD2's infilling by mask rate, on held-out prompts:

```
rate 0.50   75.5% success        rate 0.85   18.1%
rate 0.70   50.0%                rate 1.00    0.0%   <- cold start
```

The model is good at completion and has never once succeeded at cold start. Editing a real protein
is infilling at a very low mask rate — 5–20% — which is off the *left* edge of that table. This
fork aims the model at what it demonstrably does well.

There is a second reason to expect this to behave better than anything measured so far. Every FILIP
guidance number to date was taken on a **half-masked canvas**, which is out of distribution for
AMPLIFY's protein encoder. Editing hands it a complete, real protein — its training distribution.
The steering we measured is probably a floor, not a ceiling.

## The five swaps (`data/swaps.jsonl`)

| swap | protein | what changes | independent oracle |
|---|---|---|---|
| `sho1_sh3_to_pdz` | P40073, 367 aa | SH3 domain → PDZ | Pfam/InterProScan: does the SH3 hit go and a PDZ hit appear? |
| `lactamase_periplasm_to_cytosol` | P26918, 254 aa | periplasm → cytoplasm, drops `signal` | SignalP: the signal peptide should disappear |
| `hox2_dimer_to_monomer` | Q84U86, 308 aa | leucine zipper removed, homodimer → monomer | coiled-coil / heptad prediction |
| `arca_cytosol_to_membrane` | P0A9Q3, 238 aa | cytosol → single-pass inner membrane | TM-helix prediction: none → exactly one |
| `trypsin_catalytic_to_dead` | P00760, 246 aa | catalytic → serine protease homolog | does the `GDSGGP` nucleophile box survive? |

Every oracle was checked for **baseline signal on the unedited protein** before being adopted:
P26918's N-terminus is a textbook signal peptide (peak Kyte-Doolittle +1.71), ArcA has no
hydrophobic segment at all (peak +1.06, below threshold), Q84U86 scores a full 4/4 leucine heptad,
and P00760 carries `GDSGGP` at position 197. Only the SH3→PDZ swap has no cheap local proxy and
needs a real HMM scan.

### How the swaps are built

Targets are **generated, never typed**. Each swap declares an ordered list of literal substitutions
with an expected hit count; `python -m src.swaps check` re-derives every target from the stored
current caption and compares. A hand-written target would be a second, silently divergent copy of a
1,500-character string — and the contrast between the two captions *is* the experiment.

Three rules the file enforces:

- **Minimal pairs.** Everything outside the declared substitutions is byte-identical. Otherwise
  measured movement could be a response to phrasing, length or vocabulary rather than to the
  property we meant to change.
- **Phrasing borrowed from real entries.** The text encoder saw SwissProt's conventions; a
  plausible-sounding string SwissProt never writes is out of distribution. Each swap records which
  accession it copied. Two consequences: there is **no "PDZ domain" UniProt keyword** (0 of 451,436
  captions have one), so that swap *removes* the keyword rather than replacing it; and a
  catalytically dead peptidase S1 **keeps its GO terms** in real entries, because GO is assigned by
  homology, so the trypsin swap leaves them alone.
- **Substitutions are anchored to exact strings.** If the CSV is updated and a caption changes, the
  build fails loudly rather than editing the wrong span.

### Controls

Generated automatically for all five, so 15 records total:

- `__null` — target caption *identical* to current. Any movement here is the machinery perturbing
  the protein for its own reasons, and sets the floor everything else must clear.
- `__decoy` — target is a real caption from an unrelated protein (P0C9F0). Separates "moved toward
  *this* caption" from "moved toward any caption that isn't the current one".

```bash
python -m src.swaps build     # write + validate data/swaps.jsonl
python -m src.swaps check     # re-derive every target, no writes
python -m src.swaps show arca_cytosol_to_membrane
python -m src.swaps list
```

## What came across from PLD2, and what did not

Carried: `model.py`, `sampler.py` (prompt/frozen-context decoding), `corruption.py`, `data.py`,
`objective.py`, `blosum.py`, `dist.py`, `filip_guidance.py`, `fold_fasta.py`, `metrics.py`,
`self_consistency.py`, `tm_align.py`, `train.py`, `reference_set.py`, `xpu_linalg_guard.py`.

Left behind: the whole alignment pipeline — `align.py`, `align_sample.py`, `align_compare.py`,
`preference.py`, `prompts.py`, and the preprocessing/sweep tooling. `tm_align.py` had its one
dependency on the alignment prompt manifest inlined so the tree's imports close on their own.

## Status and next steps

`data/swaps.jsonl` is built and validated. Nothing else is implemented yet.

**The next step is the separability check, and it is a gate.** Before any editing code: score each
*original* protein against both its current and its target caption. If `s(x₀,c₀) − s(x₀,c₁)` is
near zero, FILIP does not distinguish the two captions and no amount of guidance will produce a
specific edit. That gap is also the **budget** — the stopping criterion is "edit until `s(x,c₁)`
reaches `s(x₀,c₀)`", so it is exactly how far guidance has to travel.

One thing that check needs: **the target captions are new text and are not in the FILIP cache.**
`filip_guidance.PromptCache` deliberately never runs the text encoder. So this fork has to load
BioLinkBERT and encode the swapped captions itself — using the same `caption_field_labels`,
`mask_field_labels` and length cap the cache was built with. Those settings are recorded in the
cache's `fingerprint.json`; read and assert them rather than assuming.
