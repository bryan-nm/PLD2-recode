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

## Generation

```bash
qsub -v PHASES=VS scripts/edit.pbs     # the two gates first -- do this before anything else
qsub scripts/edit.pbs                  # verify, separability, edit, report
qsub -v PHASES=R scripts/edit.pbs      # re-read the table, no GPU
```

The loop, per swap: the protein starts fully committed; each round asks TAG where it is most
improvable (`gain[i] = max_a Δ[i,a] − Δ[i,x_i]`, free — the gradient is computed anyway to bias the
logits), unfreezes the top-k in both tracks, and redecodes them under guidance with everything else
frozen. It stops when the protein matches the target caption about as well as it originally matched
its own. `--contrast` puts the *current* caption in the softmax bank, which makes the objective
"raise the target at the expense of the current" rather than "raise the target" — the push rather
than a drift, with no new loss function.

### Pre-diffusion masking (arms 1 and 2)

FILIP is a **late-interaction** model: `filip_score_matrix` forms a per-(residue, text-token)
similarity matrix and then averages it twice. Those averages are a badly attenuated readout — the
t→p half divides by ~280 text tokens, so rewriting a whole caption field moves the total by a couple
of percent, and the p→t half can move the *other* way at the same time because a residue whose best
token was deleted simply finds the next best one. Measured: an unrelated caption moves the score
**0.28**, a minimally edited one **0.006**.

Read the matrix locally instead — only the columns of the tokens a swap *deletes* — and the same
model turns out to know exactly where its caption's claims live. On the Sho1 homolog E7QE10,
fraction of the top-20 aligned residues inside the annotated region:

| tokens | region | in-region | chance |
|---|---|---|---|
| `multi` `-` `pass` | TM helices | **100%** | 32% |
| `membrane` | TM helices | **75%** | 32% |
| `sh3 domain` (FAMILY NAMES) | SH3 domain | **90%** | 17% |
| `sh3 domain` (SUBUNIT) | SH3 domain | **95%** | 17% |

So `--select filip` masks the residues the deleted phrase points at, before any decoding, and lets
the model refill them. No backward pass — cheaper than the gradient selector it replaces.

```
a1   --select filip  --gamma 0   --rounds 1     aligned mask, unguided
a2   --select filip  --gamma G   --rounds 1     aligned mask, guided
a4   --select random --gamma 0   --rounds 1     the floor: same count, no information
tag  --select tag    --gamma G   --rounds R     the original iterative gradient selector
```

**a1 vs a4** says whether localisation bought anything. **a2 vs a1** is the only place guidance has
to justify itself — and four of the six swaps are *removals*, which a prior may well do unaided.

The `hit` column reports what fraction of the residues actually unfrozen fell inside the annotated
region, against chance. That is the arm's own premise: if `hit` sits at chance, nothing downstream
is worth reading.

### Two gates, and they come first

**V — captions round-trip** (`src.captions verify`). A swapped target is text nobody has encoded,
so this fork loads BioLinkBERT itself. mini-embed registers the caption field labels as single
special tokens, and an encoding made without that registration still produces a perfectly
well-formed tensor — one that simply lives somewhere else in the space. *Nothing errors.* So the
fingerprint is asserted, and then captions that *are* cached are re-encoded and compared. Below
1.0 cosine, stop.

**S — separability** (`--rounds 0`). Score each untouched protein against every caption in the
experiment. If `s(x₀,c₀) − s(x₀,c₁) ≲ 0` for a swap, FILIP does not distinguish those two captions
on that protein and the row means nothing however it moves — `src.edit` says so explicitly rather
than reporting a quiet zero. That gap is also the **budget**: it is exactly how far guidance has to
travel. Phase S is the edit loop with the rounds turned off, so it costs one scoring pass.

### Reading the table

```
swap                     kind             sep  d target  d current     spec   edited   ident  rnds
arca_cytosol_to_membrane swap         +0.2294   +0.2122    -0.0993  +0.2162   34/238  85.9%     6
arca..._to_membrane__decoy control_decoy -0.3584  +0.0000   +0.0000  +0.0000    0/238 100.0%     0
```

`spec` is the column that matters: the target's gain minus the mean gain across every *other*
swap's target caption. A protein that drifts toward every caption in the bank has not been steered,
and only this can tell the difference — which is why every round scores against the whole bank. A
swap whose `spec` does not beat its own `__decoy` row moved toward captions in general.

**And none of it is evidence on its own: the score being reported is the score being optimised.**
What settles a row is its oracle — SignalP, a TM-helix call, `GDSGGP`, a coiled-coil call, an HMM
scan. Those are not implemented yet; `python -m src.swaps show <id>` prints what each one is.

## Status

Implemented and smoke-tested end to end on CPU: the swap definitions, the caption encoder and its
round-trip check, the edit loop, the separability gate, the summary table, and `scripts/edit.pbs`.

`src/stub_filip.py` replaces FILIP with a toy objective (caption → random unit vector over the 20
residues; score = cosine to the protein's composition) so the whole loop runs on a laptop without
the Aurora assets. It is a real objective, not a constant — the TAG surrogate is its exact
first-order gradient — which is what makes the smoke test mean anything: the loop demonstrably
drives the target score up and the current score down, nulls make zero edits at 100% identity, and
swaps beat their decoy rows on `spec`. Every stubbed result is labelled as such in the table.

**Not implemented: the oracles, and folding the edits.** That is what turns a FILIP number into
evidence.
