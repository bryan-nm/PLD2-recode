"""Invariants for caption-directed editing:  python -m src.tests_edit

The one that matters most is section 1. FILIP's alignment matrix is indexed by VALID token, and my
first reading of it used the raw token index instead -- one off on the text axis. Under that
convention "sh3 domain" appeared to point at 10-25% of the SH3 domain, which reads as "the model
does not localise" rather than "you indexed it wrong", and would have killed this whole direction.
Corrected, it is 90-95% against a 17% chance rate. The fixture pins that down so the convention can
never drift again.
"""
import csv
import os

import numpy as np
import torch

from src.align_positions import changed_token_indices, positions_for_tokens, region_overlap

fails, checks = [], 0


def check(name, ok, extra=""):
    global checks
    checks += 1
    if not ok:
        fails.append(f"{name}{(' -- ' + extra) if extra else ''}")
        print(f"  FAIL {name} {extra}")


# ---------------------------------------------------- 1. the index convention, against real data
FIX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "fixtures")
npy = os.path.join(FIX, "E7QE10.npy")
if os.path.exists(npy):
    sim = np.load(npy)                                    # [n_valid_protein, n_valid_text]
    tt = [r for r in csv.DictReader(open(os.path.join(FIX, "E7QE10_text_tokens.tsv")),
                                    delimiter="\t") if r.get("index")]
    pt = [r for r in csv.DictReader(open(os.path.join(FIX, "E7QE10_protein_tokens.tsv")),
                                    delimiter="\t") if r.get("index")]
    n_t = sum(int(r["valid"]) for r in tt)
    n_p = sum(int(r["valid"]) for r in pt)
    check("fixture is indexed by VALID tokens on both axes", sim.shape == (n_p, n_t),
          f"{sim.shape} vs ({n_p}, {n_t})")
    seq = "".join(r["token"] for r in pt if int(r["valid"]))
    # plot_index IS the valid-token index; that is the column the matrix uses.
    plot = {int(r["index"]): int(r["plot_index"]) for r in tt}

    def cols(raw):
        return [plot[i] for i in raw if plot.get(i, -1) >= 0]

    TM = (28, 146)                                        # four TM helices
    SH3 = (seq.find("ALYPY"), len(seq))                   # canonical SH3 N-terminal motif
    check("SH3 motif is where UniProt says", SH3[0] == 306, f"found at {SH3[0]}")

    def hit(raw_cols, region, k=20):
        c = cols(raw_cols)
        v = sim[:, c].max(axis=1)
        top = np.argsort(-v)[:k]
        return sum(region[0] <= int(i) < region[1] for i in top) / k

    for name, raw, region, floor in (("multi/-/pass -> TM", [173, 174, 175], TM, 0.90),
                                     ("membrane -> TM",
                                      [int(r["index"]) for r in tt if r["token"] == "membrane"],
                                      TM, 0.60),
                                     ("sh3 domain (FAMILY) -> SH3", [277, 278, 279], SH3, 0.70),
                                     ("sh3 domain (SUBUNIT) -> SH3", [140, 141, 142], SH3, 0.70)):
        f = hit(raw, region)
        chance = (region[1] - region[0]) / len(seq)
        check(f"{name} beats chance", f >= floor and f > 2 * chance,
              f"{f:.0%} in region, chance {chance:.0%}")
else:
    print(f"  (no fixture at {npy}; skipping the index-convention checks)")

# ---------------------------------------------------- 2. the token diff
class _Tok:
    def __call__(self, text, **kw):
        return {"input_ids": [abs(hash(w)) % 9973 for w in text.split()]}


cur = "PROTEIN NAME: sho1 SUBCELLULAR LOCATION: cell membrane multi-pass membrane protein END"
tgt = "PROTEIN NAME: sho1 SUBCELLULAR LOCATION: cytoplasm END"
rem, add = changed_token_indices(_Tok(), cur, tgt)
cw, tw = cur.split(), tgt.split()
check("removed tokens are the deleted words",
      {cw[i] for i in rem} == {"cell", "membrane", "multi-pass", "protein"},
      f"{[cw[i] for i in rem]}")
check("added tokens are the inserted words", {tw[i] for i in add} == {"cytoplasm"},
      f"{[tw[i] for i in add]}")
check("an identical pair changes nothing", changed_token_indices(_Tok(), cur, cur) == ([], []))

# ---------------------------------------------------- 3. the selector recovers a planted region
L, T = 200, 6
s = torch.full((1, L, T), -1.0)
s[0, 120:140, 2] = 1.0                                    # token 2 points at residues 120-139
editable = torch.ones(1, L, dtype=torch.bool)
p = positions_for_tokens(s, [2], k=20, dilate=0, editable=editable)
check("selector recovers the planted region", int(p.sum()) == 20 and bool(p[0, 120:140].all()))
check("selector ignores tokens it was not asked about",
      int(positions_for_tokens(s, [0], k=20, dilate=0, editable=editable)[0, 120:140].sum()) < 20)
check("selector never exceeds its budget",
      all(int(positions_for_tokens(s, [2], k=kk, dilate=d, editable=editable).sum()) <= kk
          for kk in (5, 20, 50) for d in (0, 2, 5)))
check("selector respects the editable mask",
      int((positions_for_tokens(s, [2], k=20, dilate=0,
                                editable=torch.zeros(1, L, dtype=torch.bool).index_fill_(
                                    1, torch.arange(150, 200), True))[0, :150]).sum()) == 0)
# dilation should make the selection contiguous around peaks
s2 = torch.full((1, L, T), -1.0)
s2[0, [50, 100, 150], 1] = 1.0
pd = positions_for_tokens(s2, [1], k=15, dilate=2, editable=editable)
runs = int(((pd[0].int()[1:] - pd[0].int()[:-1]) == 1).sum()) + int(pd[0, 0])
check("dilation yields a few contiguous stretches, not isolated picks", runs <= 4, f"{runs} runs")

# ---------------------------------------------------- 4. the overlap statistic
ov = region_overlap(torch.zeros(1, L, dtype=torch.bool).index_fill_(
    1, torch.arange(120, 140), True), 120, 140)
check("overlap is 1.0 when fully inside", abs(ov[0] - 1.0) < 1e-9)
ov2 = region_overlap(torch.zeros(1, L, dtype=torch.bool).index_fill_(
    1, torch.arange(100, 140), True), 120, 140)
check("overlap is 0.5 when half inside", abs(ov2[0] - 0.5) < 1e-9, f"{ov2}")

print(f"\n{checks - len(fails)}/{checks} checks pass")
if fails:
    print("FAILURES:")
    for f in fails:
        print("  -", f)
    raise SystemExit(1)
