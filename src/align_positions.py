"""Which residues does a caption phrase point at?

The edit loop's original position selector was the TAG gradient: "where would changing a residue
most raise the target caption's log-probability". That is the right question only if the classifier
can score the distinction at all, and measured, it cannot -- a minimally edited caption moves
FILIP's aggregate score 0.006 where an unrelated caption moves it 0.28.

But the aggregate is not the model. FILIP is a LATE-INTERACTION model: it forms a per-(protein
position, text token) similarity matrix and then averages it twice. Averaging over ~280 text tokens
is what destroys the signal, not the absence of one. Read the matrix locally -- take only the
columns of the text tokens that the swap CHANGES -- and ask which residues they point at.

MEASURED ON Sho1 (E7QE10), fraction of the top-20 aligned residues inside the annotated region:

    multi / - / pass      100%  inside the transmembrane helices   (chance 32%)
    membrane               75%  inside the transmembrane helices   (chance 32%)
    sh3 domain (FAMILY)    90%  inside the SH3 domain              (chance 17%)
    sh3 domain (SUBUNIT)   95%  inside the SH3 domain              (chance 17%)

So the model knows perfectly well where its caption's claims live in the sequence. The information
was being averaged away before anyone looked at it.

WHICH TOKENS. For a swap that REMOVES a property -- four of the six here do -- the informative
tokens are the ones deleted from the CURRENT caption: they point at the residues implementing the
thing we want gone. Tokens ADDED by the target caption have no protein correlate yet, so their
alignment says only "where does this protein already most resemble the new claim", which is where
you would install it. `side` picks; "removed" is the default because it is the well-posed one.

THE DIFF IS OVER TOKEN IDS, not characters. Mapping character spans onto wordpieces needs offset
mapping and is quietly wrong when a field label is registered as a single special token. Diffing
the two token-id sequences directly is exact, needs nothing from the tokenizer, and yields the
indices in the coordinates the similarity matrix is actually indexed by.
"""
from __future__ import annotations
import difflib

import torch


def changed_token_indices(tok, current: str, target: str, max_len: int = None):
    """-> (removed indices into `current`'s tokens, added indices into `target`'s tokens).

    Both index the VALID (non-special) token sequence, which is the axis
    FilipGuidance.align_to_text's similarity matrix uses.
    """
    kw = dict(add_special_tokens=False)
    if max_len:
        kw.update(truncation=True, max_length=max_len)
    a = tok(current, **kw)["input_ids"]
    b = tok(target, **kw)["input_ids"]
    removed, added = [], []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes():
        if op in ("delete", "replace"):
            removed += list(range(i1, i2))
        if op in ("insert", "replace"):
            added += list(range(j1, j2))
    return removed, added


def positions_for_tokens(sim: torch.Tensor, token_idx, k: int, dilate: int = 0,
                         editable: torch.Tensor = None):
    """(B, L) bool: the k canvas positions most aligned with `token_idx`.

    A position scores as its BEST match over the changed tokens, not its mean. A phrase is several
    wordpieces ("multi", "-", "pass") and a residue implementing it need only answer to one of
    them; averaging would dilute a sharp hit against the neighbouring pieces in exactly the way the
    aggregate score dilutes everything.

    `dilate` widens each selected position by +/- that many residues before the count is taken.
    Domains are contiguous and the alignment is peaky, so a handful of scattered peaks describe a
    region the model is pointing at rather than the whole of it.
    """
    B, L, _ = sim.shape
    if not len(token_idx):
        return torch.zeros(B, L, dtype=torch.bool, device=sim.device)
    idx = torch.as_tensor(sorted(set(int(t) for t in token_idx if 0 <= int(t) < sim.shape[2])),
                          device=sim.device)
    score = sim[:, :, idx].max(dim=2).values                       # [B, L]
    if editable is not None:
        score = score.masked_fill(~editable, float("-inf"))
    score = torch.nan_to_num(score, neginf=-1e9)

    if dilate <= 0:
        order = score.argsort(dim=1, descending=True)
        rank = order.argsort(dim=1)
        picked = rank < k
    else:
        # Grow the seeds outward until the budget is spent, so the selection is a few contiguous
        # stretches rather than k isolated residues.
        picked = torch.zeros(B, L, dtype=torch.bool, device=sim.device)
        n_seed = max(1, k // (2 * dilate + 1))
        order = score.argsort(dim=1, descending=True)
        rank = order.argsort(dim=1)
        seeds = rank < n_seed
        for d in range(-dilate, dilate + 1):
            picked |= torch.roll(seeds, d, dims=1)
        if editable is not None:
            picked &= editable
        # Trim to budget by score, so dilation never smuggles in more than was asked for.
        over = picked.sum(dim=1) > k
        if bool(over.any()):
            s2 = score.masked_fill(~picked, float("-inf"))
            r2 = s2.argsort(dim=1, descending=True).argsort(dim=1)
            picked &= (r2 < k)
    if editable is not None:
        picked &= editable
    return picked


def region_overlap(picked: torch.Tensor, lo: int, hi: int):
    """Fraction of the selected positions inside [lo, hi). The validation statistic -- the one that
    said the alignment beats chance 5x on Sho1's SH3 domain."""
    n = picked.sum(dim=1).clamp_min(1)
    inside = picked[:, lo:hi].sum(dim=1)
    return (inside.float() / n.float()).tolist()
