"""A toy stand-in for FILIP, so the edit loop can be exercised without the Aurora assets.

    python -m src.edit --smoke --stub-filip --device cpu --select filip

NOT A MOCK THAT RETURNS CONSTANTS. Both things the loop asks of FILIP are modelled by a real, if
trivial, objective:

    every caption WORD hashes to a fixed unit vector c_w over the 20 residues
    sim(position p, word w) = c_w[residue at p]        <- the late-interaction matrix
    s(protein, caption)     = cos(composition(x), mean_w c_w)   <- the aggregate score

So a word genuinely "points at" the residues it likes, which is what `--select filip` reads, and
the aggregate genuinely responds to edits, which is what the stopping criterion reads. A stub that
returned noise would exercise the same lines and prove nothing: the smoke test asserts the selector
finds the planted residues and the loop drives the score, and neither assertion means anything
unless the stub has structure to find.

It is also loud. Anything that swaps out the thing being measured has to announce itself, or a
stubbed run and a real one look identical in a log six weeks later.
"""
from __future__ import annotations
import hashlib

import torch

_CACHE = {}


def _word_vec(word: str, d: int = 20) -> torch.Tensor:
    """A fixed unit vector per word. hashlib, not hash(): Python randomises str hashing per
    process unless PYTHONHASHSEED is set, so hash() would differ across ranks."""
    if word not in _CACHE:
        h = hashlib.sha256(word.encode()).digest()
        g = torch.Generator().manual_seed(int.from_bytes(h[:8], "big") % (2 ** 63))
        v = torch.randn(d, generator=g)
        _CACHE[word] = v / v.norm()
    return _CACHE[word]


class _Tok:
    """Whitespace tokenizer, mirroring only the call signature align_positions needs."""

    def __call__(self, text, **kw):
        return {"input_ids": [int.from_bytes(hashlib.sha256(w.encode()).digest()[:4], "big")
                              for w in text.split()]}


class StubCaptionEncoder:
    """Mirrors captions.CaptionEncoder. `encode` -> ([M, T, 20], [M, T] mask), one row per word."""

    tok = _Tok()
    max_len = None

    def __init__(self, *_, verbose: bool = True, **__):
        self.fingerprint = {"stub": True}
        if verbose:
            print("[stub] caption encoder is a STUB: every word hashes to a random unit vector "
                  "over the 20 residues. No text encoder is loaded.", flush=True)

    def encode(self, texts, text_proj=None, device=None):
        toks = [t.split() for t in texts]
        T = max(len(t) for t in toks)
        z = torch.zeros(len(toks), T, 20)
        m = torch.zeros(len(toks), T, dtype=torch.bool)
        for i, ws in enumerate(toks):
            for j, w in enumerate(ws):
                z[i, j] = _word_vec(w)
            m[i, :len(ws)] = True
        dev = device or torch.device("cpu")
        return z.to(dev), m.to(dev)

    def hidden(self, texts):
        return self.encode(texts)


class _Filip:
    text_proj = staticmethod(lambda x: x)


class StubGuidance:
    """Mirrors the FilipGuidance surface that src/edit.py uses."""

    filip = _Filip()

    def __init__(self, cfg, device, *, gamma: float = 10.0, verbose: bool = True, **__):
        self.cfg, self.device, self.gamma = cfg, device, float(gamma)
        self._z_t = self._z_bank = None
        self.calls = 0
        if verbose:
            print(f"[stub] GUIDANCE IS A STUB (gamma={gamma}). Scores are composition cosines and "
                  f"alignments are per-word residue preferences, not FILIP. Any result from this "
                  f"run is about the loop, never about proteins.", flush=True)

    def set_target_encoded(self, z_t, mask_t):
        self._z_t = self._pool(z_t, mask_t)

    def set_bank_encoded(self, z_bank, mask_bank):
        self._z_bank = None if z_bank is None else self._pool(z_bank, mask_bank)

    @staticmethod
    def _pool(z, mask):
        """[M, T, 20] -> [20], the caption's mean word vector (first caption only)."""
        v = (z[0] * mask[0].unsqueeze(-1)).sum(0) / mask[0].sum().clamp_min(1)
        return v / v.norm().clamp_min(1e-9)

    @staticmethod
    def _composition(aa_canvas: torch.Tensor) -> torch.Tensor:
        f = torch.zeros(aa_canvas.shape[0], 20, device=aa_canvas.device)
        res = aa_canvas.clamp(max=20)
        for a in range(20):
            f[:, a] = (res == a).sum(dim=1).float()
        return f / f.sum(dim=1, keepdim=True).clamp_min(1.0)

    @torch.no_grad()
    def raw_scores(self, aa_canvas, z_t, mask_t=None):
        f = self._composition(aa_canvas)
        f = f / f.norm(dim=1, keepdim=True).clamp_min(1e-9)
        if mask_t is None:
            mask_t = torch.ones(z_t.shape[:2], dtype=torch.bool, device=z_t.device)
        c = (z_t * mask_t.unsqueeze(-1)).sum(1) / mask_t.sum(1, keepdim=True).clamp_min(1)
        c = (c / c.norm(dim=1, keepdim=True).clamp_min(1e-9)).to(aa_canvas.device)
        return f @ c.T

    @torch.no_grad()
    def align_to_text(self, aa_canvas, z_t, mask_t):
        """[B, L, T]: sim(position, word) = that word's preference for the residue sitting there."""
        res = aa_canvas.clamp(max=19)
        c = z_t[0].to(aa_canvas.device)                                # [T, 20]
        sim = c.T[res]                                                 # [B, L, T]
        live = aa_canvas < 20                                          # residues only
        sim = sim.masked_fill(~live.unsqueeze(-1), float("-inf"))
        return sim.masked_fill(~mask_t[0].to(aa_canvas.device).view(1, 1, -1), float("-inf"))

    @torch.no_grad()
    def tag_delta(self, aa_canvas):
        B, L = aa_canvas.shape
        f = self._composition(aa_canvas)
        fn = f / f.norm(dim=1, keepdim=True).clamp_min(1e-9)
        c = self._z_t.to(aa_canvas.device)
        s = (fn * c).sum(dim=1, keepdim=True)
        grad = (c.unsqueeze(0) - s * fn) / f.norm(dim=1, keepdim=True).clamp_min(1e-9)
        return grad.unsqueeze(1).expand(B, L, 20).contiguous() / max(L, 1)

    def __call__(self, canvas, logits):
        self.calls += 1
        out = logits.clone()
        out[..., :20] = out[..., :20] + self.gamma * self.tag_delta(canvas)
        return out
