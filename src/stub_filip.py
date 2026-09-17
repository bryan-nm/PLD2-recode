"""A toy stand-in for FILIP, so the edit loop can be exercised without the Aurora assets.

    python -m src.edit --smoke --stub-filip --device cpu

NOT A MOCK THAT RETURNS CONSTANTS. The objective here is real, just trivial: every caption is a
fixed random unit vector c over the 20 residues, a protein is its residue-composition vector f, and

    s(x, caption) = cos(f(x), c)

That is enough to make the loop's mechanics testable rather than merely runnable -- the score
genuinely responds to edits, the TAG surrogate is the true first-order gradient of it, so position
selection, the edit budget and the stopping criterion all have something to bite on. A stub that
returned noise would exercise the same lines and prove nothing: `src/tests_edit.py` asserts the
loop actually drives this objective up, which only means something because the objective is real.

It is also loud. Anything that swaps out the thing being measured has to announce itself, or a
stubbed run and a real one look identical in a log six weeks later.
"""
from __future__ import annotations
import hashlib

import torch


def _caption_vec(text: str, d: int = 20) -> torch.Tensor:
    """A fixed unit vector per caption. Deterministic across processes -- hashlib, not hash(),
    which Python randomises per process unless PYTHONHASHSEED is set."""
    h = hashlib.sha256(text.encode()).digest()
    g = torch.Generator().manual_seed(int.from_bytes(h[:8], "big") % (2 ** 63))
    v = torch.randn(d, generator=g)
    return v / v.norm()


class StubCaptionEncoder:
    """Mirrors captions.CaptionEncoder. `encode` -> ([M, 1, 20], [M, 1] mask)."""

    def __init__(self, *_, verbose: bool = True, **__):
        self.fingerprint = {"stub": True}
        if verbose:
            print("[stub] caption encoder is a STUB: captions are hashed to random unit vectors "
                  "over the 20 residues. No text encoder is loaded.", flush=True)

    def encode(self, texts, text_proj=None, device=None):
        z = torch.stack([_caption_vec(t) for t in texts]).unsqueeze(1)      # [M, 1, 20]
        m = torch.ones(z.shape[0], 1, dtype=torch.bool)
        dev = device or torch.device("cpu")
        return z.to(dev), m.to(dev)

    def hidden(self, texts):
        return self.encode(texts)


class _Proj:
    def __call__(self, x):
        return x


class _Filip:
    text_proj = _Proj()


class StubGuidance:
    """Mirrors the FilipGuidance surface that src/edit.py uses.

    s(x, c) = cos(composition(x), c), and tag_delta is its exact first-order term: swapping the
    residue at position i to `a` changes the composition by (e_a - e_{x_i}) / L, so to first order
    the score changes by <grad_f s, e_a - e_{x_i}> / L. edit.py subtracts the incumbent's value, so
    only the e_a half has to be returned here.
    """

    filip = _Filip()

    def __init__(self, cfg, device, *, gamma: float = 10.0, verbose: bool = True, **__):
        self.cfg, self.device, self.gamma = cfg, device, float(gamma)
        self._z_t = self._z_bank = None
        self.calls = 0
        if verbose:
            print(f"[stub] GUIDANCE IS A STUB (gamma={gamma}). Scores are composition cosines, not "
                  f"FILIP. Any result from this run is about the loop, never about proteins.",
                  flush=True)

    # --- the bits edit.py calls -------------------------------------------
    def set_target_encoded(self, z_t, mask_t):
        self._z_t = z_t.reshape(-1)[:20].to(self.device)

    def set_bank_encoded(self, z_bank, mask_bank):
        self._z_bank = None if z_bank is None else z_bank.reshape(-1)[:20].to(self.device)

    @staticmethod
    def _composition(aa_canvas: torch.Tensor) -> torch.Tensor:
        """[B, 20] normalised residue counts. Only the 20 residues count; EOS/PAD/MASK do not."""
        B = aa_canvas.shape[0]
        f = torch.zeros(B, 20, device=aa_canvas.device)
        res = aa_canvas.clamp(max=20)
        for a in range(20):
            f[:, a] = (res == a).sum(dim=1).float()
        return f / f.sum(dim=1, keepdim=True).clamp_min(1.0)

    @torch.no_grad()
    def raw_scores(self, aa_canvas, z_t, mask_t=None):
        f = self._composition(aa_canvas)                                   # [B, 20]
        c = z_t.reshape(z_t.shape[0], -1)[:, :20].to(aa_canvas.device)     # [M, 20]
        f = f / f.norm(dim=1, keepdim=True).clamp_min(1e-9)
        c = c / c.norm(dim=1, keepdim=True).clamp_min(1e-9)
        return f @ c.T                                                     # [B, M]

    @torch.no_grad()
    def tag_delta(self, aa_canvas):
        B, L = aa_canvas.shape
        f = self._composition(aa_canvas)
        fn = f / f.norm(dim=1, keepdim=True).clamp_min(1e-9)
        c = (self._z_t / self._z_t.norm().clamp_min(1e-9)).to(aa_canvas.device)
        s = (fn * c).sum(dim=1, keepdim=True)                              # [B, 1]
        # d cos(f, c) / d f, then the per-position effect of adding one residue.
        grad = (c.unsqueeze(0) - s * fn) / f.norm(dim=1, keepdim=True).clamp_min(1e-9)
        return grad.unsqueeze(1).expand(B, L, 20).contiguous() / max(L, 1)

    def __call__(self, canvas, logits):
        self.calls += 1
        d = self.tag_delta(canvas)
        out = logits.clone()
        out[..., :20] = out[..., :20] + self.gamma * d
        return out
