"""Encode NEW caption text the way the precomputed cache was encoded.

    python -m src.captions verify --n 16       # the only check that matters; see below

WHY THIS EXISTS. src/filip_guidance.PromptCache deliberately never runs the text encoder: every
caption it serves was embedded once by mini-embed's precompute, and guidance only needs the
768->16 projection head. That is exactly right for conditioning on an EXISTING protein's caption,
and useless here -- a swapped target caption is text nobody has ever encoded. So this fork loads
BioLinkBERT and encodes it.

THE RISK IS SILENT INCOMPATIBILITY, not failure. Tokenisation of these captions is not the
tokeniser's default: mini-embed registers the caption field labels ("PROTEIN NAME:", "FUNCTION:",
...) as single special tokens, and masks them out of the valid-token mask at encode time. Encode a
new caption without that registration and it still produces a perfectly well-formed [L, 768] tensor
-- one that lives somewhere else in the space than every cached caption, and whose scores against a
protein would be quietly meaningless. Nothing would error.

So two guards, and the second is the real one:

  1. FINGERPRINT. The cache records the encoder path, length cap, special-mask flags and the exact
     field-label set it was built with. Those are read and asserted, not assumed.
  2. ROUND TRIP. `verify` takes captions that ARE in the cache, encodes them here from their CSV
     text, and compares against the stored embedding. If this path is configured differently in any
     way that matters, the cosine falls off 1.0 and says so. A fingerprint can only catch the
     settings someone thought to record; this catches the rest.
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import torch

from config import FILIP_CACHE, MINI_EMBED_REPO, SWISSPROT_COLS, SWISSPROT_CSV
from .filip_guidance import _mini_embed


def load_fingerprint(cache_dir: str = FILIP_CACHE) -> dict:
    p = os.path.join(cache_dir, "fingerprint.json")
    if not os.path.exists(p):
        raise SystemExit(
            f"no {p}. The cache records how it was built; without it there is no way to encode a "
            f"new caption compatibly, and an incompatible encoding fails silently rather than "
            f"loudly. Point --cache at a cache built by mini-embed's src/precompute.py.")
    return json.load(open(p))


class CaptionEncoder:
    """Mirrors PromptCache's interface so either can feed FilipGuidance.

    `encode(texts, text_proj, device) -> (z [M, Lt, D], mask [M, Lt])`, projected into the shared
    space exactly as PromptCache.encode does.
    """

    def __init__(self, cache_dir: str = FILIP_CACHE, repo: str = None, device=None,
                 verbose: bool = True):
        self.mods = _mini_embed(repo or MINI_EMBED_REPO)
        self.cfg = self.mods["config"].default_cfg()
        self.device = device or torch.device("cpu")
        fp = load_fingerprint(cache_dir)
        self.fingerprint = fp

        # --- guard 1: the settings the cache says it used ---------------------------------
        want_labels = sorted(str(l) for l in self.cfg.data.caption_field_labels)
        have_labels = sorted(str(l) for l in fp.get("text_field_labels", []))
        if have_labels != want_labels:
            raise SystemExit(
                f"caption field labels disagree with the cache.\n  cache : {have_labels}\n"
                f"  config: {want_labels}\nRegistering these changes the TOKENISATION itself, so "
                f"an encoding made under a different set is not comparable with the cached ones.")
        if fp.get("text_encoder_path") != self.cfg.model.text_encoder_path:
            raise SystemExit(f"cache was built with text encoder {fp.get('text_encoder_path')!r}, "
                             f"config says {self.cfg.model.text_encoder_path!r}")
        self.max_len = int(fp.get("max_text_tokens", self.cfg.data.max_text_tokens))
        self.mask_specials = bool(fp.get("mask_text_specials", True))
        self.mask_field_labels = bool(fp.get("mask_text_field_labels", True))

        self.model, self.tok = self.mods["encoders"].load_text_encoder(
            self.cfg.model.text_encoder_path, self.device, self.cfg.data.caption_field_labels)
        if verbose:
            print(f"[captions] text encoder {self.cfg.model.text_encoder_path}\n"
                  f"[captions] max_len={self.max_len} mask_specials={self.mask_specials} "
                  f"mask_field_labels={self.mask_field_labels} "
                  f"field_labels={len(want_labels)} (fingerprint-matched)", flush=True)

    @torch.no_grad()
    def hidden(self, texts):
        """-> (h [M, Lt, 768], valid mask [M, Lt]) straight from the encoder."""
        return self.mods["encoders"].encode_text_batch(
            self.model, self.tok, list(texts), self.device, self.max_len,
            mask_specials=self.mask_specials, mask_field_labels=self.mask_field_labels)

    @torch.no_grad()
    def encode(self, texts, text_proj, device=None):
        """-> (z [M, Lt, D], mask [M, Lt]) in the shared space. Same contract as PromptCache."""
        h, m = self.hidden(texts)
        dev = device or self.device
        return text_proj(h.to(dev).float()), m.to(dev)


def _cached_captions(cache_dir, csv_path, n, stride):
    """(rows, texts) for `n` captions spread across the corpus, read from the CSV by cache row."""
    ids_path = os.path.join(cache_dir, "pair_ids.json")
    if not os.path.exists(ids_path):
        raise SystemExit(f"no {ids_path}")
    pair_ids = json.load(open(ids_path))
    rows = list(range(0, min(len(pair_ids), n * stride), stride))[:n]
    want = {r: pair_ids[r] for r in rows}
    id_col, _, text_col = SWISSPROT_COLS
    import csv as _csv
    _csv.field_size_limit(10 ** 9)
    texts, seen = {}, 0
    with open(csv_path, newline="") as f:
        for i, r in enumerate(_csv.DictReader(f)):
            if i in want:
                if r[id_col] != want[i]:
                    raise SystemExit(
                        f"row {i}: CSV says {r[id_col]!r}, pair_ids.json says {want[i]!r}. The "
                        f"cache was built from a different CSV, so cache rows do not name these "
                        f"proteins.")
                texts[i] = r[text_col]
                seen += 1
                if seen == len(want):
                    break
    return rows, [texts[r] for r in rows]


def verify(cache_dir=FILIP_CACHE, csv_path=None, n=16, stride=9973, device="cpu", repo=None):
    """Encode cached captions afresh and compare against the stored embeddings."""
    csv_path = csv_path or SWISSPROT_CSV
    dev = torch.device(device)
    enc = CaptionEncoder(cache_dir, repo, dev)
    rows, texts = _cached_captions(cache_dir, csv_path, n, stride)
    print(f"[captions] round-tripping {len(rows)} cached captions (rows {rows[:4]}...)", flush=True)

    cache = enc.mods["data"].PackedPerTokenCache(cache_dir, "text", enc.cfg.model.text_hidden)
    fresh_h, fresh_m = enc.hidden(texts)
    worst, report = 1.0, []
    for k, r in enumerate(rows):
        h_c, m_c = cache.get(int(r))
        a = fresh_h[k][fresh_m[k]].float().cpu()
        b = h_c[m_c.bool()].float().cpu()
        if a.shape != b.shape:
            report.append((r, None, f"shape {tuple(a.shape)} vs cached {tuple(b.shape)}"))
            worst = 0.0
            continue
        cos = float(torch.nn.functional.cosine_similarity(
            a.reshape(1, -1), b.reshape(1, -1)).item())
        worst = min(worst, cos)
        report.append((r, cos, ""))
    for r, cos, msg in report:
        print(f"  row {r:>7}  " + (f"cosine {cos:.6f}" if cos is not None else "") + f"  {msg}")
    ok = worst > 0.999
    print(f"\n[captions] worst cosine {worst:.6f} -> "
          + ("MATCH: new captions will be encoded compatibly with the cache."
             if ok else
             "MISMATCH. This encoding path is NOT the one that built the cache, so a score "
             "against a fresh caption is not comparable with one against a cached caption. Do not "
             "run generation until this is 1.0."))
    return 0 if ok else 1


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("verify",))
    ap.add_argument("--cache", default=FILIP_CACHE)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--repo", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--stride", type=int, default=9973, help="prime, to spread across the corpus")
    a = ap.parse_args()
    raise SystemExit(verify(a.cache, a.csv, a.n, a.stride, a.device, a.repo))


if __name__ == "__main__":
    main()
