"""Caption-directed protein editing: push a protein from its own caption toward a target one.

    python -m src.edit --device xpu                      # every swap in data/swaps.jsonl
    python -m src.edit --only arca_cytosol_to_membrane --n 8
    python -m src.edit --smoke --device cpu              # tiny random model, every code path

THE LOOP. The protein starts fully committed -- this is editing, not generation -- and each round:

  1. asks the classifier where it is most improvable. TAG already computes d log p(target | x) with
     respect to every (position, residue) in order to bias the logits, so
         gain[i] = max_a delta[i, a] - delta[i, x_i]
     is free, and it is the natural answer to "which residues are holding this protein back".
  2. unfreezes the top-k of those, in BOTH tracks, and redecodes them under guidance with
     everything else frozen. That is one ordinary sampler call: the machinery for frozen context
     already exists, and a prompted decode cannot move the boundary or touch a frozen residue.
  3. rescores, and stops when the protein matches the target caption about as well as it originally
     matched its own.

WHY THAT STOPPING RULE. FILIP scores are not calibrated in absolute terms -- there is no score at
which a protein "is" its caption -- so the only meaningful threshold is a per-protein reference
point, and s(x0, current) is the obvious one: it is how well this particular protein matches a
caption that is TRUE of it, measured by this particular model. --stop-frac moves the bar.

WHY EDITING SHOULD WORK WHERE COLD START DID NOT. Two measurements. PLD2's infilling runs 75.5%
success at mask rate 0.50 and 0.0% at 1.0, and editing 5-20% of a protein sits off the easy end of
that curve. And every FILIP guidance number so far was taken on a half-masked canvas, which is out
of distribution for AMPLIFY's protein encoder; here it reads a complete protein, which is what it
was trained on.

SEPARABILITY IS MEASURED, NOT ASSUMED. Round 0 scores the untouched protein against every caption
in the experiment -- its own, its target, and every other swap's target. s(x0, current) - s(x0,
target) is the gap guidance has to close, and if it is ~0 for a swap then FILIP does not
distinguish those two captions and that row's result means nothing, however it moves. The same
matrix gives specificity for free: a protein that drifts toward every caption has not been steered.

BOTH CAPTIONS ARE ENCODED FRESH, including the current one, which IS in the cache. Mixing a cached
embedding with a freshly encoded one would put any systematic difference between the two paths
directly into the contrast being measured -- which is the one number this experiment is about.
src/captions.py verifies the fresh path reproduces the cache, so encoding both costs nothing.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

import torch

from config import CFG, CKPT_DIR, FILIP_CACHE, FILIP_CKPT
from .data import AA, DI
from .dist import init_distributed
from .model import LoopedDiffusionLM
from .sampler import aa_track, decode_seqs, decode_struct, generate
from .swaps import read as read_swaps
from .train import find_latest_ckpt

try:
    import intel_extension_for_pytorch as ipex
except Exception:
    ipex = None

_AA_ID = {c: i for i, c in enumerate(AA)}
_DI_ID = {c: i for i, c in enumerate(DI)}


def _decode(model, dev, **kw):
    """Every call into the PLD2 model goes through here, under autocast.

    ipex.optimize(dtype=bfloat16) rewrites the weights to bf16 while the canvas stays long and the
    embeddings come out fp32, so a forward outside autocast dies on `expected self and mat2 to have
    the same dtype, but got: float != c10::BFloat16` -- deep inside the attention qkv projection,
    where it reads as a model bug rather than a missing context manager. Every other caller in PLD2
    wraps generate() this way; this wrapper exists so there is one place to forget it rather than
    two, and neither of them is a bare call site.

    The FILIP scoring path deliberately does NOT go through here: AMPLIFY is never ipex-optimised,
    so it stays fp32, and the separability numbers are the one quantity in this experiment worth
    keeping at full precision.
    """
    with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=torch.bfloat16,
                                         enabled=dev.type in ("xpu", "cuda")):
        return generate(model, device=str(dev), **kw)


def encode_canvas(seq, mcfg, canvas, di=None, n_tracks=2):
    """(K, canvas) long, laid out exactly as training does: [AA* EOS PAD*] in track 0, and the 3Di
    track padded from the boundary INCLUSIVE with no EOS of its own."""
    L, K = len(seq), max(1, n_tracks)
    tok = torch.full((K, canvas), mcfg.pad_token_id, dtype=torch.long)
    tok[0, :L] = torch.tensor([_AA_ID[c] for c in seq], dtype=torch.long)
    tok[0, L] = mcfg.eos_token_id
    if K > 1 and di:
        tok[1, :len(di)] = torch.tensor([_DI_ID.get(c, 0) for c in di[:L]], dtype=torch.long)
    return tok


def prime_structure(model, tok, mcfg, dcfg, dev, n, steps=None, seed=0):
    """Decode the 3Di track with the amino acids held fixed -> (B,K,L) canvas.

    The model's own structural reading of the protein, which the edit rounds then use as frozen
    context outside the positions they touch. The alternative -- folding the five originals and
    running foldseek -- would be a better structure and the WRONG one: what the edit needs as
    context is the structure track this model believes goes with this sequence, in its own
    vocabulary, since that is what it will condition on.
    """
    L = int((tok[0] == mcfg.eos_token_id).nonzero()[0, 0])
    B = n
    prompt = tok.unsqueeze(0).expand(B, -1, -1).contiguous().to(dev)
    given = torch.zeros_like(prompt, dtype=torch.bool)
    given[:, 0] = True                      # the whole amino-acid track is context
    given[:, 1:, L:] = True                 # the structure track's PAD tail; residues are free
    torch.manual_seed(seed)
    cv, _ = _decode(model, dev, Lmax=dcfg.canvas, batch_size=B,
                    n_steps=steps or dcfg.canvas, temperature=CFG.opt.sample_temperature,
                    gumbel_temp=CFG.opt.sample_gumbel_temp,
                    rep_penalty=0.0, max_run=0, min_len=CFG.opt.sample_min_len,
                    prompt=prompt, prompt_mask=given)
    return cv, L


def pick_positions(gain, k, editable):
    """(B,L) bool selecting each row's top-k improvable residue positions."""
    g = gain.masked_fill(~editable, float("-inf"))
    order = g.argsort(dim=1, descending=True)
    rank = order.argsort(dim=1)
    return (rank < k) & editable


def run_swap(rec, model, guide, capenc, bank_z, bank_mask, bank_names, mcfg, dcfg, dev, a):
    """-> a trajectory dict for one swap record."""
    t0 = time.perf_counter()
    B, L0 = a.n, rec["length"]
    tok = encode_canvas(rec["sequence"], mcfg, dcfg.canvas, n_tracks=mcfg.n_tracks)

    # --- the model's own structure track for the untouched protein -------------------------
    cv, L = prime_structure(model, tok, mcfg, dcfg, dev, B, a.prime_steps, a.seed)
    assert L == L0, f"{rec['swap_id']}: EOS at {L}, sequence is {L0}"

    z_cur, m_cur = capenc.encode([rec["caption_current"]], guide.filip.text_proj, dev)
    z_tgt, m_tgt = capenc.encode([rec["caption_target"]], guide.filip.text_proj, dev)
    guide.set_target_encoded(z_tgt, m_tgt)
    # The contrastive denominator: with likelihood="softmax_bank" this makes the objective
    # "raise the target's score at the expense of the current one" rather than "raise the target's
    # score", which is the difference between editing and drifting.
    guide.set_bank_encoded(z_cur if a.contrast else None, m_cur if a.contrast else None)

    def scores(canvas):
        return guide.raw_scores(aa_track(canvas), bank_z, bank_mask)        # [B, n_bank]

    s0 = scores(cv)
    i_cur, i_tgt = bank_names.index(rec["swap_id"] + "/current"), \
        bank_names.index(rec["swap_id"] + "/target")
    # THE BAR IS AN INTERPOLATION ALONG THE GAP, not a fraction of a score. FILIP similarities can
    # be negative, and `s(x0,current) * frac` moves the bar UP when the score is negative -- the
    # opposite of "part of the way there". Anchoring at the target's own starting score makes
    # stop_frac mean "close this fraction of the separability gap" for either sign, and 1.0 still
    # reproduces exactly the rule this experiment is defined by: edit until the protein matches the
    # target caption as well as it originally matched its own.
    s_cur0, s_tgt0 = float(s0[:, i_cur].mean()), float(s0[:, i_tgt].mean())
    sep = s_cur0 - s_tgt0
    stop_at = s_tgt0 + a.stop_frac * sep

    if sep <= 0 and rec["kind"] == "swap":
        # NOT A FAILURE, AND THE MOST IMPORTANT THING THIS RUN CAN SAY. The stopping rule is "reach
        # what the original already scores against its own caption", so a target that starts at or
        # above that bar is already there and the loop correctly does nothing. What it means is
        # that the classifier does not separate these two captions on this protein -- so any edit
        # made toward this target would be chasing noise, and any movement reported for this row
        # would mean nothing. Read it as the separability gate firing, not as a bug.
        print(f"[edit] {rec['swap_id']}: separability {sep:+.4f} <= 0 -- the target caption "
              f"already scores at or above the current one on the untouched protein, so there is "
              f"no gap to close and no edit to make. This row is uninformative by construction.",
              flush=True)

    budget = max(1, int(round(a.max_edit_frac * L)))
    pos = torch.arange(dcfg.canvas, device=dev)
    is_res = (pos < L).unsqueeze(0).expand(B, -1)
    ever = torch.zeros(B, dcfg.canvas, dtype=torch.bool, device=dev)
    active = torch.ones(B, dtype=torch.bool, device=dev)
    traj = [dict(round=0, n_edited=[0] * B, scores=s0.float().cpu().tolist(),
                 seqs=decode_seqs(cv, mcfg)[0])]

    for rnd in range(1, a.rounds + 1):
        s = scores(cv)
        active &= (s[:, i_tgt] < stop_at) & (ever.sum(1) < budget)
        if not bool(active.any()):
            break
        delta = guide.tag_delta(aa_track(cv))                                # [B, L, 20]
        cur = aa_track(cv).clamp(max=19).unsqueeze(-1)
        gain = delta.max(-1).values - delta.gather(-1, cur).squeeze(-1)      # [B, L]
        # Once the budget is spent a row may only refine what it has already touched, so an edit
        # cannot quietly spread across the whole protein one round at a time.
        spent = ever.sum(1, keepdim=True) >= budget
        editable = is_res & active.unsqueeze(1) & torch.where(spent, ever, is_res)
        picked = pick_positions(gain, a.k, editable)
        if not bool(picked.any()):
            break

        given = torch.ones_like(cv, dtype=torch.bool)
        given[:, :, :] = ~picked.unsqueeze(1)          # free the picks in BOTH tracks
        given[:, :, L:] = True                         # EOS and the PAD tail stay context
        torch.manual_seed(a.seed + 7919 * rnd)
        cv, _ = _decode(model, dev, Lmax=dcfg.canvas, batch_size=B,
                        n_steps=max(2 * a.k, 8),
                        temperature=a.temperature, gumbel_temp=CFG.opt.sample_gumbel_temp,
                        rep_penalty=CFG.opt.sample_rep_penalty,
                        rep_periods=CFG.opt.sample_rep_periods, max_run=CFG.opt.sample_max_run,
                        min_len=CFG.opt.sample_min_len, guidance_fn=guide,
                        prompt=cv, prompt_mask=given)
        ever |= picked
        seqs = decode_seqs(cv, mcfg)[0]
        traj.append(dict(round=rnd, n_edited=ever.sum(1).cpu().tolist(),
                         scores=scores(cv).float().cpu().tolist(), seqs=seqs,
                         picked=[p.nonzero().flatten().cpu().tolist() for p in picked]))

    final = decode_seqs(cv, mcfg)[0]
    dis = decode_struct(cv, mcfg)[0] if mcfg.n_tracks == 2 else [None] * B
    ident = [sum(x == y for x, y in zip(rec["sequence"], f)) / max(L, 1) for f in final]
    return dict(
        swap_id=rec["swap_id"], kind=rec["kind"], accession=rec["accession"], length=L,
        stub=bool(getattr(a, "stub_filip", False)),
        n=B, rounds_run=len(traj) - 1, budget=budget, gamma=a.gamma, contrast=bool(a.contrast),
        separability=sep, stop_at=stop_at, bank=bank_names,
        s0=s0.float().cpu().tolist(), s_final=traj[-1]["scores"],
        identity_to_original=ident, n_edited=ever.sum(1).cpu().tolist(),
        original=rec["sequence"], edited=final, edited_3di=dis,
        trajectory=traj, seconds=round(time.perf_counter() - t0, 1))


def build_bank(recs, capenc, text_proj, dev):
    """Every caption in the experiment, encoded once -> (z, mask, names).

    Scoring against all of them on every round costs one matrix multiply and buys the whole
    specificity analysis after the fact: a protein that moves toward its target AND toward five
    unrelated targets has not been steered, it has drifted.
    """
    texts, names = [], []
    for r in recs:
        for tag, key in (("current", "caption_current"), ("target", "caption_target")):
            texts.append(r[key])
            names.append(f"{r['swap_id']}/{tag}")
    z, m = capenc.encode(texts, text_proj, dev)
    return z, m, names


def summarize(paths):
    """The table. Reads whatever per-rank JSONLs exist, so it works mid-run.

    `d target` and `d current` are the two halves of the push. `spec` is the one that matters:
    (target gained) - (mean gain across every OTHER swap's target caption). A protein that drifts
    toward every caption in the bank has not been steered, and only this column can tell the
    difference -- which is why every round scores against the whole bank.
    """
    import glob as glob_
    recs = []
    for pat in paths:
        for p in sorted(glob_.glob(pat)):
            recs += [json.loads(l) for l in open(p) if l.strip()]
    if not recs:
        raise SystemExit(f"nothing to summarize in {paths}")
    if any(r.get("stub") for r in recs):
        print("!! STUBBED RUN: these scores come from src/stub_filip.py, not FILIP. !!\n")
    print(f"{'swap':<38} {'kind':<14} {'sep':>8} {'d target':>9} {'d current':>10} "
          f"{'spec':>8} {'edited':>9} {'ident':>7} {'rnds':>5}")
    print("-" * 114)
    for r in sorted(recs, key=lambda x: (x["swap_id"].split("__")[0], x["kind"])):
        names, s0, sf = r["bank"], torch.tensor(r["s0"]), torch.tensor(r["s_final"])
        base = r["swap_id"].split("__")[0]
        i_t = names.index(base + "/target") if base + "/target" in names else None
        i_c = names.index(base + "/current") if base + "/current" in names else None
        if i_t is None or i_c is None:
            continue
        d = (sf - s0).mean(0)
        others = [j for j, nm in enumerate(names)
                  if nm.endswith("/target") and not nm.startswith(base + "/")]
        spec = float(d[i_t] - (d[others].mean() if others else 0.0))
        print(f"{r['swap_id']:<38} {r['kind']:<14} {r['separability']:>+8.4f} "
              f"{float(d[i_t]):>+9.4f} {float(d[i_c]):>+10.4f} {spec:>+8.4f} "
              f"{sum(r['n_edited']) / len(r['n_edited']):>5.0f}/{r['length']:<3} "
              f"{sum(r['identity_to_original']) / len(r['identity_to_original']):>6.1%} "
              f"{r['rounds_run']:>5}")
    print("\n[edit] sep <= 0 means the classifier does not separate that swap's two captions on "
          "the untouched\n[edit] protein; that row is uninformative however it moves. A swap whose "
          "`spec` does not beat its\n[edit] own __decoy row moved toward captions in general, not "
          "toward this one.")


def main():
    sys.stdout.reconfigure(line_buffering=True)
    dcfg = CFG.data
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", default=CFG.device)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default="runs/edits.jsonl")
    ap.add_argument("--swaps", default=None)
    ap.add_argument("--only", default=None, help="colon-separated swap ids")
    ap.add_argument("--kinds", default="swap:control_null:control_decoy")
    ap.add_argument("--n", type=int, default=4, help="replicates per swap, batched")
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--k", type=int, default=8, help="positions edited per round")
    ap.add_argument("--max-edit-frac", type=float, default=0.20)
    ap.add_argument("--stop-frac", type=float, default=1.0,
                    help="stop when s(x,target) has closed this fraction of the gap between its "
                         "starting value and s(x0,current). 1.0 = all the way.")
    ap.add_argument("--gamma", type=float, default=10.0)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--contrast", action=argparse.BooleanOptionalAction, default=True,
                    help="put the CURRENT caption in the softmax bank (the push, not just a pull)")
    ap.add_argument("--guide-mode", default="tag", choices=("tag", "deg"))
    ap.add_argument("--guide-chunk", type=int, default=4)
    ap.add_argument("--filip-ckpt", default=FILIP_CKPT)
    ap.add_argument("--filip-cache", default=FILIP_CACHE)
    ap.add_argument("--prime-steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--canvas", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--stub-filip", action="store_true",
                    help="replace FILIP and the text encoder with a toy composition objective "
                         "(src/stub_filip.py). Exercises the loop without the Aurora assets; "
                         "every number it produces is about the loop, never about proteins.")
    ap.add_argument("--no-ipex", action="store_true")
    ap.add_argument("--summarize", nargs="*", default=None,
                    help="read result JSONL(s) (globs ok) and print the table; no model is loaded")
    a = ap.parse_args()

    if a.summarize is not None:
        return summarize(a.summarize or [a.out, a.out[:-6] + ".rank*.jsonl"])

    env = init_distributed(a.device, no_dist=True)
    dev = env.device
    rank, world = env.rank, env.world_size
    mcfg = CFG.model_config()
    if a.smoke:
        mcfg.d_model, mcfg.n_heads, mcfg.d_ff = 128, 4, 384
        mcfg.n_upstream, mcfg.n_middle, mcfg.n_downstream = 1, 2, 1
        mcfg.n_recurrence, mcfg.checkpoint_chunk, mcfg.grad_checkpoint = 1, 2, False
    if a.canvas:
        dcfg.canvas = a.canvas

    recs = read_swaps(a.swaps) if a.swaps else read_swaps()
    kinds = {t for t in a.kinds.split(":") if t}
    recs = [r for r in recs if r["kind"] in kinds]
    if a.only:
        want = {t for t in a.only.split(":") if t}
        recs = [r for r in recs if r["swap_id"] in want]
    if not recs:
        raise SystemExit("no swap records selected")
    too_long = [r for r in recs if r["length"] + 1 > dcfg.canvas]
    if too_long:
        raise SystemExit(f"{[r['swap_id'] for r in too_long]} exceed the {dcfg.canvas} canvas")

    model = LoopedDiffusionLM(mcfg).to(dev).eval()
    if a.smoke:
        print(f"[edit] SMOKE: random init, d_model={mcfg.d_model}, canvas={dcfg.canvas}. The "
              f"sequences are noise; what is tested is that the loop runs.", flush=True)
    else:
        ckpt = a.ckpt or find_latest_ckpt(CKPT_DIR)
        if not ckpt or not os.path.exists(ckpt):
            raise SystemExit(f"no checkpoint (looked at {a.ckpt or CKPT_DIR})")
        st = torch.load(ckpt, map_location=dev, weights_only=True)
        model.load_state_dict(st["model"])
        if ipex is not None and dev.type == "xpu" and not a.no_ipex:
            model = ipex.optimize(model, dtype=torch.bfloat16)
        if rank == 0:
            print(f"[edit] policy {ckpt} (step {st.get('step', '?')}) on {dev}", flush=True)

    if a.stub_filip:
        from .stub_filip import StubCaptionEncoder, StubGuidance
        guide = StubGuidance(mcfg, dev, gamma=a.gamma, verbose=(rank == 0))
        capenc = StubCaptionEncoder(verbose=(rank == 0))
    else:
        from .captions import CaptionEncoder
        from .filip_guidance import FilipGuidance
        guide = FilipGuidance(mcfg, dev, ckpt=a.filip_ckpt, cache_dir=a.filip_cache,
                              gamma=a.gamma, mode=a.guide_mode, likelihood="softmax_bank",
                              chunk=a.guide_chunk, verbose=(rank == 0))
        capenc = CaptionEncoder(a.filip_cache, device=dev, verbose=(rank == 0))
    bank_z, bank_mask, bank_names = build_bank(recs, capenc, guide.filip.text_proj, dev)
    if rank == 0:
        print(f"[edit] {len(recs)} record(s) x {a.n} replicate(s) | gamma={a.gamma} "
              f"contrast={'on' if a.contrast else 'OFF'} | budget={a.max_edit_frac:.0%} "
              f"| {a.k} positions/round x {a.rounds} rounds | bank of {len(bank_names)} captions",
              flush=True)

    mine = [r for i, r in enumerate(recs) if i % world == rank]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    out_path = a.out if world == 1 else f"{a.out[:-6]}.rank{rank:03d}.jsonl"
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            try:
                done.add(json.loads(line)["swap_id"])
            except Exception:
                continue

    with open(out_path, "a") as fh:
        for r in mine:
            if r["swap_id"] in done:
                print(f"[edit] {r['swap_id']}: already done", flush=True)
                continue
            res = run_swap(r, model, guide, capenc, bank_z, bank_mask, bank_names,
                           mcfg, dcfg, dev, a)
            fh.write(json.dumps(res) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
            i_c = bank_names.index(r["swap_id"] + "/current")
            i_t = bank_names.index(r["swap_id"] + "/target")
            s0, sf = torch.tensor(res["s0"]), torch.tensor(res["s_final"])
            print(f"[edit] {res['swap_id']:<38} sep {res['separability']:+.4f} | "
                  f"target {float(s0[:, i_t].mean()):+.4f} -> {float(sf[:, i_t].mean()):+.4f} "
                  f"(bar {res['stop_at']:+.4f}) | current {float(s0[:, i_c].mean()):+.4f} -> "
                  f"{float(sf[:, i_c].mean()):+.4f} | edited "
                  f"{sum(res['n_edited']) / len(res['n_edited']):.0f}/{res['length']} | "
                  f"ident {sum(res['identity_to_original']) / len(res['identity_to_original']):.1%}"
                  f" | {res['rounds_run']} rounds, {res['seconds']:.0f}s", flush=True)
    print(f"[edit] rank {rank}: wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
