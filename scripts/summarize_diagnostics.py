#!/usr/bin/env python
"""Summarize a run's diagnostics.jsonl into the divergence-relevant view.

Reads what rgrl.diagnostics wrote and answers, in order, the questions that
distinguish the ways a run can destabilize:

  1. WHEN did it turn? The step where loss stopped improving, next to the step
     where the gradient norm first spiked. Grad spike BEFORE the loss rise means
     one bad optimizer step; grad rise AFTER means the loss is driving the
     gradients, i.e. a feedback loop already underway.
  2. Is the clip threshold doing the work? clip_frac near 1.0 means
     max_grad_norm, not the LR schedule, sets the effective step size.
  3. WHICH block moved? Per-module update/param and trust ratios at the turn
     versus early on -- a module whose update_ratio climbs while its grad does
     not is Adam's second moment decaying, not a data spike.

    python scripts/summarize_diagnostics.py <run_dir>/diagnostics.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("jsonl", type=Path)
    ap.add_argument("--top", type=int, default=8, help="modules to show, by update_ratio")
    args = ap.parse_args()

    recs = load(args.jsonl)
    if not recs:
        raise SystemExit(f"{args.jsonl} is empty")

    print(f"{len(recs)} records, steps {recs[0]['step']}..{recs[-1]['step']}\n")
    print(f"{'step':>8} {'loss':>9} {'lr':>9} {'gnorm':>9} {'gmax':>9} {'clip':>6} {'upd/par':>9} {'nf':>3} {'spk':>4}")
    for r in recs:
        d = r.get("detail") or {}
        ur = d.get("update_ratio")
        print(
            f"{r['step']:>8} {r['loss']:>9.4f} {r['lr']:>9.2e} "
            f"{r['grad_norm_mean']:>9.3g} {r['grad_norm_max']:>9.3g} "
            f"{r['clip_frac']:>6.2f} {(f'{ur:.2e}' if ur is not None else '-'):>9} "
            f"{r['n_nonfinite_grad']:>3} {r['n_spikes']:>4}"
        )

    best = min(recs, key=lambda r: r["loss"])
    print(f"\nbest loss {best['loss']:.4f} at step {best['step']}")
    worst = max(recs, key=lambda r: r["loss"])
    if worst["loss"] > 2 * best["loss"]:
        print(
            f"loss rose to {worst['loss']:.4f} by step {worst['step']} "
            f"({worst['loss'] / best['loss']:.1f}x the best)"
        )
    spikes = [r for r in recs if r["n_spikes"]]
    if spikes:
        print(
            f"first grad spike at step {spikes[0].get('first_spike_step')} "
            f"(record {spikes[0]['step']}); {sum(r['n_spikes'] for r in recs)} spikes total"
        )
        rel = "BEFORE" if spikes[0]["step"] <= best["step"] else "AFTER"
        print(f"  -> spike came {rel} the best-loss step: "
              + ("suspect a single bad update / clip failure"
                 if rel == "BEFORE" else "suspect a feedback loop already underway"))
    else:
        print("no grad spikes: divergence was smooth, not a single bad step")
    nf = sum(r["n_nonfinite_grad"] for r in recs)
    if nf:
        print(f"*** {nf} non-finite gradient steps -- with clipping on, one poisons every param")
    clipped = [r for r in recs if r["clip_frac"] > 0.5]
    if clipped:
        print(
            f"clip_frac > 0.5 in {len(clipped)}/{len(recs)} records (from step "
            f"{clipped[0]['step']}): max_grad_norm is setting the step size"
        )

    det = [r for r in recs if r.get("detail", {}).get("by_module")]
    if len(det) >= 2:
        first, last = det[0], det[-1]
        print(f"\nper-module, step {first['step']} -> {last['step']} "
              "(grad | upd/param | trust):")
        mods = last["detail"]["by_module"]
        order = sorted(mods, key=lambda m: -(mods[m].get("update_ratio") or 0))
        for m in order[: args.top]:
            a = first["detail"]["by_module"].get(m, {})
            b = mods[m]
            fmt = lambda x: f"{x:.3g}" if x is not None else "-"
            print(
                f"  {m:<28} {fmt(a.get('grad')):>9} -> {fmt(b.get('grad')):>9} | "
                f"{fmt(a.get('update_ratio')):>9} -> {fmt(b.get('update_ratio')):>9} | "
                f"{fmt(a.get('trust_ratio')):>6} -> {fmt(b.get('trust_ratio')):>6}"
            )
        pn0 = first["detail"].get("param_norm")
        pn1 = last["detail"].get("param_norm")
        if pn0 and pn1:
            print(f"  total param norm {pn0:.4g} -> {pn1:.4g} ({pn1 / pn0:.2f}x)")


if __name__ == "__main__":
    main()
