"""Optimization diagnostics for training runs that destabilize.

Motivation: two sim-bottles baselines learned well and then *diverged* -- train
loss 0.027 -> 1.03 with val tracking it (0.033 -> 1.12), i.e. the fit to the
TRAINING distribution collapsed ~40x, so it was not a generalization gap. The
existing log line (loss / lr / it-per-s) contains nothing that can distinguish
the candidate causes, so a diverged run had to be re-run from scratch to learn
anything. This records, cheaply and continuously, the quantities that separate
them:

  grad_norm       pre-clip global gradient norm. The primary signal: a spike
                  train ahead of the loss rise means the optimizer took one bad
                  step; a slow ramp means a feedback loop.
  clip_frac       fraction of steps whose norm exceeded max_grad_norm. If this
                  sits near 1.0 the clip threshold is the de-facto step size and
                  the loss is being driven by it, not by the LR schedule.
  update_ratio    ||delta p|| / ||p||, measured across the real opt.step(). The
                  scale-free "how far did the weights actually move" number;
                  healthy late training is ~1e-4 or below. Adam can produce a
                  large update from *small* grads when the second moment decays,
                  which grad_norm alone cannot show.
  trust_ratio     max over params of |exp_avg| / (sqrt(exp_avg_sq) + eps). Adam's
                  effective per-element step in units of LR. Blows up when
                  exp_avg_sq collapses -- the classic slow-divergence mechanism.
  param_norm      global weight norm; a monotone climb points at weight-decay /
                  LR interaction rather than a data spike.

Everything above is global and costs a few small reductions per step. Per-module
breakdowns (which block is amplifying) are recorded only every ``diag_every``
steps, and on a spike, where the extra cost does not matter.

Output is one JSON object per record in ``diagnostics.jsonl`` in the run dir, so
a diverged run can be diagnosed after the fact instead of re-run.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import torch

# Group parameters by their first few name components. Deep enough to separate
# the vision backbone's blocks from the DiT's, shallow enough that the
# per-module record stays a handful of lines rather than hundreds.
_MODULE_DEPTH = 2


def _module_of(name: str) -> str:
    return ".".join(name.split(".")[:_MODULE_DEPTH])


class TrainDiagnostics:
    """Records optimization health to ``<out>/diagnostics.jsonl``.

    Cheap path (every step): global grad norm, clip rate, loss.
    Detailed path (every ``diag_every`` steps, and on spikes): per-module grad
    and update norms, Adam trust ratios, parameter norms.
    """

    def __init__(
        self,
        out: Path,
        params: Iterable[tuple[str, torch.nn.Parameter]],
        opt: torch.optim.Optimizer,
        diag_every: int,
        spike_factor: float,
        max_grad_norm: float | None,
        enabled: bool = True,
    ):
        self.enabled = enabled
        if not enabled:
            return
        # Materialize once: named_parameters() rebuilds a generator each call,
        # and this is touched every step.
        self._params = [(n, p) for n, p in params if p.requires_grad]
        self._opt = opt
        self._path = Path(out) / "diagnostics.jsonl"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self.diag_every = diag_every
        self.spike_factor = spike_factor
        self.max_grad_norm = max_grad_norm
        # Running state between flushes.
        self._n = 0
        self._n_clipped = 0
        self._n_nonfinite_grad = 0
        self._gn_sum = 0.0
        self._gn_max = 0.0
        self._gn_max_step = -1
        self._ema_gn: float | None = None  # spike reference
        self._n_spikes = 0
        self._first_spike_step: int | None = None
        self._snapshot: list[torch.Tensor] | None = None
        self._pending_detail = False
        self._grad_by_module: dict[str, float] = {}

    # ---- per-step, cheap -------------------------------------------------

    def before_step(self, step: int) -> None:
        """Snapshot weights when this step will need an update_ratio."""
        if not self.enabled:
            return
        self._pending_detail = self.diag_every > 0 and (step + 1) % self.diag_every == 0
        if self._pending_detail:
            # Flat clone of the params we will measure; freed at record().
            self._snapshot = [p.detach().clone() for _, p in self._params]

    def record_grad_norm(self, step: int, total_norm: torch.Tensor | float | None) -> float:
        """Log the pre-clip global grad norm.

        ``total_norm`` is what ``clip_grad_norm_`` already returned (pre-clip, so
        it is exactly the quantity of interest and costs nothing extra). Pass
        None when clipping is off and it will be computed here.
        """
        if not self.enabled:
            return 0.0
        if total_norm is None:
            # An infinite threshold makes clip_grad_norm_ a pure measurement: it
            # returns the true norm and its clip coefficient is clamped to 1, so
            # gradients are left untouched.
            total_norm = torch.nn.utils.clip_grad_norm_(
                [p for _, p in self._params], float("inf")
            )
        gn = float(total_norm) if not isinstance(total_norm, torch.Tensor) else total_norm.item()
        self._n += 1
        if not math.isfinite(gn):
            self._n_nonfinite_grad += 1
            # A non-finite norm poisons every downstream statistic (and, with
            # clipping on, every gradient via the clip coefficient), so flag the
            # step but keep it out of the sums.
            self._pending_detail = True
            return gn
        self._gn_sum += gn
        if gn > self._gn_max:
            self._gn_max, self._gn_max_step = gn, step
        if self.max_grad_norm is not None and gn > self.max_grad_norm:
            self._n_clipped += 1
        # Spike test against a slow EMA of the norm: catches "one bad step"
        # onsets that a per-window mean would average away.
        if self._ema_gn is None:
            self._ema_gn = gn
        else:
            if gn > self.spike_factor * self._ema_gn:
                self._n_spikes += 1
                if self._first_spike_step is None:
                    self._first_spike_step = step
                self._pending_detail = True
            self._ema_gn = 0.99 * self._ema_gn + 0.01 * gn
        if self._pending_detail:
            # Must happen HERE, not in flush(): this runs before opt.step(), so
            # gradients are still alive (flush() sees them already zeroed) and the
            # weights are still pre-update. A spike detected above also lands
            # here, so take its snapshot now rather than losing update_ratio.
            self._grad_by_module = self._norms_by_module(
                lambda p: p.grad if p.grad is not None else None
            )
            if self._snapshot is None:
                self._snapshot = [p.detach().clone() for _, p in self._params]
        return gn

    def _norms_by_module(self, pick) -> dict[str, float]:
        """Grouped L2 norm of whatever ``pick`` returns for each parameter."""
        acc: dict[str, float] = {}
        with torch.no_grad():
            for name, p in self._params:
                t = pick(p)
                if t is None:
                    continue
                mod = _module_of(name)
                acc[mod] = acc.get(mod, 0.0) + float(t.detach().float().norm() ** 2)
        return {k: math.sqrt(v) for k, v in acc.items()}

    # ---- periodic / on-spike, detailed -----------------------------------

    def flush(self, step: int, loss: float, lr: float) -> dict | None:
        """Write a record. Call at the existing log cadence."""
        if not self.enabled or self._n == 0:
            return None
        rec = {
            "step": step,
            "loss": loss,
            "lr": lr,
            "grad_norm_mean": self._gn_sum / max(1, self._n - self._n_nonfinite_grad),
            "grad_norm_max": self._gn_max,
            "grad_norm_max_step": self._gn_max_step,
            "clip_frac": self._n_clipped / self._n,
            "n_nonfinite_grad": self._n_nonfinite_grad,
            "n_spikes": self._n_spikes,
            "first_spike_step": self._first_spike_step,
        }
        if self._pending_detail:
            rec["detail"] = self._detail()
        self._write(rec)
        self._n = self._n_clipped = self._n_nonfinite_grad = 0
        self._gn_sum = 0.0
        self._gn_max = 0.0
        self._gn_max_step = -1
        self._n_spikes = 0
        self._pending_detail = False
        self._snapshot = None
        self._grad_by_module = {}
        return rec

    def _detail(self) -> dict:
        """Per-module grad/update/param norms and Adam trust ratios."""
        with torch.no_grad():
            g_by_mod = self._grad_by_module
            p_sq: dict[str, float] = {}
            u_sq: dict[str, float] = {}
            trust: dict[str, float] = {}
            total_p_sq = total_u_sq = 0.0
            for i, (name, p) in enumerate(self._params):
                mod = _module_of(name)
                pn = float(p.detach().float().norm() ** 2)
                p_sq[mod] = p_sq.get(mod, 0.0) + pn
                total_p_sq += pn
                if self._snapshot is not None:
                    un = float((p.detach() - self._snapshot[i]).float().norm() ** 2)
                    u_sq[mod] = u_sq.get(mod, 0.0) + un
                    total_u_sq += un
                st = self._opt.state.get(p)
                if st and "exp_avg" in st and "exp_avg_sq" in st:
                    r = (
                        st["exp_avg"].detach().float().abs()
                        / (st["exp_avg_sq"].detach().float().sqrt() + 1e-8)
                    ).max()
                    trust[mod] = max(trust.get(mod, 0.0), float(r))
            det = {
                "param_norm": math.sqrt(total_p_sq),
                "by_module": {
                    mod: {
                        "grad": g_by_mod.get(mod),
                        "param": math.sqrt(v),
                        "update": math.sqrt(u_sq[mod]) if mod in u_sq else None,
                        # Scale-free: how far this block moved relative to its size.
                        "update_ratio": (
                            math.sqrt(u_sq[mod] / v) if mod in u_sq and v > 0 else None
                        ),
                        "trust_ratio": trust.get(mod),
                    }
                    for mod, v in sorted(p_sq.items())
                },
            }
            if self._snapshot is not None and total_p_sq > 0:
                det["update_ratio"] = math.sqrt(total_u_sq / total_p_sq)
            return det

    def _write(self, rec: dict) -> None:
        with self._path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
