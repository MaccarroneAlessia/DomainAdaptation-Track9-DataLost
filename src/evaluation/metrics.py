"""
metrics.py — Training Metrics Logger & Evaluator
=================================================
Persona 3 — Weighting & Evaluation Strategist

Tracks per-epoch:
  • target accuracy (head_tgt)
  • source influence ratio  S1 / (S1 + S2)   [from DynamicWeighter]
  • per-component losses: cls_s1, cls_s2, cls_tgt, adv, pseudo
  • average prediction entropy on target

All data is stored in plain Python lists → easy to serialise (JSON / pickle)
and plot with matplotlib.

Usage
-----
    from evaluation.metrics import MetricsLogger, compute_entropy

    logger = MetricsLogger()

    # inside training loop:
    logger.log_batch(w1=0.6, w2=0.4, loss_dict={...})

    # at epoch end:
    logger.end_epoch(target_acc=42.3)

    # save / plot:
    logger.save("experiments/logs/run_01.json")
    fig = logger.plot()
    fig.savefig("figures/training_curves.png", dpi=150)
"""

from __future__ import annotations

import json
import time
import math
from pathlib import Path
from typing import Any

import torch


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def compute_entropy(logits: torch.Tensor) -> float:
    """
    Mean Shannon entropy of a batch of logits.

    Args
    ----
    logits : [B, C]  — raw (un-normalised) class scores

    Returns
    -------
    float  — mean entropy in nats (per sample)
    """
    with torch.no_grad():
        probs = logits.softmax(dim=-1).clamp(min=1e-9)
        entropy = -(probs * probs.log()).sum(dim=-1)  # [B]
        return entropy.mean().item()


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Top-1 accuracy (%)."""
    with torch.no_grad():
        preds = logits.argmax(dim=-1)
        return (preds == labels).float().mean().item() * 100.0


# ──────────────────────────────────────────────────────────────────────────────
# MetricsLogger
# ──────────────────────────────────────────────────────────────────────────────

class MetricsLogger:
    """
    Accumulates batch-level statistics and aggregates them at epoch boundaries.

    Attributes (per-epoch lists, after calling end_epoch)
    -------------------------------------------------------
    epochs         : list[int]
    target_acc     : list[float]    — % accuracy on target (head_tgt)
    influence_s1   : list[float]    — mean w1 over all batches in epoch
    influence_s2   : list[float]    — mean w2
    loss_total     : list[float]
    loss_cls_s1    : list[float]
    loss_cls_s2    : list[float]
    loss_cls_tgt   : list[float]
    loss_adv       : list[float]
    loss_pseudo    : list[float]
    entropy_tgt    : list[float]    — mean prediction entropy on target
    epoch_duration : list[float]    — seconds per epoch
    """

    _LOSS_KEYS = ("total", "cls_s1", "cls_s2", "cls_tgt", "adv", "pseudo")

    def __init__(self, run_name: str = "run"):
        self.run_name = run_name
        self._reset_accumulators()

        # Per-epoch history
        self.epochs:         list[int]   = []
        self.target_acc:     list[float] = []
        self.influence_s1:   list[float] = []
        self.influence_s2:   list[float] = []
        self.entropy_tgt:    list[float] = []
        self.epoch_duration: list[float] = []
        self.losses: dict[str, list[float]] = {k: [] for k in self._LOSS_KEYS}

        self._epoch_start = time.time()

    # ── batch-level ──────────────────────────────────────────────────────────

    def log_batch(
        self,
        w1: float,
        w2: float,
        loss_dict: dict[str, float],
        entropy: float | None = None,
    ):
        """
        Call once per training batch.

        Parameters
        ----------
        w1, w2    : source weights from DynamicWeighter (should sum to ~1)
        loss_dict : keys are a subset of {"total","cls_s1","cls_s2","cls_tgt","adv","pseudo"}
        entropy   : optional pre-computed target entropy for this batch
        """
        self._acc_w1.append(float(w1))
        self._acc_w2.append(float(w2))
        for k in self._LOSS_KEYS:
            if k in loss_dict:
                self._acc_losses[k].append(float(loss_dict[k]))
        if entropy is not None:
            self._acc_entropy.append(float(entropy))

    # ── epoch-level ──────────────────────────────────────────────────────────

    def end_epoch(
        self,
        target_acc: float,
        epoch: int | None = None,
    ):
        """
        Finalise the current epoch.  Call after iterating the full eval loader.

        Parameters
        ----------
        target_acc : float  — accuracy (%) on target set
        epoch      : int    — epoch index; auto-increments if None
        """
        ep = epoch if epoch is not None else (len(self.epochs) + 1)
        self.epochs.append(ep)
        self.target_acc.append(float(target_acc))

        # Source influence
        if self._acc_w1:
            self.influence_s1.append(sum(self._acc_w1) / len(self._acc_w1))
            self.influence_s2.append(sum(self._acc_w2) / len(self._acc_w2))
        else:
            self.influence_s1.append(0.5)
            self.influence_s2.append(0.5)

        # Losses
        for k in self._LOSS_KEYS:
            vals = self._acc_losses[k]
            self.losses[k].append(sum(vals) / len(vals) if vals else 0.0)

        # Entropy
        self.entropy_tgt.append(
            sum(self._acc_entropy) / len(self._acc_entropy)
            if self._acc_entropy else math.nan
        )

        # Timing
        now = time.time()
        self.epoch_duration.append(now - self._epoch_start)
        self._epoch_start = now

        self._reset_accumulators()

    def _reset_accumulators(self):
        self._acc_w1:     list[float] = []
        self._acc_w2:     list[float] = []
        self._acc_losses: dict[str, list[float]] = {k: [] for k in self._LOSS_KEYS}
        self._acc_entropy: list[float] = []

    # ── serialisation ────────────────────────────────────────────────────────

    def state_dict(self) -> dict[str, Any]:
        return {
            "run_name":       self.run_name,
            "epochs":         self.epochs,
            "target_acc":     self.target_acc,
            "influence_s1":   self.influence_s1,
            "influence_s2":   self.influence_s2,
            "entropy_tgt":    self.entropy_tgt,
            "epoch_duration": self.epoch_duration,
            "losses":         self.losses,
        }

    def load_state_dict(self, d: dict[str, Any]):
        for k, v in d.items():
            setattr(self, k, v)

    def save(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.state_dict(), f, indent=2)
        print(f"[MetricsLogger] saved → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "MetricsLogger":
        with open(path) as f:
            d = json.load(f)
        obj = cls(run_name=d.get("run_name", "run"))
        obj.load_state_dict(d)
        return obj

    # ── console summary ───────────────────────────────────────────────────────

    def print_last(self):
        if not self.epochs:
            print("[MetricsLogger] No epochs logged yet.")
            return
        ep = self.epochs[-1]
        acc = self.target_acc[-1]
        w1  = self.influence_s1[-1]
        w2  = self.influence_s2[-1]
        ent = self.entropy_tgt[-1]
        lt  = self.losses["total"][-1]
        print(
            f"[Epoch {ep:>3d}] "
            f"Acc={acc:.2f}%  "
            f"w(S1/S2)={w1:.3f}/{w2:.3f}  "
            f"H={ent:.4f}  "
            f"L_tot={lt:.4f}"
        )

    # ── plotting ──────────────────────────────────────────────────────────────

    def plot(self, figsize: tuple[int, int] = (14, 10)):
        """
        Returns a matplotlib Figure with 4 subplots:
          1. Target accuracy evolution
          2. Source influence ratio (S1 vs S2)
          3. Loss components
          4. Target entropy
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
        except ImportError:
            raise ImportError("matplotlib is required for plotting. pip install matplotlib")

        fig = plt.figure(figsize=figsize)
        fig.suptitle(f"Training Curves — {self.run_name}", fontsize=14, fontweight="bold")
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

        ep = self.epochs

        # 1 — Target accuracy
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(ep, self.target_acc, color="#2196F3", linewidth=2, marker="o", ms=4)
        ax1.set_title("Target Accuracy (head_tgt)")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy (%)")
        ax1.grid(True, alpha=0.3)
        if self.target_acc:
            best_ep = ep[self.target_acc.index(max(self.target_acc))]
            ax1.axvline(best_ep, color="#2196F3", linestyle="--", alpha=0.5,
                        label=f"Best @ ep {best_ep}")
            ax1.legend(fontsize=8)

        # 2 — Source influence
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.stackplot(ep, self.influence_s1, self.influence_s2,
                      labels=["Source 1 (HMDB)", "Source 2 (UCF)"],
                      colors=["#FF5722", "#4CAF50"], alpha=0.7)
        ax2.set_title("Source Influence Ratio")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Weight")
        ax2.set_ylim(0, 1)
        ax2.legend(fontsize=8, loc="upper right")
        ax2.grid(True, alpha=0.3)

        # 3 — Loss components
        ax3 = fig.add_subplot(gs[1, 0])
        loss_styles = {
            "total":   ("#212121", "-",  2.5),
            "cls_s1":  ("#FF5722", "--", 1.5),
            "cls_s2":  ("#4CAF50", "--", 1.5),
            "cls_tgt": ("#2196F3", ":",  1.5),
            "adv":     ("#9C27B0", "-.", 1.5),
            "pseudo":  ("#FF9800", "-.", 1.5),
        }
        for key, (color, ls, lw) in loss_styles.items():
            vals = self.losses.get(key, [])
            if any(v != 0.0 for v in vals):
                ax3.plot(ep, vals, color=color, linestyle=ls, linewidth=lw, label=key)
        ax3.set_title("Loss Components")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Loss")
        ax3.legend(fontsize=7, ncol=2)
        ax3.grid(True, alpha=0.3)

        # 4 — Entropy
        ax4 = fig.add_subplot(gs[1, 1])
        valid_ent = [(e, h) for e, h in zip(ep, self.entropy_tgt) if not math.isnan(h)]
        if valid_ent:
            ep_e, ent_e = zip(*valid_ent)
            ax4.plot(ep_e, ent_e, color="#9C27B0", linewidth=2, marker="s", ms=4)
        ax4.set_title("Target Prediction Entropy")
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("H (nats)")
        ax4.grid(True, alpha=0.3)

        return fig


# ──────────────────────────────────────────────────────────────────────────────
# Comparative Report
# ──────────────────────────────────────────────────────────────────────────────

def comparative_table(
    runs: dict[str, "MetricsLogger"],
    latex: bool = False,
) -> str:
    """
    Build a text/LaTeX table comparing multiple runs.

    Parameters
    ----------
    runs   : dict  name → MetricsLogger
    latex  : bool  if True, return LaTeX tabular; else plain text

    Example
    -------
    >>> table = comparative_table({
    ...     "Baseline":     logger_base,
    ...     "Multi-Source": logger_ms,
    ...     "Weighted DA":  logger_w,
    ... })
    >>> print(table)
    """
    headers = ["Run", "Best Acc (%)", "Last Acc (%)", "Min Entropy", "Mean w(S1)", "Mean w(S2)"]
    rows = []
    for name, lg in runs.items():
        best_acc = max(lg.target_acc) if lg.target_acc else float("nan")
        last_acc = lg.target_acc[-1]  if lg.target_acc else float("nan")
        ents = [h for h in lg.entropy_tgt if not math.isnan(h)]
        min_ent  = min(ents) if ents else float("nan")
        mean_w1  = sum(lg.influence_s1) / len(lg.influence_s1) if lg.influence_s1 else float("nan")
        mean_w2  = sum(lg.influence_s2) / len(lg.influence_s2) if lg.influence_s2 else float("nan")
        rows.append([name, f"{best_acc:.2f}", f"{last_acc:.2f}",
                     f"{min_ent:.4f}", f"{mean_w1:.3f}", f"{mean_w2:.3f}"])

    if latex:
        col_fmt = "l" + "r" * (len(headers) - 1)
        lines = [
            r"\begin{tabular}{" + col_fmt + r"}",
            r"\toprule",
            " & ".join(headers) + r" \\",
            r"\midrule",
        ]
        for r in rows:
            lines.append(" & ".join(r) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        return "\n".join(lines)

    # Plain text
    col_w = [max(len(h), max((len(r[i]) for r in rows), default=0))
             for i, h in enumerate(headers)]
    sep = "+-" + "-+-".join("-" * w for w in col_w) + "-+"
    fmt_row = lambda r: "| " + " | ".join(v.ljust(w) for v, w in zip(r, col_w)) + " |"
    lines = [sep, fmt_row(headers), sep]
    for r in rows:
        lines.append(fmt_row(r))
    lines.append(sep)
    return "\n".join(lines)