"""
evaluation/evaluator.py  (aggiornato)
======================================
Contiene DUE classi pubbliche:

  1. Evaluator                    — pipeline completa (richiede modello + loader)
  2. DynamicEvaluationStrategist  — adapter leggero usato da main.py
                                    NON modifica trainer.py né il resto del codice

DynamicEvaluationStrategist espone esattamente l'API già chiamata in main.py:

    strategist = DynamicEvaluationStrategist(temperature=0.5)
    w1, w2 = strategist.compute_dynamic_weights(feat_tgt, feat_s1, feat_s2)
    strategist.log_batch_metrics(epoch, batch_idx, w1, w2)
    strategist.update_accuracy_evolution(epoch, acc_head, acc_ens)
    strategist.generate_plots()
    strategist.generate_markdown_report()

Internamente delega tutto a MetricsLogger e CosineWeighter — nessuna
logica duplicata.
"""

from __future__ import annotations

import os
import math
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from evaluation.weighting import CosineWeighter, AttentionWeighter, CentroidTracker
from evaluation.metrics   import MetricsLogger, compute_entropy, compute_accuracy, comparative_table


# ══════════════════════════════════════════════════════════════════════════════
# 1. Evaluator  (pipeline completa, opzionale — per uso avanzato)
# ══════════════════════════════════════════════════════════════════════════════

class Evaluator:
    """
    Pipeline di valutazione completa.
    Da usare se vuoi un oggetto che gestisce autonomamente modello + loader.

    Per l'integrazione con main.py usa invece DynamicEvaluationStrategist.
    """

    def __init__(
        self,
        model:             nn.Module,
        device:            torch.device,
        weighter:          CosineWeighter | AttentionWeighter,
        centroid_tracker:  CentroidTracker,
        logger:            MetricsLogger,
        domain_id_target:  int = 2,
    ):
        self.model            = model
        self.device           = device
        self.weighter         = weighter
        self.centroid_tracker = centroid_tracker
        self.logger           = logger
        self.domain_id_target = domain_id_target

    @torch.no_grad()
    def evaluate(self, eval_loader: DataLoader, epoch: int,
                 loss_dict_epoch: dict | None = None) -> float:
        self.model.eval()

        c1_ready = self.centroid_tracker.centroids[0] is not None
        c2_ready = self.centroid_tracker.centroids[1] is not None
        use_weighting = c1_ready and c2_ready

        all_logits, all_labels, all_w1, all_w2, all_ent = [], [], [], [], []

        for frames, labels, _ in eval_loader:
            frames = frames.to(self.device)
            labels = labels.to(self.device)
            cls_logits, _, embeddings, _ = self.model(frames, domain=self.domain_id_target)

            all_logits.append(cls_logits.cpu())
            all_labels.append(labels.cpu())
            all_ent.append(compute_entropy(cls_logits))

            if use_weighting:
                c1 = self.centroid_tracker.get(0, device=self.device)
                c2 = self.centroid_tracker.get(1, device=self.device)
                w1, w2 = self.weighter(embeddings, c1, c2)
                all_w1.append(w1.item())
                all_w2.append(w2.item())

        logits_cat = torch.cat(all_logits)
        labels_cat = torch.cat(all_labels)
        acc = compute_accuracy(logits_cat, labels_cat)

        mean_w1 = sum(all_w1) / len(all_w1) if all_w1 else 0.5
        mean_w2 = sum(all_w2) / len(all_w2) if all_w2 else 0.5
        mean_ent = sum(all_ent) / len(all_ent) if all_ent else math.nan

        self.logger.log_batch(w1=mean_w1, w2=mean_w2,
                              loss_dict=loss_dict_epoch or {}, entropy=mean_ent)
        self.logger.end_epoch(target_acc=acc, epoch=epoch)
        self.model.train()
        return acc

    @torch.no_grad()
    def collect_embeddings(self, loader, domain_id: int,
                           max_samples: int = 2000) -> tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        embs, labs = [], []
        total = 0
        for frames, labels, _ in loader:
            if total >= max_samples:
                break
            frames = frames.to(self.device)
            _, _, embeddings, _ = self.model(frames, domain=domain_id)
            embs.append(embeddings.cpu())
            labs.append(labels)
            total += frames.size(0)
        self.model.train()
        return torch.cat(embs)[:max_samples], torch.cat(labs)[:max_samples]

    @staticmethod
    def plot_tsne(embeddings_dict: dict, title: str = "t-SNE",
                  perplexity: int = 30, random_state: int = 42):
        try:
            from sklearn.manifold import TSNE
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            import numpy as np
        except ImportError:
            raise ImportError("pip install scikit-learn matplotlib")

        colors  = ["#FF5722", "#4CAF50", "#2196F3", "#9C27B0"]
        markers = ["o", "s", "^", "D"]

        all_embs   = torch.cat([v[0] for v in embeddings_dict.values()]).numpy()
        all_labels = torch.cat([v[1] for v in embeddings_dict.values()]).numpy()
        dom_ids    = []
        for i, (_, (e, _)) in enumerate(embeddings_dict.items()):
            dom_ids.extend([i] * len(e))
        dom_ids = __import__("numpy").array(dom_ids)

        coords = TSNE(n_components=2, perplexity=perplexity,
                      random_state=random_state).fit_transform(all_embs)

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle(title, fontsize=14, fontweight="bold")
        for i, name in enumerate(embeddings_dict.keys()):
            mask = dom_ids == i
            axes[0].scatter(coords[mask, 0], coords[mask, 1],
                            c=colors[i % len(colors)], marker=markers[i % len(markers)],
                            label=name, alpha=0.5, s=12, linewidths=0)
        axes[0].set_title("by Domain"); axes[0].legend(fontsize=9)
        axes[0].set_xticks([]); axes[0].set_yticks([])

        unique_cls = sorted(set(all_labels.tolist()))
        cmap = cm.get_cmap("tab20", len(unique_cls))
        for j, cls in enumerate(unique_cls):
            mask = all_labels == cls
            axes[1].scatter(coords[mask, 0], coords[mask, 1],
                            c=[cmap(j)], alpha=0.4, s=10, linewidths=0)
        axes[1].set_title("by Class")
        axes[1].set_xticks([]); axes[1].set_yticks([])
        plt.tight_layout()
        return fig


# ══════════════════════════════════════════════════════════════════════════════
# 2. DynamicEvaluationStrategist  ← questo è ciò che main.py importa
# ══════════════════════════════════════════════════════════════════════════════

class DynamicEvaluationStrategist:
    """
    Adapter leggero che espone l'API chiamata da main.py e delega
    internamente a MetricsLogger + CosineWeighter.

    Non richiede né modello né loader — riceve solo i tensori già
    estratti da main.py durante la sua evaluate_target().

    API pubblica (esattamente quella già chiamata in main.py)
    ---------------------------------------------------------
    compute_dynamic_weights(feat_tgt, feat_s1, feat_s2) -> (w1, w2)
    log_batch_metrics(epoch, batch_idx, w1, w2)
    update_accuracy_evolution(epoch, acc_head, acc_ens)
    generate_plots(output_dir)
    generate_markdown_report(output_dir)
    """

    def __init__(
        self,
        temperature:  float = 0.5,
        run_name:     str   = "",
        output_dir:   str   = "experiments/logs",
    ):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name   = run_name or f"run_{ts}"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._weighter = CosineWeighter(temperature=temperature)
        self._logger   = MetricsLogger(run_name=self.run_name)

        # Buffer per aggregare le batch-metrics prima di chiudere l'epoca
        # (end_epoch viene chiamato in update_accuracy_evolution)
        self._pending_w1:    list[float] = []
        self._pending_w2:    list[float] = []
        self._pending_ent:   list[float] = []
        self._pending_losses: dict[str, list[float]] = {}

        # Per tabella comparativa multi-run
        self._acc_head_history: list[float] = []
        self._acc_ens_history:  list[float] = []

    # ── API chiamata da main.py ───────────────────────────────────────────────

    def compute_dynamic_weights(
        self,
        feat_tgt: torch.Tensor,   # [B, D]  feature del batch target
        feat_s1:  torch.Tensor,   # [B, D] o [D]  feature/centroide S1
        feat_s2:  torch.Tensor,   # [B, D] o [D]  feature/centroide S2
    ) -> tuple[float, float]:
        """
        Calcola (w1, w2) via cosine similarity.
        Accetta sia batch [B,D] che centroidi [D] per S1/S2.
        """
        c1 = feat_s1.mean(dim=0) if feat_s1.dim() == 2 else feat_s1
        c2 = feat_s2.mean(dim=0) if feat_s2.dim() == 2 else feat_s2
        tgt = feat_tgt if feat_tgt.dim() == 2 else feat_tgt.unsqueeze(0)

        with torch.no_grad():
            w1, w2 = self._weighter(tgt, c1, c2)
        return w1.item(), w2.item()

    def log_batch_metrics(
        self,
        epoch:     int,
        batch_idx: int,
        w1:        float,
        w2:        float,
        loss_dict: dict[str, float] | None = None,
        entropy:   float | None = None,
    ):
        """
        Accumula le metriche di un singolo batch.
        Chiamata una volta per batch dentro evaluate_target() di main.py.
        """
        self._pending_w1.append(w1)
        self._pending_w2.append(w2)
        if entropy is not None:
            self._pending_ent.append(entropy)
        if loss_dict:
            for k, v in loss_dict.items():
                self._pending_losses.setdefault(k, []).append(v)

    def update_accuracy_evolution(
        self,
        epoch:    int,
        acc_head: float,
        acc_ens:  float,
    ):
        """
        Chiude l'epoca corrente nel logger con l'accuracy appena calcolata.
        Chiamata una volta per epoca in main.py dopo aver iterato eval_loader.
        """
        # Aggrega i pending di questo epoch in un unico log_batch
        mean_w1  = sum(self._pending_w1)  / len(self._pending_w1)  if self._pending_w1  else 0.5
        mean_w2  = sum(self._pending_w2)  / len(self._pending_w2)  if self._pending_w2  else 0.5
        mean_ent = sum(self._pending_ent) / len(self._pending_ent) if self._pending_ent else math.nan
        loss_agg = {k: sum(v) / len(v) for k, v in self._pending_losses.items()}

        self._logger.log_batch(w1=mean_w1, w2=mean_w2,
                               loss_dict=loss_agg, entropy=mean_ent)
        self._logger.end_epoch(target_acc=acc_head, epoch=epoch)
        self._logger.print_last()

        self._acc_head_history.append(acc_head)
        self._acc_ens_history.append(acc_ens)

        # Reset accumulatori per prossima epoch
        self._pending_w1.clear()
        self._pending_w2.clear()
        self._pending_ent.clear()
        self._pending_losses.clear()

    # ── Output ────────────────────────────────────────────────────────────────

    def generate_plots(self, output_dir: str | None = None):
        """
        Salva le 4 curve di training (accuracy, weighting, loss, entropy)
        più il confronto head vs ensemble in figures/.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("[DynamicEvaluationStrategist] matplotlib non disponibile, skip plots.")
            return

        out = Path(output_dir) if output_dir else self.output_dir
        out.mkdir(parents=True, exist_ok=True)
        fig_dir = out.parent.parent / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        # 1. Curve di training (4 subplot dal MetricsLogger)
        fig = self._logger.plot()
        path1 = fig_dir / f"training_curves_{self.run_name}.png"
        fig.savefig(path1, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[DynamicEvaluationStrategist] Plot salvato → {path1}")

        # 2. Head vs Ensemble accuracy
        if self._acc_head_history and self._acc_ens_history:
            ep = list(range(1, len(self._acc_head_history) + 1))
            fig2, ax = plt.subplots(figsize=(9, 4))
            ax.plot(ep, self._acc_head_history, label="head_tgt",  color="#2196F3", lw=2)
            ax.plot(ep, self._acc_ens_history,  label="ensemble",  color="#FF9800", lw=2, ls="--")
            ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
            ax.set_title(f"Head vs Ensemble — {self.run_name}", fontweight="bold")
            ax.legend(); ax.grid(True, alpha=0.3)
            fig2.tight_layout()
            path2 = fig_dir / f"head_vs_ensemble_{self.run_name}.png"
            fig2.savefig(path2, dpi=150, bbox_inches="tight")
            plt.close(fig2)
            print(f"[DynamicEvaluationStrategist] Plot salvato → {path2}")

        # 3. Source weighting stackplot standalone
        if self._logger.epochs:
            fig3, ax = plt.subplots(figsize=(9, 3))
            ax.stackplot(self._logger.epochs,
                         self._logger.influence_s1,
                         self._logger.influence_s2,
                         labels=["S1 (HMDB)", "S2 (UCF)"],
                         colors=["#FF5722", "#4CAF50"], alpha=0.75)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Epoch"); ax.set_ylabel("Weight")
            ax.set_title(f"Dynamic Source Weighting — {self.run_name}", fontweight="bold")
            ax.legend(loc="upper right", fontsize=9); ax.grid(True, alpha=0.2)
            fig3.tight_layout()
            path3 = fig_dir / f"source_weighting_{self.run_name}.png"
            fig3.savefig(path3, dpi=150, bbox_inches="tight")
            plt.close(fig3)
            print(f"[DynamicEvaluationStrategist] Plot salvato → {path3}")

    def generate_markdown_report(self, output_dir: str | None = None):
        """
        Scrive un report .md con tabella comparativa e statistiche finali.
        """
        out = Path(output_dir) if output_dir else self.output_dir
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"report_{self.run_name}.md"

        lg = self._logger
        best_acc  = max(lg.target_acc) if lg.target_acc else 0.0
        last_acc  = lg.target_acc[-1]  if lg.target_acc else 0.0
        ents      = [h for h in lg.entropy_tgt if not math.isnan(h)]
        min_ent   = min(ents) if ents else float("nan")
        mean_w1   = sum(lg.influence_s1) / len(lg.influence_s1) if lg.influence_s1 else 0.5
        mean_w2   = sum(lg.influence_s2) / len(lg.influence_s2) if lg.influence_s2 else 0.5
        n_epochs  = len(lg.epochs)

        # Tabella plain-text per confronto
        table_txt = comparative_table({self.run_name: lg})

        lines = [
            f"# Evaluation Report — {self.run_name}",
            f"",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"",
            f"## Summary",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Epochs completed | {n_epochs} |",
            f"| Best target acc (head_tgt) | **{best_acc:.2f}%** |",
            f"| Last target acc | {last_acc:.2f}% |",
            f"| Min prediction entropy | {min_ent:.4f} |",
            f"| Mean weight S1 (HMDB) | {mean_w1:.3f} |",
            f"| Mean weight S2 (UCF) | {mean_w2:.3f} |",
            f"",
            f"## Accuracy Evolution",
            f"",
        ]

        if lg.epochs:
            lines.append("| Epoch | Target Acc (%) | w(S1) | w(S2) | Entropy |")
            lines.append("|-------|---------------|-------|-------|---------|")
            for i, ep in enumerate(lg.epochs):
                acc_v = lg.target_acc[i] if i < len(lg.target_acc) else "-"
                w1_v  = lg.influence_s1[i] if i < len(lg.influence_s1) else "-"
                w2_v  = lg.influence_s2[i] if i < len(lg.influence_s2) else "-"
                ent_v = lg.entropy_tgt[i]  if i < len(lg.entropy_tgt)  else float("nan")
                ent_s = f"{ent_v:.4f}" if not math.isnan(ent_v) else "-"
                acc_s = f"{acc_v:.2f}" if isinstance(acc_v, float) else str(acc_v)
                w1_s  = f"{w1_v:.3f}"  if isinstance(w1_v,  float) else str(w1_v)
                w2_s  = f"{w2_v:.3f}"  if isinstance(w2_v,  float) else str(w2_v)
                lines.append(f"| {ep} | {acc_s} | {w1_s} | {w2_s} | {ent_s} |")

        lines += [
            f"",
            f"## Comparative Table (plain-text)",
            f"",
            f"```",
            table_txt,
            f"```",
            f"",
            f"## Loss Components (last epoch)",
            f"",
        ]

        if lg.epochs:
            lines.append("| Loss | Value |")
            lines.append("|------|-------|")
            for k, vals in lg.losses.items():
                if vals and any(v != 0.0 for v in vals):
                    lines.append(f"| {k} | {vals[-1]:.4f} |")

        path.write_text("\n".join(lines), encoding="utf-8")
        print(f"[DynamicEvaluationStrategist] Report salvato → {path}")

    # ── Accesso diretto al logger (per uso avanzato / notebook) ──────────────

    @property
    def logger(self) -> MetricsLogger:
        return self._logger

    def save_logger(self, path: str | None = None):
        p = path or str(self.output_dir / f"metrics_{self.run_name}.json")
        self._logger.save(p)