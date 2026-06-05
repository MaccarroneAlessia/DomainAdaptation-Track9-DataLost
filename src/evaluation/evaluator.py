from __future__ import annotations

import os
import math
import random
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from evaluation.weighting import CosineWeighter, AttentionWeighter, CentroidTracker
from evaluation.metrics import MetricsLogger, compute_entropy, compute_accuracy, comparative_table


class FastEvaluator:
    """
    Versione snella della pipeline di valutazione.
    - Valutazione veloce ogni epoca (subset)
    - Valutazione completa solo alla fine
    """
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        weighter: CosineWeighter | AttentionWeighter,
        centroid_tracker: CentroidTracker,
        logger: MetricsLogger,
        domain_id_target: int = 2,
        fast_eval_size: int = 500,
        full_eval_size: int = 2000,
    ):
        self.model = model
        self.device = device
        self.weighter = weighter
        self.centroid_tracker = centroid_tracker
        self.logger = logger
        self.domain_id_target = domain_id_target
        self.fast_eval_size = fast_eval_size
        self.full_eval_size = full_eval_size
        
        self.full_results = {"epochs": [], "acc_head": [], "acc_ensemble": []}

    def _create_subset_loader(self, full_loader: DataLoader, num_samples: int) -> DataLoader:
        dataset = full_loader.dataset
        total = len(dataset)
        indices = random.sample(range(total), min(num_samples, total))
        subset = Subset(dataset, indices)
        
        return DataLoader(
            subset,
            batch_size=full_loader.batch_size,
            shuffle=False,
            num_workers=full_loader.num_workers,
            pin_memory=full_loader.pin_memory,
        )

    @torch.no_grad()
    def evaluate_fast(self, eval_loader: DataLoader, epoch: int, loss_dict_epoch: dict | None = None) -> dict:
        """Valutazione VELOCE su subset del target."""
        self.model.eval()
        
        fast_loader = self._create_subset_loader(eval_loader, self.fast_eval_size)
        
        c1_ready = self.centroid_tracker.centroids[0] is not None
        c2_ready = self.centroid_tracker.centroids[1] is not None
        use_weighting = c1_ready and c2_ready

        all_logits_head, all_logits_ens, all_labels = [], [], []
        all_w1, all_w2, all_ent = [], [], []

        for frames, labels, _ in fast_loader:
            frames = frames.to(self.device)
            labels = labels.to(self.device)
            cls_logits, _, embeddings, ensemble_probs = self.model(frames, domain=self.domain_id_target)

            all_logits_head.append(cls_logits.cpu())
            all_logits_ens.append(ensemble_probs.cpu())
            all_labels.append(labels.cpu())
            all_ent.append(compute_entropy(cls_logits))

            if use_weighting and embeddings is not None:
                c1 = self.centroid_tracker.get(0, device=self.device)
                c2 = self.centroid_tracker.get(1, device=self.device)
                w1, w2 = self.weighter(embeddings, c1, c2)
                all_w1.append(w1.item())
                all_w2.append(w2.item())

        logits_head_cat = torch.cat(all_logits_head)
        logits_ens_cat = torch.cat(all_logits_ens)
        labels_cat = torch.cat(all_labels)
        
        acc_head = compute_accuracy(logits_head_cat, labels_cat)
        acc_ens = compute_accuracy(logits_ens_cat, labels_cat)

        mean_w1 = sum(all_w1) / len(all_w1) if all_w1 else 0.5
        mean_w2 = sum(all_w2) / len(all_w2) if all_w2 else 0.5
        mean_ent = sum(all_ent) / len(all_ent) if all_ent else math.nan

        self.logger.log_batch(
            w1=mean_w1, 
            w2=mean_w2,
            loss_dict=loss_dict_epoch or {}, 
            entropy=mean_ent
        )
        self.logger.end_epoch(target_acc=acc_head, epoch=epoch)
        
        self.model.train()
        return {"acc_head": acc_head, "acc_ensemble": acc_ens, "entropy": mean_ent}

    @torch.no_grad()
    def evaluate_full(self, eval_loader: DataLoader, epoch: int) -> dict:
        """Valutazione COMPLETA su subset significativo del target."""
        print(f"\n  [FULL EVAL] Esecuzione valutazione su {self.full_eval_size} clip...")
        self.model.eval()
        
        full_loader = self._create_subset_loader(eval_loader, self.full_eval_size)
        
        correct_head = correct_ens = total = 0

        for frames, labels, _ in full_loader:
            frames = frames.to(self.device)
            labels = labels.to(self.device)
            cls_logits, _, _, ensemble_probs = self.model(frames, domain=self.domain_id_target)

            correct_head += (cls_logits.argmax(-1) == labels).sum().item()
            correct_ens += (ensemble_probs.argmax(-1) == labels).sum().item()
            total += labels.size(0)

        acc_head = (correct_head / total * 100) if total > 0 else 0.0
        acc_ens = (correct_ens / total * 100) if total > 0 else 0.0

        print(f"  [FULL EVAL] Epoca {epoch+1} — head_tgt: {acc_head:.2f}% | ensemble: {acc_ens:.2f}%")
        
        self.full_results["epochs"].append(epoch + 1)
        self.full_results["acc_head"].append(acc_head)
        self.full_results["acc_ensemble"].append(acc_ens)
        
        self.model.train()
        return {"acc_head": acc_head, "acc_ensemble": acc_ens}


class DynamicEvaluationStrategist:
    """
    Strategist leggero che espone l'API chiamata da main.py.
    Versione snella con valutazione veloce.
    """
    def __init__(
        self,
        temperature: float = 0.5,
        run_name: str = "",
        output_dir: str = "experiments/logs",
        fast_eval_size: int = 500,
        full_eval_size: int = 2000,
    ):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = run_name or f"run_{ts}"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.fast_eval_size = fast_eval_size
        self.full_eval_size = full_eval_size

        self._weighter = CosineWeighter(temperature=temperature)
        self._logger = MetricsLogger(run_name=self.run_name)
        
        self._centroids = {0: None, 1: None}
        self._centroid_momentum = 0.99

        self._pending_w1: list[float] = []
        self._pending_w2: list[float] = []
        self._pending_ent: list[float] = []
        self._pending_losses: dict[str, list[float]] = {}

        self._acc_head_history: list[float] = []
        self._acc_ens_history: list[float] = []
        
        self._full_eval_results = {"epoch": [], "acc_head": [], "acc_ensemble": []}
        self._model = None

    def update_centroids(self, source_id: int, embeddings: torch.Tensor):
        if embeddings is None or embeddings.numel() == 0:
            return
            
        batch_mean = embeddings.detach().mean(dim=0).cpu()
        
        if self._centroids[source_id] is None:
            self._centroids[source_id] = batch_mean
        else:
            self._centroids[source_id] = (
                self._centroid_momentum * self._centroids[source_id] +
                (1 - self._centroid_momentum) * batch_mean
            )

    def get_centroids(self, device: torch.device) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        c1 = self._centroids[0].to(device) if self._centroids[0] is not None else None
        c2 = self._centroids[1].to(device) if self._centroids[1] is not None else None
        return c1, c2

    def compute_dynamic_weights(
        self,
        feat_tgt: torch.Tensor,
        feat_s1: torch.Tensor | None = None,
        feat_s2: torch.Tensor | None = None,
    ) -> tuple[float, float]:
        with torch.no_grad():
            tgt = feat_tgt.detach()
            
            if feat_s1 is None or feat_s2 is None:
                c1 = self._centroids[0]
                c2 = self._centroids[1]
                if c1 is None or c2 is None:
                    return 0.5, 0.5
                c1 = c1.to(tgt.device)
                c2 = c2.to(tgt.device)
            else:
                c1 = feat_s1.detach().mean(dim=0)
                c2 = feat_s2.detach().mean(dim=0)
            
            avg_tgt = tgt.mean(dim=0)
            
            sim1 = F.cosine_similarity(avg_tgt.unsqueeze(0), c1.unsqueeze(0), eps=1e-8).item()
            sim2 = F.cosine_similarity(avg_tgt.unsqueeze(0), c2.unsqueeze(0), eps=1e-8).item()
            
            tau = 0.5
            w1 = math.exp(sim1 / tau) / (math.exp(sim1 / tau) + math.exp(sim2 / tau))
            w2 = 1.0 - w1
            
        return w1, w2

    def log_batch_metrics(
        self,
        epoch: int,
        batch_idx: int,
        w1: float,
        w2: float,
        loss_dict: dict[str, float] | None = None,
        entropy: float | None = None,
    ):
        self._pending_w1.append(w1)
        self._pending_w2.append(w2)
        if entropy is not None:
            self._pending_ent.append(entropy)
        if loss_dict:
            for k, v in loss_dict.items():
                self._pending_losses.setdefault(k, []).append(v)

    def update_accuracy_evolution(
        self,
        epoch: int,
        acc_head: float,
        acc_ens: float,
    ):
        mean_w1 = sum(self._pending_w1) / len(self._pending_w1) if self._pending_w1 else 0.5
        mean_w2 = sum(self._pending_w2) / len(self._pending_w2) if self._pending_w2 else 0.5
        mean_ent = sum(self._pending_ent) / len(self._pending_ent) if self._pending_ent else math.nan
        loss_agg = {k: sum(v) / len(v) for k, v in self._pending_losses.items()}

        self._logger.log_batch(w1=mean_w1, w2=mean_w2, loss_dict=loss_agg, entropy=mean_ent)
        self._logger.end_epoch(target_acc=acc_head, epoch=epoch)

        self._acc_head_history.append(acc_head)
        self._acc_ens_history.append(acc_ens)

        self._pending_w1.clear()
        self._pending_w2.clear()
        self._pending_ent.clear()
        self._pending_losses.clear()

    def record_full_eval(self, epoch: int, acc_head: float, acc_ens: float):
        self._full_eval_results["epoch"].append(epoch + 1)
        self._full_eval_results["acc_head"].append(acc_head)
        self._full_eval_results["acc_ensemble"].append(acc_ens)

    def set_model(self, model: nn.Module):
        self._model = model

    def generate_plots(self, output_dir: str | None = None):
        out = Path(output_dir) if output_dir else self.output_dir
        out.mkdir(parents=True, exist_ok=True)
        fig_dir = out.parent.parent / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        # 1. Training curves
        fig = self._logger.plot()
        if fig is not None:
            path1 = fig_dir / f"training_curves_{self.run_name}.png"
            fig.savefig(path1, dpi=150, bbox_inches="tight")
            plt.close(fig)

        # 2. Head vs Ensemble
        if self._acc_head_history and self._acc_ens_history:
            ep = list(range(1, len(self._acc_head_history) + 1))
            fig2, ax = plt.subplots(figsize=(9, 4))
            ax.plot(ep, self._acc_head_history, label="head_tgt (fast)", color="#2196F3", lw=2)
            ax.plot(ep, self._acc_ens_history, label="ensemble (fast)", color="#FF9800", lw=2, ls="--")
            
            if self._full_eval_results["epoch"]:
                ax.scatter(
                    self._full_eval_results["epoch"], 
                    self._full_eval_results["acc_head"],
                    color="#2196F3", marker="o", s=50, zorder=5,
                    label="head_tgt (full)"
                )
                ax.scatter(
                    self._full_eval_results["epoch"], 
                    self._full_eval_results["acc_ensemble"],
                    color="#FF9800", marker="s", s=50, zorder=5,
                    label="ensemble (full)"
                )
            
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy (%)")
            ax.set_title(f"Head vs Ensemble — {self.run_name}", fontweight="bold")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig2.tight_layout()
            path2 = fig_dir / f"head_vs_ensemble_{self.run_name}.png"
            fig2.savefig(path2, dpi=150, bbox_inches="tight")
            plt.close(fig2)

        # 3. Source weighting
        if self._logger.epochs:
            fig3, ax = plt.subplots(figsize=(9, 3))
            ax.stackplot(self._logger.epochs,
                         self._logger.influence_s1,
                         self._logger.influence_s2,
                         labels=["S1 (HMDB)", "S2 (UCF)"],
                         colors=["#FF5722", "#4CAF50"], alpha=0.75)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Weight")
            ax.set_title(f"Dynamic Source Weighting — {self.run_name}", fontweight="bold")
            ax.legend(loc="upper right", fontsize=9)
            ax.grid(True, alpha=0.2)
            fig3.tight_layout()
            path3 = fig_dir / f"source_weighting_{self.run_name}.png"
            fig3.savefig(path3, dpi=150, bbox_inches="tight")
            plt.close(fig3)

    def generate_markdown_report(self, output_dir: str | None = None):
        out = Path(output_dir) if output_dir else self.output_dir
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"report_{self.run_name}.md"

        lg = self._logger
        best_acc = max(lg.target_acc) if lg.target_acc else 0.0
        last_acc = lg.target_acc[-1] if lg.target_acc else 0.0
        ents = [h for h in lg.entropy_tgt if not math.isnan(h)]
        min_ent = min(ents) if ents else float("nan")
        mean_w1 = sum(lg.influence_s1) / len(lg.influence_s1) if lg.influence_s1 else 0.5
        mean_w2 = sum(lg.influence_s2) / len(lg.influence_s2) if lg.influence_s2 else 0.5
        n_epochs = len(lg.epochs)
        
        best_full = max(self._full_eval_results["acc_head"]) if self._full_eval_results["acc_head"] else best_acc

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
            f"| Best target acc (fast eval) | **{best_acc:.2f}%** |",
            f"| Best target acc (full eval) | **{best_full:.2f}%** |",
            f"| Last target acc | {last_acc:.2f}% |",
            f"| Min prediction entropy | {min_ent:.4f} |",
            f"| Mean weight S1 (HMDB) | {mean_w1:.3f} |",
            f"| Mean weight S2 (UCF) | {mean_w2:.3f} |",
            f"",
            f"## Accuracy Evolution (Fast Evaluation - every epoch)",
            f"",
        ]

        if lg.epochs:
            lines.append("| Epoch | Target Acc (%) | w(S1) | w(S2) | Entropy |")
            lines.append("|-------|---------------|-------|-------|---------|")
            for i, ep in enumerate(lg.epochs):
                acc_v = lg.target_acc[i] if i < len(lg.target_acc) else "-"
                w1_v = lg.influence_s1[i] if i < len(lg.influence_s1) else "-"
                w2_v = lg.influence_s2[i] if i < len(lg.influence_s2) else "-"
                ent_v = lg.entropy_tgt[i] if i < len(lg.entropy_tgt) else float("nan")
                ent_s = f"{ent_v:.4f}" if not math.isnan(ent_v) else "-"
                acc_s = f"{acc_v:.2f}" if isinstance(acc_v, float) else str(acc_v)
                w1_s = f"{w1_v:.3f}" if isinstance(w1_v, float) else str(w1_v)
                w2_s = f"{w2_v:.3f}" if isinstance(w2_v, float) else str(w2_v)
                lines.append(f"| {ep} | {acc_s} | {w1_s} | {w2_s} | {ent_s} |")

        if self._full_eval_results["epoch"]:
            lines += [
                f"",
                f"## Full Evaluation Results",
                f"",
                "| Epoch | Head Acc (%) | Ensemble Acc (%) |",
                "|-------|--------------|------------------|",
            ]
            for ep, ah, ae in zip(
                self._full_eval_results["epoch"],
                self._full_eval_results["acc_head"],
                self._full_eval_results["acc_ensemble"]
            ):
                lines.append(f"| {ep} | {ah:.2f} | {ae:.2f} |")

        lines += [
            f"",
            f"## Comparative Table",
            f"",
            f"```",
            comparative_table({self.run_name: lg}),
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

    @property
    def logger(self) -> MetricsLogger:
        return self._logger

    def save_logger(self, path: str | None = None):
        p = path or str(self.output_dir / f"metrics_{self.run_name}.json")
        self._logger.save(p)

    @torch.no_grad()
    def collect_embeddings(self, loader, domain_name: str, max_samples: int = 500) -> tuple[torch.Tensor, torch.Tensor]:
        """Raccoglie embeddings per visualizzazione t-SNE."""
        if self._model is None:
            raise RuntimeError("Model not set. Call set_model() first.")
        
        self._model.eval()
        embs, labs = [], []
        total = 0
        
        for frames, labels, _ in loader:
            if total >= max_samples:
                break
            frames = frames.to(next(self._model.parameters()).device)
            _, _, embeddings, _ = self._model(frames, domain=2)
            embs.append(embeddings.cpu())
            labs.append(labels)
            total += frames.size(0)
        
        self._model.train()
        return torch.cat(embs)[:max_samples], torch.cat(labs)[:max_samples]

    @staticmethod
    def plot_tsne(
        embeddings_dict: dict,
        title: str = "t-SNE Visualization",
        perplexity: int = 30,
        random_state: int = 42,
        save_path: str | None = None,
    ):
        try:
            from sklearn.manifold import TSNE
            import numpy as np
            import matplotlib.cm as cm
        except ImportError:
            print("scikit-learn richiesto per t-SNE. Installa con: pip install scikit-learn")
            return None

        colors_domain = ["#FF5722", "#4CAF50", "#2196F3"]
        markers = ["o", "s", "^"]
        domain_names = ["HMDB-51 (Source 1)", "UCF-101 (Source 2)", "Kinetics (Target)"]

        all_embs = []
        all_labels = []
        domain_ids = []
        
        for i, (name, (embs, labels)) in enumerate(embeddings_dict.items()):
            if embs is not None and len(embs) > 0:
                all_embs.append(embs.numpy() if torch.is_tensor(embs) else embs)
                all_labels.append(labels.numpy() if torch.is_tensor(labels) else labels)
                domain_ids.extend([i] * len(embs))

        if not all_embs:
            print("Nessun embedding valido per t-SNE")
            return None

        all_embs_np = np.concatenate(all_embs, axis=0)
        all_labels_np = np.concatenate(all_labels, axis=0)
        domain_ids_np = np.array(domain_ids)

        print(f"  Esecuzione t-SNE su {len(all_embs_np)} campioni...")
        tsne = TSNE(n_components=2, perplexity=min(perplexity, len(all_embs_np)-1),
                    random_state=random_state, verbose=0)
        coords = tsne.fit_transform(all_embs_np)

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle(title, fontsize=14, fontweight="bold")

        for i in range(3):
            mask = domain_ids_np == i
            if mask.any():
                axes[0].scatter(
                    coords[mask, 0], coords[mask, 1],
                    c=colors_domain[i], marker=markers[i],
                    label=domain_names[i], alpha=0.5, s=15, linewidths=0
                )
        axes[0].set_title("by Domain")
        axes[0].legend(fontsize=9, loc="best")
        axes[0].set_xticks([])
        axes[0].set_yticks([])

        unique_cls = sorted(set(all_labels_np.tolist()))[:20]
        cmap = cm.get_cmap("tab20", len(unique_cls))
        for j, cls in enumerate(unique_cls):
            mask = all_labels_np == cls
            if mask.any():
                axes[1].scatter(
                    coords[mask, 0], coords[mask, 1],
                    c=[cmap(j)], alpha=0.4, s=10, linewidths=0
                )
        axes[1].set_title("by Class (first 20 classes)")
        axes[1].set_xticks([])
        axes[1].set_yticks([])

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  t-SNE salvato → {save_path}")
        
        return fig