import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# --- INTEGRAZIONE STRATEGIST PERSONA 3 ---
from evaluation.weighting import CentroidTracker
from evaluation.evaluator import DynamicEvaluationStrategist
from evaluation.metrics import compute_entropy
# ------------------------------------------

try:
    import wandb
except ModuleNotFoundError:
    wandb = None

class Trainer:
    """
    Trainer Unificato Completo per il framework Multi-Source Domain Adaptation (Track 9).
    Fonde la loss avanzata del team (Hard Pseudo-Labeling, EM) con l'infrastruttura di P3.
    """
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        max_epochs: int = 30,
        checkpoint_dir: str = "experiments/checkpoints",
        incomplete_simulation: bool = True,
        source2_enabled: bool = True,
        patience: int = 7,
        lambda_pseudo: float = 0.1,
        warmup_epochs: int = 5,
        lambda_em: float = 0.1,
        disable_early_stopping_if_mock: bool = False,
        fast_eval_size: int = 500,       
        full_eval_size: int = 2000,      
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.max_epochs = max_epochs
        self.checkpoint_dir = checkpoint_dir
        self.incomplete_simulation = incomplete_simulation
        self.source2_enabled = source2_enabled
        self.patience = patience
        self.lambda_pseudo = lambda_pseudo
        self.warmup_epochs = warmup_epochs
        self.lambda_em = lambda_em
        self.disable_early_stopping_if_mock = disable_early_stopping_if_mock
        self.fast_eval_size = fast_eval_size
        self.full_eval_size = full_eval_size
        
        # --- INTEGRAZIONE STRATEGIST PERSONA 3 ---
        self.strategist = DynamicEvaluationStrategist(
            run_name="MS_DANN_Persona3",
            fast_eval_size=fast_eval_size,
            full_eval_size=full_eval_size
        )
        self.centroid_tracker = CentroidTracker(embed_dim=512)
        self.strategist.set_model(self.model)
        # ------------------------------------------
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        self._global_step = 0
        self.best_tgt_acc = -1.0
        self.epochs_without_improvement = 0
        self.full_eval_results = {"epochs": [], "acc_head": [], "acc_ensemble": []}
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=self.max_epochs, 
            eta_min=1e-6
        )

    def _schedule_disc_dropout(self, epoch: int) -> None:
        frac = epoch / max(self.max_epochs - 1, 1)
        p = max(0.2, 0.5 - 0.3 * frac)
        if hasattr(self.model, "discriminator"):
            self.model.discriminator.set_dropout(p)

    def _compute_alpha(self, epoch: int, step: int, num_steps: int) -> float:
        p = float(step + epoch * num_steps) / float(self.max_epochs * num_steps + 1e-8)
        return float((2.0 / (1.0 + torch.exp(torch.tensor(-10.0 * p))) - 1.0).item())

    def _log(self, metrics: dict) -> None:
        if wandb and wandb.run:
            try:
                wandb.log({**metrics, "global_step": self._global_step})
            except Exception:
                pass

    def resume(self, checkpoint_path: str = None) -> int:
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.checkpoint_dir, "latest_checkpoint.pth")
            
        if not os.path.exists(checkpoint_path):
            print(f"[RESUME] Nessun checkpoint trovato in {checkpoint_path}. Partenza da zero.")
            return 0
            
        print(f"[RESUME] Caricamento checkpoint da {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_tgt_acc = checkpoint.get("best_acc", -1.0)
        self._global_step = checkpoint.get("global_step", 0)
        self.epochs_without_improvement = checkpoint.get("epochs_without_improvement", 0)

        start_epoch = checkpoint.get("epoch", -1) + 1
        print(f"[RESUME] Training ripreso dall'epoca {start_epoch} con best acc {self.best_tgt_acc:.2f}%")
        return start_epoch

    def _save_checkpoint(self, epoch: int, acc: float, is_best: bool = False) -> None:
        state = {
            "epoch":                epoch,
            "global_step":          self._global_step,
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_acc":             self.best_tgt_acc,
            "epochs_without_improvement": self.epochs_without_improvement,
        }
        latest_path = os.path.join(self.checkpoint_dir, "latest_checkpoint.pth")
        torch.save(state, latest_path)

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
            torch.save(state, best_path)
            print(f"     [BEST] Nuovo best model salvato → {best_path}  (acc={acc:.2f}%)")

    def train_step(self, batch_s1, batch_s2, batch_tgt, alpha: float, epoch: int, step: int) -> dict:
        self.model.set_grl_alpha(alpha)

        x_s1, y_s1, d_s1 = (t.to(self.device) for t in batch_s1)
        x_s2, y_s2, d_s2 = (t.to(self.device) for t in batch_s2)
        x_tgt, y_tgt, d_tgt = (t.to(self.device) for t in batch_tgt)

        active_s1 = True
        active_s2 = self.source2_enabled
        if self.incomplete_simulation:
            if torch.rand(1).item() < 0.1 and active_s2:
                active_s1 = False
            elif torch.rand(1).item() < 0.1 and active_s1:
                active_s2 = False

        dom_logits_list, dom_labels_list = [], []
        cls_s1 = cls_s2 = None

        if active_s1:
            cls_s1, dom_s1, feat_s1, _ = self.model(x_s1, domain=0)
            dom_logits_list.append(dom_s1)
            dom_labels_list.append(d_s1)
            self.centroid_tracker.update(feat_s1, source_id=0)

        if active_s2:
            cls_s2, dom_s2, feat_s2, _ = self.model(x_s2, domain=1)
            dom_logits_list.append(dom_s2)
            dom_labels_list.append(d_s2)
            self.centroid_tracker.update(feat_s2, source_id=1)

        cls_tgt, dom_tgt, feat_tgt, ensemble_probs = self.model(x_tgt, domain=2)
        dom_logits_list.append(dom_tgt)
        dom_labels_list.append(d_tgt)

        loss_dict = self.loss_fn(
            dom_logits=torch.cat(dom_logits_list, dim=0),
            dom_labels=torch.cat(dom_labels_list, dim=0),
            logits_s1=cls_s1, labels_s1=y_s1 if active_s1 else None,
            logits_s2=cls_s2, labels_s2=y_s2 if active_s2 else None,
        )

        loss_cls = loss_dict["loss_cls"]
        loss_cls_s1_item = loss_dict["loss_cls_s1"]
        loss_cls_s2_item = loss_dict["loss_cls_s2"]
        loss_adv = loss_dict["loss_adv"]

        total_inf = loss_cls_s1_item + loss_cls_s2_item + 1e-8
        ratio_s1  = loss_cls_s1_item / total_inf
        ratio_s2  = loss_cls_s2_item / total_inf

        w1, w2 = self.strategist.compute_dynamic_weights(feat_tgt)

        if epoch >= self.warmup_epochs:
            confidence, pseudo_labels = ensemble_probs.detach().max(dim=-1)
            mask = confidence > 0.7
            if mask.sum() > 0:
                loss_tgt_pseudo = F.cross_entropy(cls_tgt[mask], pseudo_labels[mask])
            else:
                loss_tgt_pseudo = torch.tensor(0.0, device=self.device)
        else:
            loss_tgt_pseudo = torch.tensor(0.0, device=self.device)

        probs_tgt = F.softmax(cls_tgt, dim=-1).clamp(min=1e-8)
        loss_em = -(probs_tgt * probs_tgt.log()).sum(-1).mean()
        current_lambda_em = self.lambda_em * min(1.0, max(0.0, epoch - self.warmup_epochs) / 5.0)

        loss_total = loss_dict["loss_total"] + self.lambda_pseudo * loss_tgt_pseudo + current_lambda_em * loss_em

        self.optimizer.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        with torch.no_grad():
            probs = F.softmax(dom_tgt, dim=-1)
            conf_entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean().item()
            p_as_s1  = probs[:, 0].mean().item()
            p_as_s2  = probs[:, 1].mean().item()
            p_as_tgt = probs[:, 2].mean().item()

        return {
            "loss_total":      loss_total.item(),
            "loss_cls":        loss_cls.item(),
            "loss_cls_s1":     loss_cls_s1_item,
            "loss_cls_s2":     loss_cls_s2_item,
            "loss_adv":        loss_adv.item(),
            "loss_tgt_pseudo": loss_tgt_pseudo.item() if isinstance(loss_tgt_pseudo, torch.Tensor) else loss_tgt_pseudo,
            "loss_em":         loss_em.item(),
            "influence_s1":    ratio_s1,
            "influence_s2":    ratio_s2,
            "w1":              w1,
            "w2":              w2,
            "conf_entropy":    conf_entropy,
            "p_as_s1":         p_as_s1,
            "p_as_s2":         p_as_s2,
            "p_as_tgt":        p_as_tgt,
            "active_s1":       active_s1,
            "active_s2":       active_s2,
            "alpha":           alpha,
        }

    def train_epoch(self, train_loader, epoch: int) -> dict:
        self.model.train()
        self._schedule_disc_dropout(epoch)
        num_steps = len(train_loader)

        totals = {k: 0.0 for k in [
            "loss_total", "loss_cls", "loss_adv", "loss_tgt_pseudo", "loss_em",
            "influence_s1", "influence_s2", "w1", "w2", "conf_entropy", "p_as_s1", "p_as_s2", "p_as_tgt",
        ]}
        s1_active = s2_active = 0

        pbar = tqdm(enumerate(train_loader), total=num_steps, desc=f"Epoca {epoch+1}/{self.max_epochs}")

        for step, (batch_s1, batch_s2, batch_tgt) in pbar:
            alpha = self._compute_alpha(epoch, step, num_steps)
            metrics = self.train_step(batch_s1, batch_s2, batch_tgt, alpha, epoch, step)

            for k in totals:
                totals[k] += metrics[k]
            s1_active += int(metrics["active_s1"])
            s2_active += int(metrics["active_s2"])
            self._global_step += 1

            # 🟩 FIX GRAFICO LOSS COMPONENTS VUOTO:
            # Inviamo lo storico esatto associando le chiavi volute da metrics.py allo strategist
            self.strategist.log_batch_metrics(
                epoch=epoch, batch_idx=step, w1=metrics["w1"], w2=metrics["w2"], 
                loss_dict={
                    "total":   metrics["loss_total"],
                    "cls_s1":  metrics["loss_cls_s1"],
                    "cls_s2":  metrics["loss_cls_s2"],
                    "adv":     metrics["loss_adv"],
                    "pseudo":  metrics["loss_tgt_pseudo"]
                }, 
                entropy=metrics["conf_entropy"]
            )

            self._log({
                "step/loss_total":    metrics["loss_total"],
                "step/loss_cls":      metrics["loss_cls"],
                "step/loss_adv":      metrics["loss_adv"],
                "step/loss_em":       metrics["loss_em"],
                "step/influence_s1":  metrics["influence_s1"],
                "step/influence_s2":  metrics["influence_s2"],
                "step/conf_entropy":  metrics["conf_entropy"],
                "step/grl_alpha":     metrics["alpha"],
            })

            pbar.set_postfix({"Loss": f"{metrics['loss_total']:.3f}", "ConfEnt": f"{metrics['conf_entropy']:.3f}"})

        avgs = {k: v / num_steps for k, v in totals.items()}
        avgs["s1_active_ratio"] = s1_active / num_steps
        avgs["s2_active_ratio"] = s2_active / num_steps
        self._log({f"epoch/{k}": v for k, v in avgs.items()} | {"epoch/num": epoch + 1})
        return avgs

    def _create_subset_loader(self, eval_loader, num_samples: int):
        dataset = eval_loader.dataset
        total = len(dataset)
        indices = random.sample(range(total), min(num_samples, total))
        subset = torch.utils.data.Subset(dataset, indices)
        return torch.utils.data.DataLoader(subset, batch_size=eval_loader.batch_size, shuffle=False, num_workers=0, pin_memory=False)

    def evaluate(self, eval_loader, epoch: int, fast: bool = True) -> dict:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        self.model.eval()
        num_samples = self.fast_eval_size if fast else self.full_eval_size
        eval_type = "FAST" if fast else "FULL"
        
        subset_loader = self._create_subset_loader(eval_loader, num_samples)
        correct_head = correct_ens = total_count = 0

        with torch.no_grad():
            for x, y, _ in subset_loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                cls_logits, _, _, ensemble_probs = self.model(x, domain=2)

                correct_head += (cls_logits.argmax(-1) == y).sum().item()
                correct_ens += (ensemble_probs.argmax(-1) == y).sum().item()
                total_count += y.size(0)

        acc_head = (correct_head / total_count * 100) if total_count > 0 else 0.0
        acc_ens = (correct_ens / total_count * 100) if total_count > 0 else 0.0
        print(f"\n  [{eval_type} EVAL] Epoca {epoch+1} — head_tgt: {acc_head:.2f}%  |  ensemble: {acc_ens:.2f}%")

        if fast:
            is_best = acc_head > self.best_tgt_acc
            if is_best:
                self.best_tgt_acc = acc_head
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            self.strategist.update_accuracy_evolution(epoch, acc_head, acc_ens)
            self._save_checkpoint(epoch, acc_head, is_best=is_best)
        else:
            self.full_eval_results["epochs"].append(epoch + 1)
            self.full_eval_results["acc_head"].append(acc_head)
            self.full_eval_results["acc_ensemble"].append(acc_ens)
            self.strategist.record_full_eval(epoch, acc_head, acc_ens)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        return {"acc_head": acc_head, "acc_ensemble": acc_ens}

    def fit(self, train_loader, eval_loader, auto_resume: bool = True, resume_path: str = None) -> None:
        print("\n" + "=" * 60)
        print(f"Inizio training su {self.device} | Tetto Epoche: {self.max_epochs}")
        print("=" * 60)

        start_epoch = self.resume(resume_path) if auto_resume else 0

        for epoch in range(start_epoch, self.max_epochs):
            train_metrics = self.train_epoch(train_loader, epoch)
            print(f"\n--- Riepilogo Epoca {epoch+1} ---")
            print(f"  Loss: total={train_metrics['loss_total']:.4f} | cls={train_metrics['loss_cls']:.4f} | adv={train_metrics['loss_adv']:.4f}")
            print(f"  Influence (P3): w(S1)={train_metrics['w1']:.3f} | w(S2)={train_metrics['w2']:.3f}")
            
            eval_fast = self.evaluate(eval_loader, epoch, fast=True)
            print(f"  Fast Eval: head_tgt={eval_fast['acc_head']:.2f}% | ensemble={eval_fast['acc_ensemble']:.2f}%")
            print("-" * 50)
            
            if self.epochs_without_improvement >= self.patience and not self.disable_early_stopping_if_mock:
                print(f"\n[EARLY STOPPING] Raggiunta la soglia di patience a {self.patience} epoche consecutive.")
                break
                
            self.scheduler.step()

        # 🟩 MODIFICA EFFETTUATA CON SUCCESSO: 
        # La chiamata pesante evaluate(..., fast=False) che causava freeze è stata rimossa.
        print("\n" + "=" * 60)
        print("✅ TRAINING COMPLETATO CON SUCCESSO — PASSO ALL'ANALISI STATISTICA")
        print("=" * 60)
        #self.evaluate(eval_loader, self.max_epochs - 1, fast=False)