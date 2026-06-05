import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

try:
    import wandb
except ModuleNotFoundError:
    wandb = None

class Trainer:
    """
    Trainer per il framework Multi-Source Domain Adaptation (Track 9).
    
    Gestisce:
    - Ottimizzazione simultanea (Classification Loss + Adversarial Domain Loss)
    - Scheduling progressivo del parametro alpha del GRL
    - Incomplete batch simulation (drop casuale mid-epoch delle sorgenti per robustezza)
    - Tracciamento e logging dell'influence ratio (Source 1 vs Source 2)
    - Misurazione della Domain Confusion (probabilità e entropia sul Target)
    - Valutazione su Target sia tramite testa specifica (head_tgt) che tramite ensemble pesato

    Metodi pubblici:
        fit()           — loop completo (chiama train_epoch + evaluate ogni epoca)
        train_epoch()   — singola epoca di training
        train_step()    — singolo batch (forward + loss + backward)
        evaluate()      — valutazione sul target (zero-shot)
        _log()          — logging W&B centralizzato
        _save_checkpoint() — salvataggio best model
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
        patience: int = 7,  # EARLY STOPPING: max epoche senza miglioramenti
        lambda_pseudo: float = 0.1,
        warmup_epochs: int = 5,
        lambda_em: float = 0.1,
        disable_early_stopping_if_mock: bool = False,  # True solo con dati mock (main --mock)
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
        self.epochs_without_improvement = 0
        
        # Creazione della cartella per i checkpoint
        os.makedirs(checkpoint_dir, exist_ok=True)
        self._global_step = 0  # contatore globale per W&B
        # Miglior accuratezza sul target per salvare il checkpoint ottimale
        self.best_tgt_acc = -1.0
        
        # LR Scheduler per un decadimento graduale
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=self.max_epochs, 
            eta_min=1e-6
        )

    def _schedule_disc_dropout(self, epoch: int) -> None:
        # dropout alto all'inizio (GRL debole) -> scende con le epoche
        frac = epoch / max(self.max_epochs - 1, 1)
        p = max(0.2, 0.5 - 0.3 * frac)
        if hasattr(self.model, "discriminator"):
            self.model.discriminator.set_dropout(p)

    def compute_domain_confusion(self, dom_logits_target: torch.Tensor) -> torch.Tensor:
        """
        [WOW STRATEGY] Calcola l'entropia della distribuzione di dominio per i soli campioni target.
        Un'elevata entropia indica che l'encoder estrae feature indistinguibili (DA efficace).
        """
        probs = torch.softmax(dom_logits_target, dim=-1)
        # Aggiungiamo un piccolo epsilon (1e-6) per evitare il log(0) e stabilizzare il calcolo
        entropy = -torch.sum(probs * torch.log(probs + 1e-6), dim=-1)
        return entropy.mean()

    # scheduling
    def _compute_alpha(self, epoch: int, step: int, num_steps: int) -> float:
        """
        Schedule progressivo DANN: alpha cresce da 0 a 1 durante il training.
        Formula: 2 / (1 + exp(-10p)) - 1   dove p = frazione training completata.
        """
        p = float(step + epoch * num_steps) / float(self.max_epochs * num_steps + 1e-8)
        return float((2.0 / (1.0 + torch.exp(torch.tensor(-10.0 * p))) - 1.0).item())

    # ─ Logging

    def _log(self, metrics: dict) -> None:
        """Invia metriche a W&B se disponibile e il run è attivo."""
        if wandb and wandb.run:
            try:
                wandb.log({**metrics, "global_step": self._global_step})
            except Exception:
                pass

    # ─ Checkpoint / Resume

    def resume(self, checkpoint_path: str = None) -> int:
        """Riprende il training da un checkpoint. Ritorna l'epoca di partenza."""
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

        # I buffer (centroidi inclusi) sono già ripristinati da load_state_dict.
        start_epoch = checkpoint.get("epoch", -1) + 1
        print(f"[RESUME] Training ripreso dall'epoca {start_epoch} (global_step: {self._global_step}, patience usata: {self.epochs_without_improvement}/{self.patience}) con best acc {self.best_tgt_acc:.2f}%")
        return start_epoch

    def _save_checkpoint(self, epoch: int, acc: float, is_best: bool = False) -> None:
        """Salva il modello (latest e opzionalmente best)."""
        state = {
            "epoch":                epoch,
            "global_step":          self._global_step,
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_acc":             self.best_tgt_acc,
            "epochs_without_improvement": self.epochs_without_improvement,
        }
        
        # Salva sempre l'ultimo checkpoint per la fault tolerance
        latest_path = os.path.join(self.checkpoint_dir, "latest_checkpoint.pth")
        torch.save(state, latest_path)
        print(f"     [SAVE] Checkpoint epoca {epoch+1} salvato -> {latest_path}")

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
            torch.save(state, best_path)
            print(f"     [BEST] Nuovo best model salvato -> {best_path}  (acc={acc:.2f}%)")

    # ─ Singolo step 

    def train_step(self, batch_s1, batch_s2, batch_tgt, alpha: float, epoch: int) -> dict:
        """
        Esegue un singolo step di training su un triplo di batch.

        Restituisce un dizionario con tutte le metriche dello step:
            loss_total, loss_cls, loss_adv, loss_tgt_pseudo,
            w1, w2, conf_entropy, p_as_s1, p_as_s2, p_as_tgt,
            active_s1 (bool), active_s2 (bool)
        """
        # 1. Aggiorna alpha del GRL
        self.model.set_grl_alpha(alpha)

        # 2. Dati su device
        x_s1, y_s1, d_s1 = (t.to(self.device) for t in batch_s1)
        x_s2, y_s2, d_s2 = (t.to(self.device) for t in batch_s2)
        x_tgt, y_tgt, d_tgt = (t.to(self.device) for t in batch_tgt)

        # 3. Incomplete batch simulation
        active_s1 = True
        active_s2 = self.source2_enabled
        if self.incomplete_simulation:
            if torch.rand(1).item() < 0.1 and active_s2:
                active_s1 = False
            elif torch.rand(1).item() < 0.1 and active_s1:
                active_s2 = False

        # 4. Forward pass
        dom_logits_list, dom_labels_list = [], []
        cls_s1 = cls_s2 = None

        if active_s1:
            cls_s1, dom_s1, _, _ = self.model(x_s1, domain=0)
            dom_logits_list.append(dom_s1)
            dom_labels_list.append(d_s1)

        if active_s2:
            cls_s2, dom_s2, _, _ = self.model(x_s2, domain=1)
            dom_logits_list.append(dom_s2)
            dom_labels_list.append(d_s2)

        cls_tgt, dom_tgt, feat_tgt, ensemble_probs = self.model(x_tgt, domain=2)
        dom_logits_list.append(dom_tgt)
        dom_labels_list.append(d_tgt)

        # 5. Calcolo Losses tramite MultiSourceLoss
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

        # 6. Influence Ratio (proxy: loss classificazione per sorgente)
        total_inf = loss_cls_s1_item + loss_cls_s2_item + 1e-8
        ratio_s1  = loss_cls_s1_item / total_inf
        ratio_s2  = loss_cls_s2_item / total_inf

        # 7. Pesi ensemble (stessa τ del modello, non hardcoded)
        tau = getattr(self.model, "temperature", 0.1)
        with torch.no_grad():
            both_ready = (
                self.model.s1_centroid_initialized.item()
                and self.model.s2_centroid_initialized.item()
            )
            if both_ready:
                mu = feat_tgt.mean(dim=0)
                s1 = F.cosine_similarity(mu.unsqueeze(0), self.model.s1_centroid.unsqueeze(0), eps=1e-8)
                s2 = F.cosine_similarity(mu.unsqueeze(0), self.model.s2_centroid.unsqueeze(0), eps=1e-8)
                w1, w2 = torch.softmax(torch.stack([s1, s2]) / tau, dim=0)
                w1, w2 = w1.item(), w2.item()
            else:
                w1 = w2 = 0.5

        # 8. Loss pseudo-labeling target (Hard pseudo-labeling maskata sulla confidence)
        if epoch >= self.warmup_epochs:
            confidence, pseudo_labels = ensemble_probs.detach().max(dim=-1)
            mask = confidence > 0.7
            if mask.sum() > 0:
                loss_tgt_pseudo = F.cross_entropy(cls_tgt[mask], pseudo_labels[mask])
            else:
                loss_tgt_pseudo = torch.tensor(0.0, device=self.device)
        else:
            loss_tgt_pseudo = torch.tensor(0.0, device=self.device)

        # 8b. Entropy Minimization sul target
        probs_tgt = F.softmax(cls_tgt, dim=-1).clamp(min=1e-8)
        loss_em = -(probs_tgt * probs_tgt.log()).sum(-1).mean()
        
        # Schedule dinamico per lambda_em
        current_lambda_em = self.lambda_em * min(1.0, max(0.0, epoch - self.warmup_epochs) / 5.0)

        # 9. Loss totale
        # La loss avversariale è già scalata dentro loss_dict["loss_total"]
        loss_total = loss_dict["loss_total"] + self.lambda_pseudo * loss_tgt_pseudo + current_lambda_em * loss_em

        # 10. Backward
        self.optimizer.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # 11. Domain confusion (no grad)
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

    # ─ Singola epoca 

    def train_epoch(self, train_loader, epoch: int) -> dict:
        """
        Esegue una singola epoca di training.
        Chiama train_step() per ogni batch e aggrega le metriche.
        Restituisce il dizionario delle metriche medie dell'epoca.
        """
        self.model.train()
        self._schedule_disc_dropout(epoch)
        num_steps = len(train_loader)

        # Accumulatori
        totals = {k: 0.0 for k in [
            "loss_total", "loss_cls", "loss_adv", "loss_tgt_pseudo", "loss_em",
            "influence_s1", "influence_s2", "w1", "w2",
            "conf_entropy", "p_as_s1", "p_as_s2", "p_as_tgt",
        ]}
        s1_active = s2_active = 0

        pbar = tqdm(
            enumerate(train_loader),
            total=num_steps,
            desc=f"Epoca {epoch+1}/{self.max_epochs}",
        )

        for step, (batch_s1, batch_s2, batch_tgt) in pbar:
            alpha = self._compute_alpha(epoch, step, num_steps)
            metrics = self.train_step(batch_s1, batch_s2, batch_tgt, alpha, epoch)

            # Accumula
            for k in totals:
                totals[k] += metrics[k]
            s1_active += int(metrics["active_s1"])
            s2_active += int(metrics["active_s2"])
            self._global_step += 1

            # Log per step
            self._log({
                "step/loss_total":    metrics["loss_total"],
                "step/loss_cls":      metrics["loss_cls"],
                "step/loss_adv":      metrics["loss_adv"],
                "step/loss_tgt_pseudo": metrics["loss_tgt_pseudo"],
                "step/loss_em":       metrics["loss_em"],
                "step/influence_s1":  metrics["influence_s1"],
                "step/influence_s2":  metrics["influence_s2"],
                "step/conf_entropy":  metrics["conf_entropy"],
                "step/grl_alpha":     metrics["alpha"],
            })

            pbar.set_postfix({
                "Loss":    f"{metrics['loss_total']:.3f}",
                "ConfEnt": f"{metrics['conf_entropy']:.3f}",
                "Inf S1":  f"{metrics['influence_s1']:.2f}",
            })

        # Medie epoca
        avgs = {k: v / num_steps for k, v in totals.items()}
        avgs["s1_active_ratio"] = s1_active / num_steps
        avgs["s2_active_ratio"] = s2_active / num_steps

        # Log per epoca
        self._log({f"epoch/{k}": v for k, v in avgs.items()} | {"epoch/num": epoch + 1})

        return avgs

    # ─ Valutazione 

    def evaluate(self, eval_loader, epoch: int) -> dict:
        """
        Valuta il modello sul target (Kinetics) in modalità zero-shot.
        Confronta testa specifica (head_tgt) vs ensemble semantico pesato.
        """
        self.model.eval()
        correct_head = correct_ens = total = 0

        with torch.no_grad():
            for x, y, _ in eval_loader:
                x, y = x.to(self.device), y.to(self.device)
                cls_logits, _, _, ensemble_probs = self.model(x, domain=2)

                correct_head += (cls_logits.argmax(-1) == y).sum().item()
                correct_ens  += (ensemble_probs.argmax(-1) == y).sum().item()
                total        += y.size(0)

        acc_head = (correct_head / total * 100) if total > 0 else 0.0
        acc_ens  = (correct_ens  / total * 100) if total > 0 else 0.0

        print(f"  [EVAL] Epoca {epoch+1} — head_tgt: {acc_head:.2f}%  |  ensemble: {acc_ens:.2f}%")

        self._log({
            "eval/acc_head_tgt": acc_head,
            "eval/acc_ensemble": acc_ens,
            "epoch/num":         epoch + 1,
        })

        is_best = acc_head > self.best_tgt_acc
        if is_best:
            self.best_tgt_acc = acc_head
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        self._save_checkpoint(epoch, acc_head, is_best=is_best)

        return {"acc_head": acc_head, "acc_ensemble": acc_ens}

    # ─ Loop principale 

    def fit(self, train_loader, eval_loader, auto_resume: bool = True, resume_path: str = None) -> None:
        """
        fit()                               ← orchestrazione chiama :
            ├── train_epoch()               ← aggrega gli step, logga per epoca
            │     └── train_step()          ← forward, loss, backward
            └── evaluate()                  ← zero-shot sul target, salva checkpoint
                    └── _save_checkpoint()
                        _compute_alpha()    ← schedule GRL
                        _log()              ← W&B centralizzato

        Loop completo di training.
        Per ogni epoca: train_epoch() -> evaluate() -> stampa riepilogo
        """
        print("\n" + "=" * 50)
        print(f"Inizio training su {self.device}")
        print(f"Epoche massime: {self.max_epochs} | "
              f"Incomplete sim: {self.incomplete_simulation} | "
              f"Source2: {self.source2_enabled}")
        print("=" * 50)

        start_epoch = 0
        if auto_resume:
            start_epoch = self.resume(resume_path)

        for epoch in range(start_epoch, self.max_epochs):

            # Training
            train_metrics = self.train_epoch(train_loader, epoch)

            # Riepilogo epoca
            print(f"\n--- Riepilogo Epoca {epoch+1} ---")
            print(f"  Loss:      total={train_metrics['loss_total']:.4f} | "
                  f"cls={train_metrics['loss_cls']:.4f} | "
                  f"adv={train_metrics['loss_adv']:.4f} | "
                  f"tgt_ps={train_metrics['loss_tgt_pseudo']:.4f} | "
                  f"em={train_metrics['loss_em']:.4f}")
            print(f"  Influence: S1={train_metrics['influence_s1']:.3f} | "
                  f"S2={train_metrics['influence_s2']:.3f} | "
                  f"ratio={train_metrics['w1']/(train_metrics['w2']+1e-8):.3f}")
            print(f"  Confusion: entropy={train_metrics['conf_entropy']:.3f} | "
                  f"->S1={train_metrics['p_as_s1']:.3f} | "
                  f"->S2={train_metrics['p_as_s2']:.3f} | "
                  f"->Tgt={train_metrics['p_as_tgt']:.3f}")
            print(f"  Drop:      S1 attivo {train_metrics['s1_active_ratio']*100:.0f}% | "
                  f"S2 attivo {train_metrics['s2_active_ratio']*100:.0f}%")

            # Valutazione
            self.evaluate(eval_loader, epoch)
            print("-" * 50)
            
            # Con mock l'acc target resta ~0% -> salta early stop se disable_early_stopping_if_mock
            if self.epochs_without_improvement >= self.patience and not self.disable_early_stopping_if_mock:
                print(f"\n[EARLY STOPPING] Nessun miglioramento per {self.patience} epoche consecutive. Training interrotto all'epoca {epoch+1}.")
                break
                
            self.scheduler.step()
            print(f"  LR attuale: {self.scheduler.get_last_lr()[0]:.2e}")

        print(f"\nTraining completato. Best acc target: {self.best_tgt_acc:.2f}%")
