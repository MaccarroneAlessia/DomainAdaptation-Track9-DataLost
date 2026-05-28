# Loss base per training multi-source:
# L_total = L_cls + lambda_adv * L_adv
# Nota: il termine pseudo-label target viene aggiunto in trainer.py.
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiSourceLoss(nn.Module):
    """
    Loss totale:  L = L_cls + lambda_adv * L_adv

    L_cls  : CrossEntropy su Source1 + Source2 (supervisione)
    L_adv  : CrossEntropy del discriminatore di dominio (avversariale)
    """
    def __init__(self, lambda_adv: float = 0.1):
        super().__init__()
        self.lambda_adv = lambda_adv
        self.ce = nn.CrossEntropyLoss(label_smoothing=0.1)

    def classification_loss(
        self,
        logits_s1: torch.Tensor = None, labels_s1: torch.Tensor = None,
        logits_s2: torch.Tensor = None, labels_s2: torch.Tensor = None,
    ) -> torch.Tensor:
        """Supervisione sui due domini sorgente."""
        loss = 0.0
        if logits_s1 is not None and labels_s1 is not None:
            loss = loss + self.ce(logits_s1, labels_s1)
        if logits_s2 is not None and labels_s2 is not None:
            loss = loss + self.ce(logits_s2, labels_s2)
        return loss

    def adversarial_loss(
        self,
        dom_logits: torch.Tensor,  # (B*3, 3) — tutti i domini concatenati
        dom_labels: torch.Tensor,  # (B*3,)
    ) -> torch.Tensor:
        """
        Il GRL fa già il lavoro dell'inversione durante la backprop.
        Qui la loss è una normale CrossEntropy sul discriminatore.
        """
        return self.ce(dom_logits, dom_labels)

    def forward(
        self,
        dom_logits: torch.Tensor,
        dom_labels: torch.Tensor,
        logits_s1: torch.Tensor = None, labels_s1: torch.Tensor = None,
        logits_s2: torch.Tensor = None, labels_s2: torch.Tensor = None,
    ) -> dict:
        device = dom_logits.device
        loss_s1 = self.ce(logits_s1, labels_s1) if logits_s1 is not None else torch.tensor(0.0, device=device)
        loss_s2 = self.ce(logits_s2, labels_s2) if logits_s2 is not None else torch.tensor(0.0, device=device)
        
        L_cls = loss_s1 + loss_s2
        L_adv = self.adversarial_loss(dom_logits, dom_labels)
        L_tot = L_cls + self.lambda_adv * L_adv

        # Restituisce dizionario → utile per il logging con W&B / tensorboard
        return {
            "loss_total": L_tot,
            "loss_cls":   L_cls,
            "loss_adv":   L_adv,
            "loss_cls_s1":  loss_s1.item(),
            "loss_cls_s2":  loss_s2.item(),
        }