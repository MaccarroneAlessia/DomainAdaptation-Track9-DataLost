# Loss base per training multi-source:
# L_total = L_cls + lambda_adv * L_adv
# Nota: il termine pseudo-label target viene aggiunto in trainer.py.
import torch
import torch.nn as nn

class MultiSourceLoss(nn.Module):
    """
    Loss totale:  L = L_cls + lambda_adv * L_adv

    L_cls  : CrossEntropy su Source1 + Source2 (supervisione)
    L_adv  : CrossEntropy del discriminatore di dominio (avversariale)
    """
    def __init__(self, lambda_adv: float = 0.1):
        super().__init__()
        self.lambda_adv = lambda_adv
        # label_smoothing=0.1 eccezionale per evitare l'overfitting delle teste sui piccoli dataset
        self.ce = nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(
        self,
        dom_logits: torch.Tensor,
        dom_labels: torch.Tensor,
        logits_s1: torch.Tensor = None, labels_s1: torch.Tensor = None,
        logits_s2: torch.Tensor = None, labels_s2: torch.Tensor = None,
    ) -> dict:
        device = dom_logits.device
        
        # Calcolo condizionale sicuro (Previene i crash mid-epoch causati dalla simulazione incompleta)
        loss_s1 = self.ce(logits_s1, labels_s1) if logits_s1 is not None and labels_s1 is not None else torch.tensor(0.0, device=device)
        loss_s2 = self.ce(logits_s2, labels_s2) if logits_s2 is not None and labels_s2 is not None else torch.tensor(0.0, device=device)
        
        L_cls = loss_s1 + loss_s2
        L_adv = self.ce(dom_logits, dom_labels)
        L_tot = L_cls + self.lambda_adv * L_adv

        # Tracciamento sicuro dei contributi numerici delle sorgenti
        s1_val = loss_s1.item()
        s2_val = loss_s2.item()
        sum_val = s1_val + s2_val
        
        inf_s1 = s1_val / sum_val if sum_val > 0 else 0.5
        inf_s2 = s2_val / sum_val if sum_val > 0 else 0.5

        return {
            "loss_total": L_tot,
            "loss_cls":   L_cls,
            "loss_adv":   L_adv,
            "loss_cls_s1": s1_val,
            "loss_cls_s2": s2_val,
            "influence_ratio_s1": inf_s1,
            "influence_ratio_s2": inf_s2
        }