# Implementa i 2 classificatori sorgente e quello target.

import torch
import torch.nn as nn

def make_head(input_dim: int, num_classes: int, dropout: float = 0.5) -> nn.Sequential:
    """Factory per una singola testa di classificazione (MLP a 2 strati)."""
    return nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(256, num_classes)
    )

# Alias per compatibilità con model.py
make_classifier = make_head

class MultiSourceClassifier(nn.Module):
    """
    Gestisce teste separate per Source 1, Source 2 e il Target.

    Tre teste di classificazione indipendenti con numero di classi diverso.

        Source 1 (HMDB):     51 classi
        Source 2 (UCF):       5 classi
        Target   (Kinetics): 400 classi
    """
    def __init__(
            self, 
            input_dim: int = 512, 
            num_classes_s1: int = 51,
            num_classes_s2: int = 5,
            num_classes_tgt: int = 400,
            dropout: float = 0.5,
        ):
        super().__init__()
        
        # Teste specifiche per sorgente (Domain-specific)
        self.source1_head = make_head(input_dim, num_classes_s1, dropout)
        self.source2_head = make_head(input_dim, num_classes_s2, dropout)
        self.target_head  = make_head(input_dim, num_classes_tgt, dropout)

        # Registro per accesso dinamico nel forward
        self._heads = {0: self.source1_head, 1: self.source2_head, 2: self.target_head}
        
    def forward(self, x: torch.Tensor, domain_id: int = None):
        """
        In fase di training possiamo selezionare la testa in base al dominio.
        In fase di test usiamo la target_head.

        domain_id: 0 = Source1, 1 = Source2, 2 = Target
        Usato sia in training (sorgenti) che in eval (target).
        """
        if domain_id not in self._heads:
            raise ValueError(f"domain_id deve essere 0, 1 o 2. Ricevuto: {domain_id}")
        return self._heads[domain_id](x)
    
if __name__ == "__main__":
    clf = MultiSourceClassifier()
    feat = torch.randn(4, 512)

    out_s1  = clf(feat, domain_id=0)
    out_s2  = clf(feat, domain_id=1)
    out_tgt = clf(feat, domain_id=2)

    print(f"S1  logits: {out_s1.shape}")   # (4, 51)
    print(f"S2  logits: {out_s2.shape}")   # (4, 5)
    print(f"Tgt logits: {out_tgt.shape}")  # (4, 400)