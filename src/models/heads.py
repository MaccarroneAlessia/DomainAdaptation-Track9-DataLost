# Implementa i 2 classificatori sorgente e quello target.

import torch
import torch.nn as nn

def make_head(input_dim: int, num_classes: int, dropout: float = 0.5) -> nn.Sequential:
    """Factory per una singola testa di classificazione (MLP a 2 strati)."""
    return nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.BatchNorm1d(256), # Garantisce che i gradienti non esplodano prima delle teste
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
        Source 2 (UCF):       101 classi
        Target   (Kinetics): 400 classi
    """
    def __init__(
            self, 
            input_dim: int = 512, 
            num_classes_s1: int = 51,
            num_classes_s2: int = 101,
            num_classes_tgt: int = 400,
            dropout: float = 0.5,
        ):
        super().__init__()
        
        # nn.ModuleDict
        # Registro per accesso dinamico nel forward
        self.heads = nn.ModuleDict({
            # Teste specifiche per sorgente (Domain-specific)
            "0": make_head(input_dim, num_classes_s1, dropout),
            "1": make_head(input_dim, num_classes_s2, dropout),
            "2": make_head(input_dim, num_classes_tgt, dropout)
        })
        
    def forward(self, x: torch.Tensor, domain_id: int) -> torch.Tensor:
        """
        In fase di training possiamo selezionare la testa in base al dominio.
        In fase di test usiamo la target_head.

        domain_id: 0 = Source1, 1 = Source2, 2 = Target
        Usato sia in training (sorgenti) che in eval (target).
        """
        dom_str = str(domain_id)
        if dom_str not in self.heads:
            raise ValueError(f"domain_id non registrato. Ricevuto: {domain_id}")
        return self.heads[dom_str](x)