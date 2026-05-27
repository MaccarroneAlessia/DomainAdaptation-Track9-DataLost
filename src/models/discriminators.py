# Implementa il Gradient Reversal Layer (GRL) e i discriminatori di dominio

import torch
import torch.nn as nn
from torch.autograd import Function

class GRLFunction(Function): # torch.autograd.Function
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(torch.tensor(alpha))
        # save_for_backward per i tensori più robusto 
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Inverte il gradiente durante la backpropagation
        # neg() * alpha → il discriminatore "spinge" l'encoder
        # verso feature domain-invarianti
        output = grad_output.neg() * ctx.saved_tensors[0].item()
        return output, None
        # None per alpha, che non è un tensore e non ha gradiente


class GRL(nn.Module):
    """
    Wrapper Module del GRL, con alpha schedulabile durante il training.
    Utile per la strategia progressiva: alpha cresce da 0 → 1 nelle prime epoche.
    """
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GRLFunction.apply(x, self.alpha)
    

class DomainDiscriminator(nn.Module):
    """
    Discriminatore di dominio per l'addestramento avversariale.
    Distinguere tra Source 1 (dom=0), Source 2 (dom=1) e Target (dom=2).

    Architettura semplice, se fosse troppo potente
    vince sull'encoder e l'addestramento diverge.
    """
    def __init__(self, input_dim: int = 512, num_domains: int = 3, dropout_p: float = 0.3):
        super().__init__()

        # GRL è ora separato -> aggiornare alpha a ogni epoca senza ricostruire il discriminatore
        self.grl = GRL(alpha=1.0)
        self.dropout = nn.Dropout(dropout_p)

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            self.dropout,
            nn.Linear(256, num_domains)
        )

    def set_alpha(self, alpha: float):
        """Permette al trainer di schedulare alpha progressivamente."""
        self.grl.alpha = alpha
        
    def set_dropout(self, p: float):
        """Permette al trainer di schedulare dinamicamente il dropout."""
        self.dropout.p = p

    def forward(self, x: torch.Tensor):
        # Passaggio attraverso il Gradient Reversal Layer
        # alpha non deve essere un parametro di forward
        x = self.grl(x)
        return self.net(x)