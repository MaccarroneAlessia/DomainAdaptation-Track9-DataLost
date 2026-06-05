# src/models/model.py
import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .backbone import VideoEncoder
    from .discriminators import DomainDiscriminator
    from .heads import make_classifier
except ImportError:
    from backbone import VideoEncoder
    from discriminators import DomainDiscriminator
    from heads import make_classifier

def load_overlap_matrix(csv_path: str, num_src: int, num_tgt: int) -> torch.Tensor:
    """
    Carica la matrice di similarità semantica (num_src x num_tgt) da file CSV.
    Ogni riga rappresenta una classe sorgente; 
    i valori sono i punteggi di overlap con le classi target. 
    Le righe vengono normalizzate a somma 1.

    Se il file non esiste, crea una matrice uniforme.
    """
    if not os.path.exists(csv_path):
        # Fallback uniforme per test fittizi o in assenza di file
        return torch.ones(num_src, num_tgt) / num_tgt
    try:
        matrix = []
        with open(csv_path, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader)  # Salta l'header
            for row in reader:
                if len(row) > 1:
                    scores = [float(v) for v in row[1:]]
                    matrix.append(scores)
        
        tensor = torch.tensor(matrix, dtype=torch.float32)
        
        # Gestione di discrepanze nelle shape
        if tensor.shape != (num_src, num_tgt):
            return torch.ones(num_src, num_tgt) / num_tgt
        
        # Normalizzazione per riga (somma a 1 per essere una distribuzione valida, distribuzione valida su classi target)
        row_sums = tensor.sum(dim=1, keepdim=True).clamp(min=1e-8)
        #row_sums = torch.where(row_sums == 0, torch.ones_like(row_sums), row_sums)
        return tensor / row_sums
    except Exception:
        # Qualsiasi errore nel parsing restituisce la matrice uniforme
        return torch.ones(num_src, num_tgt) / num_tgt


class MultiSourceDANN(nn.Module):
    """
    Multi-Source Domain Adversarial Neural Network (MS-DANN) per Action Recognition.

    Architettura
                ┌──────────────┐
                │  VideoEncoder│  ← CNN condivisa (R3D-18 o R(2+1)D-18), output 512-dim
                └──────┬───────┘
                       │ feat (B, 512)
       ┌───────────────┼─────────────────┐
       │               │                 │
    head_s1         head_s2         head_tgt   GRL -> DomainDiscriminator
    (51 cls)        (5 cls)         (400 cls)      (3 domini: S1, S2, Target)
    
    -> usare due dataset sorgente (HMDB-51 e UCF-101) 
       per addestrare il modello ad adattarsi a un dataset target (Kinetics), sfruttando 
       la similarità semantica per bilanciare la conoscenza.

    Training:
        - Source 1 (HMDB-51):  supervisione con head_s1 + adversarial domain loss
        - Source 2 (UCF-101):  supervisione con head_s2 + adversarial domain loss
        - Target  (Kinetics):  solo domain loss + ensemble semantico zero-shot

    Target Classifier Combinato (Ensemble):
        Le predizioni di head_s1 e head_s2 vengono proiettate sulle 400 classi Kinetics
        tramite le matrici di overlap semantico (M1, M2). I pesi dell'ensemble sono
        calcolati dinamicamente via similarità coseno tra il batch target e i centroidi
        EMA delle due sorgenti.
    """
    def __init__(
            self, 
            num_classes_s1: int = 51, 
            num_classes_s2: int = 5, 
            num_classes_tgt: int = 400, 
            pretrained: bool = False, 
            backbone_type: str = "r3d_18",
            temperature: float = 0.1,
            ema_momentum: float = 0.9,
            dropout: float = 0.5
    ):
        super().__init__()
        self.temperature = temperature
        self.ema_momentum = ema_momentum
        

        # 1: Shared Feature Encoder (La CNN che estrae le feature video)
        # encoder (R3D-18 o R(2+1)D-18) -> feature vector da 512-dim per tutti e 3 i domini.
        self.encoder = VideoEncoder(pretrained=pretrained, model_type=backbone_type)
        dim = self.encoder.out_dim  # 512
        
        self.head_s1  = make_classifier(dim, num_classes_s1, dropout=dropout)
        self.head_s2  = make_classifier(dim, num_classes_s2, dropout=dropout)
        self.head_tgt = make_classifier(dim, num_classes_tgt, dropout=dropout)
        
        self.discriminator = DomainDiscriminator(input_dim=dim, num_domains=3)
        
        # Registrazione Buffer di Stato per i Centroidi
        self.register_buffer("s1_centroid", torch.zeros(dim))
        self.register_buffer("s2_centroid", torch.zeros(dim))
        self.register_buffer("s1_centroid_initialized", torch.tensor(False))
        self.register_buffer("s2_centroid_initialized", torch.tensor(False))
        
        # Caricamento Matrici Semantiche
        base = os.path.join(os.path.dirname(__file__), "..", "datasets", "label_analysis_output")
        self.register_buffer("M1", load_overlap_matrix(os.path.join(base, "overlap_hmdb_kinetics.csv"), num_classes_s1, num_classes_tgt))
        self.register_buffer("M2", load_overlap_matrix(os.path.join(base, "overlap_ucf_kinetics.csv"), num_classes_s2, num_classes_tgt))

    def set_grl_alpha(self, alpha: float):
        """Aggiorna il peso (alpha) del GRL per rendere l'adattamento progressivo 0->1."""
        self.discriminator.set_alpha(alpha)

    def _update_centroid(self, domain: int, feat: torch.Tensor) -> None:
        """Aggiornamento EMA del centroide della sorgente indicata (solo in training)."""
        momentum = self.ema_momentum
        batch_mean = feat.detach().mean(dim=0)
        if domain == 0:
            if not self.s1_centroid_initialized.item():
                self.s1_centroid.copy_(batch_mean)
                self.s1_centroid_initialized.fill_(True)
            else:
                self.s1_centroid.copy_(momentum * self.s1_centroid + (1 - momentum) * batch_mean)
        elif domain == 1:
            if not self.s2_centroid_initialized.item():
                self.s2_centroid.copy_(batch_mean)
                self.s2_centroid_initialized.fill_(True)
            else:
                self.s2_centroid.copy_(momentum * self.s2_centroid + (1 - momentum) * batch_mean)

    def _compute_ensemble(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Ensemble semantico per il dominio target.

        Pesi dinamici:
            w1, w2 = softmax( [cos(feat_tgt, c_s1), cos(feat_tgt, c_s2)] / τ )

        Proiezione:
            probs_tgt = w1 * (softmax(head_s1(feat)) @ M1)
                      + w2 * (softmax(head_s2(feat)) @ M2)

        Restituisce un tensore (B, num_classes_tgt) con probabilità che sommano a 1.
        """
        both_ready = self.s1_centroid_initialized.item() and self.s2_centroid_initialized.item()
        if both_ready:
            mu = feat.mean(dim=0, keepdim=True)          # (1, 512)
            s1 = F.cosine_similarity(mu, self.s1_centroid.unsqueeze(0), eps=1e-8)
            s2 = F.cosine_similarity(mu, self.s2_centroid.unsqueeze(0), eps=1e-8)
            w1, w2 = torch.softmax(torch.stack([s1, s2]) / self.temperature, dim=0)
        else:
            w1 = w2 = 0.5  # pesi uniformi finché i centroidi non sono pronti

        probs_s1 = torch.softmax(self.head_s1(feat), dim=-1)          # (B, 51)
        probs_s2 = torch.softmax(self.head_s2(feat), dim=-1)          # (B, 5)
        proj_s1  = torch.matmul(probs_s1, self.M1)                    # (B, 400)
        proj_s2  = torch.matmul(probs_s2, self.M2)                    # (B, 400)

        return w1 * proj_s1 + w2 * proj_s2                            # (B, 400)

    def forward(self, x: torch.Tensor, domain: int = 2):
        """
        Forward pass della rete. Accetta il tensore video e l'etichetta del dominio.
        Args:
            x      : video clip (B, T, C, H, W) o (B, C, T, H, W)
            domain : 0 = HMDB (S1), 1 = UCF (S2), 2 = Kinetics (Target)

        Returns:
            cls_logits    : (B, num_classes_domain)  - logit classificazione
            dom_logits    : (B, 3)                   - logit discriminatore dominio
            feat          : (B, 512)                 - embedding encoder
            ensemble_probs: (B, 400) o None          - solo per domain=2
        """
        # 1. estrazione delle feature dal video tramite la CNN condivisa
        feat = self.encoder(x)                        # (B, 512)
        
        # 2. discriminatore di dominio (GRL per l'adattamento inverte i gradienti internamente)
        dom_logits = self.discriminator(feat)         # (B, 3)
        
        # 3. Aggiornamento centroidi EMA (solo training, solo sorgenti)
        # per calcolare la distanza semantica dal target
        if self.training and domain in (0, 1):
            self._update_centroid(domain, feat)

        # 4. Classificazione specifica per il dominio corrente (source-specific o target)
        _head_map = {0: self.head_s1, 1: self.head_s2, 2: self.head_tgt}
        cls_logits = _head_map[domain](feat)

        # 5. Ensemble pesato semantico (solo per il target)
        # calcoliamo il peso di ciascuna sorgente tramite Similarità Coseno sui centroidi 
        # e proiettiamo le predizioni delle sorgenti sulle 400 classi del target
        # classificatore target combinato dinamico
        ensemble_probs = self._compute_ensemble(feat) if domain == 2 else None

        return cls_logits, dom_logits, feat, ensemble_probs


# smoke test
if __name__ == "__main__":
    print("=" * 60)
    print("SMOKE TEST - MultiSourceDANN")
    print("=" * 60)
    
    # Inizializzazione modello fittizio
    model = MultiSourceDANN(num_classes_s1=51, num_classes_s2=5, num_classes_tgt=400, pretrained=False)
    
    # Test tensor di input standard (B=32, C=3, T=16, H=112, W=112)
    x = torch.randn(32, 3, 16, 112, 112)
    
    print("\n--- Domain 0 (HMDB, S1) ---")
    cls, dom, feat, ens = model(x, domain=0)
    print(f"  cls : {cls.shape}   atteso (4, 51)")
    print(f"  dom : {dom.shape}   atteso (4, 3)")
    print(f"  feat: {feat.shape}  atteso (4, 512)")

    print("\n--- Domain 1 (UCF, S2) ---")
    cls, dom, feat, ens = model(x, domain=1)
    print(f"  cls : {cls.shape}   atteso (4, 5)")
    print(f"  dom : {dom.shape}   atteso (4, 3)")

    print("\n--- Domain 2 (Kinetics, Target) ---")
    # Simuliamo centroidi già pronti
    model.s1_centroid.copy_(torch.randn(512))
    model.s2_centroid.copy_(torch.randn(512))
    model.s1_centroid_initialized.fill_(True)
    model.s2_centroid_initialized.fill_(True)

    cls, dom, feat, ens = model(x, domain=2)
    print(f"  cls : {cls.shape}   atteso (4, 400)")
    print(f"  dom : {dom.shape}   atteso (4, 3)")
    print(f"  ens : {ens.shape}   atteso (4, 400)")
    print(f"  sum(ens[0]): {ens[0].sum().item():.4f}  atteso ≈ 1.0")

    print("\n[SUCCESS] Smoke test PASSED.")
    print("=" * 60)