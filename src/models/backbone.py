# classe dell'encoder (es. R3D-18)
# Il cluster fornisce immagini Apptainer con PyTorch e pesi pre-caricati in /shared/sifs/latest.sif

import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights, r2plus1d_18, R2Plus1D_18_Weights

class VideoEncoder(nn.Module):
    """
    CNN 3D encoder condiviso che rispetta il contratto dell'interfaccia 
    producendo un embedding di self.out_dim = 512

    1. Encoder basato su R3D-18 per l'estrazione di feature da clip video.
    
    R3D-18 (ResNet 3D) perché riesce a catturare
    le feature spazio-temporali dei video

    2. R(2+1)D-18 stato dell'arte 

    - restituisce un vettore da 512 dimensioni (embedding) 
    che cattura sia l'aspetto visivo (spaziale) che il movimento (temporale)
    che poi passeremo ai classificatori

    L'input che ci arriva dal dataset è di forma (B, 16, 3, 112, 112), ma
    torchvision si aspetta (B, C, T, H, W).
    """
    def __init__(self, pretrained: bool = True, model_type: str = "r3d_18"):
        super(VideoEncoder, self).__init__()
        # caricamendo dei pesi preaddestrati di R3D-18 
        # (nota : sul cluster i pesi sono già pre-scaricati nel container SIF per evitare problemi di rete)
        #weights = R3D_18_Weights.DEFAULT if pretrained else None
        #self.backbone = r3d_18(weights=weights)

        # Scelta dell'architettura: R3D-18 (standard) oppure R(2+1)D-18 (stato dell'arte)
        if model_type == "r2plus1d_18":
            weights = R2Plus1D_18_Weights.DEFAULT if pretrained else None
            self.backbone = r2plus1d_18(weights=weights)
        else:
            weights = R3D_18_Weights.DEFAULT if pretrained else None
            self.backbone = r3d_18(weights=weights)
        
        # out_dim come attributo pubblico per parte 3 e
        # per collegare correttamente l'encoder alle teste di classificazione in model.py
        self.out_dim = 512

        # l'ultimo strato fully connected (fc) sostituito con un'identità
        # la rete non fa classificazione, ma si ferma all'estrazione 
        # del vettore di feature spazio-temporali da 512-dim -> EMBEDDING
        self.backbone.fc = nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape standard: (B, C, T, H, W) -> (Batch, 3, 16, 112, 112)
        # Se l'input proviene da datasets.py ed è (B, T, C, H, W) con T=16 e C=3:
        # x.shape[2] == 3 indica che il canale (3) è nella terza posizione (indice 2)
        if len(x.shape) == 5 and x.shape[2] == 3 and x.shape[1] != 3:
            # R3D-18 vuole (B, C, T, H, W), quindi permutiamo da (B, T, C, H, W)
            x = x.permute(0, 2, 1, 3, 4)
        
        # Passiamo il tensore al modello
        features = self.backbone(x)
        return features

if __name__ == "__main__":
    # Test rapido per entrambe le varianti
    print("--- Test R3D-18 ---")
    model_r3d = VideoEncoder(pretrained=False, model_type="r3d_18")
    # 1. Test shape da datasets.py: (B, T, C, H, W)
    dummy_input = torch.randn(2, 3, 16, 112, 112)
    print(f"Test 1. Loader shape (2, 16, 3, 112, 112) -> Output R3D-18 shape: {model_r3d(dummy_input).shape}")

    # 2. Test shape standard / test fittizi: (B, C, T, H, W)
    dummy_input_standard = torch.randn(2, 3, 16, 112, 112)
    output_standard = model_r3d(dummy_input_standard)
    print(f"Test 2. Standard shape (2, 3, 16, 112, 112) -> Output shape: {output_standard.shape}")
    

    print("\n--- Test R(2+1)D-18 (State of the Art) ---")
    model_r2plus = VideoEncoder(pretrained=False, model_type="r2plus1d_18")
    print(f"Output R(2+1)D-18 shape: {model_r2plus(dummy_input).shape}")
    


    print(f"out_dim attributo pubblico: {model_r3d.out_dim}") # 512
