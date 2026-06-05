# Encoder video condiviso

import torch
import torch.nn as nn
import urllib.request
from torchvision.models.video import r3d_18, R3D_18_Weights, r2plus1d_18, R2Plus1D_18_Weights
from torchvision.models.video import mc3_18, MC3_18_Weights, s3d, S3D_Weights
import torchvision.models.video as video_models
import pytorchvideo.models.x3d as x3d_models

# ---  Video Transformers ---
from torchvision.models.video import mvit_v1_b, MViT_V1_B_Weights
from torchvision.models.video import swin3d_t, Swin3D_T_Weights
import torch.nn.functional as F

class Handcrafted3DCNN(nn.Module):
    """
    Rete convoluzionale 3D leggera costruita da zero.
    Progettata per estrarre feature spazio-temporali su dataset piccoli 
    senza causare overfitting estremo.
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Input: (B, 3, 16, 112, 112)
            # Block 1
            nn.Conv3d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)), # -> (B, 32, 16, 56, 56)

            # Block 2
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2)), # -> (B, 64, 8, 28, 28)

            # Block 3
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2)), # -> (B, 128, 4, 14, 14)

            # Block 4
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2)), # -> (B, 256, 2, 7, 7)

            # Block 5
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1))      # -> (B, 512, 1, 1, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return torch.flatten(x, 1) # -> (B, 512)


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
    def __init__(self, pretrained: bool = True, model_type: str = "handcrafted"):
        
        super(VideoEncoder, self).__init__()
        self.model_type = model_type
        self.out_dim = 512

        # ... logica di inizializzazione della backbone ...
        # Se model_type == "s3d" -> raw_dim = 1024
        # Se model_type == "r2plus1d_18" -> raw_dim = 512
        raw_dim = 512

        if model_type == "r2plus1d_18":
            weights = R2Plus1D_18_Weights.DEFAULT if pretrained else None
            self.backbone = video_models.r2plus1d_18(weights=weights)
            self.backbone.fc = nn.Identity()
        elif model_type == "handcrafted":
            print("Inizializzazione della rete Handcrafted3DCNN...")
            self.backbone = Handcrafted3DCNN()
            if pretrained:
                print("Warning: Handcrafted3DCNN ignorera' l'impostazione pretrained=True.")
        elif model_type == "mc3_18":
            weights = MC3_18_Weights.DEFAULT if pretrained else None
            self.backbone = video_models.mc3_18(weights=weights)
            self.backbone.fc = nn.Identity()
        elif model_type == "s3d":
            raw_dim = 1024
            weights = S3D_Weights.DEFAULT if pretrained else None
            self.backbone = video_models.s3d(weights=weights)
            self.backbone.classifier = nn.Identity()
            if hasattr(self.backbone, 'avgpool'):
                self.backbone.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        elif model_type == "x3d_s":
            # Fallback robusto per ambiente offline del cluster
            try:
                self.backbone = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_s', pretrained=pretrained)
                raw_dim = self.backbone.blocks[5].proj.in_features
                self.backbone.blocks[5].proj = nn.Identity()
            except Exception as e:
                print(f"[{model_type}] Fallback a S3D causa mancanza librerie offline: {e}")
                raw_dim = 1024
                weights = S3D_Weights.DEFAULT if pretrained else None
                self.backbone = video_models.s3d(weights=weights)
                self.backbone.classifier = nn.Identity()
                if hasattr(self.backbone, 'avgpool'):
                    self.backbone.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
                self.model_type = "s3d"

        # --- Video Transformers ---
        elif model_type == "mvit":
            # Multiscale Vision Transformer (MViT)
            raw_dim = 768 
            weights = MViT_V1_B_Weights.DEFAULT if pretrained else None
            self.backbone = video_models.mvit_v1_b(weights=weights)
            self.backbone.head[1] = nn.Identity() # Rimuove il layer di classificazione

        elif model_type == "swin3d":
            # Video Swin Transformer
            raw_dim = 768
            weights = Swin3D_T_Weights.DEFAULT if pretrained else None
            self.backbone = video_models.swin3d_t(weights=weights)
            self.backbone.head = nn.Identity() # Rimuove il layer di classificazione

        else: # Default r3d_18
            weights = R3D_18_Weights.DEFAULT if pretrained else None
            self.backbone = video_models.r3d_18(weights=weights)
            self.backbone.fc = nn.Identity()

        # Strato lineare comune per l'allineamento dimensionale richiesto dal DANN
        # Ottimizzazione: per i Transformers usiamo Dropout al 50% per mitigare il forte rischio di overfitting
        drop_prob = 0.5 if model_type in ["mvit", "swin3d"] else 0.0
        self.dropout = nn.Dropout(p=drop_prob)
        self.proj = nn.Linear(raw_dim, 512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Adattamento shape dinamica da Dataloader (B, T, C, H, W) -> (B, C, T, H, W)
        if len(x.shape) == 5 and x.shape[2] == 3 and x.shape[1] == 16:
            x = x.permute(0, 2, 1, 3, 4).contiguous()
        
        # I Transformers (MViT/Swin3D) tipicamente richiedono/performano meglio con input spaziale 224x224.
        # Effettuiamo un'interpolazione al volo se necessario.
        if self.model_type in ["mvit", "swin3d"] and x.shape[-1] < 224:
            x = F.interpolate(x, size=(x.shape[2], 224, 224), mode='trilinear', align_corners=False)
        
        # Passiamo il tensore al modello
        features = self.backbone(x)
        
        # Applichiamo la proiezione lineare finale comune con l'aggiunta di Dropout
        features = torch.flatten(features, 1)
        features = self.dropout(features)
        features = self.proj(features)
            
        return features

if __name__ == "__main__":
    # Test rapido per entrambe le varianti
    print("--- Test R3D-18 ---")
    model_r3d = VideoEncoder(pretrained=False, model_type="r3d_18")
    # 1. Test shape da datasets.py: (B, T, C, H, W)
    dummy_input = torch.randn(2, 16, 3, 112, 112)
    print(f"Test 1. Loader shape (2, 16, 3, 112, 112) -> Output R3D-18 shape: {model_r3d(dummy_input).shape}")

    # 2. Test shape standard / test fittizi: (B, C, T, H, W)
    dummy_input_standard = torch.randn(2, 3, 16, 112, 112)
    output_standard = model_r3d(dummy_input_standard)
    print(f"Test 2. Standard shape (2, 3, 16, 112, 112) -> Output shape: {output_standard.shape}")
    

    print("\n--- Test R(2+1)D-18 (State of the Art) ---")
    model_r2plus = VideoEncoder(pretrained=False, model_type="r2plus1d_18")
    print(f"Output R(2+1)D-18 shape: {model_r2plus(dummy_input).shape}")
    


    print("\n--- Test Handcrafted3DCNN ---")
    model_hand = VideoEncoder(pretrained=False, model_type="handcrafted")
    print(f"Output Handcrafted shape: {model_hand(dummy_input).shape}")

    print(f"\nout_dim attributo pubblico: {model_hand.out_dim}") # 512
