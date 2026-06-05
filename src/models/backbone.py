# Encoder video condiviso

import os 
import torch
import torch.nn as nn
import urllib.request
from torchvision.models.video import r3d_18, R3D_18_Weights, r2plus1d_18, R2Plus1D_18_Weights
from torchvision.models.video import mc3_18, MC3_18_Weights, s3d, S3D_Weights
import torchvision.models.video as video_models
import pytorchvideo.models.x3d as x3d_models

# ---  Video Transformers ---
import timm
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
    def __init__(self, pretrained: bool = True, model_type: str = "handcrafted", weights_path: str = None):
        
        super(VideoEncoder, self).__init__()
        self.model_type = model_type
        self.out_dim = 512
        self.weights_path = weights_path

        # Se usiamo pesi custom, disabilitiamo il pre-training di default su Kinetics
        if self.weights_path is not None:
            pretrained = False
            print(f"[{model_type}] Pesi di torchvision (Kinetics) disabilitati. Verranno usati pesi custom.")

        # ... logica di inizializzazione della backbone ...
        # Se model_type == "s3d" -> raw_dim = 1024
        # Se model_type == "r2plus1d_18" -> raw_dim = 512
        raw_dim = 512

        # r2plus1d_18 preaddestrato su Kinetics-400 (pesi su target) [UPPER BOUND TECNICO]
        if model_type == "r2plus1d_18":
            print("Caricamento r2plus1d_18 preaddestrato su Kinetics-400 (pesi su target)...")
            weights = R2Plus1D_18_Weights.DEFAULT if pretrained else None # pesi su Kinetics-400
            self.backbone = video_models.r2plus1d_18(weights=weights)
            self.backbone.fc = nn.Identity()
        elif model_type == "resnet18":
            print("Caricamento ResNet18 preaddestrato su Kinetics-400 (pesi su target)...")
            weights = R3D_18_Weights.DEFAULT if pretrained else None # pesi su Kinetics-400
            self.backbone = video_models.r3d_18(weights=weights)
            self.backbone.fc = nn.Identity()

        # no data leak
        elif model_type == "r2plus1d_34_ig65m":
            print("Caricamento R(2+1)D_34 preaddestrato su IG-65M (Puro, No Kinetics) via PyTorch Hub...")
            # torch.hub per scaricare r2plus1d_34_clip8_ig65m_from_scratch-9bae36ae.pth automaticamente
            self.backbone = torch.hub.load('moabitcoin/ig65m-pytorch', 'r2plus1d_34_8_ig65m', num_classes=487, pretrained=pretrained, trust_repo=True)
            raw_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif model_type == "handcrafted":
            print("Inizializzazione della rete Handcrafted3DCNN from scratch (no pretraining)...")
            self.backbone = Handcrafted3DCNN()
            if pretrained:
                print("Warning: Handcrafted3DCNN ignorera' l'impostazione pretrained=True.")
        elif model_type == "mc3_18":
            print("Caricamento MC3-18 da zero (no pretraining)...")
            weights = MC3_18_Weights.DEFAULT if pretrained else None
            self.backbone = video_models.mc3_18(weights=weights)
            self.backbone.fc = nn.Identity()
        elif model_type == "s3d":
            print("Caricamento S3D da zero (no pretraining)...")
            raw_dim = 1024
            weights = S3D_Weights.DEFAULT if pretrained else None
            self.backbone = video_models.s3d(weights=weights)
            self.backbone.classifier = nn.Identity()
            if hasattr(self.backbone, 'avgpool'):
                self.backbone.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        elif model_type == "x3d_s":
            print("Caricamento X3D_S...")
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
        #elif model_type == "mvit":
            # print("Caricamento MViT preaddestrato sul target ...")
            # Multiscale Vision Transformer (MViT)
            #raw_dim = 768 
            #weights = MViT_V1_B_Weights.DEFAULT if pretrained else None
            #self.backbone = video_models.mvit_v1_b(weights=weights)
            #self.backbone.head[1] = nn.Identity() # Rimuove il layer di classificazione

        #elif model_type == "swin3d":
            #print("Caricamento Swin3D preaddestrato su Kinetics-400 (pesi su target)...")
            # Video Swin Transformer
            #raw_dim = 768
            #weights = Swin3D_T_Weights.DEFAULT if pretrained else None
            #self.backbone = video_models.swin3d_t(weights=weights)
            #self.backbone.head = nn.Identity() # Rimuove il layer di classificazione

        elif model_type == "swin_2d_imagenet":
            print("Caricamento Swin 2D preaddestrato su ImageNet-1K (Puro, No Kinetics) via TIMM...")
            try:
                # Usa TIMM per scaricare uno Swin Transformer 2D. num_classes=0 rimuove la testa.
                self.backbone = timm.create_model("swin_small_patch4_window7_224.ms_in1k", pretrained=pretrained, num_classes=0)
                raw_dim = self.backbone.num_features
                self.is_2d_transformer = True
            except ImportError:
                raise ImportError("Per usare 'swin_2d_imagenet' devi installare timm.")

        elif model_type == "mvit_2d_imagenet":
            print("Caricamento MViTv2 2D preaddestrato su ImageNet-1K (Puro, No Kinetics) via TIMM...")
            try:
                # Usa TIMM per scaricare un MViT 2D addestrato su ImageNet. num_classes=0 rimuove la testa.
                self.backbone = timm.create_model("mvitv2_small.fb_in1k", pretrained=pretrained, num_classes=0)
                raw_dim = self.backbone.num_features
                self.is_2d_transformer = True
            except ImportError:
                raise ImportError("Per usare 'mvit_2d_imagenet' devi installare timm.")
            
        else: # Default handcrafted 
            print("Inizializzazione della rete Handcrafted3DCNN from scratch (no pretraining)...")
            self.backbone = Handcrafted3DCNN()
            if pretrained:
                print("Warning: Handcrafted3DCNN ignorera' l'impostazione pretrained=True.")

        # --- Caricamento Pesi Custom Alternativi (No Kinetics) ---
        if self.weights_path is not None:
            if os.path.exists(self.weights_path):
                print(f"Caricamento pesi custom da: {self.weights_path}...")
                state_dict = torch.load(self.weights_path, map_location='cpu')
                # Gestione chiavi 'state_dict' (spesso i checkpoint hanno i pesi dentro una sottochiave)
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                elif 'model_state' in state_dict:
                    state_dict = state_dict['model_state']
                
                # Caricamento permissivo (strict=False) per ignorare il layer finale del classificatore
                missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
                print(f"Pesi caricati. Chiavi ignorate/mancanti: {len(missing) + len(unexpected)}")
            else:
                print(f"ERRORE: File pesi non trovato: {self.weights_path}")

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
        # Interpolazione condizionale per architetture che necessitano di risoluzioni maggiori
        if self.model_type in ["mvit", "swin3d"] and x.shape[-1] < 224:
            x = F.interpolate(x, size=(x.shape[2], 224, 224), mode='trilinear', align_corners=False)
        elif getattr(self, "model_type", None) == "x3d_s" and x.shape[-1] < 160:
            # X3D_S crasha con input 112x112 perché le pooling interne riducono la risoluzione a 4x4,
            # ma il layer finale ha un kernel 5x5. Interpoliamo a 160x160 (che è il suo default).
            x = F.interpolate(x, size=(x.shape[2], 160, 160), mode='trilinear', align_corners=False)
        
        # Gestione speciale per Transformer 2D su ImageNet (No Kinetics)
        if getattr(self, "is_2d_transformer", False):
            # x è (B, C, T, H, W). Lo portiamo a (B*T, C, H, W) per passarlo in una rete 2D
            B, C, T_dim, H, W = x.shape
            x_2d = x.permute(0, 2, 1, 3, 4).reshape(B * T_dim, C, H, W)
            
            if H < 224 or W < 224:
                x_2d = F.interpolate(x_2d, size=(224, 224), mode='bilinear', align_corners=False)
                
            features_2d = self.backbone(x_2d) # (B*T, raw_dim)
            features_2d = self.dropout(features_2d)
            features_2d = self.proj(features_2d) # (B*T, 512)
            
            # Media sul tempo (Temporal Average Pooling)
            features = features_2d.view(B, T_dim, -1).mean(dim=1) # (B, 512)
            return features

        # Passiamo il tensore al modello standard 3D
        features = self.backbone(x)
        
        # Applichiamo la proiezione lineare finale comune con l'aggiunta di Dropout
        features = torch.flatten(features, 1)
        features = self.dropout(features)
        features = self.proj(features)
            
        return features

if __name__ == "__main__":
    # Test rapido per entrambe le varianti
    print("--- Test R3D-18 ---")
    model_r3d = VideoEncoder(pretrained=False, model_type="r3d_18", weights_path=None)
    # 1. Test shape da datasets.py: (B, T, C, H, W)
    dummy_input = torch.randn(2, 16, 3, 112, 112)
    print(f"Test 1. Loader shape (2, 16, 3, 112, 112) -> Output R3D-18 shape: {model_r3d(dummy_input).shape}")

    # 2. Test shape standard / test fittizi: (B, C, T, H, W)
    dummy_input_standard = torch.randn(2, 3, 16, 112, 112)
    output_standard = model_r3d(dummy_input_standard)
    print(f"Test 2. Standard shape (2, 3, 16, 112, 112) -> Output shape: {output_standard.shape}")
    

    print("\n--- Test R(2+1)D-18 (State of the Art) ---")
    model_r2plus = VideoEncoder(pretrained=False, model_type="r2plus1d_18", weights_path=None)
    print(f"Output R(2+1)D-18 shape: {model_r2plus(dummy_input).shape}")
    


    print("\n--- Test Handcrafted3DCNN ---")
    model_hand = VideoEncoder(pretrained=False, model_type="handcrafted", weights_path=None)
    print(f"Output Handcrafted shape: {model_hand(dummy_input).shape}")

    print(f"\nout_dim attributo pubblico: {model_hand.out_dim}") # 512
