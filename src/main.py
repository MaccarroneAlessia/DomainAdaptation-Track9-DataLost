# src/main.py
# apptainer run --nv /shared/sifs/latest.sif python src/main.py --config experiments/configs/base_config.yaml

import argparse
import copy
import os
import random
import numpy as np
import torch
import torch.optim as optim
import yaml

from datasets.datasets import build_dataloaders
from models.model      import MultiSourceDANN
from training.losses   import MultiSourceLoss
from training.trainer  import Trainer


def load_config(base: str, override: str = None) -> dict:
    with open(base, 'r') as f:
        cfg = yaml.safe_load(f)
    if override:
        with open(override, 'r') as f:
            ov = yaml.safe_load(f)
        for section, values in ov.items():
            if isinstance(values, dict):
                cfg.setdefault(section, {}).update(values)
            else:
                cfg[section] = values
    return cfg


# --- COMPONENTI MOCK PER DEBUG E DRY-RUN OFFLINE -----------------------------
class MockBatch:
    def __init__(self, batch_size: int, num_classes: int, domain_id: int):
        self.x = torch.randn(batch_size, 16, 3, 112, 112)
        self.y = torch.randint(0, num_classes, (batch_size,))
        self.d = torch.full((batch_size,), domain_id, dtype=torch.long)
        
    def __getitem__(self, idx):
        return (self.x, self.y, self.d)[idx]


class MockMultiSourceDataLoader:
    def __init__(self, batch_size: int = 8, num_steps: int = 5):
        self.batch_size = batch_size
        self.num_steps = num_steps
        
    def __iter__(self):
        for _ in range(self.num_steps):
            yield (
                MockBatch(self.batch_size, 51, 0),
                MockBatch(self.batch_size, 5, 1),
                MockBatch(self.batch_size, 400, 2)
            )
            
    def __len__(self):
        return self.num_steps
        
    def get_target_eval_loader(self, batch_size: int = 16):
        class MockEvalLoader:
            def __init__(self, bs: int, steps: int):
                self.bs = bs
                self.steps = steps
            def __iter__(self):
                for _ in range(self.steps):
                    yield (
                        torch.randn(self.bs, 16, 3, 112, 112),
                        torch.randint(0, 400, (self.bs,)),
                        torch.full((self.bs,), 2, dtype=torch.long)
                    )
            def __len__(self):
                return self.steps
        return MockEvalLoader(batch_size, 2)


# --- FUNZIONI DI VALUTAZIONE PROTOCOLLO DOMAIN ADAPTATION -------------------
def evaluate_target(model, eval_loader, device, dtype=torch.float32):
    """Valutazione target: confronta l'head_tgt nativa con lo Zero-Shot Ensemble semantico."""
    model.eval()
    correct_head = correct_ens = total = 0
    with torch.no_grad():
        for frames, labels, _ in eval_loader:
            # Corretto il cast al dtype hardware (bfloat16 o fp32) per evitare crash di tipo peso/input
            frames = frames.to(device=device, dtype=dtype)
            labels = labels.to(device=device, dtype=torch.long)
            
            cls_logits, _, _, ensemble_probs = model(frames, domain=2)
            
            correct_head += (cls_logits.argmax(-1) == labels).sum().item()
            
            if ensemble_probs is not None:
                correct_ens += (ensemble_probs.argmax(-1) == labels).sum().item()
            else:
                correct_ens += 0
                
            total += labels.size(0)
            
    if total == 0:
        return 0.0, 0.0
    return 100.0 * correct_head / total, 100.0 * correct_ens / total


def evaluate_source_only_entropy(model, eval_loader, device, dtype=torch.float32):
    """Misura il livello di confusione/adattamento calcolando l'entropia della head_s1 sul target."""
    model.eval()
    entropy_list = []
    
    with torch.no_grad():
        for frames, _, _ in eval_loader:
            frames = frames.to(device=device, dtype=dtype)
            cls_on_tgt, _, _, _ = model(frames, domain=0)
            
            # Calcolo numericamente stabile dell'entropia di Shannon via log_softmax
            probs = torch.softmax(cls_on_tgt, dim=-1)
            log_probs = torch.log_softmax(cls_on_tgt, dim=-1)
            entropy = -(probs * log_probs).sum(-1).mean().item()
            entropy_list.append(entropy)
            
    return sum(entropy_list) / len(entropy_list) if entropy_list else 0.0


# --- PIPELINE DI CONFIGURAZIONE ED ESECUZIONE ESPERIMENTI ---------------------
def run_experiment(run_name, cfg, loader, eval_loader, hmdb_map, ucf_map, kin_map, device, is_mock=False, auto_resume=True):
    """Costruisce modello+trainer, esegue training e ritorna metriche target."""
    print(f"\n=== RUN: {run_name} ===")
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    ablation_cfg = cfg.get("ablation", {})

    model = MultiSourceDANN(
        num_classes_s1=len(hmdb_map),
        num_classes_s2=len(ucf_map),
        num_classes_tgt=len(kin_map),
        pretrained=model_cfg.get("pretrained", False),
        backbone_type=model_cfg.get("encoder", "r2plus1d_18"),
        temperature=model_cfg.get("temperature", 0.1),
        ema_momentum=model_cfg.get("ema_momentum", 0.9),
        dropout=model_cfg.get("dropout", 0.5),
    ).to(device)

    # Configurazione dinamica della precisione hardware rilevata
    target_dtype = torch.float32
    if cfg.get("hardware", {}).get("precision") == "bfloat16":
        model = model.to(torch.bfloat16)
        target_dtype = torch.bfloat16
        print("--> Attivata precisione nativa bfloat16 per nodi Ampere/L40S.")
        
    print(f"Modello su {device} - parametri totali: {sum(p.numel() for p in model.parameters()):,}")

    loss_fn = MultiSourceLoss(lambda_adv=train_cfg.get("lambda_adv", 0.1))
    optimizer = optim.Adam(
        model.parameters(),
        lr=train_cfg.get("learning_rate", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )

    resnet_name = model_cfg.get("encoder", "r2plus1d_18")
    run_tag = f"{resnet_name}_{run_name}".replace(" ", "_").replace("(", "").replace(")", "").lower()
    checkpoint_path = cfg["paths"].get("checkpoint", "experiments/checkpoints")
    checkpoint_dir = os.path.join(checkpoint_path, run_tag)

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        max_epochs=train_cfg.get("max_epochs", 30),
        checkpoint_dir=checkpoint_dir,
        incomplete_simulation=ablation_cfg.get("incomplete_simulation", True),
        source2_enabled=ablation_cfg.get("source2_enabled", True),
        patience=train_cfg.get("patience", 7),
        lambda_pseudo=train_cfg.get("lambda_pseudo", 0.1),
        warmup_epochs=train_cfg.get("warmup_epochs", 5),
        lambda_em=train_cfg.get("lambda_em", 0.1),
        disable_early_stopping_if_mock=is_mock,
    )
    
    # Avvio del loop di training
    trainer.fit(train_loader=loader, eval_loader=eval_loader, auto_resume=auto_resume)

    # Valutazione finale passando il dtype corretto per prevenire i crash di inferenza
    acc_head, acc_ens = evaluate_target(model, eval_loader, device, dtype=target_dtype)
    src_entropy = evaluate_source_only_entropy(model, eval_loader, device, dtype=target_dtype)
    
    return {
        "acc_head_tgt": acc_head,
        "acc_ens_tgt": acc_ens,
        "entropy_s1_on_tgt": src_entropy,
        "best_acc_head_tgt": trainer.best_tgt_acc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",          default="experiments/configs/base_config.yaml")
    parser.add_argument("--config-override", default=None)   
    parser.add_argument("--mock",            action="store_true", help="Usa dati fittizi/simulati per dry-run offline")
    parser.add_argument("--compare-da",      action="store_true", help="Confronta backbone-only (no DA) vs DA sul target")
    args = parser.parse_args()

    cfg = load_config(args.config, args.config_override)

    # --- Riproducibilità Rigorosa ---------------------------------------------
    seed = cfg.get("hardware", {}).get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    # --- Configurazione Device Hardware --------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Caricamento Dati (Real Loader o Mock Mode) ---------------------------
    if args.mock:
        print("\n[MOCK MODE ACTIVE] Generazione tensori sintetici ad alta fedeltà...")
        hmdb_map = {f"class_{i}": i for i in range(51)}
        ucf_map = {f"class_{i}": i for i in range(5)}
        kin_map = {f"class_{i}": i for i in range(400)}
        loader = MockMultiSourceDataLoader(batch_size=cfg["data"]["batch_size"], num_steps=5)
        eval_loader = loader.get_target_eval_loader(batch_size=16)
        # Sovrascrive il numero di epoche per rendere il test istantaneo
        cfg["training"]["max_epochs"] = min(cfg["training"]["max_epochs"], 2)
    else:
        loader, hmdb_map, ucf_map, kin_map = build_dataloaders(
            cfg["paths"]["data_root"],
            batch_size=cfg["data"].get("batch_size", 8),
            num_workers=cfg["data"].get("num_workers", 4),
        )
        eval_loader = loader.get_target_eval_loader(batch_size=16)

    print(f"Mappatura Classi completata - S1: {len(hmdb_map)} | S2: {len(ucf_map)} | Target: {len(kin_map)}")

    # --- Esecuzione degli Esperimenti ----------------------------------------
    if args.compare_da:
        cfg_no_da = copy.deepcopy(cfg)
        cfg_no_da.setdefault("training", {})
        cfg_no_da["training"]["lambda_adv"] = 0.0
        cfg_no_da["training"]["lambda_pseudo"] = 0.0

        res_no_da = run_experiment("Backbone-only_no_DA", cfg_no_da, loader, eval_loader, hmdb_map, ucf_map, kin_map, device, args.mock)
        res_da = run_experiment("Domain_Adaptation_Active", cfg, loader, eval_loader, hmdb_map, ucf_map, kin_map, device, args.mock)

        print("\n" + "="*40 + " CONFRONTO STRUTTURALE TARGET " + "="*40)
        print(f"Miglior Accuratezza head_tgt | Baseline no-DA: {res_no_da['best_acc_head_tgt']:.2f}% | MS-DANN con DA: {res_da['best_acc_head_tgt']:.2f}%")
        print(f"Entropia di S1 su Target    | Baseline no-DA: {res_no_da['entropy_s1_on_tgt']:.4f} | MS-DANN con DA: {res_da['entropy_s1_on_tgt']:.4f}")
        print("="*108)
    else:
        res = run_experiment("Framework_Training_Run", cfg, loader, eval_loader, hmdb_map, ucf_map, kin_map, device, args.mock)
        print("\n=== Analisi Metrica di Allineamento ===")
        print(f"Entropia media head_s1 -> target: {res['entropy_s1_on_tgt']:.4f}")
        print("(Nota: Un valore di entropia più elevato indica una maggiore invarianza computazionale dei domini)")


if __name__ == "__main__":
    main()