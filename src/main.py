# main.py (dentro src/)

# dalla root del progetto
# apptainer run --nv /shared/sifs/latest.sif python src/main.py --config experiments/configs/base_config.yaml
from datasets.datasets import build_dataloaders
from models.model      import MultiSourceDANN
from training.losses   import MultiSourceLoss
from training.trainer  import Trainer

import argparse
import copy
import torch
import torch.optim as optim
import yaml
import numpy as np
import random


def load_config(base: str, override: str = None) -> dict:
    with open(base) as f:
        cfg = yaml.safe_load(f)
    if override:
        with open(override) as f:
            ov = yaml.safe_load(f)
        for section, values in ov.items():
            if isinstance(values, dict):
                cfg.setdefault(section, {}).update(values)
            else:
                cfg[section] = values
    return cfg


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

###
def evaluate_target(model, eval_loader, device):
    """Valutazione target: head_tgt vs ensemble semantico."""
    model.eval()
    correct_head = correct_ens = total = 0
    with torch.no_grad():
        for frames, labels, _ in eval_loader:
            frames = frames.to(device)
            labels = labels.to(device)
            cls_logits, _, _, ensemble_probs = model(frames, domain=2)
            correct_head += (cls_logits.argmax(-1) == labels).sum().item()
            correct_ens += (ensemble_probs.argmax(-1) == labels).sum().item()
            total += labels.size(0)
    if total == 0:
        return 0.0, 0.0
    return 100.0 * correct_head / total, 100.0 * correct_ens / total


def evaluate_source_only_entropy(model, eval_loader, device):
    """Baseline: head_s1 applicata al target (senza DA)."""
    model.eval()
    entropy_list = []
    with torch.no_grad():
        for frames, _, _ in eval_loader:
            frames = frames.to(device)
            cls_on_tgt, _, _, _ = model(frames, domain=0)
            entropy = -(cls_on_tgt.softmax(-1) * cls_on_tgt.log_softmax(-1)).sum(-1).mean().item()
            entropy_list.append(entropy)
    return sum(entropy_list) / len(entropy_list) if entropy_list else 0.0


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
    ).to(device)

    loss_fn = MultiSourceLoss(lambda_adv=train_cfg.get("lambda_adv", 0.1))
    optimizer = optim.Adam(
        model.parameters(),
        lr=train_cfg.get("learning_rate", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )

    checkpoint_path = cfg["paths"].get("checkpoint", "experiments/checkpoints")
    checkpoint_path = os.path.join(checkpoint_path, run_name.replace(" ", "_").lower())

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        max_epochs=train_cfg.get("max_epochs", 30),
        checkpoint_dir=checkpoint_path,
        incomplete_simulation=ablation_cfg.get("incomplete_simulation", True),
        source2_enabled=ablation_cfg.get("source2_enabled", True),
        patience=train_cfg.get("patience", 7),
        lambda_pseudo=train_cfg.get("lambda_pseudo", 0.1),
        disable_early_stopping_if_mock=is_mock,
    )
    trainer.fit(train_loader=loader, eval_loader=eval_loader, auto_resume=auto_resume)

    acc_head, acc_ens = evaluate_target(model, eval_loader, device)
    src_entropy = evaluate_source_only_entropy(model, eval_loader, device)
    return {
        "acc_head_tgt": acc_head,
        "acc_ens_tgt": acc_ens,
        "entropy_s1_on_tgt": src_entropy,
        "best_acc_head_tgt": trainer.best_tgt_acc,
    }
###

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",          default="experiments/configs/base_config.yaml")
    parser.add_argument("--config-override", default=None)   # es. model_v1.yaml
    parser.add_argument("--mock",            action="store_true", help="Usa dati fittizi/simulati per testare il pipeline offline")
    parser.add_argument("--compare-da",      action="store_true", help="Confronta backbone-only (no DA) vs DA sul target")
    args = parser.parse_args()

    cfg = load_config(args.config, args.config_override)

    # --- Riproducibilità -----------------------------------------------------
    seed = cfg["hardware"]["seed"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # --- Device --------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Dati (Persona 1 / Mock Mode) ----------------------------------------
    if args.mock:
        print("\n[MOCK MODE] Inizializzazione dati simulati per dry-run...")
        hmdb_map = {f"class_{i}": i for i in range(51)}
        ucf_map = {f"class_{i}": i for i in range(5)}
        kin_map = {f"class_{i}": i for i in range(400)}
        loader = MockMultiSourceDataLoader(batch_size=cfg["data"]["batch_size"], num_steps=5)
        eval_loader = loader.get_target_eval_loader(batch_size=16)
        # Forza meno epoche per il dry-run veloce
        cfg["training"]["max_epochs"] = min(cfg["training"]["max_epochs"], 2)
    else:
        loader, hmdb_map, ucf_map, kin_map = build_dataloaders(
            cfg["paths"]["data_root"],
            batch_size=cfg["data"]["batch_size"],
        )
        eval_loader = loader.get_target_eval_loader(batch_size=16)

    print(f"Classi — S1: {len(hmdb_map)} | S2: {len(ucf_map)} | Tgt: {len(kin_map)}")

    def train_and_report(current_cfg: dict, tag: str):
        # run singolo (stessa pipeline di main.py)
        print(f"\n=== {tag} ===")

        model_cfg = current_cfg.get("model", {})
        train_cfg = current_cfg.get("training", {})
        model = MultiSourceDANN(
            num_classes_s1  = len(hmdb_map),
            num_classes_s2  = len(ucf_map),
            num_classes_tgt = len(kin_map),
            pretrained      = model_cfg.get("pretrained", False),
            backbone_type   = model_cfg.get("encoder", "r2plus1d_18"),
            temperature     = model_cfg.get("temperature", 0.1),
            ema_momentum    = model_cfg.get("ema_momentum", 0.9),
        ).to(device)
        print(f"Modello su {device} — parametri: {sum(p.numel() for p in model.parameters()):,}")

        loss_fn   = MultiSourceLoss(lambda_adv=train_cfg.get("lambda_adv", 0.1))
        optimizer = optim.Adam(
            model.parameters(),
            lr           = train_cfg.get("learning_rate", 1e-4),
            weight_decay = train_cfg.get("weight_decay", 1e-4),
        )

        ablation_cfg = current_cfg.get("ablation", {})
        checkpoint_base = current_cfg["paths"].get("checkpoint", "experiments/checkpoints")
        checkpoint_path = os.path.join(checkpoint_base, tag.replace(" ", "_").replace("(", "").replace(")", "").lower())

        trainer = Trainer(
            model                 = model,
            loss_fn               = loss_fn,
            optimizer             = optimizer,
            device                = device,
            max_epochs            = train_cfg.get("max_epochs", 30),
            checkpoint_dir        = checkpoint_path,
            incomplete_simulation = ablation_cfg.get("incomplete_simulation", True),
            source2_enabled       = ablation_cfg.get("source2_enabled", True),
            patience              = train_cfg.get("patience", 7),
            lambda_pseudo         = train_cfg.get("lambda_pseudo", 0.1),
            disable_early_stopping_if_mock = args.mock,
        )
        trainer.fit(train_loader=loader, eval_loader=eval_loader)

        # baseline: head_s1 sul target (entropy)
        model.eval()
        entropy_list = []
        with torch.no_grad():
            for frames, _, _ in eval_loader:
                frames = frames.to(device)
                cls_on_tgt, _, _, _ = model(frames, domain=0)
                entropy = -(cls_on_tgt.softmax(-1) * cls_on_tgt.log_softmax(-1)).sum(-1).mean().item()
                entropy_list.append(entropy)
        avg_entropy = sum(entropy_list) / len(entropy_list) if entropy_list else 0.0

        return trainer.best_tgt_acc, avg_entropy

    if args.compare_da:
        cfg_no_da = copy.deepcopy(cfg)
        cfg_no_da.setdefault("training", {})
        cfg_no_da["training"]["lambda_adv"] = 0.0
        cfg_no_da["training"]["lambda_pseudo"] = 0.0

        best_no_da, ent_no_da = train_and_report(cfg_no_da, "Backbone-only (no DA)")
        best_da, ent_da = train_and_report(cfg, "Domain Adaptation")

        print("\n=== Confronto Target ===")
        print(f"Best acc head_tgt | no-DA: {best_no_da:.2f}% | DA: {best_da:.2f}%")
        print(f"Entropy S1→target | no-DA: {ent_no_da:.4f} | DA: {ent_da:.4f}")
    else:
        _, avg_entropy = train_and_report(cfg, "Training")
        print("\n=== Baseline source-only (senza DA) ===")
        print(f"Entropia media head_s1 → target: {avg_entropy:.4f}")
        print(f"(valore alto = encoder non adattato; con DA dovrebbe scendere)")


if __name__ == "__main__":
    main()






# python src/main.py --config experiments/configs/base_config.yaml


# scp -r .\src\* mcclss01m52d960x@gcluster.dmi.unict.it:~/DomainAdaptation-Track9-DataLost/src/

# ssh codice
# srun --account=dl-course-q2 --partition=dl-course-q2 --qos=gpu-medium --gres=gpu:1 --gres=shard:5632 --pty bash
# cd ~/DomainAdaptation-Track9-DataLost
# apptainer shell --nv /shared/sifs/latest.sif
# srun --account <coda> --partition <coda> --qos=gpu-small --gres=gpu:1 --pty bash
# python main.py --config experiments/configs/base_config.yaml

#    python main.py \
#    --config          experiments/configs/base_config.yaml \
#    --config-override experiments/configs/model_v1.yaml