# main.py (dentro src/)

# dalla root del progetto
# apptainer run --nv /shared/sifs/latest.sif python src/main.py --config experiments/configs/base_config.yaml
from datasets.datasets import build_dataloaders
from models.model      import MultiSourceDANN
from training.losses   import MultiSourceLoss
from training.trainer  import Trainer


import argparse
import torch
import torch.optim as optim
import yaml
import os
import sys


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",          default="experiments/configs/base_config.yaml")
    parser.add_argument("--config-override", default=None)   # es. model_v1.yaml
    parser.add_argument("--mock",            action="store_true", help="Usa dati fittizi/simulati per testare il pipeline offline")
    args = parser.parse_args()

    cfg = load_config(args.config, args.config_override)

    # --- Riproducibilità -----------------------------------------------------
    torch.manual_seed(cfg["hardware"]["seed"])

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

    # --- Modello -------------------------------------------------------------
    model = MultiSourceDANN(
        num_classes_s1  = len(hmdb_map),
        num_classes_s2  = len(ucf_map),
        num_classes_tgt = len(kin_map),
        pretrained      = cfg["model"].get("pretrained", False),
        backbone_type   = cfg["model"].get("encoder", "r2plus1d_18")
    )
    model = model.to(device)
    print(f"Modello su {device} — parametri: {sum(p.numel() for p in model.parameters()):,}")

    # --- Loss e Ottimizzatore ------------------------------------------------
    loss_fn   = MultiSourceLoss(lambda_adv=cfg["training"]["lambda_adv"])
    optimizer = optim.Adam(
        model.parameters(),
        lr           = cfg["training"]["learning_rate"],
        weight_decay = cfg["training"]["weight_decay"],
    )

    # --- Training ------------------------------------------------------------
    ablation_cfg = cfg.get("ablation", {})
    incomplete_sim = ablation_cfg.get("incomplete_simulation", True)
    source2_ok = ablation_cfg.get("source2_enabled", True)
    checkpoint_path = cfg["paths"].get("checkpoint", "experiments/checkpoints")

    trainer = Trainer(
        model                 = model,
        loss_fn               = loss_fn,
        optimizer             = optimizer,
        device                = device,
        max_epochs            = cfg["training"]["max_epochs"],
        checkpoint_dir        = checkpoint_path,
        incomplete_simulation = incomplete_sim,
        source2_enabled       = source2_ok,
    )
    trainer.fit(train_loader=loader, eval_loader=eval_loader)

    # ── Baseline source-only (dopo il training) ───────────────────────────────
    print("\n=== Baseline source-only (senza DA) ===")
    model.eval()
    entropy_list = []
    with torch.no_grad():
        for frames, _, _ in eval_loader:
            frames = frames.to(device)
            cls_on_tgt, _, _, _ = model(frames, domain=0)  # head_s1 sul target
            entropy = -(cls_on_tgt.softmax(-1) * cls_on_tgt.log_softmax(-1)).sum(-1).mean().item()
            entropy_list.append(entropy)

    avg_entropy = sum(entropy_list) / len(entropy_list)
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