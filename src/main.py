from datasets.datasets import build_dataloaders
from models.model      import MultiSourceDANN
from training.losses   import MultiSourceLoss
from training.trainer  import Trainer
from evaluation.evaluator import DynamicEvaluationStrategist
from evaluation.metrics   import compute_entropy, MetricsLogger, comparative_table

import argparse
import copy
import os
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


def evaluate_target(model, eval_loader, device, eval_strategist=None, epoch=0, trainer_ref=None):
    """Valutazione target: head_tgt vs ensemble semantico con weighting dinamico."""
    model.eval()
    correct_head = correct_ens = total = 0

    mapped_losses = {"total": 0.0, "cls_tgt": 0.0, "adv": 0.0, "pseudo": 0.0}
    if trainer_ref is not None:
        total_val = getattr(trainer_ref, 'last_loss_total', 0.0)
        cls_val   = getattr(trainer_ref, 'last_loss_cls', 0.0)
        adv_val   = getattr(trainer_ref, 'last_loss_adv', 0.0)
        ps_val    = getattr(trainer_ref, 'last_loss_tgt_ps', 0.0)
        
        if total_val or cls_val or adv_val or ps_val:
            mapped_losses = {
                "total": total_val,
                "cls_tgt": cls_val,
                "adv": adv_val,
                "pseudo": ps_val
            }

    # Spegnimento dei worker paralleli per l'evaluation
    original_workers = eval_loader.num_workers if hasattr(eval_loader, 'num_workers') else 0
    if hasattr(eval_loader, 'num_workers'):
        eval_loader.num_workers = 0
    if hasattr(eval_loader, 'pin_memory'):
        eval_loader.pin_memory = False

    with torch.no_grad():
        for batch_idx, (frames, labels, _) in enumerate(eval_loader):
            frames = frames.to(device)
            labels = labels.to(device)

            cls_logits, domain_logits, features, ensemble_probs = model(frames, domain=2)
            
            cls_logits = cls_logits.detach()
            features = features.detach()
            ensemble_probs = ensemble_probs.detach()

            if eval_strategist is not None:
                c1_ready = model.s1_centroid_initialized.item() if hasattr(model, 's1_centroid_initialized') else False
                c2_ready = model.s2_centroid_initialized.item() if hasattr(model, 's2_centroid_initialized') else False
                if c1_ready and c2_ready:
                    c1 = model.s1_centroid.detach()
                    c2 = model.s2_centroid.detach()
                else:
                    c1 = features.mean(dim=0).detach()
                    c2 = features.mean(dim=0).detach()

                w_s1, w_s2 = eval_strategist.compute_dynamic_weights(features, c1, c2)
                ent = compute_entropy(cls_logits)

                eval_strategist.log_batch_metrics(
                    epoch, batch_idx, w_s1, w_s2, 
                    loss_dict=mapped_losses, 
                    entropy=ent
                )

            correct_head += (cls_logits.argmax(-1) == labels).sum().item()
            correct_ens  += (ensemble_probs.argmax(-1) == labels).sum().item()
            total        += labels.size(0)

    # Ripristino dello stato originale del loader
    if hasattr(eval_loader, 'num_workers'):
        eval_loader.num_workers = original_workers

    if total == 0:
        return 0.0, 0.0

    acc_head = 100.0 * correct_head / total
    acc_ens  = 100.0 * correct_ens  / total

    if eval_strategist is not None:
        eval_strategist.update_accuracy_evolution(epoch, acc_head, acc_ens)

    return acc_head, acc_ens


def train_and_report(current_cfg: dict, tag: str, loader, eval_loader, hmdb_map, ucf_map, kin_map, device, is_mock=False):
    print(f"\n=== {tag} ===")
    model_cfg    = current_cfg.get("model", {})
    train_cfg    = current_cfg.get("training", {})
    ablation_cfg = current_cfg.get("ablation", {})

    eval_strategist = DynamicEvaluationStrategist(
        temperature = model_cfg.get("temperature", 0.5),
        run_name    = tag,
    )

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

    checkpoint_base = current_cfg["paths"].get("checkpoint", "experiments/checkpoints")
    checkpoint_path = os.path.join(
        checkpoint_base,
        tag.replace(" ", "_").replace("(", "").replace(")", "").lower()
    )

    max_epochs = train_cfg.get("max_epochs", 30)
    
    trainer = Trainer(
        model                          = model,
        loss_fn                        = loss_fn,
        optimizer                      = optimizer,
        device                         = device,
        max_epochs                     = max_epochs,  
        checkpoint_dir                 = checkpoint_path,
        incomplete_simulation          = ablation_cfg.get("incomplete_simulation", True),
        source2_enabled                = ablation_cfg.get("source2_enabled", True),
        patience                       = train_cfg.get("patience", 7),
        lambda_pseudo                  = train_cfg.get("lambda_pseudo", 0.1),
        disable_early_stopping_if_mock = is_mock,
    )
    
    # Esegui il training (fit gestisce già il loop interno sulle epoche)
    trainer.fit(train_loader=loader, eval_loader=eval_loader, auto_resume=True)
    
    # Valutazione finale dopo il training
    acc_head, acc_ens = evaluate_target(
        model, eval_loader, device,
        eval_strategist=eval_strategist,
        epoch=max_epochs,
        trainer_ref=trainer
    )
    print(f"Target Acc Finale (Head): {acc_head:.2f}% | Target Acc (Ensemble): {acc_ens:.2f}%")

    eval_strategist.generate_plots()
    eval_strategist.generate_markdown_report()

    model.eval()
    entropy_list = []
    with torch.no_grad():
        for frames, _, _ in eval_loader:
            frames = frames.to(device)
            cls_on_tgt, _, _, _ = model(frames, domain=0)
            entropy = -(cls_on_tgt.softmax(-1) * cls_on_tgt.log_softmax(-1)).sum(-1).mean().item()
            entropy_list.append(entropy)
    avg_entropy = sum(entropy_list) / len(entropy_list) if entropy_list else 0.0

    return trainer.best_tgt_acc, avg_entropy, tag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",          default="experiments/configs/base_config.yaml")
    parser.add_argument("--config-override", default=None)
    parser.add_argument("--mock",            action="store_true")
    parser.add_argument("--compare-da",      action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config, args.config_override)

    seed = cfg["hardware"]["seed"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device rilevato per la sessione: {device}")

    if args.mock:
        print("\n[MOCK MODE] Inizializzazione dati simulati...")
        hmdb_map = {f"class_{i}": i for i in range(51)}
        ucf_map  = {f"class_{i}": i for i in range(5)}
        kin_map  = {f"class_{i}": i for i in range(400)}
        loader      = MockMultiSourceDataLoader(batch_size=cfg["data"]["batch_size"], num_steps=5)
        eval_loader = loader.get_target_eval_loader(batch_size=16)
        cfg["training"]["max_epochs"] = min(cfg["training"]["max_epochs"], 2)
    else:
        loader, hmdb_map, ucf_map, kin_map = build_dataloaders(
            cfg["paths"]["data_root"],
            batch_size=cfg["data"]["batch_size"],
        )
        eval_loader = loader.get_target_eval_loader(batch_size=16)

    print(f"Classi Caricate — S1 (HMDB): {len(hmdb_map)} | S2 (UCF): {len(ucf_map)} | Tgt (Kinetics): {len(kin_map)}")

    if args.compare_da:
        cfg_no_da = copy.deepcopy(cfg)
        cfg_no_da.setdefault("training", {})
        cfg_no_da["training"]["lambda_adv"]    = 0.0
        cfg_no_da["training"]["lambda_pseudo"] = 0.0

        best_no_da, ent_no_da, tag_no_da = train_and_report(
            cfg_no_da, "Baseline_No_DA", loader, eval_loader, hmdb_map, ucf_map, kin_map, device, args.mock
        )
        
        best_da, ent_da, tag_da = train_and_report(
            cfg, "Weighted_DA", loader, eval_loader, hmdb_map, ucf_map, kin_map, device, args.mock
        )

        print("\n=== Confronto Target Terminato ===")
        print(f"Best acc head_tgt | no-DA: {best_no_da:.2f}% | DA: {best_da:.2f}%")
        print(f"Entropy S1->target | no-DA: {ent_no_da:.4f} | DA: {ent_da:.4f}")

        print("\n[INFO] Rilevamento dei file di log per la generazione del Report di sintesi...")
        try:
            fn_no_da = tag_no_da.replace(" ", "_").replace("(", "").replace(")", "")
            fn_da    = tag_da.replace(" ", "_").replace("(", "").replace(")", "")
            
            logger_no_da = MetricsLogger.load(f"experiments/logs/metrics_{fn_no_da}.json")
            logger_da    = MetricsLogger.load(f"experiments/logs/metrics_{fn_da}.json")
            
            runs_dict = {
                "Baseline (No DA)": logger_no_da,
                "Weighted DA": logger_da
            }
            
            table_md    = comparative_table(runs_dict, latex=False)
            table_latex = comparative_table(runs_dict, latex=True)
            
            os.makedirs("docs", exist_ok=True)
            with open("docs/REPORT.md", "w", encoding="utf-8") as f:
                f.write("# Relazione Finale di Valutazione — Domain Adaptation Track 9\n\n")
                f.write("Questo report mette a confronto le performance del modello base con e senza l'attivazione ")
                f.write("dei meccanismi di allineamento avversariale e pesatura dinamica geometrica dei centroidi.\n\n")
                f.write("## Tabella Comparativa delle Performance\n\n")
                f.write(table_md)
                f.write("\n\n## Codice Tabella in Formato LaTeX (Pronto per Paper / Presentazione)\n\n")
                f.write("```latex\n")
                f.write(table_latex)
                f.write("\n```\n")
                
            print("📈 [MINIMUM OBJECTIVE] File 'docs/REPORT.md' generato con successo!")
            
        except Exception as e:
            print(f"[ATTENZIONE] Errore durante la creazione del report comparativo aggregato: {e}")

    else:
        best_acc, avg_entropy, _ = train_and_report(
            cfg, "Training_Singolo", loader, eval_loader, hmdb_map, ucf_map, kin_map, device, args.mock
        )
        print("\n=== Pipeline Conclusa ===")
        print(f"Migliore accuratezza Target: {best_acc:.2f}%")
        print(f"Entropia media head_s1 -> target: {avg_entropy:.4f}")


if __name__ == "__main__":
    main()
    
# python src/main.py --config experiments/configs/base_config.yaml


# scp -r .\src\* vllmtt02t20b429t@gcluster.dmi.unict.it:~/DomainAdaptation-Track9-DataLost/src/

# ssh codice
# srun --account=dl-course-q2 --partition=dl-course-q2 --qos=gpu-medium --gres=gpu:1 --gres=shard:5632 --pty bash
# cd ~/DomainAdaptation-Track9-DataLost
# apptainer shell --nv /shared/sifs/latest.sif
# srun --account <coda> --partition <coda> --qos=gpu-small --gres=gpu:1 --pty bash
# python main.py --config experiments/configs/base_config.yaml

#    python main.py \
#    --config          experiments/configs/base_config.yaml \
#    --config-override experiments/configs/model_v1.yaml

