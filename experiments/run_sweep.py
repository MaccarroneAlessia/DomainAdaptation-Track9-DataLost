# W&B sweep: wandb sweep experiments/configs/sweep.yaml && wandb agent <sweep_id>
# esempio che l'agente di W&B userà per lanciare gli esperimenti
# inizializza lo state object da wandb.config e lo inietta nell'ottimizzatore e nel loop di training

# wandb sweep experiments/configs/sweep.yaml
# restituisce id
# wandb agent utente/progetto/ID
# wandb sweep run_sweep.py

import sys
import os
import torch
import torch.optim as optim
import wandb

# pacchetto src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from main import MockMultiSourceDataLoader  # stesso mock del dry-run
from models.model import MultiSourceDANN
from training.losses import MultiSourceLoss
from training.trainer import Trainer

def main():
    # Inizializza run W&B (i parametri verranno iniettati dallo sweep agent)
    wandb.init()
    c = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Inizializza Modello con iperparametri statici (o dinamici se aggiunti allo sweep)
    model = MultiSourceDANN(
        num_classes_s1=51,
        num_classes_s2=5,
        num_classes_tgt=400,
        pretrained=getattr(c, "pretrained", False),
        backbone_type=getattr(c, "encoder", "r2plus1d_18"),
        temperature=getattr(c, "temperature", 0.1),
        ema_momentum=getattr(c, "ema_momentum", 0.9),
    ).to(device)

    # 2. Inizializza Loss e Ottimizzatore con iperparametri statici (o dinamici se aggiunti allo sweep)
    loss_fn = MultiSourceLoss(lambda_adv=c.lambda_adv)
    optimizer = optim.Adam(model.parameters(), lr=c.learning_rate)

    loader = MockMultiSourceDataLoader(batch_size=c.batch_size, num_steps=8)
    eval_loader = loader.get_target_eval_loader(batch_size=16)

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        max_epochs=getattr(c, "max_epochs", 3),
        checkpoint_dir=f"experiments/checkpoints/sweep_{wandb.run.id}",
        patience=getattr(c, "patience", 99),
        lambda_pseudo=getattr(c, "lambda_pseudo", 0.1),
        disable_early_stopping_if_mock=True,
    )

    trainer.fit(loader, eval_loader, auto_resume=False)
    wandb.log({"eval/acc_head_tgt": trainer.best_tgt_acc})


if __name__ == "__main__":
    main()
