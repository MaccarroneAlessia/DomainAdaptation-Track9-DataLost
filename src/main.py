#!/usr/bin/env python3
"""
main.py — Entry point unificato per Multi-Source Domain Adaptation (Track 9)
Versione definitiva: ripulito dai cicli di inferenza video duplicati e dai typos.
"""

import os
import sys
import argparse
import yaml
import torch
import random
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.datasets import build_dataloaders
from models.model import MultiSourceDANN
from training.losses import MultiSourceLoss
from training.trainer import Trainer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str, override_path: str = None) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if override_path and os.path.exists(override_path):
        with open(override_path, 'r') as f:
            override = yaml.safe_load(f)
        
        def deep_merge(base, overrides):
            for key, value in overrides.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_merge(base[key], value)
                else:
                    base[key] = value
        deep_merge(config, override)
        print(f"[CONFIG] Override applicato da {override_path}")
    
    return config


def create_mock_loaders(batch_size: int = 8, num_classes: dict = None):
    from torch.utils.data import TensorDataset, DataLoader
    
    if num_classes is None:
        num_classes = {'source1': 51, 'source2': 5, 'target': 400}
    
    def mock_data(num_samples, num_classes, seq_len=16, channels=3, h=112, w=112):
        x = torch.randn(num_samples, channels, seq_len, h, w)
        y = torch.randint(0, num_classes, (num_samples,))
        d = torch.zeros(num_samples, dtype=torch.long)
        return TensorDataset(x, y, d)
    
    s1_ds = mock_data(200, num_classes['source1'])
    s2_ds = mock_data(200, num_classes['source2'])
    tgt_ds = mock_data(200, num_classes['target'])
    
    loader_s1 = DataLoader(s1_ds, batch_size=batch_size, shuffle=True)
    loader_s2 = DataLoader(s2_ds, batch_size=batch_size, shuffle=True)
    loader_tgt = DataLoader(tgt_ds, batch_size=batch_size, shuffle=False)
    
    class MockMultiLoader:
        def __iter__(self):
            for b1, b2, bt in zip(loader_s1, loader_s2, loader_tgt):
                yield b1, b2, bt
        def __len__(self):
            return min(len(loader_s1), len(loader_s2), len(loader_tgt))
        def get_target_eval_loader(self, batch_size=None, num_workers=None):
            return DataLoader(tgt_ds, batch_size=batch_size or 8, shuffle=False)
    
    return MockMultiLoader(), None, None, None


def main():
    parser = argparse.ArgumentParser(description="Multi-Source Domain Adaptation Training")
    parser.add_argument('--config', type=str, required=True, help="Config file path")
    parser.add_argument('--config-override', type=str, default=None, help="Override config file")
    parser.add_argument('--mock', action='store_true', help="Use mock data for testing")
    parser.add_argument('--experiment', type=str, default="training_singolo", help="Experiment name")
    parser.add_argument('--source2-enabled', action='store_true', default=True)
    parser.add_argument('--fast-eval-size', type=int, default=500)
    parser.add_argument('--full-eval-size', type=int, default=2000)
    parser.add_argument('--skip-tsne', action='store_true', help="Skip t-SNE generation")
    parser.add_argument('--quick', action='store_true', help="Modalità super veloce")
    
    args = parser.parse_args()
    cfg = load_config(args.config, args.config_override)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device rilevato per la sessione: {device}")
    
    set_seed(cfg.get('hardware', {}).get('seed', 42))
    
    checkpoint_dir = os.path.join(cfg['paths']['checkpoint'], args.experiment)
    log_dir = os.path.join(cfg['paths']['output_dir'], args.experiment)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Caricamento dati
    if args.mock:
        print("\n=== Modalità MOCK ===")
        num_classes = {
            'source1': cfg['model']['num_classes']['source1'],
            'source2': cfg['model']['num_classes']['source2'],
            'target': cfg['model']['num_classes']['target']
        }
        train_loader, hmdb_map, ucf_map, kin_map = create_mock_loaders(
            batch_size=cfg['data']['batch_size'],
            num_classes=num_classes
        )
    else:
        print("\n=== Caricamento dati reali ===")
        train_loader, hmdb_map, ucf_map, kin_map = build_dataloaders(
            data_root=cfg['paths']['data_root'],
            batch_size=cfg['data']['batch_size'],
            num_workers=cfg['data']['num_workers']
        )
    
    model = MultiSourceDANN(
        num_classes_s1=len(hmdb_map) if hmdb_map else cfg['model']['num_classes']['source1'],
        num_classes_s2=len(ucf_map) if ucf_map else cfg['model']['num_classes']['source2'],
        num_classes_tgt=len(kin_map) if kin_map else cfg['model']['num_classes']['target'],
        pretrained=cfg['model']['pretrained'],
        backbone_type=cfg['model'].get('encoder', 'r3d_18'),
        temperature=cfg['model']['temperature'],
        ema_momentum=cfg['model']['ema_momentum']
    ).to(device)
    
    print(f"\n=== {args.experiment} ===")
    print(f"Modello: {sum(p.numel() for p in model.parameters()):,} parametri")
    
    loss_fn = MultiSourceLoss(lambda_adv=cfg['training'].get('lambda_adv', 0.1))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['training']['learning_rate'], weight_decay=cfg['training']['weight_decay'])
    
    # Dataloader di validazione isolato a worker=0 per eliminare i freeze su Linux
    eval_loader = train_loader.get_target_eval_loader(batch_size=cfg['data']['batch_size'], num_workers=0)
    
    trainer = Trainer(
        model=model, loss_fn=loss_fn, optimizer=optimizer, device=device,
        max_epochs=cfg['training']['max_epochs'], checkpoint_dir=checkpoint_dir,
        incomplete_simulation=True, source2_enabled=args.source2_enabled,
        patience=cfg['training'].get('patience', 7), # 🟩 CORRETTO E SINCRONIZZATO DEFINITIVO
        lambda_pseudo=cfg['training']['lambda_pseudo'],
        warmup_epochs=cfg['training'].get('warmup_epochs', 5),
        lambda_em=cfg['training'].get('lambda_em', 0.1),
        disable_early_stopping_if_mock=args.mock,
        fast_eval_size=args.fast_eval_size, full_eval_size=args.full_eval_size
    )
    
    # Training nativo unificato
    trainer.fit(train_loader, eval_loader, auto_resume=True)

    #print("\n" + "=" * 60)
    #print("🧪 AVVIO GENERAZIONE REPORT STRUTTURATO (ULTRA-FAST)")
    #print("=" * 60)
    
    torch.cuda.empty_cache()
    
    # Plotta l'evoluzione ad area dei pesi e le curve dello strategist di P3
    trainer.strategist.generate_plots(output_dir=log_dir)
    
    try:
        # Invocazione automatica del nuovo script ad hoc post_training_analysis.py
        from evaluation.post_training_analysis import main as run_p3_analysis
        run_p3_analysis()
    except Exception as e:
        print(f"⚠️ Nota: Generazione fallback locale dei log strutturati. ({e})")
        trainer.strategist.generate_markdown_report(output_dir=log_dir)


if __name__ == "__main__":
    main()