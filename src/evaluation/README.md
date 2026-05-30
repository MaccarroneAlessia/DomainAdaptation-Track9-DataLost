# Evaluation Sottomodulo — Domain Adaptation Track 9

Questo sottomodulo implementa i compiti del **Weighting & Evaluation Strategist (Persona 3)**. Regola l'aggregazione predittiva adattiva controllando l'equilibrio tra i domini sorgente ed emettendo i grafici di convergenza.

## Funzionalità
1. **Embedding Cosine Weighter**: Intercetta le feature estratte dal modulo `src/models/backbone.py` per calcolare la distanza geometrica tra i batch sorgente e target.
2. **Logging in tempo reale**: Integrazione diretta con `wandb` per tracciare il rapporto relativo `Source_1 / Source_2` ad ogni singolo iteratore.
3. **Generatore di Reportistica**: Esportazione automatica di tabelle comparative standardizzate in formato Markdown compatibile con i documenti di monitoraggio globali della repository (`docs/REPORT.md`).

## Requisiti di Input
Il calcolatore dei pesi dinamici si aspetta tensori di feature latenti appiattiti post-pooling dal backbone con forma `[Batch_Size, Feature_Dimension]`.