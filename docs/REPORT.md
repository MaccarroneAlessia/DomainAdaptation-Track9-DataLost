# Multi-source Domain Adaptation per il Riconoscimento di Azioni

- **Group ID**: DataLost
- **Project ID**: Track 9

---

## 1. Introduction and Objective

Il progetto affronta la **Multi-source Domain Adaptation (MSDA)** per il riconoscimento di azioni in video. Disponiamo di due dataset sorgente etichettati, **HMDB-51** e **UCF-101**, e di un dataset target non etichettato, un sottoinsieme di **Kinetics-400**. L'obiettivo e addestrare un modello che combini la conoscenza delle due sorgenti per classificare azioni nel target, gestendo dinamicamente quanto "fidarsi" di ciascuna sorgente.

Il problema e rilevante perche annotare video e costoso: sfruttare piu sorgenti etichettate gia esistenti per operare su un dominio nuovo e non annotato e un'esigenza pratica diffusa. La difficolta del setting *multi-source* rispetto al caso a sorgente singola e che le sorgenti non sono equivalenti: una puo essere piu informativa dell'altra rispetto al target, e il modello deve riconoscerlo.

Ipotesi di partenza: (i) l'allineamento adversariale delle distribuzioni riduce il domain shift e migliora l'accuratezza sul target; (ii) una pesatura dinamica delle sorgenti sfrutta meglio le due sorgenti rispetto a un loro uso indifferenziato. La verifica di queste ipotesi ha richiesto un passaggio metodologico cruciale: con un backbone pre-addestrato sullo stesso dataset del target, le ipotesi sembravano smentite a causa di un *information leakage*; eliminando il leakage con un backbone ImageNet inflated (Sezione 3), la domain adaptation si e rivelata effettivamente utile (Sezione 5). L'identificazione e la rimozione del leakage costituiscono uno dei contributi principali del lavoro.

## 2. Contribution and Added Value

Abbiamo costruito un sistema MSDA composto da un encoder condiviso, classificatori specifici per sorgente, un discriminatore di dominio addestrato in modo adversariale tramite *Gradient Reversal Layer* (GRL), e un meccanismo di **ensemble pesato dinamicamente** che combina le predizioni delle sorgenti in base alla loro affidabilita sul target.

Valore aggiunto rispetto all'esecuzione di codice esistente:

- **Meccanismo di pesatura dinamica per confidenza.** Combiniamo i classificatori delle sorgenti con pesi calcolati per ogni batch target in base alla confidenza (entropia) delle loro predizioni. Questo produce anche il **rapporto di influenza** tra le sorgenti (obiettivo 4).
- **Setting a tre dataset** con spazio di etichette condiviso costruito ad hoc, che richiede una mappatura non banale tra tassonomie eterogenee.
- **Simulazione di sorgenti mancanti (source drop).**
- **Studio dell'effetto del peso adversariale** e **studio di scalabilita** sul numero di sorgenti.
- **Diagnosi e rimozione del leakage del backbone**: abbiamo identificato che il backbone pre-addestrato su Kinetics introduceva un leakage verso il target, e lo abbiamo eliminato adottando un backbone ImageNet inflato a 3D (I3D), ottenendo un setting di valutazione onesto in cui la domain adaptation si dimostra efficace.
- **Analisi critica e diagnosi di fallimenti**: abbiamo individuato e corretto un difetto del meccanismo di pesatura iniziale (collasso a pesi uniformi) e l'effetto dell'overconfidence indotta dallo sbilanciamento di classe.

Per ragioni computazionali e metodologiche operiamo su **feature pre-estratte** da un backbone congelato, concentrando il lavoro sull'allineamento delle distribuzioni piuttosto che sull'apprendimento di feature visive.

## 3. Data Used

**Provenienza.**
- **HMDB-51** (sorgente): clip organizzate in cartelle per classe. Nel dataset fornito ogni clip e una cartella di frame JPG gia estratti.
- **UCF-101** (sorgente): 101 classi, clip in formato video `.avi`, organizzate in cartelle per classe.
- **Kinetics-400** (target, sottoinsieme `kinetics400_5per`): clip video `.mp4` in cartelle per classe, usate senza etichette in addestramento (le etichette servono solo per la valutazione).

**Spazio di etichette condiviso.** I tre dataset hanno tassonomie diverse; la DA closed-set richiede uno spazio comune. Abbiamo costruito l'**intersezione** delle tre tassonomie, ispirandoci al benchmark UCF-HMDB (Chen et al., 2019). Risultano **11 classi** condivise:

`climb, golf, kick_ball, pullup, punch, pushup, ride_bike, ride_horse, shoot_ball, shoot_bow, walk`.

La classe `fencing` (presente nel benchmark UCF-HMDB) e stata **esclusa** perche assente dalla tassonomia di Kinetics-400 (l'unico nome vicino, `sword fighting`, denota un'azione diversa). Abbiamo preferito 11 classi pulite a una mappatura semantica forzata. Per ogni classe canonica abbiamo definito la corrispondenza con il nome grezzo di ciascun dataset (es. `shoot_ball` -> `Basketball` in UCF, `shooting basketball` in Kinetics). Alcuni accoppiamenti non banali e da giustificare: `climb` -> UCF `RopeClimbing` / Kinetics `climbing a rope`; `kick_ball` -> `SoccerPenalty` / `kicking soccer ball`; `walk` -> `WalkingWithDog` / `walking the dog` (accoppiamento piu debole). La mappatura e verificata automaticamente contro i dati prima dell'estrazione.

**Statistiche** (clip con feature estratte, dopo il filtraggio sulle 11 classi):

| Dataset | Clip totali | Note sul bilanciamento |
| :--- | :---: | :--- |
| HMDB-51 | 1684 | bilanciato (~100-130/classe) **tranne `walk` con 548 clip** |
| UCF-101 | 1457 | ben bilanciato (100-164/classe) |
| Kinetics (target) | 303 | piccolo e sbilanciato (12-45 clip/classe) |

Lo sbilanciamento di `walk` in HMDB e la ridotta dimensione del target Kinetics sono fattori rilevanti per l'interpretazione dei risultati (Sezione 5).

**Preprocessing e scelta del backbone.** Non addestriamo su video grezzi: usiamo un backbone 3D congelato come estrattore di feature, estratte una sola volta e salvate su disco (tutto l'addestramento opera su questi vettori, eliminando la decodifica video dal ciclo di training). Per i dataset video (UCF, Kinetics) la lettura avviene in streaming con PyAV, leggendo solo i 16 frame necessari per evitare di caricare l'intero video in memoria.

La scelta del backbone e stata oggetto di una revisione importante. Inizialmente avevamo usato **R3D-18 pre-addestrato su Kinetics-400**. Poiche il nostro target e un sottoinsieme di Kinetics, questo introduceva un **information leakage**: le feature erano gia allineate al target, gonfiando le prestazioni zero-shot (baseline a 0.756) e lasciando un domain shift minimo da correggere. Su indicazione del relatore abbiamo adottato un backbone privo di questo problema: una **ResNet-50 pre-addestrata su ImageNet** (immagini, mai Kinetics), "gonfiata" a 3D con la tecnica I3D (Carreira & Zisserman, 2017; inflation dei kernel 2D->3D con inizializzazione temporale centrata e trasferimento dei pesi ImageNet). Il backbone inflated produce feature da 2048 dimensioni e, non avendo mai visto Kinetics, fornisce un setting onesto in cui il domain shift e reale. Per ogni clip campioniamo 16 frame, ridimensioniamo, center-crop 224x224 e normalizziamo con le costanti ImageNet. **Tutti i risultati riportati nella Sezione 5 usano questo backbone**; il confronto con il vecchio backbone Kinetics e discusso per evidenziare l'effetto del leakage.

## 4. Methodology and Architecture

### 4.1 Panoramica

Quattro componenti che condividono lo stesso spazio di embedding:

1. **encoder condiviso** `E` (comune a tutti i domini);
2. **un classificatore per sorgente** (HMDB e UCF), sullo spazio di etichette condiviso;
3. **discriminatore di dominio** addestrato adversarialmente tramite GRL;
4. **ensemble pesato** che, in inferenza sul target, combina i classificatori con pesi dinamici per confidenza.

```
                          +-> Classificatore_HMDB -> logit
feature --> Encoder E ----+-> Classificatore_UCF  -> logit
            (condiviso)   +-> GRL -> Discriminatore di dominio
                               |
                               v
                    Ensemble pesato per confidenza
                               |
                               v
                    Predizione finale sul Target
```

### 4.2 Encoder condiviso

L'encoder mappa il vettore di feature (512-dim) in uno spazio di embedding ridotto (256-dim). E una rete completamente connessa con due livelli lineari, batch normalization, ReLU e dropout. Essendo condiviso, e il luogo in cui avviene l'allineamento: deve produrre rappresentazioni discriminative per la classificazione e invarianti rispetto al dominio.

### 4.3 Classificatori per sorgente

A valle dell'encoder, ogni sorgente ha la propria testa lineare verso le 11 classi, addestrata con cross-entropy sulle etichette della rispettiva sorgente. Teste separate (invece di una condivisa) permettono di catturare le specificita di ciascuna sorgente e di pesarle in inferenza.

### 4.4 Allineamento adversariale: Gradient Reversal Layer

L'allineamento segue Ganin & Lempitsky (2015). Un **discriminatore di dominio** cerca di predire da quale dominio proviene un embedding; l'encoder cerca di ingannarlo. Se il discriminatore non riesce piu a distinguere i domini, l'encoder ha reso le distribuzioni sovrapponibili (domain shift ridotto). Il **GRL** implementa il gioco: nel forward e l'identita, nel backward moltiplica il gradiente per -lambda. Cosi il discriminatore impara a riconoscere il dominio, ma l'encoder riceve il gradiente invertito e impara a confonderlo, in un'unica fase di ottimizzazione.

Il coefficiente lambda segue la schedulazione DANN, lambda = 2/(1 + e^(-gamma*p)) - 1, dove p e la frazione di addestramento: parte da 0 e cresce verso 1, lasciando che l'encoder impari prima a classificare e introducendo gradualmente la pressione di allineamento. Abbiamo reso configurabile un peso moltiplicativo `adversarial_weight` per studiare l'effetto dell'intensita dell'allineamento (Sezione 5).

### 4.5 Ensemble pesato per confidenza

In inferenza sul target combiniamo le predizioni dei due classificatori con pesi dinamici. **Scelta di design e sua revisione:** inizialmente i pesi erano basati sulla similarita coseno tra il centroide del batch target e i centroidi delle sorgenti nello spazio di embedding. Abbiamo verificato sperimentalmente che questo approccio **collassava a pesi uniformi (50/50)**: dopo il ReLU finale dell'encoder tutti gli embedding giacciono nello stesso "cono" positivo, rendendo i centroidi delle sorgenti quasi identici (cosine similarity ~0.999) e privi di potere discriminante.

Abbiamo quindi sostituito il criterio con una **pesatura per confidenza**: per ogni batch target, ciascun classificatore di sorgente produce predizioni di cui calcoliamo l'entropia media; una sorgente con predizioni piu confidenti (entropia bassa) riceve peso maggiore, tramite softmax con temperatura. La predizione finale e la combinazione pesata dei logit. Questo meccanismo produce anche il **rapporto di influenza Source 1 vs Source 2** (obiettivo 4). La motivazione: una sorgente che produce predizioni nette sul target e ritenuta piu affidabile per quel batch.

### 4.6 Funzione di perdita e logica di addestramento

La perdita combina la cross-entropy di classificazione (somma sulle sorgenti attive) e la cross-entropy del discriminatore di dominio (sorgenti + target), modulata da lambda tramite GRL. A ogni passo: si estrae un batch da ciascuna sorgente e uno dal target; si calcolano le perdite; si esegue un passo di Adam sulla perdita totale.

### 4.7 Baseline

Per quantificare il domain shift, addestriamo un encoder con un **unico** classificatore sull'unione delle sorgenti, **senza** allineamento, valutato zero-shot sul target. Il divario sorgenti/target misura il domain shift ed e il termine di paragone.

### 4.8 Obiettivi aggiuntivi

- **Source drop**: con probabilita configurabile, una sorgente viene scartata a ogni passo (mai entrambe), simulando batch incompleti.
- **Studio del peso adversariale**: confronto a adversarial_weight = 0.0, 0.3, 1.0.
- **Studio di scalabilita**: confronto tra solo HMDB, solo UCF, ed entrambe.
- **Bilanciamento di classe**: sottocampionamento della classe sovrarappresentata `walk` (cap_per_class) e valutazione tramite macro-accuracy (media delle accuratezze per-classe), per misurare e correggere il bias indotto dallo sbilanciamento.

## 5. Results and Discussion

Tutti i risultati usano il backbone ImageNet inflated (Sezione 3), privo di leakage verso il target.

**Table 1**: Risultati quantitativi sul dominio target (Kinetics)

| Model | Target Acc | Macro-Acc | Influenza HMDB/UCF |
| :--- | :---: | :---: | :---: |
| Baseline (no DA) | 0.667 | - | - |
| MSDA adv=0.0 | 0.683 | 0.679 | 0.54 / 0.46 |
| MSDA adv=0.3 | 0.680 | 0.685 | 0.63 / 0.37 |
| MSDA adv=1.0 | 0.713 | 0.718 | 0.64 / 0.36 |
| **MSDA adv=1.0 + `walk` bilanciata** | **0.759** | **0.748** | 0.50 / 0.50 |

**Table 2**: Confronto tra backbone (effetto del leakage), accuratezza sul target

| Configurazione | Backbone Kinetics (con leakage) | Backbone ImageNet (senza leakage) |
| :--- | :---: | :---: |
| Baseline (no DA) | 0.756 | 0.667 |
| MSDA adv=1.0 | 0.700 | 0.713 |

**Table 3**: Studio di scalabilita (backbone ImageNet) - accuratezza sul target

| Sorgenti | N. sorgenti | Target Accuracy |
| :--- | :---: | :---: |
| solo HMDB-51 | 1 | 0.419 |
| solo UCF-101 | 1 | 0.690 |
| **HMDB-51 + UCF-101** | 2 | **0.713** |

**Discussione.**

*Il leakage del backbone era reale.* Passando da R3D-18/Kinetics a ResNet-50/ImageNet inflated, la baseline zero-shot scende da 0.756 a 0.667 (Table 2). Questo conferma che parte dell'accuratezza precedente derivava da informazione trapelata dal pre-training su Kinetics, di cui il target e un sottoinsieme. Con il backbone ImageNet il domain shift sorgenti->target e reale e la valutazione e onesta.

*La domain adaptation funziona nel setting onesto.* Questo e il risultato centrale. Con il backbone ImageNet, la MSDA con allineamento adversariale pieno (adv=1.0) raggiunge 0.713, **superando la baseline** (0.667) di circa 5 punti. La direzione e opposta a quella osservata con il backbone Kinetics, dove l'allineamento peggiorava le prestazioni (0.700 < 0.756): la spiegazione e che con feature gia allineate (leakage) l'allineamento distorceva rappresentazioni buone, mentre con feature ImageNet esiste un vero shift che l'allineamento corregge. L'eliminazione del leakage e quindi la condizione che rende efficace la domain adaptation.

*Relazione con l'intensita dell'allineamento.* La progressione adv=0.0->0.3->1.0 da 0.683->0.680->0.713: l'allineamento pieno e il migliore e supera sia la baseline sia l'assenza di allineamento, ma la relazione non e strettamente monotona (adv=0.3 e leggermente sotto adv=0.0). Attribuiamo la non-monotonia al rumore dovuto alla ridotta dimensione del target (303 clip), che rende differenze di pochi punti poco significative.

*Il bilanciamento di `walk` e il miglior intervento.* Sottocampionando la classe `walk` sovrarappresentata in HMDB (548 -> 150 clip) e con adv=1.0, l'accuratezza sale a **0.759** (macro-acc 0.748), il valore migliore in assoluto, +9 punti sulla baseline. Inoltre l'influenza delle sorgenti torna bilanciata (0.50/0.50) da 0.64/0.36: lo sbilanciamento di `walk` rendeva HMDB artificialmente piu confidente (overconfidence), e il meccanismo di pesatura per confidenza la premiava indebitamente; bilanciando, la pesatura si riequilibra. Questo fenomeno si osserva in modo consistente con entrambi i backbone, a conferma della sua robustezza.

*Scalabilita: il multi-source ora aiuta.* Lo studio di scalabilita (rifatto con il backbone ImageNet) mostra: solo HMDB 0.419, solo UCF 0.690, HMDB+UCF **0.713**. La combinazione delle due sorgenti **supera la migliore sorgente singola**. Questo e l'opposto di quanto osservato con il backbone Kinetics (dove combinare peggiorava): senza leakage le due sorgenti portano informazione complementare e il multi-source produce un guadagno reale, confermando l'ipotesi alla base del progetto. Il source drop (drop_prob=0.3) non degrada significativamente le prestazioni, indicando robustezza all'assenza intermittente di una sorgente.

*Analisi per-classe.* L'accuratezza varia molto tra classi: azioni visivamente distintive (`ride_bike`, `climb`, `pushup`) sono le piu accurate, mentre le classi con pochi esempi nel target o semanticamente ambigue (`kick_ball`, `shoot_bow`) sono le piu deboli. Il bilanciamento di `walk` riduce gli errori spuri verso quella classe, migliorando l'equita tra le classi (macro-accuracy).

## 6. Conclusion and Limitations

Abbiamo implementato un sistema MSDA completo (encoder condiviso, classificatori per sorgente, allineamento adversariale con GRL, ensemble pesato per confidenza) e ne abbiamo studiato il comportamento sul trasferimento HMDB+UCF -> Kinetics. Il percorso del lavoro si articola in tre fasi che ne costituiscono il contributo principale. (1) Con un backbone R3D-18 pre-addestrato su Kinetics, la domain adaptation sembrava inutile o dannosa (baseline 0.756, MSDA 0.700). (2) Abbiamo diagnosticato la causa in un *information leakage*: il target e un sottoinsieme di Kinetics, quindi il backbone aveva gia visto il dominio target. (3) Su indicazione del relatore abbiamo eliminato il leakage adottando un backbone ResNet-50/ImageNet gonfiato a 3D (I3D inflation). Nel setting onesto risultante, la domain adaptation **funziona**: la MSDA con allineamento (0.713) supera la baseline (0.667), la combinazione delle due sorgenti supera la migliore sorgente singola (0.713 vs 0.690), e il bilanciamento della classe `walk` porta il risultato migliore (0.759, macro-acc 0.748), riequilibrando anche l'influenza delle sorgenti. Il lavoro mostra quindi sia l'importanza metodologica di evitare il leakage nella valutazione della domain adaptation, sia l'efficacia dell'allineamento adversariale, del multi-source e del bilanciamento delle classi in un setting corretto.

Limitazioni:

- **Pesatura per confidenza vulnerabile all'overconfidence.** Pesare per confidenza puo favorire una sorgente sicura ma inaccurata; lo si e visto con lo sbilanciamento di `walk`. Una pesatura calibrata (es. temperature scaling) o basata su un piccolo insieme di validazione target sarebbe piu robusta.
- **Backbone ImageNet non specializzato per video.** Eliminando il leakage usiamo un backbone addestrato su immagini statiche: cattura bene l'aspetto spaziale ma non la dinamica temporale, il che limita l'accuratezza assoluta. Un backbone pre-addestrato su un dataset video diverso da Kinetics (es. Something-Something) unirebbe assenza di leakage e modellazione temporale.
- **Setting closed-set** su 11 classi: le classi presenti solo in alcuni domini sono escluse.
- **Feature congelate**: nessun fine-tuning end-to-end.
- **Target ridotto** (303 clip), che rende le metriche rumorose e puo spiegare la non-monotonia rispetto al peso adversariale.

Esperimenti futuri: backbone video non-Kinetics; pesatura calibrata o per-classe; fine-tuning parziale del backbone; setting open-set; ribilanciamento sistematico di tutte le classi sorgente.

## 7. Additional Information

### 7.1 Contribution Breakdown

[DA COMPLETARE da parte vostra: chi ha fatto cosa.]

- **[Nome 1]**: ...
- **[Nome 2]**: ...
- **[Nome 3]**: ...

### 7.2 Use of Artificial Intelligence

[DA COMPLETARE in modo veritiero. Traccia:]

Abbiamo utilizzato strumenti di assistenza basati su IA per: stesura dello scaffold del codice e del boilerplate (struttura dei moduli, data loading, script di training e valutazione), supporto al debugging (es. risoluzione di un OOM nella lettura video, correzione del meccanismo di pesatura), adattamento del codice all'ambiente di calcolo (cluster offline con container Apptainer), e supporto alla redazione della documentazione e di parti del report. Le decisioni architetturali, la diagnosi e interpretazione dei risultati e la responsabilita complessiva del lavoro sono nostre.

---

### Riferimenti

- Y. Ganin, V. Lempitsky, *Unsupervised Domain Adaptation by Backpropagation*, ICML 2015.
- M.-H. Chen et al., *Temporal Attentive Alignment for Large-Scale Video Domain Adaptation*, ICCV 2019.
- D. Tran et al., *A Closer Look at Spatiotemporal Convolutions for Action Recognition*, CVPR 2018.
