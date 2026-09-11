# Mappa del repo — `overrefusal-in-posttraining`

Documento di riferimento: cosa fa ogni file, in che ordine si usano, dove
leggono/scrivono i dati. Aggiornato ispezionando il codice reale (non solo
il README, che in alcuni punti è disallineato — vedi sezione **Bug e
disallineamenti trovati** in fondo).

---

## 1. Pipeline in ordine di esecuzione

```
run_experiment.py            STEP 1 — genera le risposte del modello
        │
        ▼
results/<model>/raw_results.csv     (prompt, response, label, source,
                                      checkpoint, predicted_refusal)
        │
        ├── plot_results.py                 plot keyword-based (no judge)
        │
        ▼
run_judge.py                 STEP 2 — giudice LLM a due assi (GA/PD)
        │
        ▼
results/<model>/raw_results.csv + colonne judge_ga, judge_pd, is_coherent, ...
        │
        ├── data/push_to_hf.py              push su HuggingFace
        └── plot_results.py                 plot judge-based
        │
        ▼
analysis/extract_and_push.py STEP 3 — estrae le attivazioni interne del
                              modello e le carica su HuggingFace (repo
                              *diversa* dai risultati testuali)
        │
        ▼
saracandu/olmo-activations   (OLMo2)
saracandu/olmo3-activations  (OLMo3)
        │
        ├── check_sources.py                 verifica cosa è presente
        ├── analysis/compute_entanglement.py, compute_centroid_cosines.py,
        │   compute_behavioural_probe.py, run_classification.py
        │                                    analisi geometrica (probe,
        │                                    entanglement, cross-transfer)
        └── analysis/plot_2d_refusal_space*.py, plot_pca_umap.py,
            plot_entanglement_curves.py       figure
```

**Nota importante**: `run_representation_analysis.py` (Exp 1-2-3,
entanglement "in linea" con GPU) importa `analysis.representation_analysis`,
che **non esiste in questo repo** — vedi bug #2 in fondo. La pipeline
geometrica realmente usata è quella basata su `extract_and_push.py` +
`analysis/compute_*.py`, non su questo script.

---

## 2. File root

| File | Cosa fa | Legge da | Scrive su |
|---|---|---|---|
| `run_experiment.py` | Genera le risposte del modello per ogni (checkpoint × dataset × system prompt); applica il refusal detector a keyword inline | `dataset_config.py` (via `data/dataset_loader.py`), pesi HF del modello | `results/<model>/raw_results.csv` |
| `run_judge.py` | Giudice LLM a due stadi (coerenza, poi GA/PD) su `raw_results.csv` esistenti | `results/<model>/raw_results.csv`, API judge (`JUDGE_API_KEY`) o modello locale | stesso CSV, con colonne aggiunte (`is_coherent`, `judge_ga`, `judge_pd`, ...) |
| `run_representation_analysis.py` | CLI per Exp 1/2/3 di geometria (entanglement, evoluzione, system-prompt sweep) | **⚠️ rotto** — vedi bug #2 | `results/<model>/geometry/*.npz` (se funzionasse) |
| `check_sources.py` | Conta righe per (checkpoint, group, source) su HF **senza** scaricare le colonne di attivazione — solo `label`/`source`/`checkpoint`, per un check economico | repo attivazioni HF (default `saracandu/olmo-activations` — **va sempre passato `--hf-repo saracandu/olmo3-activations --model-family olmo3` per OLMo3**) | stdout / niente file |
| `filter_missing_sources.py` | Filtra un `raw_results.csv` alle sole righe che mancano ancora nelle attivazioni pushate, per evitare di rigenerare forward-pass GPU già fatti | `results/olmo2/raw_results.csv` | CSV filtrato (path passato con `--out` o simile) |
| `explore_comprehension_decision.py` | Esplorazione pura (nessun fit): distribuzioni, distanze tra centroidi, PCA colorata per gruppo-gold vs comportamento. Gestisce sia il path "flat" che quello "nested" delle attivazioni (vedi bug #4 su duplice struttura) | repo attivazioni HF, entrambe le strutture di path | `results/<model>/geometry/explore/*.csv`, `*.png` |
| `test_loaders.py` | Verifica che tutti i loader dei dataset in `dataset_config.py` funzionino (n righe, distribuzione label) | HF Hub (dataset pubblici) | stdout |
| `test_olmo_loader.py` | Unit test (mock) per `models/olmo_loader.py` — `_build_prompt`, gestione system prompt | — | — |
| `exps-status.md` | **Log di lavoro/risultati**, non codice — riassume gli esperimenti fatti su OLMo2 (entanglement, boundary margin, probe) con la motivazione della rimozione di `beavertails` dalle analisi | — | — |
| `README.md` | Descrizione generale del progetto | — | — |
| `requirements.txt` | Dipendenze Python | — | — |
| `.gitignore` | Esclude `.env`, `.overenv/`, `__pycache__/`, `slurm_outputs/`, `figures/` dal versionamento | — | — |

### Configurazioni modello

| File | Modello | Note |
|---|---|---|
| `config.py` | OLMo 1 (7B) | Richiede `ai2-olmo>=0.4.0`, venv **`.venv`** (diverso dagli altri) |
| `config_olmo2.py` | OLMo 2 (7B) | venv `.venv-olmo23` |
| `config_olmo3.py` | OLMo 3 Instruct | venv `.venv-olmo23`. Nota: nome modello `Olmo-3-...` (case misto, intenzionale) |
| `config_olmo3_think.py` | OLMo 3 Think (con blocco `<think>...</think>`) | ⚠️ **rotto**, vedi bug #1 |
| `dataset_config.py` | Registro unico di tutti i dataset benchmark (`ALL_DATASETS`), importato da ogni `config*.py` | — |

---

## 3. `analysis/`

| File | Cosa fa | Repo/file di destinazione |
|---|---|---|
| `extract_and_push.py` (v3) | Estrae le attivazioni residual-stream per OLMo2/OLMo3 a layer selezionati (8,16,19,24,26,31), posizioni `last_prompt`/`post_instr_k`/`first_gen`, push incrementale su HF | Scrive su `--hf-repo` passato esplicitamente (obbligatorio, `required=True`) — **niente default**, a differenza degli script di lettura |
| `compute_entanglement.py` | Calcola entanglement `cos(v_ref, v_over)` e boundary margin per layer/checkpoint. Gruppi: `harmful` (label=1), `pseudo_harm` (or_bench+false_reject), `harmless` (resto label=0) | legge `HF_REPO="saracandu/olmo-activations"` di default → scrive `results/olmo2/geometry/ent_*.csv` |
| `compute_centroid_cosines.py` | Coseno tra centroidi delle attivazioni `first_gen` tra coppie di checkpoint (base→sft→dpo→final) | stesso default repo → `results/olmo2/geometry/centroid_cosines.csv` |
| `compute_behavioural_probe.py` | Probe logistica che predice `predicted_refusal` da `first_gen`; poi cross-checkpoint transfer (train su base, test su sft/dpo/final) | → `results/olmo2/geometry/behavioral_probe*.csv` |
| `run_classification.py` | Probe logistica 3 classi (harmful/pseudo_harm/harmless) con cross-val e cross-checkpoint transfer | → `results/olmo2/classifiers/clf3_results*.csv` |
| `plot_2d_refusal_space.py` | Proietta le attivazioni su `v_ref`/`v_over`, grid layer×checkpoint, colorato per gruppo gold | → `figures/` |
| `plot_2d_refusal_space_behavioral.py` | Come sopra ma split anche per `predicted_refusal` | → `figures/` |
| `plot_entanglement_curves.py` | Curve entanglement + boundary margin per (token_position × method) | legge i CSV di `compute_entanglement.py` → `results/olmo2/geometry/plots/` |
| `plot_pca_umap.py` | Griglia PCA/UMAP layer×checkpoint, 3 categorie, opzionale split behavioural | → `figures/pca_umap/` |
| `plot_results.py` | Plot unificati (refusal rate, judge breakdown, FP/FN, heatmap) da `raw_results.csv`, sia keyword-based che judge-based | legge `results/<model>/raw_results.csv` → `figures/` o dir passata |
| `compare_categories.py`, `compare_models.py` | Confronti aggregati fra OLMo2/OLMo3/OLMo3-Think, letti da `results/<model>/raw_results.csv` | → `results/plots_comparison/` |
| `source_filters.py` | Esclude `beavertails` (6.96% errore di etichetta stimato, Zhu et al. 2024) dalle analisi — usato con `--exclude-sources` | — |

---

## 4. `data/`, `evaluation/`, `models/`

| File | Cosa fa |
|---|---|
| `data/dataset_loader.py` | Carica ogni dataset benchmark in un DataFrame unificato (`prompt`, `label`, `category`, `source`), applicando le regole di `dataset_config.py` (`over_refusal`→label 0, `harmful`→label 1, `mixed`→da colonna) |
| `data/push_to_hf.py` | Push dei `raw_results.csv` giudicati verso una repo HF di **risultati testuali** (dataset con split per famiglia di modello) — **diversa** dalla repo delle attivazioni |
| `evaluation/refusal_detector.py` | Rilevamento rifiuto via regex/keyword (`REFUSAL_PATTERNS` da `config.py`), più eventuale fallback LLM-judge |
| `evaluation/metrics.py` | Calcola metriche FP/FN (keyword-based) e GA/PD (judge-based) da un DataFrame di risultati |
| `evaluation/llm_judge.py` | Pipeline giudice a due stadi: coerenza, poi GA (0-3)/PD (0-1) |
| `evaluation/eval_by_checkpoint.py` | Aggrega i giudizi GA/PD per checkpoint × source × label gold |
| `models/olmo_loader.py` | Wrapper per caricare un checkpoint OLMo/HF e generare in batch; un modello alla volta per evitare OOM |

---

## 5. Script SLURM (`*.sh`)

Tutti assumono `cd /u/scandussio/overrefusal-in-posttraining` e un venv attivato (`.overenv` o simile) — **path hardcoded al cluster specifico**, da adattare se si cambia macchina/utente.

| Script | Scopo | Note |
|---|---|---|
| `run_experiment.sh` | Genera tracce, 8 job array (checkpoint × system-prompt) | |
| `run_experiment_missing.sh` | Genera solo le tracce mancanti (OLMo2, `--datasets` deve combaciare esattamente con `dataset_config.py`, altrimenti fallisce silenziosamente) | |
| `run_experiment_olmo3_missing.sh` | Come sopra ma per OLMo3, entrambi i system prompt in un solo job per checkpoint | |
| `run_exp_olmo3.sh` | Genera tracce OLMo3, salta `base__mistral_safety` | |
| `base_run_experiment.sh` | Rerun mirato del solo checkpoint `base__none` | |
| `run_extract_olmo2_missing.sh` | Estrae le attivazioni mancanti per OLMo2 → `saracandu/olmo-activations` | Dipende da un job precedente che genera `wildguard` (`--dependency=afterok`) |
| `extract_acts.sh` | Estrazione attivazioni OLMo2, 8 checkpoint-tag (incl. `mistral_safety`) | |
| `extract_olmo3.sh` | Estrazione attivazioni OLMo3 → **`saracandu/olmo3-activations`** (repo separata apposta, stesso hidden_size/num_layers di OLMo2 → rischio di collisione silenziosa se finissero sulla stessa repo) | |
| `collect_acts.sh` | Job array 0-71: 2 famiglie × 4 checkpoint × 9 dataset | |
| `run_judge_olmo2.sh` | Giudice LLM su OLMo2 (backend API Orfeo) | **⚠️ contiene credenziali in chiaro**, vedi bug #3 |
| `run_judge_olmo3.sh` | Giudice LLM su OLMo3, `--time=48:00:00` | **⚠️ stesso problema credenziali** |
| `run_entanglement.sh` | Lancia `compute_entanglement.py` per le 4 combinazioni (position × method) | |
| `run_classification.sh` | Lancia `run_classification.py --exclude-sources beavertails` | |
| `run_behavioral_probe.sh` | Lancia `compute_behavioural_probe.py --exclude-sources beavertails` | |
| `generate_figures.sh` | Rigenera tutte le figure 2D senza `beavertails` | |

---

## 6. Dove sono davvero i dati (destinazioni)

| Cosa | Dove | Note |
|---|---|---|
| Tracce + giudizio OLMo2 | `results/olmo2/raw_results.csv` (locale, **non in questo zip**) | poi push opzionale su HF via `data/push_to_hf.py` |
| Tracce + giudizio OLMo3 | `results/olmo3/raw_results.csv` (locale) | idem |
| Attivazioni OLMo2 | **`saracandu/olmo-activations`** (HF dataset) | path sia flat (`data/{ckpt}/*.parquet`) sia nested v3 (`data/olmo2/{ckpt}/*.parquet`) — vedi bug #4 |
| Attivazioni OLMo3 | **`saracandu/olmo3-activations`** (HF dataset, repo separata) | **il default di quasi tutti gli script è la repo di OLMo2** — va sempre passato `--hf-repo saracandu/olmo3-activations` esplicitamente |
| Figure | `figures/` (in `.gitignore`, locale) | |
| Log SLURM | `slurm_outputs/` (in `.gitignore`, locale) | |

---

## 7. Bug e disallineamenti trovati (da sistemare)

1. **`config_olmo3_think.py` import rotto**: `from datasets_config import ALL_DATASETS` (plurale) ma il file si chiama `dataset_config.py` (singolare, confermato anche dal docstring del file stesso). OLMo3-Think non parte finché non si corregge in `from dataset_config import ALL_DATASETS`.

2. **`run_representation_analysis.py` importa un modulo inesistente**: `from analysis.representation_analysis import (...)` — quel file non è in questo repo (e nemmeno `analysis/plot_entanglement.py`, citato nel README). Questo script è dead code allo stato attuale: o va rimosso, o vanno recuperati/riscritti i moduli mancanti.

3. **Credenziali hardcoded** in `run_judge_olmo2.sh` e `run_judge_olmo3.sh` (client_id + password per l'API Orfeo, in chiaro nello script). Da spostare in variabile d'ambiente o file protetto (`~/.orfeo_token`), e da rigenerare/revocare visto che sono già stati esposti in questo file.

4. **Doppia struttura di path per le attivazioni**, per motivi storici, non ancora unificata nel codice:
   - flat: `data/{checkpoint}/*.parquet` (5 dataset originali: or_bench, false_reject, harmbench, toxicchat, beavertails) — ha le colonne `post_instr_0/1/2`
   - nested v3: `data/{model_family}/{checkpoint}/*.parquet` (5 dataset nuovi: jailbreakbench, xstest, alpaca, advbench, wildguard) — **non ha** `post_instr_*`, solo `last_prompt`/`first_gen`

   Solo `check_sources.py` ed `explore_comprehension_decision.py` gestiscono esplicitamente entrambi i path (`--model-family`/`--nested-only`); gli altri script di `analysis/` (`compute_entanglement.py`, `compute_centroid_cosines.py`, `compute_behavioural_probe.py`, `run_classification.py`) leggono **solo il path flat** — quindi ignorano silenziosamente i 5 dataset nuovi se non aggiornati.

5. **README disallineato col codice reale**: cita `push_to_hf.py` come file di root (è in `data/`), `analysis/plot_entanglement.py` (non esiste, il file reale è `plot_entanglement_curves.py`), `analysis/representation_analysis.py` (non esiste), e usa `datasets_config.py` nello schema mentre il file reale è `dataset_config.py`. Andrebbe riscritto per riflettere lo stato attuale del repo.

6. **`exps-status.md`** è un log di risultati/note di lavoro, non documentazione di codice — consiglio di spostarlo in una cartella tipo `notes/` o `docs/experiments/` per non confonderlo con la documentazione strutturale del repo.

---

## 8. Pulizia eseguita

- ✅ Spostato `exps-status.md` → `docs/exps-status.md`
- ✅ Corretto l'import in `config_olmo3_think.py` (`datasets_config` → `dataset_config`)
- ✅ Messo in quarantena `run_representation_analysis.py` → `_broken/run_representation_analysis.py.bak` (vedi `_broken/README.md` per la motivazione; da reintegrare solo se si ritrova `analysis/representation_analysis.py`)
- ✅ Aggiornato il README (diagramma pipeline, struttura repo, tabella "dove sono i dati")

## 9. Ancora da fare (richiede accesso al cluster, non eseguibile da qui)

- **Rigenerare/revocare le credenziali Orfeo** esposte in chiaro in `run_judge_olmo2.sh`/`run_judge_olmo3.sh` — non le ho toccate perché non posso rigenerarle da qui, ma vanno considerate compromesse. Sostituirle con `$(cat ~/.orfeo_token)` o variabili d'ambiente non versionate.
- **Unificare la lettura flat+nested** delle attivazioni negli script di `analysis/` che oggi leggono solo il path flat (`compute_entanglement.py`, `compute_centroid_cosines.py`, `compute_behavioural_probe.py`, `run_classification.py`) — per ora solo `check_sources.py` ed `explore_comprehension_decision.py` gestiscono entrambe le strutture.
