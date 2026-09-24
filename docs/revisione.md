# Revisione del codice vecchio (settembre 2026)

Cosa è emerso riscrivendo il repo, e cosa resta da verificare sul cluster.
Ogni punto dice cosa cambia per il paper.

## Emerso

**1. La tabella comportamentale di `exps-status.md` non era del giudice.**
`metrics_by_group.csv` non è prodotto da nessuno script del repo. Ricostruito
dai dati: recall e compliance usano il keyword detector (`predicted_refusal`)
e includono beavertails. Rifatta con il giudice, senza beavertails, xstest
diviso (`scripts/behavior.py`, OLMo2):

| | rifiuta harmful | risponde pseudo | risponde harmless | incoerenti |
|---|---|---|---|---|
| base | 0.459 | 0.823 | 0.821 | 0.416 |
| sft | 0.933 | 0.557 | 0.852 | 0.009 |
| dpo | 0.918 | 0.649 | 0.902 | 0.003 |
| final | 0.912 | 0.665 | 0.900 | 0.003 |

La storia regge (SFT crea l'over-refusal, DPO lo recupera in parte, final ≈ DPO)
ma con numeri diversi: dopo SFT si risponde al 56% delle pseudo, non al 66%.

**2. `first_gen` è letto dopo la decisione.** In `extract_and_push.py`
`first_gen_idx = len(prompt_ids)`, cioè il primo token della risposta già
scritto. Lo stato lì codifica la parola scelta. Il collasso delle pseudo
rifiutate verso v_ref a first_gen, e il 96% della probe comportamentale,
possono venire in parte da quel token. La decisione avviene sull'ultimo token
del template (`pre_gen`), già presente su HF come ultima colonna `post_instr`.

**3. Tre script, una sola misura, numeri diversi.** `boundary_margin_n = 2t - 1`
(verificato in `tests/test_core.py`), e `cos_pseudo_harmless` di explore è
l'entanglement. Ma leggevano dati diversi (xstest dentro o fuori, repo flat o
nested): per base, layer 8, first_gen, un calcolo dà +0.15 e l'altro -0.34.

**4. Probe a tre classi e identità del dataset.** I gruppi coincidono con le
fonti, quindi il >90% può riflettere lo stile del dataset. `probe.py` riporta
ora l'accuratezza leave-one-source-out. Inoltre il vecchio "transfer" base→base
era accuratezza sul training set.

**5. Coseni tra centroidi grezzi vicini a 1 per costruzione.** Lo stream
residuo ha una grossa componente comune. `drift.py` riporta anche i coseni
delle direzioni v_ref e v_over, dove quella componente si cancella.

**6. Minori.** Il giudice locale crashava sempre (`use_reasoning_fallback`
non definito); quello API era a posto (1 chiamata fallita su 42.624). Le run
mistral_safety non avevano WildGuard (ora escluse del tutto). Il base vede il
prompt grezzo, gli altri il chat template: base vs SFT mescola training e
formato.

## Da verificare sul cluster (un comando ciascuno)

**A. Estrazione e generazione vedevano gli stessi token?** La generazione
tokenizzava con i token speciali di default, l'estrazione vecchia con
`add_special_tokens=False`. Se il tokenizer aggiunge un BOS, i contesti
differiscono di un token.

```bash
python -c "from transformers import AutoTokenizer as T; t=T.from_pretrained('allenai/OLMo-2-1124-7B-SFT'); print(t('ciao').input_ids, t('ciao', add_special_tokens=False).input_ids)"
```

**B. Il base in estrazione usava un template?** La vecchia estrazione applicava
il template se il tokenizer ne aveva uno, la generazione no (per nome).

```bash
python -c "from transformers import AutoTokenizer as T; print(bool(T.from_pretrained('allenai/OLMo-2-1124-7B').chat_template))"
```

Se A o B danno una differenza, le attivazioni vanno ri-estratte con
`scripts/extract.py` (che usa la stessa funzione della generazione) su un
repo nuovo. Se no, quelle attuali vanno bene.

## Verifiche fatte (24 settembre)

**A = True, B = False.** Generazione ed estrazione dei modelli chat vedevano
gli stessi token: le loro attivazioni sono affidabili. I tokenizer dei base
non hanno template.

**Il base era misto, in entrambe le famiglie.** Il vecchio `olmo_loader.py`
dava al base la cornice `User: {prompt}\nAssistant:` (resta commentata nel
file), poi è passato al testo grezzo. Le fonti generate o estratte prima e
dopo quel cambio hanno condizioni diverse. Le risposte generate con la
cornice si riconoscono perché il base inventa nuovi turni `\nUser:`.

| base | generazione | estrazione |
|---|---|---|
| OLMo2, OR-Bench, FalseReject, ToxicChat, HarmBench | cornice | cornice (3 token) |
| OLMo2, XSTest, Alpaca, AdvBench, JBB | cornice | grezzo |
| OLMo2, WildGuard | grezzo | grezzo |
| OLMo3, OR-Bench, FalseReject, ToxicChat, HarmBench | cornice | grezzo |
| OLMo3, le altre | grezzo | grezzo |

Quindi la riga "base" delle tabelle di comportamento e tutta la geometria
del base (anche OLMo3, che sembrava pulito) non sono affidabili. SFT, DPO e
final non sono toccati.

**Scelta:** il base usa la cornice `User: … Assistant:` per tutte le fonti,
in generazione ed estrazione (`config.BASE_FRAME`, `prompts.py`), e la
risposta viene tagliata al primo `\nUser:` inventato. Si rifà con
`sbatch jobs/redo_base.sh olmo2` e `olmo3`. Tra base e chat si confrontano
`last_prompt`, `pre_gen` (ruolo uguale, token diverso: `:` contro `\n`) e
`first_gen`; i token intermedi del template no.

**Token del template.** OLMo2 spezza `<|assistant|>` in cinque pezzi: le
posizioni sono `\n < | assistant | > \n`. I due `|` sono poco informativi e
danno misure instabili: non vanno riportati. OLMo3: `<|im_end|> \n
<|im_start|> assistant \n`. `config.TEMPLATE_TOKENS` li nomina nelle figure.

**OLMo3 ha un system prompt nascosto.** Senza system prompt esplicito, il
template ne inserisce uno di default ("You are a helpful function-calling AI
assistant..."). Va dichiarato nel paper.

## Cosa dicono i testi (raw_results.csv)

**"Rifiuto" è una categoria mista.** Con la definizione binaria, rifiutato è
tutto ciò che non è una risposta piena. Tra le pseudo rifiutate:

| | rifiuto netto | risponde ma prende distanza | risposta parziale |
|---|---|---|---|
| OLMo2 SFT | 0.68 | 0.25 | 0.08 |
| OLMo2 DPO | 0.47 | 0.38 | 0.15 |
| OLMo3 SFT | 0.64 | 0.27 | 0.08 |
| OLMo3 DPO | 0.33 | 0.60 | 0.07 |

DPO non solo fa rispondere di più: trasforma molti rifiuti netti in
risposte con presa di distanza. I rifiuti netti sulle pseudo passano da circa
30% a 16% (OLMo2) e da 26% a 11% (OLMo3): l'effetto di DPO è più forte di
quanto dica il tasso binario. Nel base i "rifiuti" sono all'80% (OLMo2)
risposte parziali, non rifiuti: le misure su v_beh del base non hanno senso.

**first_gen è dopo la decisione.** Il 64-65% delle risposte rifiutate inizia
con "I'm"; quelle accettate con "1", "Certainly", "In".

**DPO cambia idea in entrambe le direzioni.** Sugli stessi prompt, OLMo2:
150 passano da rifiutati ad accettati, 35 al contrario (su 1244). OLMo3: 164
e 63 (su 1242). Il recupero netto è reale, ma circa il 15% dei prompt cambia
esito.

**I giudizi, letti a campione, sono sensati**, ma alcuni prompt di
FalseReject sono discutibili come "sicuri" (per esempio manipolare persone
nominate): il rifiuto lì non è per forza un errore.

## Controlli prompt per prompt sulle attivazioni

`scripts/inspect_cell.py` (job: `jobs/inspect.sh`): t di ogni prompt per tipo
di risposta, test di permutazione dentro ogni fonte, controllo sulla
lunghezza, i prompt agli estremi, e i prompt che DPO fa cambiare.
