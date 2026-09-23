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
