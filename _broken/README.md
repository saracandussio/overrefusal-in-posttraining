# File quarantenati

## `run_representation_analysis.py.bak`

Importa `from analysis.representation_analysis import (...)` e presumibilmente
`analysis/plot_entanglement.py` (citato nel vecchio README) — nessuno dei due
file esiste in questo repo. Non è chiaro se siano stati persi in un merge
precedente o se non siano mai stati versionati.

La pipeline di analisi geometrica **realmente funzionante e usata** è quella
basata su:

```
analysis/extract_and_push.py          → estrae le attivazioni, le carica su HF
analysis/compute_entanglement.py      → entanglement + boundary margin
analysis/compute_centroid_cosines.py  → coseni tra centroidi cross-checkpoint
analysis/compute_behavioural_probe.py → probe comportamentale
analysis/run_classification.py        → probe 3-classi
analysis/plot_2d_refusal_space*.py, plot_pca_umap.py, plot_entanglement_curves.py
```

Se `analysis/representation_analysis.py` viene ritrovato (es. in un vecchio
commit o backup), va reintegrato prima di rimuovere questa cartella. Altrimenti,
questo script va considerato superato e rimosso in una pulizia futura.
