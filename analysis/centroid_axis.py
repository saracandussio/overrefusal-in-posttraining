#!/usr/bin/env python3
"""
centroid_axis.py  (v2)

Posizione del centroide pseudo-harmful sull'asse harmless -> harmful,
nello spazio pieno (niente PCA, niente dipendenza dalla proiezione).

    t        = proiezione sull'asse.  0 = harmless, 1 = harmful, 0.5 = mediano
    off_axis = componente ortogonale all'asse, normalizzata sulla lunghezza
               dell'asse. Se e' grande, "sta in mezzo" e' una descrizione
               sbagliata: il centroide sta fuori dal segmento.

Entrambe invarianti a rotazione, quindi confrontabili fra checkpoint e fra
posizioni — cosa che i pannelli PCA non sono, perche' ogni pannello ha la
sua base.

NOVITA' v2
  - bootstrap ~6x piu' veloce: il resample usa pesi multinomiali e un
    matvec (w @ X) invece di ricopiare le righe con fancy indexing, e i
    resample di harmless/harmful sono condivisi fra i sottogruppi pseudo
    della stessa cella.
  - progress a schermo con ETA, cosi' non sembra piantato.
  - --per-source: calcola t ENTRO ciascuna fonte pseudo. Serve perche' le
    fonti hanno tassi di rifiuto diversi: una separazione acc/ref aggregata
    puo' essere interamente effetto di fonte, e una non-separazione puo'
    essere due effetti di fonte che si cancellano. Se il segno di
    t(ref) - t(acc) e' coerente dentro tutte le fonti, la fonte non spiega
    il risultato.

Uso:
    python analysis/centroid_axis.py --model-family olmo2 \
        --checkpoints base__none sft__none dpo__none final__none \
        --layers 8 19 26 31 --positions last_prompt first_gen \
        --exclude-sources beavertails \
        --raw-results-csv results/olmo2/raw_results.csv \
        --per-source --bootstrap 300 \
        --out results/olmo2/geometry/centroid_axis.csv
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

PSEUDO_SOURCES = {"or_bench", "false_reject", "xstest"}


def assign_group(df):
    g = pd.Series("harmless", index=df.index)
    g[df["label"] == 1] = "harmful"
    g[(df["label"] == 0) & (df["source"].isin(PSEUDO_SOURCES))] = "pseudo_harm"
    return g


def load_token(cli):
    if cli:
        return cli
    p = Path("~/.hf_token").expanduser()
    return p.read_text().strip() if p.exists() else None


def data_paths(ckpt, model_family, nested_only):
    flat = f"data/{ckpt}/*.parquet"
    nested = f"data/{model_family}/{ckpt}/*.parquet" if model_family else None
    if nested_only:
        if not nested:
            raise SystemExit("--nested-only richiede --model-family")
        return [nested]
    return [flat, nested] if nested else [flat]


def load(args, token):
    from datasets import load_dataset

    cols = ["label", "source", "checkpoint", "prompt", "predicted_refusal"]
    cols += [f"layer_{l}_{p}" for l in args.layers for p in args.positions]

    dfs = []
    for ck in args.checkpoints:
        parts = []
        for path in data_paths(ck, args.model_family, args.nested_only):
            try:
                ds = load_dataset(args.hf_dataset, data_files={"train": path},
                                  split="train", token=token)
            except Exception as e:
                print(f"  [skip] {ck} {path}: {type(e).__name__}: {str(e)[:90]}")
                continue
            keep = [c for c in cols if c in ds.column_names]
            missing = [c for c in cols if c not in ds.column_names
                       and c.startswith("layer_")]
            if missing:
                print(f"  [warn] {ck} {path}: mancano {len(missing)} colonne "
                      f"(es. {missing[0]})")
            parts.append(ds.select_columns(keep).to_pandas())
        if not parts:
            continue
        d = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
        print(f"  {ck}: {len(d)} righe")
        dfs.append(d)

    if not dfs:
        raise SystemExit("Nessun dato caricato.")
    df = pd.concat(dfs, ignore_index=True)

    if args.exclude_sources:
        n0 = len(df)
        df = df[~df["source"].isin(set(args.exclude_sources))].reset_index(drop=True)
        print(f"[.] Escluse {args.exclude_sources}: {n0} -> {len(df)}")

    df["group"] = assign_group(df)

    if args.raw_results_csv:
        import sys, os
        sys.path.insert(0, os.getcwd())
        from analysis.judge_utils import attach_judge_refusal
        n0 = len(df)
        df = attach_judge_refusal(df, args.raw_results_csv, drop_missing=True)
        df["predicted_refusal"] = df["judge_refusal"]
        print(f"[.] Refusal dal judge: {n0} -> {len(df)}")
    elif "predicted_refusal" not in df.columns:
        raise SystemExit("Manca 'predicted_refusal'; passa --raw-results-csv.")

    df["predicted_refusal"] = df["predicted_refusal"].astype(bool)
    return df


def stack(d, col):
    """float32: dimezza memoria e banda, irrilevante per una media."""
    X = np.stack(d[col].values).astype(np.float32)
    return X[np.linalg.norm(X, axis=1) > 0]


def t_off_from_centroids(a, b, c):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    axis = b - a
    n2 = float(axis @ axis)
    if n2 == 0:
        return np.nan, np.nan
    v = c - a
    t = float(v @ axis) / n2
    off = float(np.linalg.norm(v - t * axis)) / np.sqrt(n2)
    return t, off


def boot_means(X, n_boot, rng):
    """
    Medie bootstrap di X -> (n_boot, d).
    Resample via pesi multinomiali + matvec: non ricopia mai le righe,
    ~6x piu' veloce del fancy indexing su matrici (5000, 4096).
    """
    n = len(X)
    W = rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot).astype(np.float32) / n
    return W @ X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dataset", default="saracandu/olmo-activations")
    ap.add_argument("--hf-token", default=None)
    ap.add_argument("--checkpoints", nargs="+",
                    default=["base__none", "sft__none", "dpo__none", "final__none"])
    ap.add_argument("--layers", nargs="+", type=int, default=[8, 19, 26, 31])
    ap.add_argument("--positions", nargs="+", default=["last_prompt", "first_gen"])
    ap.add_argument("--exclude-sources", nargs="*", default=None)
    ap.add_argument("--model-family", default=None, choices=[None, "olmo2", "olmo3"])
    ap.add_argument("--nested-only", action="store_true")
    ap.add_argument("--raw-results-csv", default=None,
                    help="Se dato, usa il judge invece di predicted_refusal.")
    ap.add_argument("--per-source", action="store_true",
                    help="Calcola t anche ENTRO ciascuna fonte pseudo.")
    ap.add_argument("--bootstrap", type=int, default=300,
                    help="N resample per l'IC 95%% (0 = niente IC). 300 basta "
                         "per un IC stabile; 1000 costa 3x.")
    ap.add_argument("--min-n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/geometry/centroid_axis.csv")
    args = ap.parse_args()

    print("[.] Carico ...")
    df = load(args, load_token(args.hf_token))
    rng = np.random.default_rng(args.seed)

    cells = [(ck, L, pos) for ck in args.checkpoints
             for L in args.layers for pos in args.positions]
    print(f"\n[.] {len(cells)} celle, bootstrap={args.bootstrap}")

    rows = []
    t_start = time.time()
    for i, (ck, L, pos) in enumerate(cells, 1):
        d = df[df.checkpoint == ck]
        col = f"layer_{L}_{pos}"
        if len(d) == 0 or col not in d.columns:
            print(f"  [{i}/{len(cells)}] {ck} L{L} {pos}: saltata", flush=True)
            continue

        A_less = stack(d[d.group == "harmless"], col)
        A_ful = stack(d[d.group == "harmful"], col)
        if min(len(A_less), len(A_ful)) < args.min_n:
            print(f"  [{i}/{len(cells)}] {ck} L{L} {pos}: n insufficiente", flush=True)
            continue

        ps = d[d.group == "pseudo_harm"]
        subs = {
            ("all", "pseudo_all"): ps,
            ("all", "pseudo_acc"): ps[~ps.predicted_refusal],
            ("all", "pseudo_ref"): ps[ps.predicted_refusal],
        }
        if args.per_source:
            for src in sorted(ps.source.unique()):
                p_s = ps[ps.source == src]
                subs[(src, "pseudo_acc")] = p_s[~p_s.predicted_refusal]
                subs[(src, "pseudo_ref")] = p_s[p_s.predicted_refusal]

        # resample di harmless/harmful calcolati UNA volta per cella e
        # riusati da tutti i sottogruppi pseudo
        M_less = M_ful = None
        if args.bootstrap:
            M_less = boot_means(A_less, args.bootstrap, rng)
            M_ful = boot_means(A_ful, args.bootstrap, rng)
        a0, b0 = A_less.mean(0), A_ful.mean(0)

        for (src, tag), sub in subs.items():
            if len(sub) < args.min_n:
                continue
            A_ps = stack(sub, col)
            if len(A_ps) < args.min_n:
                continue
            t, off = t_off_from_centroids(a0, b0, A_ps.mean(0))
            row = dict(checkpoint=ck, layer=L, pos=pos, source=src, grp=tag,
                       n=len(A_ps), t=t, off_axis=off)
            if args.bootstrap:
                M_ps = boot_means(A_ps, args.bootstrap, rng)
                ts = np.empty(args.bootstrap)
                offs = np.empty(args.bootstrap)
                for k in range(args.bootstrap):
                    ts[k], offs[k] = t_off_from_centroids(M_less[k], M_ful[k], M_ps[k])
                row.update(t_lo=float(np.percentile(ts, 2.5)),
                           t_hi=float(np.percentile(ts, 97.5)),
                           off_lo=float(np.percentile(offs, 2.5)),
                           off_hi=float(np.percentile(offs, 97.5)))
            rows.append(row)

        el = time.time() - t_start
        eta = el / i * (len(cells) - i)
        print(f"  [{i}/{len(cells)}] {ck} L{L} {pos}  "
              f"({el:.0f}s, ETA {eta:.0f}s)", flush=True)

    if not rows:
        raise SystemExit("Nessuna cella calcolabile (controlla --min-n).")

    r = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    r.to_csv(args.out, index=False)

    pd.set_option("display.width", 220)
    agg = r[r.source == "all"]
    print("\n=== t : 0 = harmless, 1 = harmful ===")
    print(agg.pivot_table(index=["pos", "layer", "checkpoint"],
                          columns="grp", values="t").round(3))
    print("\n=== off_axis : 0 = esattamente sul segmento harmless-harmful ===")
    print(agg.pivot_table(index=["pos", "layer", "checkpoint"],
                          columns="grp", values="off_axis").round(3))

    if args.bootstrap:
        print("\n=== t(ref) vs t(acc), aggregato ===")
        for (pos, L, ck), g in agg.groupby(["pos", "layer", "checkpoint"]):
            gi = g.set_index("grp")
            if not {"pseudo_acc", "pseudo_ref"} <= set(gi.index):
                continue
            acc, ref = gi.loc["pseudo_acc"], gi.loc["pseudo_ref"]
            sep = "SI" if (acc.t_hi < ref.t_lo or ref.t_hi < acc.t_lo) else "no"
            print(f"  {pos:<12} L{L:<3} {ck:<14} "
                  f"acc={acc.t:.3f}[{acc.t_lo:.3f},{acc.t_hi:.3f}]  "
                  f"ref={ref.t:.3f}[{ref.t_lo:.3f},{ref.t_hi:.3f}]  "
                  f"delta={ref.t - acc.t:+.3f}   IC disgiunti: {sep}")

    if args.per_source:
        ps_ = r[r.source != "all"]
        if len(ps_):
            piv = ps_.pivot_table(index=["pos", "layer", "checkpoint", "source"],
                                  columns="grp", values="t")
            if {"pseudo_acc", "pseudo_ref"} <= set(piv.columns):
                piv["delta"] = piv["pseudo_ref"] - piv["pseudo_acc"]
            print("\n=== t ENTRO fonte  (delta = ref - acc; segno coerente fra "
                  "le fonti => non e' effetto di fonte) ===")
            print(piv.round(3))

    print(f"\n[ok] {args.out}")


if __name__ == "__main__":
    main()