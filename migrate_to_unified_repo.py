#!/usr/bin/env python3
"""
migrate_to_unified_repo.py

Copia gli shard Parquet delle attivazioni COSI' COME SONO, da:

    saracandu/olmo-activations   data/{ckpt}/*.parquet          (OLMo2, 5 source)
    saracandu/olmo-activations   data/olmo2/{ckpt}/*.parquet    (OLMo2, 5 source)
    saracandu/olmo3-activations  data/olmo3/{ckpt}/*.parquet    (OLMo3)

verso un'unica repo, con path sempre prefissati dalla famiglia:

    saracandu/overrefusal-activations
        data/olmo2/{ckpt}/shard_*.parquet
        data/olmo3/{ckpt}/shard_*.parquet

Nessuna ricostruzione, nessun merge, nessun ricalcolo: le colonne
(inclusi TUTTI i post_instr_*, in numero diverso fra le fonti — 7 nel
flat OLMo2, 3 nel nested, 5 in OLMo3) restano quelle originali, shard
per shard. E' un download+reupload, non un load_dataset(): non tocca lo
schema, quindi non puo' silenziosamente perdere colonne.

Uso:
    python -m migrate_to_unified_repo \\
        --target-repo saracandu/overrefusal-activations --dry-run

    python -m migrate_to_unified_repo \\
        --target-repo saracandu/overrefusal-activations
"""
import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from analysis.olmo_data import CHECKPOINTS, hf_token

# Sorgenti per famiglia: (repo, prefisso path). Path SEMPRE espansi con
# list_repo_files, mai con data_files/glob di load_dataset — e' proprio il
# meccanismo che oggi ha causato ore di errori fuorvianti.
SOURCES = {
    "olmo2": [
        ("saracandu/olmo-activations", "data/{ckpt}"),
        ("saracandu/olmo-activations", "data/olmo2/{ckpt}"),
    ],
    "olmo3": [
        ("saracandu/olmo3-activations", "data/olmo3/{ckpt}"),
    ],
}


def list_shards(api: HfApi, repo: str, prefix: str) -> list[str]:
    all_files = api.list_repo_files(repo, repo_type="dataset")
    return sorted(f for f in all_files
                  if f.startswith(prefix + "/") and f.endswith(".parquet"))


def peek_sources(repo: str, shards: list[str], token: str) -> "pd.Series":
    """
    Conta le righe per source leggendo TUTTI gli shard, ma solo le colonne
    'source'/'category' via pyarrow — non l'intero parquet (le attivazioni
    sono la quasi totalita' del peso del file, quindi il costo resta basso
    anche leggendo ogni shard).
    """
    import pandas as pd
    import pyarrow.parquet as pq

    frames = []
    for shard in shards:
        local = hf_hub_download(repo, shard, repo_type="dataset", token=token)
        pf = pq.ParquetFile(local)
        cols = [c for c in ("source", "category") if c in pf.schema_arrow.names]
        frames.append(pq.read_table(local, columns=cols).to_pandas())
    if not frames:
        return pd.Series(dtype=int)
    d = pd.concat(frames, ignore_index=True)
    return d["source"].value_counts()


def migrate_family(family: str, target_repo: str, dry_run: bool, token: str,
                   check_sources: bool = False) -> None:
    print(f"\n{'='*60}\n{family}\n{'='*60}")
    api = HfApi(token=token)
    if not dry_run:
        api.create_repo(target_repo, repo_type="dataset", exist_ok=True, private=True)

    for ck in CHECKPOINTS:
        shards_by_source = []
        for repo, prefix_tmpl in SOURCES[family]:
            prefix = prefix_tmpl.format(ckpt=ck)
            shards = list_shards(api, repo, prefix)
            if shards:
                shards_by_source.append((repo, shards))
                print(f"  {ck}: {len(shards)} shard da {repo}/{prefix}/")
                if check_sources:
                    counts = peek_sources(repo, shards, token)
                    print(f"      source = {counts.to_dict()}")

        total = sum(len(s) for _, s in shards_by_source)
        if total == 0:
            print(f"  [skip] {family}/{ck}: nessuno shard trovato in nessuna sorgente")
            continue
        if dry_run:
            continue

        staging = Path(tempfile.mkdtemp(prefix=f"mig_{family}_{ck}_"))
        n = 0
        for repo, shards in shards_by_source:
            for shard in shards:
                local = hf_hub_download(repo, shard, repo_type="dataset", token=token)
                dest = staging / f"shard_{n:05d}.parquet"
                shutil.copy(local, dest)
                n += 1

        path_in_repo = f"data/{family}/{ck}"
        print(f"  upload: {n} shard -> {target_repo}/{path_in_repo}/")
        api.upload_folder(folder_path=str(staging), path_in_repo=path_in_repo,
                          repo_id=target_repo, repo_type="dataset",
                          commit_message=f"Migrate {family}/{ck}: {n} shard, schema originale")
        shutil.rmtree(staging, ignore_errors=True)

    print(f"[ok] {family} -> {target_repo}/data/{family}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-repo", required=True)
    ap.add_argument("--families", nargs="+", default=list(SOURCES),
                    choices=list(SOURCES))
    ap.add_argument("--dry-run", action="store_true",
                    help="conta gli shard per sorgente, non scrive nulla")
    ap.add_argument("--check-sources", action="store_true",
                    help="oltre a contare gli shard, campiona i primi 3 per "
                         "sorgente e stima le righe per 'source' — attivo "
                         "automaticamente con --dry-run, opzionale altrimenti")
    args = ap.parse_args()

    token = hf_token()
    if token is None:
        raise SystemExit("Nessun token trovato (HF_TOKEN o ~/.hf_token).")

    for fam in args.families:
        migrate_family(fam, args.target_repo, args.dry_run, token,
                       check_sources=args.check_sources or args.dry_run)

    if not args.dry_run:
        print(f"\n[ok] Migrazione completa: "
              f"https://huggingface.co/datasets/{args.target_repo}")


if __name__ == "__main__":
    main()