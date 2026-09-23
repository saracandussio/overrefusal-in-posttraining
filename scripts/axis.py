"""Q2/Q3 — Where do pseudo-harmful prompts sit on the refusal axis, and what
separates the ones the model refuses?

One row per checkpoint x layer x position x pseudo subset (all pseudo, then
each pseudo source alone with --per-source) with every measure in
overrefusal.geometry and its bootstrap interval. This single script replaces
compute_entanglement.py, centroid_axis.py and the centroid tables of
explore_comprehension_decision.py, which computed the same quantities
three times on slightly different data.

Per-source rows matter: sources refuse at different rates, so a gap between
refused and answered prompts could be a source effect. If the sign of
t_refused - t_answered holds within every source, it is not.

    python scripts/axis.py --family olmo2 --bootstrap 300 --per-source
    python scripts/axis.py --family olmo3 --layers 16 24 --positions pre_gen first_gen
"""
import pandas as pd

from overrefusal import activations, cli, config, geometry, refusal
from overrefusal.io import save_atomic

MIN_PER_SIDE = 20  # fewer refused (or answered) prompts than this: no v_beh


def main():
    p = cli.parser(__doc__, activations=True)
    p.add_argument("--bootstrap", type=int, default=300)
    p.add_argument("--per-source", action="store_true")
    args = p.parse_args()

    df = activations.load(args.family, args.layers, args.positions, args.checkpoints)
    df = refusal.attach(df, pd.read_csv(config.raw_results_csv(args.family)))
    print(f"dropped {df.attrs['n_dropped_unjudged']} rows without a usable judgement")

    rows = []
    cells = [(c, L, p) for c in args.checkpoints for L in args.layers
             for p in activations.expand(df, L, args.positions)]
    for ckpt, layer, pos in cells:
        column = activations.col(layer, pos)
        d = df[(df.checkpoint == ckpt) & activations.has(df, column)]
        if d.empty:  # e.g. template tokens for the base model
            continue
        X = lambda mask: activations.matrix(d[mask], column)
        pseudo = d.group == "pseudo_harm"
        subsets = {"all": pseudo}
        if args.per_source:
            subsets |= {s: pseudo & (d.source == s) for s in sorted(d.source[pseudo].unique())}

        for name, mask in subsets.items():
            n_ref, n_acc = int((mask & (d.refused == 1)).sum()), int((mask & (d.refused == 0)).sum())
            if min(n_ref, n_acc) < MIN_PER_SIDE:
                print(f"  skip {ckpt} L{layer} {pos} {name}: refused={n_ref} answered={n_acc}")
                continue
            m = geometry.cell(X(d.group == "harmless"), X(d.group == "harmful"),
                              X(mask & (d.refused == 1)), X(mask & (d.refused == 0)),
                              n_boot=args.bootstrap)
            rows.append({"checkpoint": ckpt, "layer": layer, "position": pos,
                         "pseudo_source": name, **m})

    out = pd.DataFrame(rows)
    save_atomic(out, config.results_dir(args.family) / "geometry" / "axis.csv")

    main_rows = out[out.pseudo_source == "all"].copy()
    main_rows["position"] = pd.Categorical(
        main_rows.position, activations.reading_order(main_rows.position), ordered=True)
    main_rows["checkpoint"] = pd.Categorical(main_rows.checkpoint, args.checkpoints, ordered=True)
    for measure in ["entanglement", "t", "off_axis", "cos_vbeh_vref", "cos_vbeh_vover_orth"]:
        print(f"\n{measure}")
        print(main_rows.pivot_table(index=["checkpoint", "position"], columns="layer",
                                    values=measure, observed=True).round(2).to_string())


if __name__ == "__main__":
    main()
