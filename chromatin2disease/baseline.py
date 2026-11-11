"""Non-graph reference: logistic regression on each trace's pairwise-distance map (missing segments filled with
the training median), with the same graphs and splits as train.py.

    python -m chromatin2disease.baseline --dataset ctcf [--split replicate --test-replicate rep3] [--out runs/<name>]

Writes config.json and metrics.json ({train, val, test: {locus: ROC-AUC}}) to the run directory.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .graphs import load_dataset
from .train import parse_args, split


def main(argv=None):
    args = parse_args(argv)
    graphs, meta, info = load_dataset(args.dataset, args.contact_scale)
    parts = dict(zip(["train", "val", "test"], split(meta, args)))
    metrics = {k: {} for k in parts}
    for locus, d in info["loci"].items():
        nb, iu = len(d["starts"]), np.triu_indices(len(d["starts"]), 1)
        rows = np.flatnonzero(meta.locus == locus)
        X = np.full((len(rows), len(iu[0])), np.nan)
        for k, i in enumerate(rows):
            b, D = graphs[i].bin.numpy() - d["offset"], np.full((nb, nb), np.nan)
            D[np.ix_(b, b)] = torch.cdist(graphs[i].pos, graphs[i].pos).numpy()
            X[k] = D[iu]
        y, mask = meta.label.to_numpy()[rows], {k: np.isin(rows, v) for k, v in parts.items()}
        X = np.where(np.isnan(X), np.nan_to_num(np.nanmedian(X[mask["train"]], 0)), X)
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000, class_weight="balanced"))
        clf.fit(X[mask["train"]], y[mask["train"]])
        for k, m in mask.items():
            if len(set(y[m])) == 2:
                metrics[k][locus] = roc_auc_score(y[m], clf.predict_proba(X[m])[:, 1])
        print(locus, "ROC-AUC", "  ".join(f"{k} {v[locus]:.4f}" for k, v in metrics.items() if locus in v))
    out = Path(args.out or f"runs/baseline_{args.dataset}_{time.strftime('%Y%m%d-%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
