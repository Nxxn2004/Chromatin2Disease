"""Average virtual-node attention per locus segment and class on the test set, to find the segments the model
relies on (proposal: "use the attention scores to prioritise the nodes"). Attention is multiplied by graph size,
so 1 means "average node". Writes attention_<locus>.csv and .png into the run directory.

    python -m chromatin2disease.interpret runs/<run>
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from .graphs import load_dataset
from .train import build_model, parse_args, split


def main(run):
    run = Path(run)
    args = parse_args([])
    vars(args).update(json.loads((run / "config.json").read_text()))
    graphs, meta, info = load_dataset(args.dataset, args.contact_scale)
    test = split(meta, args)[2]
    model = build_model(info, args)
    model.load_state_dict(torch.load(run / "best.pt"))
    model.eval()

    rows = []
    with torch.no_grad():
        for b in DataLoader([graphs[i] for i in test], batch_size=256):
            g = b.batch.numpy()
            alpha = model(b, return_attention=True)[1].numpy()
            rows.append(pd.DataFrame({"bin": b.bin.numpy(), "label": b.y.numpy()[g].astype(int),
                                      "attention": alpha * np.bincount(g)[g]}))
    df = pd.concat(rows)
    for locus, d in info["loci"].items():
        sub = df[(df.bin >= d["offset"]) & (df.bin < d["offset"] + len(d["starts"]))]
        if sub.empty:
            continue
        sub = sub.assign(start=np.array(d["starts"])[sub.bin - d["offset"]])
        names = meta[meta.locus == locus].groupby("label").label_name.first()
        tab = sub.pivot_table(index="start", columns="label", values="attention").rename(columns=names)
        tab.to_csv(run / f"attention_{locus}.csv")
        ax = tab.set_index(tab.index / 1e6).plot(figsize=(10, 3), title=f"{locus}: readout attention per segment")
        ax.set(xlabel="genomic position (Mb)", ylabel="attention x graph size")
        ax.figure.tight_layout()
        ax.figure.savefig(run / f"attention_{locus}.png", dpi=150)
    print(f"wrote attention tables and plots to {run}")


if __name__ == "__main__":
    main(sys.argv[1])
