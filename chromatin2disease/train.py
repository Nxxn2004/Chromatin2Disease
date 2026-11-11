"""Train the chromatin GAT and report test ROC-AUC per task and locus.

    python -m chromatin2disease.train --dataset ctcf                    # CTCF degraded vs untreated
    python -m chromatin2disease.train --dataset axin2 --layers 1        # interpretable 1-layer model
    python -m chromatin2disease.train --dataset all                     # both tasks, shared encoder
    python -m chromatin2disease.train --dataset ctcf --split replicate --test-replicate rep3
    python -m chromatin2disease.train --dataset ctcf --act relu --no-attn-leaky   # simpler variants
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

from .graphs import load_dataset
from .model import ChromatinGAT, random_rotation
from .sources import DATASETS, TASKS


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="ctcf", choices=DATASETS)
    p.add_argument("--split", default="random", choices=["random", "replicate"])
    p.add_argument("--test-replicate", default="rep2")
    p.add_argument("--contact-scale", type=float, default=1.0, help="multiplies the contact threshold")
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--act", default="elu", choices=["elu", "leaky_relu", "relu", "sigmoid"], help="GAT layer non-linearity")
    p.add_argument("--no-attn-leaky", dest="attn_leaky", action="store_false", help="drop the LeakyReLU in the attention score")
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="run directory (default runs/<dataset>_<time>)")
    return p.parse_args(argv)


def split(meta, args):
    """Stratified 70/15/15 split, or test = one replicate and 15% of the rest for validation."""
    idx, strat = meta.index.to_numpy(), (meta.task + "|" + meta.locus + "|" + meta.label.astype(str)).to_numpy()
    if args.split == "random":
        rest, test = train_test_split(idx, test_size=0.15, stratify=strat, random_state=args.seed)
    else:
        test, rest = idx[meta.replicate == args.test_replicate], idx[meta.replicate != args.test_replicate]
    train, val = train_test_split(rest, test_size=0.15 / (0.85 if args.split == "random" else 1),
                                  stratify=strat[rest], random_state=args.seed)
    return np.sort(train), np.sort(val), np.sort(test)


def build_model(info, args):
    return ChromatinGAT(info["n_bins"], len(TASKS), args.hidden, args.layers, args.dropout, args.act, args.attn_leaky)


@torch.no_grad()
def predict(model, graphs):
    model.eval()
    return torch.cat([torch.sigmoid(model(b)) for b in DataLoader(graphs, batch_size=256)]).numpy()


def evaluate(meta, prob):
    """ROC-AUC per task and per locus."""
    df = meta.assign(prob=prob)
    return {k: roc_auc_score(g.label, g.prob) for col in ["task", "locus"] for k, g in df.groupby(col)}


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    out = Path(args.out or f"runs/{args.dataset}_{time.strftime('%Y%m%d-%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))

    graphs, meta, info = load_dataset(args.dataset, args.contact_scale)
    parts = dict(zip(["train", "val", "test"], split(meta, args)))
    data = {k: [graphs[i] for i in v] for k, v in parts.items()}
    print(" | ".join(f"{k} {len(v)}" for k, v in parts.items()), "graphs")

    # class-balanced loss: each class of each task gets equal total weight
    tr = meta.loc[parts["train"]]
    weight = torch.ones(len(TASKS), 2)
    for (t, label), n in tr.groupby(["task", "label"]).size().items():
        weight[TASKS.index(t), label] = (tr.task == t).sum() / (2 * n)

    model = build_model(info, args)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    tasks = meta.task.unique()
    best, bad = -1, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for b in DataLoader(data["train"], batch_size=args.batch_size, shuffle=True):
            b.pos = random_rotation(b.pos, b.batch, b.num_graphs)
            loss = F.binary_cross_entropy_with_logits(model(b), b.y, weight=weight[b.task, b.y.long()])
            opt.zero_grad()
            loss.backward()
            opt.step()
        val = evaluate(meta.loc[parts["val"]], predict(model, data["val"]))
        score = np.mean([val[t] for t in tasks])
        print(f"epoch {epoch:3d}  loss {loss.item():.4f}  val AUC {score:.4f}  "
              + "  ".join(f"{k} {v:.3f}" for k, v in val.items() if k not in tasks), flush=True)
        if score > best:
            best, bad = score, 0
            torch.save(model.state_dict(), out / "best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                break

    model.load_state_dict(torch.load(out / "best.pt"))
    prob = {k: predict(model, v) for k, v in data.items()}
    metrics = {k: evaluate(meta.loc[parts[k]], p) for k, p in prob.items()}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    meta.loc[parts["test"]].assign(prob=prob["test"]).to_csv(out / "test_predictions.csv", index=False)
    print("test ROC-AUC: " + "  ".join(f"{k} {v:.4f}" for k, v in metrics["test"].items()), f"\nsaved to {out}")


if __name__ == "__main__":
    main()
