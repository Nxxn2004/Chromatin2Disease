"""Chromatin traces -> PyTorch Geometric graphs, one graph per traced chromosome.

Nodes  imaged DNA segments of the trace (only segments imaged in every file of the locus)
       x   = [genomic position in locus (0-1), distance to trace centroid / locus scale]
       pos = centred xyz / locus scale (rotated randomly during training)
       bin = segment id, unique across loci (0 unused)
Edges  both directions: consecutive observed segments (backbone) and pairs closer than the contact threshold
       = contact_scale x median distance between genomically adjacent segments, as in Hafner et al. 2023
       (251 / 222 nm for CTCF chr3 / chr6 untreated; the paper reports 256 / 238 nm)

Locus scale = median radius of gyration. Spots farther than 4 scales from their trace's median are dropped, and
traces with < 4 segments or < 30% of the locus are skipped. Delete ``cache/`` after changing this file.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch, Data

from .fofct import read_core_table
from .sources import ROOT, TASKS, load_sources


def radius_of_gyration(xyz):
    return np.sqrt(((xyz - xyz.mean(0)) ** 2).sum(1).mean())


def build_graphs(sources, contact_scale=1.0, min_coverage=0.3, min_nodes=4, outlier=4.0):
    spots = pd.concat([read_core_table(p).assign(source=i, locus=l) for i, (p, l) in enumerate(zip(sources.path, sources.locus))])

    loci, offset = {}, 1  # segment grid of each locus: starts imaged in every file
    for locus, g in spots.groupby("locus"):
        starts = sorted(set.intersection(*(set(f.start) for _, f in g.groupby("source"))))
        loci[locus] = dict(starts=starts, offset=offset)
        offset += len(starts)
    grid = pd.DataFrame([(l, s, i) for l, d in loci.items() for i, s in enumerate(d["starts"])], columns=["locus", "start", "bin"])
    spots = spots.merge(grid, on=["locus", "start"]).sort_values(["source", "trace_id", "bin"], kind="stable")
    traces = [(s, t, g.locus.iat[0], g.bin.to_numpy(), g[["x", "y", "z"]].to_numpy())
              for (s, t), g in spots.groupby(["source", "trace_id"])]

    for locus, d in loci.items():  # locus scale, then outlier removal, then contact threshold
        d["scale"] = np.median([radius_of_gyration(x) for _, _, l, b, x in traces if l == locus and len(b) >= min_nodes])
    for k, (s, t, l, b, x) in enumerate(traces):
        if len(b) > 2:
            keep = np.linalg.norm(x - np.median(x, 0), axis=1) <= outlier * loci[l]["scale"]
            traces[k] = (s, t, l, b[keep], x[keep])
    for locus, d in loci.items():
        steps = [np.linalg.norm(np.diff(x, axis=0)[np.diff(b) == 1], axis=1) for _, _, l, b, x in traces if l == locus]
        d["contact"] = contact_scale * np.median(np.concatenate(steps))

    graphs, meta = [], []
    for s, t, locus, b, xyz in traces:
        d, src, n = loci[locus], sources.iloc[s], len(b)
        nb = len(d["starts"])
        if n < min_nodes or n / nb < min_coverage:
            continue
        centred = xyz - xyz.mean(0)
        dist = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
        backbone = np.abs(np.arange(n)[:, None] - np.arange(n)) == 1
        i, j = np.nonzero(backbone | ((dist < d["contact"]) & ~np.eye(n, dtype=bool)))
        graphs.append(Data(
            x=torch.tensor(np.stack([b / (nb - 1), np.linalg.norm(centred, axis=1) / d["scale"]], 1), dtype=torch.float),
            pos=torch.tensor(centred / d["scale"], dtype=torch.float),
            bin=torch.tensor(b + d["offset"]),
            edge_index=torch.tensor(np.stack([i, j])),
            y=torch.tensor([float(src.label)]),
            task=torch.tensor([TASKS.index(src.task)])))
        meta.append(dict(task=src.task, locus=locus, label=src.label, label_name=src.label_name,
                         replicate=src.replicate, experiment=src.experiment, file=src.file, trace_id=t, n_nodes=n))
    meta = pd.DataFrame(meta)
    for locus, d in loci.items():
        print(f"{locus}: {len(d['starts'])} segments, contact < {d['contact']:.0f} nm, "
              f"{(meta.locus == locus).sum()} traces kept")
    return graphs, meta, dict(n_bins=offset, loci=loci)


def load_dataset(dataset, contact_scale=1.0, cache="cache"):
    """(graphs, meta, info) for a dataset, cached on disk."""
    sources = load_sources(dataset)
    files = [Path(p).relative_to(ROOT).as_posix() for p in sources.path]  # same key on every machine
    key = hashlib.sha1(json.dumps([files, contact_scale]).encode()).hexdigest()[:10]
    path = Path(cache) / f"{dataset}_{key}.pt"
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        graphs, meta, info = build_graphs(sources, contact_scale)
        torch.save((Batch.from_data_list(graphs), meta, info), path)
    batch, meta, info = torch.load(path, weights_only=False)
    return batch.to_data_list(), meta, info
