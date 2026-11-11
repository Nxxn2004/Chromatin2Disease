"""Tests for the FOF-CT reader, graph construction and model."""
import textwrap

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from chromatin2disease.fofct import read_core_table
from chromatin2disease.graphs import build_graphs
from chromatin2disease.model import ChromatinGAT, random_rotation


def write(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(textwrap.dedent(body).lstrip())
    return p


def test_axin2_style_header_row_and_readout_column(tmp_path):
    df = read_core_table(write(tmp_path, "a.csv", """
        ##Table_namespace=4dn_FOF-CT_core
        spotID,traceID,x,y,z,Readout,Chrom,Chrom_Start,Chrom_End
        1,1,1544.8,1433.0,2445.9,13-16,chr17,65514012,65522015
        2,1,1421.4,1379.1,2274.9,17-20,chr17,65522016,65530019
        """))
    assert df.start.tolist() == [65514012, 65522016] and df.x.iat[0] == 1544.8


def test_hafner_style_trailing_commas_quoted_comments_cell_id(tmp_path):
    df = read_core_table(write(tmp_path, "b.csv", """
        ##Table_namespace=4dn_FOF-CT_core,,,,,,,,
        "#Software_Authors: Mateo, LJ; Boettiger, AN",,,,,,,
        628,9,2267.0,2195.4,5449.9,chr3,33390004,33420004,17
        629,9,2270.0,2190.4,5440.9,chr3,33420005,33450005,17
        """))
    assert df.trace_id.tolist() == ["9", "9"] and df.end.iat[1] == 33450005


def synthetic_sources(tmp_path, n_traces=12, n_bins=10):
    rng, rows = np.random.default_rng(0), []
    for label in (0, 1):
        lines = []
        for t in range(n_traces):
            walk = np.cumsum(rng.normal(scale=200 + 100 * label, size=(n_bins, 3)), axis=0)
            lines += [f"{t * n_bins + b},{t},{x:.1f},{y:.1f},{z:.1f},chr1,{30_000 * b},{30_000 * b + 29_999}"
                      for b, (x, y, z) in enumerate(walk)]
        (tmp_path / f"{label}.csv").write_text("\n".join(lines))
        rows.append(dict(path=str(tmp_path / f"{label}.csv"), task="ctcf", locus="ctcf_chr1", label=label,
                         label_name=f"class{label}", replicate="rep1", experiment=f"exp{label}", file=f"f{label}"))
    return pd.DataFrame(rows)


def test_graphs_and_model(tmp_path):
    graphs, meta, info = build_graphs(synthetic_sources(tmp_path))
    assert len(graphs) == 24 and set(meta.label) == {0, 1}
    assert {(0, 1), (1, 0)} <= set(map(tuple, graphs[0].edge_index.T.tolist()))  # backbone, both directions
    batch = next(iter(DataLoader(graphs, batch_size=8)))
    for layers, act, leaky in [(0, "elu", True), (1, "sigmoid", False), (3, "relu", True), (2, "leaky_relu", False)]:
        model = ChromatinGAT(info["n_bins"], hidden=16, num_layers=layers, act=act, attn_leaky=leaky)
        logits, alpha = model(batch, return_attention=True)
        assert logits.shape == (8,)
        assert torch.allclose(torch.zeros(8).index_add_(0, batch.batch, alpha), torch.ones(8), atol=1e-5)


def test_random_rotation_preserves_distances():
    pos, batch = torch.randn(20, 3), torch.tensor([0] * 10 + [1] * 10)
    rot = random_rotation(pos, batch, 2)
    for g in (0, 1):
        assert torch.allclose(torch.cdist(pos[batch == g], pos[batch == g]),
                              torch.cdist(rot[batch == g], rot[batch == g]), atol=1e-4)
