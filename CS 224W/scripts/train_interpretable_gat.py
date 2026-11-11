#!/usr/bin/env python3
"""Train an interpretable edge-aware GAT on the Hafner CTCF-AID traces.

This is a deliberately small preliminary model:

* Graphs contain observed loci only and must contain at least 37 of 73 loci.
* Edges are non-genomically-adjacent contacts closer than 238 nm.
* Node features are a paper CTCF-boundary-site flag and normalized degree.
* The primary edge feature states whether the edge spans an intervening paper
  CTCF boundary site. Normalized distance is available only as an ablation.
* Two contact-only GAT layers run before an attention readout.
* The final graph logit is exactly a sum of signed node contributions.

The default split is a preliminary random 70/15/15 split stratified by
condition and replicate. It is not a substitute for replicate-held-out
evaluation in the final project.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv
from torch_geometric.utils import scatter, softmax


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "scripts" / "audit_hafner_missingness.py"
SPEC = importlib.util.spec_from_file_location("hafner_missingness", SOURCE_PATH)
SOURCE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SOURCE)

CONTACT_THRESHOLD_NM = 238.0
MIN_NODES = 37

# Original readout numbers reported for chromosome-6 CBS hubs in Hafner et al.
# Methods: readouts 15, 23, 37, 46, and 57. The public CSV's 73 canonical loci
# correspond to original readouts 4 through 76.
PAPER_CBS_READOUTS = (15, 23, 37, 46, 57)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def spans_intervening_cbs(first_readout: int, second_readout: int) -> float:
    low, high = sorted((first_readout, second_readout))
    return float(any(low < boundary < high for boundary in PAPER_CBS_READOUTS))


def build_graph(
    mask: list[bool],
    coordinates: list[tuple[float, float, float] | None],
    label: int,
    replicate: int,
    graph_id: int,
) -> Data:
    canonical_indices = [index for index, observed in enumerate(mask) if observed]
    local_index = {canonical: local for local, canonical in enumerate(canonical_indices)}
    degrees = [0] * len(canonical_indices)
    directed_edges: list[tuple[int, int]] = []
    directed_attributes: list[tuple[float, float]] = []
    undirected_distances: list[float] = []
    undirected_spanning: list[float] = []
    threshold_squared = CONTACT_THRESHOLD_NM**2

    for offset, first in enumerate(canonical_indices):
        xyz_first = coordinates[first]
        assert xyz_first is not None
        for second in canonical_indices[offset + 1 :]:
            # The model's edges mean nonadjacent 3D contacts, not polymer bonds.
            if second - first == 1:
                continue
            xyz_second = coordinates[second]
            assert xyz_second is not None
            distance_squared = sum((a - b) ** 2 for a, b in zip(xyz_first, xyz_second))
            if distance_squared >= threshold_squared:
                continue

            distance_nm = math.sqrt(distance_squared)
            first_local = local_index[first]
            second_local = local_index[second]
            first_readout = first + 4
            second_readout = second + 4
            spans_boundary = spans_intervening_cbs(first_readout, second_readout)
            attribute = (distance_nm / CONTACT_THRESHOLD_NM, spans_boundary)

            directed_edges.extend(((first_local, second_local), (second_local, first_local)))
            directed_attributes.extend((attribute, attribute))
            degrees[first_local] += 1
            degrees[second_local] += 1
            undirected_distances.append(distance_nm / CONTACT_THRESHOLD_NM)
            undirected_spanning.append(spans_boundary)

    if not directed_edges:
        raise ValueError("Retained graph unexpectedly contains no contact edges")

    node_features = []
    locus_ids = []
    for local, canonical in enumerate(canonical_indices):
        original_readout = canonical + 4
        paper_cbs = float(original_readout in PAPER_CBS_READOUTS)
        normalized_degree = degrees[local] / max(1, len(canonical_indices) - 1)
        node_features.append((paper_cbs, normalized_degree))
        locus_ids.append(canonical)

    edge_index = torch.tensor(directed_edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(directed_attributes, dtype=torch.float32)
    data = Data(
        x=torch.tensor(node_features, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([float(label)], dtype=torch.float32),
        locus_id=torch.tensor(locus_ids, dtype=torch.long),
        replicate=torch.tensor([replicate], dtype=torch.long),
        graph_id=torch.tensor([graph_id], dtype=torch.long),
    )

    cbs_degrees = [
        degree
        for degree, canonical in zip(degrees, canonical_indices)
        if canonical + 4 in PAPER_CBS_READOUTS
    ]
    data.summary_features = torch.tensor(
        [[
            len(canonical_indices),
            len(undirected_distances),
            2 * len(undirected_distances) / len(canonical_indices),
            mean(cbs_degrees) if cbs_degrees else 0.0,
            mean(undirected_spanning) if undirected_spanning else 0.0,
            mean(undirected_distances) if undirected_distances else 0.0,
        ]],
        dtype=torch.float32,
    )
    return data


def load_dataset() -> tuple[list[Data], list[dict[str, object]]]:
    dataset: list[Data] = []
    provenance: list[dict[str, object]] = []
    graph_id = 0
    for label, replicate, filename in SOURCE.FILES:
        raw_graphs = SOURCE.load_graphs(ROOT / "data" / "hafner2023" / filename)
        retained = 0
        for mask, coordinates in raw_graphs:
            if sum(mask) < MIN_NODES:
                continue
            dataset.append(build_graph(mask, coordinates, label, replicate, graph_id))
            graph_id += 1
            retained += 1
        provenance.append(
            {
                "file": filename,
                "condition": "ctcf_depleted" if label else "untreated",
                "replicate": replicate,
                "raw_graphs": len(raw_graphs),
                "retained_graphs": retained,
            }
        )
    return dataset, provenance


def make_split(dataset: list[Data], seed: int) -> tuple[list[int], list[int], list[int]]:
    indices = np.arange(len(dataset))
    labels = np.array([int(data.y.item()) for data in dataset])
    replicates = np.array([int(data.replicate.item()) for data in dataset])
    strata = np.array([f"{label}_{replicate}" for label, replicate in zip(labels, replicates)])
    train_indices, remainder_indices = train_test_split(
        indices,
        test_size=0.30,
        random_state=seed,
        stratify=strata,
    )
    remainder_strata = strata[remainder_indices]
    validation_indices, test_indices = train_test_split(
        remainder_indices,
        test_size=0.50,
        random_state=seed,
        stratify=remainder_strata,
    )
    return train_indices.tolist(), validation_indices.tolist(), test_indices.tolist()


class InterpretableGAT(nn.Module):
    """Two contact-only GAT layers with an exactly additive final readout."""

    def __init__(self, hidden_dim: int = 8, edge_dim: int = 1, dropout: float = 0.10) -> None:
        super().__init__()
        self.dropout = dropout
        self.gat1 = GATConv(
            in_channels=2,
            out_channels=hidden_dim,
            heads=1,
            concat=False,
            edge_dim=edge_dim,
            add_self_loops=True,
            fill_value=0.0,
            residual=True,
        )
        self.gat2 = GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=1,
            concat=False,
            edge_dim=edge_dim,
            add_self_loops=True,
            fill_value=0.0,
            residual=True,
        )
        self.readout_attention = nn.Linear(hidden_dim, 1)
        self.node_score = nn.Linear(hidden_dim, 1, bias=False)
        self.graph_bias = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        return_explanations: bool = False,
    ):
        h = F.elu(self.gat1(x, edge_index, edge_attr))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.elu(self.gat2(h, edge_index, edge_attr))

        attention = softmax(self.readout_attention(h).squeeze(-1), batch)
        signed_node_score = self.node_score(h).squeeze(-1)
        contribution = attention * signed_node_score
        logits = scatter(contribution, batch, dim=0, reduce="sum") + self.graph_bias
        if return_explanations:
            return logits, attention, signed_node_score, contribution
        return logits


def graph_loader(dataset: list[Data], indices: list[int], batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        [dataset[index] for index in indices],
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = probabilities >= 0.5
    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions)),
    }


@torch.no_grad()
def evaluate(
    model: InterpretableGAT,
    loader: DataLoader,
    include_distance: bool,
    collect_nodes: bool = False,
) -> tuple[dict[str, float], dict[str, list[float]] | None]:
    model.eval()
    labels: list[float] = []
    logits: list[float] = []
    node_records: dict[str, list[float]] | None = None
    if collect_nodes:
        node_records = defaultdict(list)

    for batch in loader:
        edge_attr = batch.edge_attr if include_distance else batch.edge_attr[:, 1:2]
        output = model(
            batch.x,
            batch.edge_index,
            edge_attr,
            batch.batch,
            return_explanations=collect_nodes,
        )
        if collect_nodes:
            batch_logits, attention, signed_score, contribution = output
            assert node_records is not None
            node_labels = batch.y[batch.batch]
            node_records["locus_id"].extend(batch.locus_id.cpu().tolist())
            node_records["label"].extend(node_labels.cpu().tolist())
            node_records["attention"].extend(attention.cpu().tolist())
            node_records["signed_score"].extend(signed_score.cpu().tolist())
            node_records["contribution"].extend(contribution.cpu().tolist())
        else:
            batch_logits = output
        labels.extend(batch.y.cpu().tolist())
        logits.extend(batch_logits.cpu().tolist())

    label_array = np.asarray(labels, dtype=float)
    probability_array = 1 / (1 + np.exp(-np.asarray(logits, dtype=float)))
    return binary_metrics(label_array, probability_array), node_records


def train_one_seed(
    dataset: list[Data],
    split: tuple[list[int], list[int], list[int]],
    seed: int,
    epochs: int,
    batch_size: int,
    hidden_dim: int,
    include_distance: bool,
) -> tuple[dict[str, object], InterpretableGAT]:
    set_seed(seed)
    train_indices, validation_indices, test_indices = split
    train_loader = graph_loader(dataset, train_indices, batch_size, True)
    validation_loader = graph_loader(dataset, validation_indices, batch_size, False)
    test_loader = graph_loader(dataset, test_indices, batch_size, False)

    model = InterpretableGAT(hidden_dim=hidden_dim, edge_dim=2 if include_distance else 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    best_state = None
    best_validation_auc = -math.inf
    best_epoch = 0
    patience = 10
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_graphs = 0
        for batch in train_loader:
            optimizer.zero_grad()
            edge_attr = batch.edge_attr if include_distance else batch.edge_attr[:, 1:2]
            logits = model(batch.x, batch.edge_index, edge_attr, batch.batch)
            loss = criterion(logits, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * batch.num_graphs
            total_graphs += batch.num_graphs

        validation_metrics, _ = evaluate(model, validation_loader, include_distance)
        epoch_record = {
            "epoch": epoch,
            "train_loss": total_loss / total_graphs,
            "validation_roc_auc": validation_metrics["roc_auc"],
        }
        history.append(epoch_record)
        print(
            f"seed={seed} epoch={epoch:03d} "
            f"loss={epoch_record['train_loss']:.4f} "
            f"val_auc={validation_metrics['roc_auc']:.4f}",
            flush=True,
        )

        if validation_metrics["roc_auc"] > best_validation_auc + 1e-4:
            best_validation_auc = validation_metrics["roc_auc"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    validation_metrics, _ = evaluate(model, validation_loader, include_distance)
    test_metrics, node_records = evaluate(
        model,
        test_loader,
        include_distance,
        collect_nodes=True,
    )
    interpretation = summarize_node_explanations(node_records)
    result = {
        "seed": seed,
        "best_epoch": best_epoch,
        "validation": validation_metrics,
        "test": test_metrics,
        "history": history,
        "interpretation": interpretation,
    }
    return result, model


def summarize_node_explanations(records: dict[str, list[float]] | None) -> dict[str, object]:
    assert records is not None
    grouped: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    grouped_by_class: dict[tuple[int, int], list[float]] = defaultdict(list)
    for locus, label, attention, score, contribution in zip(
        records["locus_id"],
        records["label"],
        records["attention"],
        records["signed_score"],
        records["contribution"],
    ):
        locus_index = int(locus)
        class_label = int(label)
        grouped[locus_index]["attention"].append(float(attention))
        grouped[locus_index]["signed_score"].append(float(score))
        grouped[locus_index]["contribution"].append(float(contribution))
        grouped_by_class[(locus_index, class_label)].append(float(contribution))

    loci = []
    for locus_index, values in grouped.items():
        original_readout = locus_index + 4
        loci.append(
            {
                "canonical_index_0_based": locus_index,
                "original_readout": original_readout,
                "chrom_start": SOURCE.LOCI[locus_index],
                "paper_cbs": original_readout in PAPER_CBS_READOUTS,
                "observations": len(values["contribution"]),
                "mean_readout_attention": mean(values["attention"]),
                "mean_signed_node_score": mean(values["signed_score"]),
                "mean_contribution_all": mean(values["contribution"]),
                "mean_contribution_untreated": mean(grouped_by_class[(locus_index, 0)]),
                "mean_contribution_depleted": mean(grouped_by_class[(locus_index, 1)]),
            }
        )

    top_auxin = sorted(loci, key=lambda row: row["mean_contribution_all"], reverse=True)[:10]
    top_untreated = sorted(loci, key=lambda row: row["mean_contribution_all"])[:10]
    cbs_rows = [row for row in loci if row["paper_cbs"]]
    non_cbs_rows = [row for row in loci if not row["paper_cbs"]]
    return {
        "positive_logit_class": "ctcf_depleted",
        "top_mean_contribution_toward_depleted": top_auxin,
        "top_mean_contribution_toward_untreated": top_untreated,
        "paper_cbs_summary": {
            "mean_attention_cbs": mean(row["mean_readout_attention"] for row in cbs_rows),
            "mean_attention_non_cbs": mean(row["mean_readout_attention"] for row in non_cbs_rows),
            "mean_contribution_cbs": mean(row["mean_contribution_all"] for row in cbs_rows),
            "mean_contribution_non_cbs": mean(row["mean_contribution_all"] for row in non_cbs_rows),
        },
        "all_loci": sorted(loci, key=lambda row: row["canonical_index_0_based"]),
    }


def split_summary(dataset: list[Data], indices: list[int]) -> dict[str, object]:
    labels = [int(dataset[index].y.item()) for index in indices]
    replicates = [int(dataset[index].replicate.item()) for index in indices]
    return {
        "graphs": len(indices),
        "untreated": labels.count(0),
        "ctcf_depleted": labels.count(1),
        "replicates": {str(rep): replicates.count(rep) for rep in sorted(set(replicates))},
    }


def baseline_results(
    dataset: list[Data],
    split: tuple[list[int], list[int], list[int]],
    seed: int,
) -> dict[str, object]:
    train_indices, _, test_indices = split
    features = np.vstack([data.summary_features.numpy() for data in dataset])
    labels = np.array([int(data.y.item()) for data in dataset])
    feature_names = [
        "observed_nodes",
        "contact_edges",
        "mean_degree",
        "mean_paper_cbs_degree",
        "fraction_edges_spanning_cbs_boundary",
        "mean_normalized_edge_distance",
    ]

    output = {}
    dummy = DummyClassifier(strategy="prior", random_state=seed)
    dummy.fit(features[train_indices, :1], labels[train_indices])
    dummy_probability = dummy.predict_proba(features[test_indices, :1])[:, 1]
    output["class_prior"] = binary_metrics(labels[test_indices], dummy_probability)

    specifications = {
        "coverage_density": [0, 1, 2],
        "mechanistic_summary_without_distance": [0, 1, 2, 3, 4],
        "mechanistic_summary_with_distance": list(range(len(feature_names))),
    }
    for name, columns in specifications.items():
        pipeline = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=seed),
        )
        pipeline.fit(features[train_indices][:, columns], labels[train_indices])
        probabilities = pipeline.predict_proba(features[test_indices][:, columns])[:, 1]
        coefficients = pipeline.named_steps["logisticregression"].coef_[0]
        output[name] = {
            **binary_metrics(labels[test_indices], probabilities),
            "standardized_coefficients": {
                feature_names[column]: float(coefficient)
                for column, coefficient in zip(columns, coefficients)
            },
        }
    return output


def aggregate_seed_metrics(seed_results: list[dict[str, object]]) -> dict[str, object]:
    metric_names = ["roc_auc", "average_precision", "accuracy", "balanced_accuracy", "f1"]
    output = {}
    for metric in metric_names:
        values = [float(result["test"][metric]) for result in seed_results]
        output[metric] = {
            "mean": float(np.mean(values)),
            "standard_deviation": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "values": values,
        }
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=8)
    parser.add_argument(
        "--include-distance",
        action="store_true",
        help="Include normalized 3D edge distance as an ablation.",
    )
    parser.add_argument("--split-seed", type=int, default=224)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "interpretable_gat_preliminary.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset, provenance = load_dataset()
    split = make_split(dataset, args.split_seed)
    train_indices, validation_indices, test_indices = split
    print(
        f"Loaded {len(dataset)} graphs; split into "
        f"{len(train_indices)}/{len(validation_indices)}/{len(test_indices)}",
        flush=True,
    )

    baselines = baseline_results(dataset, split, args.split_seed)
    seed_results = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = ROOT / "models"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        result, model = train_one_seed(
            dataset,
            split,
            seed,
            args.epochs,
            args.batch_size,
            args.hidden_dim,
            args.include_distance,
        )
        seed_results.append(result)
        variant = "with_distance" if args.include_distance else "no_distance"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "node_features": ["paper_cbs", "normalized_degree"],
                "edge_features": (
                    ["normalized_distance", "spans_intervening_paper_cbs"]
                    if args.include_distance
                    else ["spans_intervening_paper_cbs"]
                ),
                "hidden_dim": args.hidden_dim,
                "seed": seed,
            },
            checkpoint_dir / f"interpretable_gat_{variant}_seed_{seed}.pt",
        )

    report = {
        "status": "preliminary_random_split_not_final_generalization_estimate",
        "dataset": {
            "graphs": len(dataset),
            "minimum_observed_nodes": MIN_NODES,
            "contact_threshold_nm": CONTACT_THRESHOLD_NM,
            "genomically_adjacent_edges_excluded": True,
            "paper_cbs_original_readouts": list(PAPER_CBS_READOUTS),
            "provenance": provenance,
        },
        "features": {
            "node": ["paper_cbs_indicator", "normalized_contact_degree"],
            "edge": (
                ["distance_nm_divided_by_238", "spans_intervening_paper_cbs"]
                if args.include_distance
                else ["spans_intervening_paper_cbs"]
            ),
            "tad_note": (
                "The CSVs do not contain authoritative per-locus TAD assignments. "
                "The second edge feature is a reproducible CBS-boundary proxy, not an exact cross-TAD label."
            ),
        },
        "split": {
            "type": "random_70_15_15_stratified_by_condition_and_replicate",
            "seed": args.split_seed,
            "train": split_summary(dataset, train_indices),
            "validation": split_summary(dataset, validation_indices),
            "test": split_summary(dataset, test_indices),
        },
        "model": {
            "contact_gat_layers": 2,
            "attention_heads": 1,
            "hidden_dim": args.hidden_dim,
            "virtual_node_during_message_passing": False,
            "readout": "attention-weighted exact additive signed node contributions",
            "positive_logit_class": "ctcf_depleted",
        },
        "baselines": baselines,
        "aggregate_test_metrics_across_seeds": aggregate_seed_metrics(seed_results),
        "seeds": seed_results,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "baselines": baselines,
        "aggregate_test_metrics_across_seeds": report["aggregate_test_metrics_across_seeds"],
    }, indent=2))


if __name__ == "__main__":
    main()
