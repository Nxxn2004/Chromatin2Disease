#!/usr/bin/env python3
"""Train the interpretable set-of-chromatin-graphs classifier.

Each chromosome trace is a variable-size contact graph. A one-layer sum GNN
produces signed, attention-weighted node contributions and a scalar graph
logit. Ten graph logits from the same condition and replicate are averaged to
produce the bag-level prediction used for training and evaluation.

This is a preliminary random trace split stratified by condition and replicate.
It deliberately does not address locus-specific missingness leakage or
replicate-held-out generalization yet; both are recorded as limitations.
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
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GraphConv
from torch_geometric.utils import scatter, softmax


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "scripts" / "audit_hafner_missingness.py"
SPEC = importlib.util.spec_from_file_location("hafner_missingness", SOURCE_PATH)
SOURCE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SOURCE)

CONTACT_THRESHOLD_NM = 238.0
MIN_NODES = 37
LOCUS_COUNT = 73
PAPER_CBS_READOUTS = (15, 23, 37, 46, 57)
NODE_FEATURES = [
    "locus_identity_one_hot_73",
    "paper_cbs_indicator",
    "normalized_contact_degree",
    "normalized_radial_position",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_graph(
    mask: list[bool],
    coordinates: list[tuple[float, float, float] | None],
    label: int,
    replicate: int,
    graph_id: int,
) -> Data:
    canonical_indices = [index for index, observed in enumerate(mask) if observed]
    local_index = {canonical: local for local, canonical in enumerate(canonical_indices)}
    xyz = np.asarray([coordinates[index] for index in canonical_indices], dtype=np.float32)
    centroid = xyz.mean(axis=0)
    radii = np.linalg.norm(xyz - centroid, axis=1)
    radius_of_gyration = float(np.sqrt(np.mean(np.square(radii))))
    if radius_of_gyration <= 0:
        radius_of_gyration = 1.0

    degrees = [0] * len(canonical_indices)
    directed_edges: list[tuple[int, int]] = []
    threshold_squared = CONTACT_THRESHOLD_NM**2
    for offset, first in enumerate(canonical_indices):
        xyz_first = coordinates[first]
        assert xyz_first is not None
        for second in canonical_indices[offset + 1 :]:
            if second - first == 1:
                continue
            xyz_second = coordinates[second]
            assert xyz_second is not None
            distance_squared = sum((a - b) ** 2 for a, b in zip(xyz_first, xyz_second))
            if distance_squared >= threshold_squared:
                continue
            first_local = local_index[first]
            second_local = local_index[second]
            directed_edges.extend(((first_local, second_local), (second_local, first_local)))
            degrees[first_local] += 1
            degrees[second_local] += 1

    if directed_edges:
        edge_index = torch.tensor(directed_edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    node_features: list[list[float]] = []
    for local, canonical in enumerate(canonical_indices):
        one_hot = [0.0] * LOCUS_COUNT
        one_hot[canonical] = 1.0
        readout = canonical + 4
        cbs = float(readout in PAPER_CBS_READOUTS)
        adjacent_observed = int(canonical - 1 in local_index) + int(canonical + 1 in local_index)
        eligible_partners = max(1, len(canonical_indices) - 1 - adjacent_observed)
        normalized_degree = degrees[local] / eligible_partners
        normalized_radial_position = float(radii[local] / radius_of_gyration)
        node_features.append(one_hot + [cbs, normalized_degree, normalized_radial_position])

    return Data(
        x=torch.tensor(node_features, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor([float(label)], dtype=torch.float32),
        locus_id=torch.tensor(canonical_indices, dtype=torch.long),
        replicate=torch.tensor([replicate], dtype=torch.long),
        graph_id=torch.tensor([graph_id], dtype=torch.long),
        observed_mask=torch.tensor([mask], dtype=torch.bool),
    )


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
    labels = np.asarray([int(data.y.item()) for data in dataset])
    replicates = np.asarray([int(data.replicate.item()) for data in dataset])
    strata = np.asarray([f"{label}_{replicate}" for label, replicate in zip(labels, replicates)])
    train, remainder = train_test_split(
        indices,
        test_size=0.30,
        random_state=seed,
        stratify=strata,
    )
    validation, test = train_test_split(
        remainder,
        test_size=0.50,
        random_state=seed,
        stratify=strata[remainder],
    )
    return train.tolist(), validation.tolist(), test.tolist()


def make_balanced_bags(
    dataset: list[Data],
    indices: list[int],
    bag_size: int,
    seed: int,
) -> list[list[int]]:
    """Create disjoint, class-balanced bags within each replicate."""
    rng = random.Random(seed)
    grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index in indices:
        label = int(dataset[index].y.item())
        replicate = int(dataset[index].replicate.item())
        grouped[(label, replicate)].append(index)

    bags: list[list[int]] = []
    for replicate in sorted({key[1] for key in grouped}):
        by_label: dict[int, list[list[int]]] = {}
        for label in (0, 1):
            values = grouped[(label, replicate)].copy()
            rng.shuffle(values)
            by_label[label] = [
                values[start : start + bag_size]
                for start in range(0, len(values) - bag_size + 1, bag_size)
            ]
        paired_count = min(len(by_label[0]), len(by_label[1]))
        for label in (0, 1):
            rng.shuffle(by_label[label])
            bags.extend(by_label[label][:paired_count])
    rng.shuffle(bags)
    return bags


def bag_summary(dataset: list[Data], bags: list[list[int]]) -> dict[str, object]:
    labels = [int(dataset[bag[0]].y.item()) for bag in bags]
    replicates = [int(dataset[bag[0]].replicate.item()) for bag in bags]
    return {
        "bags": len(bags),
        "graphs": sum(len(bag) for bag in bags),
        "untreated_bags": labels.count(0),
        "ctcf_depleted_bags": labels.count(1),
        "bags_by_replicate": {
            str(rep): replicates.count(rep) for rep in sorted(set(replicates))
        },
    }


def iter_bag_minibatches(
    dataset: list[Data],
    bags: list[list[int]],
    bags_per_batch: int,
    shuffle: bool,
    seed: int,
):
    order = list(range(len(bags)))
    if shuffle:
        random.Random(seed).shuffle(order)
    for start in range(0, len(order), bags_per_batch):
        selected = [bags[index] for index in order[start : start + bags_per_batch]]
        graphs = [dataset[index] for bag in selected for index in bag]
        graph_batch = Batch.from_data_list(graphs)
        labels = torch.tensor(
            [float(dataset[bag[0]].y.item()) for bag in selected],
            dtype=torch.float32,
        )
        yield graph_batch, labels, selected


class InterpretableBagGNN(nn.Module):
    """One sum-message layer and attention readout with exact contributions."""

    def __init__(self, hidden_dim: int = 8, dropout: float = 0.10) -> None:
        super().__init__()
        self.dropout = dropout
        self.message_layer = GraphConv(
            in_channels=LOCUS_COUNT + 3,
            out_channels=hidden_dim,
            aggr="add",
            bias=True,
        )
        self.readout_attention = nn.Linear(hidden_dim, 1)
        self.node_score = nn.Linear(hidden_dim, 1, bias=False)
        self.graph_bias = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        return_explanations: bool = False,
    ):
        h = F.elu(self.message_layer(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        attention = softmax(self.readout_attention(h).squeeze(-1), batch)
        signed_node_score = self.node_score(h).squeeze(-1)
        contribution = attention * signed_node_score
        graph_logits = scatter(contribution, batch, dim=0, reduce="sum") + self.graph_bias
        if return_explanations:
            return graph_logits, attention, signed_node_score, contribution
        return graph_logits


def graph_to_bag_logits(graph_logits: torch.Tensor, bag_size: int) -> torch.Tensor:
    if graph_logits.numel() % bag_size:
        raise ValueError("Every minibatch bag must contain the fixed bag size")
    return graph_logits.reshape(-1, bag_size).mean(dim=1)


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
    model: InterpretableBagGNN,
    dataset: list[Data],
    bags: list[list[int]],
    bag_size: int,
    bags_per_batch: int,
    collect_nodes: bool = False,
) -> tuple[dict[str, float], dict[str, float], dict[str, list[float]] | None, float]:
    model.eval()
    bag_labels: list[float] = []
    bag_logits_all: list[float] = []
    graph_labels: list[float] = []
    graph_logits_all: list[float] = []
    node_records: dict[str, list[float]] | None = defaultdict(list) if collect_nodes else None
    reconstruction_error = 0.0

    for graph_batch, labels, selected in iter_bag_minibatches(
        dataset, bags, bags_per_batch, False, 0
    ):
        output = model(
            graph_batch.x,
            graph_batch.edge_index,
            graph_batch.batch,
            return_explanations=collect_nodes,
        )
        if collect_nodes:
            graph_logits, attention, signed_score, contribution = output
            assert node_records is not None
            rebuilt = scatter(contribution, graph_batch.batch, dim=0, reduce="sum") + model.graph_bias
            reconstruction_error = max(
                reconstruction_error,
                float(torch.max(torch.abs(rebuilt - graph_logits)).item()),
            )
            node_labels = graph_batch.y[graph_batch.batch]
            node_records["locus_id"].extend(graph_batch.locus_id.cpu().tolist())
            node_records["label"].extend(node_labels.cpu().tolist())
            node_records["attention"].extend(attention.cpu().tolist())
            node_records["signed_score"].extend(signed_score.cpu().tolist())
            node_records["contribution"].extend(contribution.cpu().tolist())
        else:
            graph_logits = output
        bag_logits = graph_to_bag_logits(graph_logits, bag_size)
        bag_labels.extend(labels.cpu().tolist())
        bag_logits_all.extend(bag_logits.cpu().tolist())
        graph_labels.extend(graph_batch.y.cpu().tolist())
        graph_logits_all.extend(graph_logits.cpu().tolist())

    bag_probabilities = 1 / (1 + np.exp(-np.asarray(bag_logits_all)))
    graph_probabilities = 1 / (1 + np.exp(-np.asarray(graph_logits_all)))
    return (
        binary_metrics(np.asarray(bag_labels), bag_probabilities),
        binary_metrics(np.asarray(graph_labels), graph_probabilities),
        node_records,
        reconstruction_error,
    )


def summarize_explanations(records: dict[str, list[float]] | None) -> dict[str, object]:
    assert records is not None
    grouped: dict[tuple[int, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for locus, label, attention, signed_score, contribution in zip(
        records["locus_id"],
        records["label"],
        records["attention"],
        records["signed_score"],
        records["contribution"],
    ):
        key = (int(locus), int(label))
        grouped[key]["attention"].append(float(attention))
        grouped[key]["signed_score"].append(float(signed_score))
        grouped[key]["contribution"].append(float(contribution))

    rows = []
    for locus in range(LOCUS_COUNT):
        per_class = {}
        for label in (0, 1):
            values = grouped[(locus, label)]
            per_class[label] = {
                "observations": len(values["contribution"]),
                "mean_attention": mean(values["attention"]) if values["attention"] else math.nan,
                "mean_signed_score": mean(values["signed_score"]) if values["signed_score"] else math.nan,
                "mean_contribution": mean(values["contribution"]) if values["contribution"] else math.nan,
            }
        rows.append(
            {
                "canonical_index_0_based": locus,
                "original_readout": locus + 4,
                "chrom_start": SOURCE.LOCI[locus],
                "paper_cbs": locus + 4 in PAPER_CBS_READOUTS,
                "untreated": per_class[0],
                "ctcf_depleted": per_class[1],
                "attention_all": np.nanmean(
                    [per_class[0]["mean_attention"], per_class[1]["mean_attention"]]
                ),
                "depleted_minus_untreated_contribution": (
                    per_class[1]["mean_contribution"] - per_class[0]["mean_contribution"]
                ),
            }
        )

    top_attention = sorted(rows, key=lambda row: row["attention_all"], reverse=True)[:10]
    top_depleted = sorted(
        rows,
        key=lambda row: row["depleted_minus_untreated_contribution"],
        reverse=True,
    )[:10]
    top_untreated = sorted(
        rows,
        key=lambda row: row["depleted_minus_untreated_contribution"],
    )[:10]
    cbs = [row for row in rows if row["paper_cbs"]]
    non_cbs = [row for row in rows if not row["paper_cbs"]]
    return {
        "positive_logit_class": "ctcf_depleted",
        "top_mean_attention": top_attention,
        "top_differential_contribution_toward_depleted": top_depleted,
        "top_differential_contribution_toward_untreated": top_untreated,
        "paper_cbs_summary": {
            "mean_attention_cbs": float(np.nanmean([row["attention_all"] for row in cbs])),
            "mean_attention_non_cbs": float(
                np.nanmean([row["attention_all"] for row in non_cbs])
            ),
            "mean_abs_differential_contribution_cbs": float(
                np.nanmean(
                    [abs(row["depleted_minus_untreated_contribution"]) for row in cbs]
                )
            ),
            "mean_abs_differential_contribution_non_cbs": float(
                np.nanmean(
                    [abs(row["depleted_minus_untreated_contribution"]) for row in non_cbs]
                )
            ),
        },
        "all_loci": rows,
    }


def train_one_seed(
    dataset: list[Data],
    train_indices: list[int],
    validation_bags: list[list[int]],
    test_bags: list[list[int]],
    seed: int,
    epochs: int,
    bag_size: int,
    bags_per_batch: int,
    hidden_dim: int,
) -> tuple[dict[str, object], InterpretableBagGNN]:
    set_seed(seed)
    model = InterpretableBagGNN(hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    best_state = None
    best_validation_auc = -math.inf
    best_epoch = 0
    patience = 12
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_bags = make_balanced_bags(dataset, train_indices, bag_size, seed * 10_000 + epoch)
        total_loss = 0.0
        total_bags = 0
        for graph_batch, labels, selected in iter_bag_minibatches(
            dataset,
            train_bags,
            bags_per_batch,
            True,
            seed * 100_000 + epoch,
        ):
            optimizer.zero_grad()
            graph_logits = model(graph_batch.x, graph_batch.edge_index, graph_batch.batch)
            bag_logits = graph_to_bag_logits(graph_logits, bag_size)
            loss = criterion(bag_logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(selected)
            total_bags += len(selected)

        validation_metrics, _, _, _ = evaluate(
            model,
            dataset,
            validation_bags,
            bag_size,
            bags_per_batch,
        )
        record = {
            "epoch": epoch,
            "training_bags": total_bags,
            "train_loss": total_loss / total_bags,
            "validation_roc_auc": validation_metrics["roc_auc"],
        }
        history.append(record)
        print(
            f"seed={seed} epoch={epoch:03d} bags={total_bags} "
            f"loss={record['train_loss']:.4f} val_auc={record['validation_roc_auc']:.4f}",
            flush=True,
        )

        if validation_metrics["roc_auc"] > best_validation_auc + 1e-4:
            best_validation_auc = validation_metrics["roc_auc"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    validation_bag, validation_graph, _, _ = evaluate(
        model, dataset, validation_bags, bag_size, bags_per_batch
    )
    test_bag, test_graph, node_records, reconstruction_error = evaluate(
        model,
        dataset,
        test_bags,
        bag_size,
        bags_per_batch,
        collect_nodes=True,
    )
    return (
        {
            "seed": seed,
            "best_epoch": best_epoch,
            "validation_bag_metrics": validation_bag,
            "validation_graph_metrics": validation_graph,
            "test_bag_metrics": test_bag,
            "test_graph_metrics": test_graph,
            "maximum_additive_readout_reconstruction_error": reconstruction_error,
            "history": history,
            "interpretation": summarize_explanations(node_records),
        },
        model,
    )


def aggregate_seed_metrics(seed_results: list[dict[str, object]], key: str) -> dict[str, object]:
    metrics = ["roc_auc", "average_precision", "accuracy", "balanced_accuracy", "f1"]
    output = {}
    for metric in metrics:
        values = [float(result[key][metric]) for result in seed_results]
        output[metric] = {
            "mean": float(np.mean(values)),
            "standard_deviation": (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            ),
            "values": values,
        }
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--bag-size", type=int, default=10)
    parser.add_argument("--bags-per-batch", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=8)
    parser.add_argument("--split-seed", type=int, default=224)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "interpretable_bag_gnn_preliminary.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset, provenance = load_dataset()
    train_indices, validation_indices, test_indices = make_split(dataset, args.split_seed)
    validation_bags = make_balanced_bags(
        dataset, validation_indices, args.bag_size, args.split_seed + 1
    )
    test_bags = make_balanced_bags(dataset, test_indices, args.bag_size, args.split_seed + 2)
    reference_train_bags = make_balanced_bags(
        dataset, train_indices, args.bag_size, args.split_seed
    )
    print(
        f"Loaded {len(dataset)} graphs; trace split "
        f"{len(train_indices)}/{len(validation_indices)}/{len(test_indices)}; "
        f"fixed bags {len(reference_train_bags)}/{len(validation_bags)}/{len(test_bags)}",
        flush=True,
    )

    seed_results = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = ROOT / "models"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        result, model = train_one_seed(
            dataset,
            train_indices,
            validation_bags,
            test_bags,
            seed,
            args.epochs,
            args.bag_size,
            args.bags_per_batch,
            args.hidden_dim,
        )
        seed_results.append(result)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "node_features": NODE_FEATURES,
                "edge_features": [],
                "hidden_dim": args.hidden_dim,
                "bag_size": args.bag_size,
                "seed": seed,
            },
            checkpoint_dir / f"interpretable_bag_gnn_seed_{seed}.pt",
        )

    report = {
        "status": "preliminary_random_trace_split_missingness_not_yet_controlled",
        "dataset": {
            "graphs": len(dataset),
            "minimum_observed_nodes": MIN_NODES,
            "canonical_loci": LOCUS_COUNT,
            "contact_threshold_nm": CONTACT_THRESHOLD_NM,
            "genomically_adjacent_edges_excluded": True,
            "paper_cbs_original_readouts": list(PAPER_CBS_READOUTS),
            "provenance": provenance,
        },
        "features": {
            "node": NODE_FEATURES,
            "edge": [],
            "missing_nodes": "omitted from each variable-size graph",
        },
        "split": {
            "type": "random_70_15_15_trace_split_stratified_by_condition_and_replicate",
            "seed": args.split_seed,
            "split_before_bagging": True,
            "bags_disjoint_within_each_fixed_validation_and_test_partition": True,
            "train": bag_summary(dataset, reference_train_bags),
            "validation": bag_summary(dataset, validation_bags),
            "test": bag_summary(dataset, test_bags),
        },
        "model": {
            "message_passing_layers": 1,
            "aggregation": "sum",
            "hidden_dim": args.hidden_dim,
            "edge_features": False,
            "virtual_node_during_message_passing": False,
            "graph_readout": "softmax node attention times signed node score",
            "bag_readout": "mean of graph logits",
            "bag_size": args.bag_size,
            "positive_logit_class": "ctcf_depleted",
        },
        "limitations": [
            "Locus-specific missingness is not controlled in this preliminary run.",
            "Replicates are stratified across partitions rather than held out.",
            "Multiple traces from the same biological cell may not be group-split when cell IDs are unavailable.",
        ],
        "aggregate_test_bag_metrics_across_seeds": aggregate_seed_metrics(
            seed_results, "test_bag_metrics"
        ),
        "aggregate_test_graph_metrics_across_seeds": aggregate_seed_metrics(
            seed_results, "test_graph_metrics"
        ),
        "seeds": seed_results,
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "test_bag_metrics": report["aggregate_test_bag_metrics_across_seeds"],
                "test_graph_metrics": report["aggregate_test_graph_metrics_across_seeds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
