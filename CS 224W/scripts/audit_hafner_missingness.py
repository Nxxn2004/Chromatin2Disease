#!/usr/bin/env python3
"""Audit label leakage from coverage masks in the 73-locus Hafner dataset."""

from __future__ import annotations

import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


ROOT = Path(__file__).resolve().parents[1]
PARSER_PATH = ROOT / "scripts" / "audit_fof_ct_coverage.py"
SPEC = importlib.util.spec_from_file_location("fof_audit", PARSER_PATH)
PARSER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PARSER)

LOCI = list(range(50_590_004, 52_750_076 + 30_001, 30_001))
LOCUS_INDEX = {locus: index for index, locus in enumerate(LOCI)}
MIN_NODES = 37
CONTACT_THRESHOLD_NM = 238.0

FILES = [
    (0, 1, "4DNFIZWUQTXF.csv"),
    (0, 2, "4DNFIKVZTN9K.csv"),
    (0, 3, "4DNFID7HZ8LX.csv"),
    (1, 1, "4DNFIJYFR8FT.csv"),
    (1, 2, "4DNFIIOVHG1I.csv"),
    (1, 3, "4DNFI4VLY3WU.csv"),
]


def load_graphs(path: Path) -> list[tuple[list[bool], list[tuple[float, float, float] | None]]]:
    _, rows, _ = PARSER.read_fof(path)
    coordinates: dict[int, list[tuple[float, float, float] | None]] = {}
    masks: dict[int, list[bool]] = {}
    for row in rows:
        locus = int(row["Chrom_Start"])
        index = LOCUS_INDEX.get(locus)
        if index is None:
            continue
        trace_id = int(row["Trace_ID"])
        if trace_id not in masks:
            masks[trace_id] = [False] * len(LOCI)
            coordinates[trace_id] = [None] * len(LOCI)
        masks[trace_id][index] = True
        coordinates[trace_id][index] = tuple(float(row[axis]) for axis in ("X", "Y", "Z"))
    return [(masks[trace_id], coordinates[trace_id]) for trace_id in sorted(masks)]


def auc(y: list[int], score: list[float]) -> float:
    order = sorted(range(len(score)), key=score.__getitem__)
    sorted_score = [score[index] for index in order]
    ranks = [0.0] * len(score)
    start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and sorted_score[end] == sorted_score[start]:
            end += 1
        average_rank = (start + 1 + end) / 2
        for index in order[start:end]:
            ranks[index] = average_rank
        start = end
    n_positive = sum(value == 1 for value in y)
    n_negative = len(y) - n_positive
    return (
        (sum(rank for rank, label in zip(ranks, y) if label == 1) - n_positive * (n_positive + 1) / 2)
        / (n_positive * n_negative)
    )


def balanced_accuracy(y: list[int], prediction: list[bool]) -> float:
    sensitivity = mean(predicted for label, predicted in zip(y, prediction) if label == 1)
    specificity = mean(not predicted for label, predicted in zip(y, prediction) if label == 0)
    return (sensitivity + specificity) / 2


def fit_naive_bayes(x: list[list[bool]], y: list[int]) -> tuple[list[float], list[float]]:
    probabilities = []
    for label in (0, 1):
        rows = [mask for mask, value in zip(x, y) if value == label]
        probabilities.append(
            [(sum(mask[index] for mask in rows) + 1) / (len(rows) + 2) for index in range(len(LOCI))]
        )
    return probabilities[0], probabilities[1]


def score_naive_bayes(x: list[list[bool]], model: tuple[list[float], list[float]]) -> list[float]:
    p0, p1 = model
    scores = []
    for mask in x:
        log_odds = 0.0
        for observed, probability0, probability1 in zip(mask, p0, p1):
            if observed:
                log_odds += math.log(probability1 / probability0)
            else:
                log_odds += math.log((1 - probability1) / (1 - probability0))
        scores.append(1 / (1 + math.exp(-max(-30, min(30, log_odds)))))
    return scores


def edge_counts(
    mask: list[bool],
    coordinates: list[tuple[float, float, float] | None],
) -> tuple[int, int, int]:
    observed = [index for index, value in enumerate(mask) if value]
    contact_pairs: set[tuple[int, int]] = set()
    threshold_squared = CONTACT_THRESHOLD_NM**2
    for offset, first in enumerate(observed):
        xyz1 = coordinates[first]
        assert xyz1 is not None
        for second in observed[offset + 1 :]:
            xyz2 = coordinates[second]
            assert xyz2 is not None
            distance_squared = sum((a - b) ** 2 for a, b in zip(xyz1, xyz2))
            if distance_squared < threshold_squared:
                contact_pairs.add((first, second))
    genomic_adjacency = {
        (index, index + 1)
        for index in range(len(mask) - 1)
        if mask[index] and mask[index + 1]
    }
    return len(contact_pairs), len(genomic_adjacency), len(contact_pairs | genomic_adjacency)


def describe(values: list[int]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    masks = []
    coords = []
    labels = []
    replicates = []
    per_file = []
    for label, replicate, filename in FILES:
        graphs = load_graphs(ROOT / "data" / "hafner2023" / filename)
        retained = [(mask, xyz) for mask, xyz in graphs if sum(mask) >= MIN_NODES]
        per_file.append(
            {
                "condition": "auxin" if label else "untreated",
                "replicate": replicate,
                "raw_graphs": len(graphs),
                "retained_graphs": len(retained),
                "retained_fraction": len(retained) / len(graphs),
            }
        )
        for mask, xyz in retained:
            masks.append(mask)
            coords.append(xyz)
            labels.append(label)
            replicates.append(replicate)

    x = masks
    y = labels
    replicate_ids = replicates
    node_counts = [sum(mask) for mask in x]

    cross_validation = []
    for held_out in (1, 2, 3):
        train_indices = [index for index, replicate in enumerate(replicate_ids) if replicate != held_out]
        test_indices = [index for index, replicate in enumerate(replicate_ids) if replicate == held_out]
        train_x = [x[index] for index in train_indices]
        train_y = [y[index] for index in train_indices]
        test_x = [x[index] for index in test_indices]
        test_y = [y[index] for index in test_indices]
        test_counts = [node_counts[index] for index in test_indices]
        probability = score_naive_bayes(test_x, fit_naive_bayes(train_x, train_y))
        cross_validation.append(
            {
                "held_out_replicate": held_out,
                "test_graphs": len(test_indices),
                "missing_mask_auc": auc(test_y, probability),
                "missing_mask_balanced_accuracy": balanced_accuracy(test_y, [value >= 0.5 for value in probability]),
                "node_count_auc": auc(test_y, test_counts),
            }
        )

    per_replicate_deltas: list[list[float]] = []
    for replicate in (1, 2, 3):
        auxin = [mask for mask, label, rep in zip(x, y, replicate_ids) if rep == replicate and label == 1]
        untreated = [mask for mask, label, rep in zip(x, y, replicate_ids) if rep == replicate and label == 0]
        delta = [
            mean(mask[index] for mask in auxin) - mean(mask[index] for mask in untreated)
            for index in range(len(LOCI))
        ]
        per_replicate_deltas.append(delta)
    mean_delta = [mean(delta[index] for delta in per_replicate_deltas) for index in range(len(LOCI))]
    top_indices = sorted(range(len(LOCI)), key=lambda index: abs(mean_delta[index]), reverse=True)[:10]
    top_locus_differences = [
        {
            "chrom_start": int(LOCI[index]),
            "mean_auxin_minus_untreated": float(mean_delta[index]),
            "replicate_differences": [delta[index] for delta in per_replicate_deltas],
        }
        for index in top_indices
    ]

    edge_metrics = [edge_counts(mask, xyz) for mask, xyz in zip(x, coords)]
    exact_cutoff = [count == MIN_NODES for count in node_counts]

    output = {
        "per_file": per_file,
        "retained": {
            "graphs": len(y),
            "untreated": y.count(0),
            "auxin": y.count(1),
            "largest_to_smallest_ratio": max(y.count(0), y.count(1)) / min(y.count(0), y.count(1)),
            "node_counts": describe(node_counts),
            "graphs_exactly_37_nodes": sum(exact_cutoff),
        },
        "missingness_only_cross_replicate": cross_validation,
        "top_paired_locus_detection_differences": top_locus_differences,
        "edges_all_retained": {
            "contact_edges": describe([values[0] for values in edge_metrics]),
            "genomic_adjacency_edges": describe([values[1] for values in edge_metrics]),
            "union_edges": describe([values[2] for values in edge_metrics]),
        },
        "edges_exactly_37_nodes": {
            "graphs": sum(exact_cutoff),
            "contact_edges": describe([values[0] for values, exact in zip(edge_metrics, exact_cutoff) if exact]),
            "genomic_adjacency_edges": describe([values[1] for values, exact in zip(edge_metrics, exact_cutoff) if exact]),
            "union_edges": describe([values[2] for values, exact in zip(edge_metrics, exact_cutoff) if exact]),
        },
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
