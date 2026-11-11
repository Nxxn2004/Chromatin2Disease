#!/usr/bin/env python3
"""Audit contact-only node degrees across all Hafner 73-locus traces."""

from __future__ import annotations

import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "scripts" / "audit_hafner_missingness.py"
SPEC = importlib.util.spec_from_file_location("hafner_missingness", SOURCE_PATH)
SOURCE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SOURCE)

PRIMARY_THRESHOLD_NM = 238.0


def contact_graph(mask, coordinates, threshold_nm: float):
    observed = [index for index, value in enumerate(mask) if value]
    adjacency = {index: set() for index in observed}
    threshold_squared = threshold_nm**2
    for offset, first in enumerate(observed):
        xyz1 = coordinates[first]
        assert xyz1 is not None
        for second in observed[offset + 1 :]:
            # Genomically adjacent loci are excluded even when physically close.
            if second - first == 1:
                continue
            xyz2 = coordinates[second]
            assert xyz2 is not None
            distance_squared = sum((a - b) ** 2 for a, b in zip(xyz1, xyz2))
            if distance_squared < threshold_squared:
                adjacency[first].add(second)
                adjacency[second].add(first)
    return observed, adjacency


def connected_components(observed, adjacency):
    unseen = set(observed)
    sizes = []
    while unseen:
        stack = [unseen.pop()]
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            new_nodes = adjacency[node] & unseen
            unseen.difference_update(new_nodes)
            stack.extend(new_nodes)
        sizes.append(size)
    return sizes


def has_triangle(observed, adjacency):
    for first in observed:
        for second in adjacency[first]:
            if second > first and any(third > second for third in adjacency[first] & adjacency[second]):
                return True
    return False


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize(records):
    all_degrees = [degree for record in records for degree in record["degrees"]]
    degree_counts = Counter(all_degrees)
    graph_count = len(records)
    node_count = len(all_degrees)
    return {
        "graphs": graph_count,
        "observed_nodes": node_count,
        "nodes_per_graph": {
            "mean": mean(record["nodes"] for record in records),
            "median": median(record["nodes"] for record in records),
            "min": min(record["nodes"] for record in records),
            "max": max(record["nodes"] for record in records),
        },
        "edges_per_graph": {
            "mean": mean(record["edges"] for record in records),
            "median": median(record["edges"] for record in records),
            "min": min(record["edges"] for record in records),
            "max": max(record["edges"] for record in records),
        },
        "node_degree": {
            "mean": mean(all_degrees),
            "median": median(all_degrees),
            "p25": percentile(all_degrees, 0.25),
            "p75": percentile(all_degrees, 0.75),
            "p90": percentile(all_degrees, 0.90),
            "max": max(all_degrees),
            "degree_0_fraction": degree_counts[0] / node_count,
            "degree_1_fraction": degree_counts[1] / node_count,
            "degree_at_least_2_fraction": sum(count for degree, count in degree_counts.items() if degree >= 2) / node_count,
            "degree_at_least_3_fraction": sum(count for degree, count in degree_counts.items() if degree >= 3) / node_count,
            "degree_at_least_5_fraction": sum(count for degree, count in degree_counts.items() if degree >= 5) / node_count,
            "histogram": dict(sorted(degree_counts.items())),
        },
        "graphs_with_any_degree_at_least_2_fraction": sum(record["max_degree"] >= 2 for record in records) / graph_count,
        "graphs_where_every_node_degree_at_most_1": sum(record["max_degree"] <= 1 for record in records),
        "graphs_with_triangle_fraction": sum(record["has_triangle"] for record in records) / graph_count,
        "largest_component_size": {
            "mean": mean(record["largest_component"] for record in records),
            "median": median(record["largest_component"] for record in records),
            "max": max(record["largest_component"] for record in records),
        },
    }


def build_records(threshold_nm: float):
    records = []
    file_summary = []
    for label, replicate, filename in SOURCE.FILES:
        graphs = SOURCE.load_graphs(ROOT / "data" / "hafner2023" / filename)
        file_summary.append(
            {
                "condition": "auxin" if label else "untreated",
                "replicate": replicate,
                "valid_traces": len(graphs),
            }
        )
        for mask, coordinates in graphs:
            observed, adjacency = contact_graph(mask, coordinates, threshold_nm)
            degrees = [len(adjacency[node]) for node in observed]
            components = connected_components(observed, adjacency)
            records.append(
                {
                    "condition": "auxin" if label else "untreated",
                    "replicate": replicate,
                    "nodes": len(observed),
                    "edges": sum(degrees) // 2,
                    "degrees": degrees,
                    "max_degree": max(degrees),
                    "has_triangle": has_triangle(observed, adjacency),
                    "largest_component": max(components),
                }
            )
    return records, file_summary


def sensitivity_summary(threshold_nm):
    records, _ = build_records(threshold_nm)
    records = [record for record in records if record["nodes"] >= 37]
    all_degrees = [degree for record in records for degree in record["degrees"]]
    return {
        "threshold_nm": threshold_nm,
        "graphs": len(records),
        "mean_degree": mean(all_degrees),
        "median_degree": median(all_degrees),
        "degree_at_least_2_fraction": sum(degree >= 2 for degree in all_degrees) / len(all_degrees),
        "graphs_with_any_degree_at_least_2_fraction": sum(record["max_degree"] >= 2 for record in records) / len(records),
    }


def main():
    records, file_summary = build_records(PRIMARY_THRESHOLD_NM)
    filtered = [record for record in records if record["nodes"] >= 37]
    output = {
        "definition": {
            "contact_threshold_nm": PRIMARY_THRESHOLD_NM,
            "canonical_loci": 73,
            "genomically_adjacent_pairs_excluded": True,
            "virtual_node_edges_excluded": True,
        },
        "files": file_summary,
        "all_valid_traces": summarize(records),
        "modeling_subset_at_least_37_nodes": summarize(filtered),
        "filtered_by_condition": {
            condition: summarize([record for record in filtered if record["condition"] == condition])
            for condition in ("untreated", "auxin")
        },
        "threshold_sensitivity_filtered_subset": [
            sensitivity_summary(threshold) for threshold in (150.0, 200.0, 238.0, 300.0)
        ],
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
