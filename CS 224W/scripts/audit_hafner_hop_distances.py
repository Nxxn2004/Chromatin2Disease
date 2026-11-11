#!/usr/bin/env python3
"""Audit shortest-path receptive fields in the Hafner contact-only graphs."""

from __future__ import annotations

import importlib.util
import json
from collections import Counter, deque
from pathlib import Path
from statistics import mean, median


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "scripts" / "audit_hafner_contact_degrees.py"
SPEC = importlib.util.spec_from_file_location("hafner_degrees", SOURCE_PATH)
SOURCE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SOURCE)


def shortest_paths(source, adjacency):
    distances = {source: 0}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        for neighbor in adjacency[node]:
            if neighbor not in distances:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)
    return distances


def main():
    exact_distances = Counter()
    total_pairs = 0
    graph_diameters = []
    largest_component_fractions = []
    graph_count = 0

    for _, _, filename in SOURCE.SOURCE.FILES:
        graphs = SOURCE.SOURCE.load_graphs(ROOT / "data" / "hafner2023" / filename)
        for mask, coordinates in graphs:
            observed, adjacency = SOURCE.contact_graph(mask, coordinates, SOURCE.PRIMARY_THRESHOLD_NM)
            if len(observed) < 37:
                continue
            graph_count += 1
            total_pairs += len(observed) * (len(observed) - 1)
            components = SOURCE.connected_components(observed, adjacency)
            largest_component_fractions.append(max(components) / len(observed))
            diameter = 0
            for node in observed:
                distances = shortest_paths(node, adjacency)
                diameter = max(diameter, max(distances.values()))
                for other in observed:
                    if other == node:
                        continue
                    exact_distances[distances.get(other, "unreachable")] += 1
            graph_diameters.append(diameter)

    cumulative = {}
    running = 0
    for hops in range(1, 7):
        running += exact_distances[hops]
        cumulative[str(hops)] = running / total_pairs

    output = {
        "definition": {
            "graphs": "all traces with at least 37 of 73 canonical loci",
            "contact_threshold_nm": SOURCE.PRIMARY_THRESHOLD_NM,
            "genomically_adjacent_pairs_excluded": True,
            "virtual_node_edges_excluded": True,
            "pair_counting": "ordered pairs of distinct observed nodes",
        },
        "graphs": graph_count,
        "ordered_node_pairs": total_pairs,
        "exact_shortest_path_fraction": {
            str(distance): count / total_pairs
            for distance, count in sorted(
                exact_distances.items(), key=lambda item: (isinstance(item[0], str), item[0])
            )
        },
        "cumulative_fraction_reachable_within_hops": cumulative,
        "unreachable_fraction": exact_distances["unreachable"] / total_pairs,
        "largest_component_fraction": {
            "mean": mean(largest_component_fractions),
            "median": median(largest_component_fractions),
        },
        "graph_diameter_over_all_components": {
            "mean": mean(graph_diameters),
            "median": median(graph_diameters),
            "min": min(graph_diameters),
            "max": max(graph_diameters),
            "histogram": dict(sorted(Counter(graph_diameters).items())),
        },
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
