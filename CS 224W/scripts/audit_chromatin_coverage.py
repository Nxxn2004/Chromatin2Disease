#!/usr/bin/env python3
"""Audit per-trace locus coverage in Bintu-style chromatin tracing CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median


def is_observed(row: dict[str, str]) -> bool:
    values = []
    for column in ("X", "Y", "Z"):
        value = row[column].strip()
        try:
            values.append(float(value))
        except ValueError:
            return False
    return all(math.isfinite(value) for value in values)


def audit(path: Path, label: str) -> dict[str, object]:
    observed_by_trace: dict[int, set[int]] = defaultdict(set)
    rows_by_trace: Counter[int] = Counter()
    locus_observed: Counter[int] = Counter()
    seen_pairs: Counter[tuple[int, int]] = Counter()
    loci: set[int] = set()
    row_count = 0
    observed_rows = 0

    with path.open(newline="", encoding="utf-8-sig") as handle:
        next(handle)  # Free-text description line before the CSV header.
        reader = csv.DictReader(handle)
        required = {"Chromosome index", "Segment index", "X", "Y", "Z"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Unexpected columns in {path}: {reader.fieldnames}")

        for row in reader:
            row_count += 1
            trace_id = int(row["Chromosome index"])
            locus_id = int(row["Segment index"])
            loci.add(locus_id)
            rows_by_trace[trace_id] += 1
            seen_pairs[(trace_id, locus_id)] += 1
            if is_observed(row):
                observed_rows += 1
                observed_by_trace[trace_id].add(locus_id)
                locus_observed[locus_id] += 1

    trace_ids = sorted(rows_by_trace)
    counts = [len(observed_by_trace[trace_id]) for trace_id in trace_ids]
    n_traces = len(trace_ids)
    n_loci = len(loci)
    thresholds = [33, 50, 55, 60, n_loci]
    threshold_counts = {
        str(threshold): {
            "graphs": sum(count >= threshold for count in counts),
            "fraction": sum(count >= threshold for count in counts) / n_traces,
        }
        for threshold in thresholds
    }
    detection_by_locus = {
        locus: locus_observed[locus] / n_traces for locus in sorted(loci)
    }
    sorted_detection = sorted(detection_by_locus.items(), key=lambda item: item[1])

    return {
        "label": label,
        "path": str(path.resolve()),
        "rows": row_count,
        "graphs": n_traces,
        "possible_loci": n_loci,
        "locus_range": [min(loci), max(loci)],
        "observed_coordinate_rows": observed_rows,
        "overall_detection_fraction": observed_rows / (n_traces * n_loci),
        "observed_nodes_per_graph": {
            "mean": mean(counts),
            "median": median(counts),
            "min": min(counts),
            "max": max(counts),
        },
        "coverage_thresholds": threshold_counts,
        "coverage_histogram": dict(sorted(Counter(counts).items())),
        "per_locus_detection": {
            "min": sorted_detection[0],
            "median": median(detection_by_locus.values()),
            "max": sorted_detection[-1],
            "five_lowest": sorted_detection[:5],
            "five_highest": sorted_detection[-5:],
        },
        "duplicate_trace_locus_rows": sum(
            count - 1 for count in seen_pairs.values() if count > 1
        ),
        "traces_with_unexpected_row_count": sum(
            rows_by_trace[trace_id] != n_loci for trace_id in trace_ids
        ),
        "_observed_loci_by_trace": {
            trace_id: observed_by_trace[trace_id] for trace_id in trace_ids
        },
        "_detection_by_locus": detection_by_locus,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "datasets",
        nargs="+",
        metavar="LABEL=CSV",
        help="Dataset label and CSV path, for example untreated=data/file.csv",
    )
    args = parser.parse_args()

    results = []
    for specification in args.datasets:
        label, separator, path_text = specification.partition("=")
        if not separator:
            parser.error(f"Expected LABEL=CSV, got {specification!r}")
        results.append(audit(Path(path_text), label))

    combined: dict[str, object] = {}
    if len(results) >= 2:
        common_loci = set.intersection(
            *(set(result["_detection_by_locus"]) for result in results)
        )
        common_by_threshold = {}
        for threshold in (0.80, 0.90, 0.95):
            retained = sorted(
                locus
                for locus in common_loci
                if all(
                    result["_detection_by_locus"][locus] >= threshold
                    for result in results
                )
            )
            common_by_threshold[str(threshold)] = {
                "count": len(retained),
                "loci": retained,
            }

        stable_core = set(common_by_threshold["0.8"]["loci"])
        stable_core_stats = {}
        for result in results:
            restricted_counts = [
                len(observed & stable_core)
                for observed in result["_observed_loci_by_trace"].values()
            ]
            stable_core_stats[result["label"]] = {
                "possible_loci": len(stable_core),
                "mean_observed": mean(restricted_counts),
                "median_observed": median(restricted_counts),
                "min_observed": min(restricted_counts),
                "full_core_graphs": sum(
                    count == len(stable_core) for count in restricted_counts
                ),
                "full_core_fraction": sum(
                    count == len(stable_core) for count in restricted_counts
                )
                / len(restricted_counts),
            }

        graph_counts = [result["graphs"] for result in results]
        combined = {
            "total_graphs": sum(graph_counts),
            "largest_to_smallest_class_ratio": max(graph_counts) / min(graph_counts),
            "common_loci_by_minimum_detection_in_every_class": common_by_threshold,
            "stable_80_percent_core": stable_core_stats,
        }

    for result in results:
        result.pop("_observed_loci_by_trace")
        result.pop("_detection_by_locus")

    print(json.dumps({"datasets": results, "combined": combined}, indent=2))


if __name__ == "__main__":
    main()
