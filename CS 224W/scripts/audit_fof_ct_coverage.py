#!/usr/bin/env python3
"""Audit sparse FOF-CT core tables by trace and genomic locus."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median


def read_fof(path: Path) -> tuple[list[str], list[dict[str, str]], int]:
    columns: list[str] | None = None
    rows: list[dict[str, str]] = []
    malformed = 0
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        for values in reader:
            if not values:
                continue
            first = values[0]
            if first.startswith("##columns=("):
                declaration = first.removeprefix("##columns=(")
                declaration += "," + ",".join(values[1:])
                declaration = declaration.rstrip(")")
                columns = [value.strip() for value in declaration.split(",")]
                continue
            if first.startswith("#"):
                continue
            if columns is None:
                raise ValueError(f"No ##columns declaration before data in {path}")
            if len(values) != len(columns):
                malformed += 1
                continue
            rows.append(dict(zip(columns, values)))
    if columns is None:
        raise ValueError(f"No ##columns declaration in {path}")
    return columns, rows, malformed


def finite_xyz(row: dict[str, str]) -> bool:
    try:
        return all(math.isfinite(float(row[axis])) for axis in ("X", "Y", "Z"))
    except (KeyError, ValueError):
        return False


def audit(
    path: Path,
    condition: str,
    replicate: str,
    canonical_loci: set[int] | None = None,
) -> dict[str, object]:
    columns, rows, malformed = read_fof(path)
    required = {"Trace_ID", "X", "Y", "Z", "Chrom", "Chrom_Start", "Chrom_End"}
    if not required.issubset(columns):
        raise ValueError(f"Missing required columns in {path}: {required - set(columns)}")

    observed_by_trace: dict[int, set[int]] = defaultdict(set)
    locus_observed: Counter[int] = Counter()
    pair_counts: Counter[tuple[int, int]] = Counter()
    loci: set[int] = set(canonical_loci or ())
    all_trace_ids = {int(row["Trace_ID"]) for row in rows}
    invalid_coordinate_rows = 0
    chrom_values: Counter[str] = Counter()
    coordinate_abs_max = 0.0

    for row in rows:
        trace_id = int(row["Trace_ID"])
        locus = int(row["Chrom_Start"])
        if canonical_loci is not None and locus not in canonical_loci:
            continue
        loci.add(locus)
        chrom_values[row["Chrom"]] += 1
        pair_counts[(trace_id, locus)] += 1
        if finite_xyz(row):
            observed_by_trace[trace_id].add(locus)
            locus_observed[locus] += 1
            coordinate_abs_max = max(
                coordinate_abs_max,
                *(abs(float(row[axis])) for axis in ("X", "Y", "Z")),
            )
        else:
            invalid_coordinate_rows += 1

    trace_ids = sorted(all_trace_ids)
    counts = [len(observed_by_trace[trace_id]) for trace_id in trace_ids]
    n_traces = len(trace_ids)
    n_loci = len(loci)
    thresholds = sorted({37, 50, 55, 60, 65, 70, 73, n_loci})
    detection = {locus: locus_observed[locus] / n_traces for locus in sorted(loci)}
    sorted_detection = sorted(detection.items(), key=lambda item: item[1])

    return {
        "condition": condition,
        "replicate": replicate,
        "path": str(path.resolve()),
        "columns": columns,
        "rows": len(rows),
        "malformed_rows": malformed,
        "invalid_coordinate_rows": invalid_coordinate_rows,
        "graphs": n_traces,
        "possible_loci_present_in_file": n_loci,
        "chromosomes": dict(chrom_values),
        "coordinate_abs_max": coordinate_abs_max,
        "observed_nodes_per_graph": {
            "mean": mean(counts),
            "median": median(counts),
            "min": min(counts),
            "max": max(counts),
        },
        "coverage_thresholds": {
            str(threshold): {
                "graphs": sum(count >= threshold for count in counts),
                "fraction": sum(count >= threshold for count in counts) / n_traces,
            }
            for threshold in thresholds
        },
        "coverage_histogram": dict(sorted(Counter(counts).items())),
        "per_locus_detection": {
            "min": sorted_detection[0],
            "median": median(detection.values()),
            "max": sorted_detection[-1],
            "five_lowest": sorted_detection[:5],
            "five_highest": sorted_detection[-5:],
        },
        "duplicate_trace_locus_rows": sum(
            count - 1 for count in pair_counts.values() if count > 1
        ),
        "traces_with_duplicate_locus": len(
            {trace_id for (trace_id, _), count in pair_counts.items() if count > 1}
        ),
        "_counts": counts,
        "_loci": loci,
        "_observed_by_trace": observed_by_trace,
        "_detection": detection,
    }


def summarize_group(results: list[dict[str, object]]) -> dict[str, object]:
    all_counts = [count for result in results for count in result["_counts"]]
    all_loci = set.union(*(result["_loci"] for result in results))
    n_graphs = len(all_counts)
    total_observed = sum(all_counts)
    return {
        "graphs": n_graphs,
        "possible_loci_union": len(all_loci),
        "overall_detection_fraction": total_observed / (n_graphs * len(all_loci)),
        "observed_nodes_per_graph": {
            "mean": mean(all_counts),
            "median": median(all_counts),
            "min": min(all_counts),
            "max": max(all_counts),
        },
        "coverage_thresholds": {
            str(threshold): {
                "graphs": sum(count >= threshold for count in all_counts),
                "fraction": sum(count >= threshold for count in all_counts) / n_graphs,
            }
            for threshold in (37, 50, 55, 60, 65, 70, 73)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "datasets",
        nargs="+",
        metavar="CONDITION:REPLICATE=CSV",
    )
    parser.add_argument("--locus-min", type=int)
    parser.add_argument("--locus-max", type=int)
    parser.add_argument("--locus-step", type=int, default=1)
    args = parser.parse_args()

    canonical_loci = None
    if args.locus_min is not None or args.locus_max is not None:
        if args.locus_min is None or args.locus_max is None:
            parser.error("--locus-min and --locus-max must be supplied together")
        canonical_loci = set(
            range(args.locus_min, args.locus_max + args.locus_step, args.locus_step)
        )

    results = []
    for specification in args.datasets:
        label, separator, path_text = specification.partition("=")
        if not separator or ":" not in label:
            parser.error(f"Expected CONDITION:REPLICATE=CSV, got {specification!r}")
        condition, replicate = label.split(":", 1)
        results.append(audit(Path(path_text), condition, replicate, canonical_loci))

    by_condition: dict[str, list[dict[str, object]]] = defaultdict(list)
    for result in results:
        by_condition[result["condition"]].append(result)
    condition_summaries = {
        condition: summarize_group(group) for condition, group in by_condition.items()
    }

    all_loci = set.union(*(result["_loci"] for result in results))
    stable_loci = {}
    for threshold in (0.5, 0.8, 0.9):
        retained = sorted(
            locus
            for locus in all_loci
            if all(result["_detection"].get(locus, 0.0) >= threshold for result in results)
        )
        stable_loci[str(threshold)] = {"count": len(retained), "loci": retained}

    graph_counts = [summary["graphs"] for summary in condition_summaries.values()]
    combined = {
        "graphs": sum(graph_counts),
        "possible_loci_union": len(all_loci),
        "largest_to_smallest_condition_ratio": max(graph_counts) / min(graph_counts),
        "stable_loci_minimum_detection_in_every_replicate": stable_loci,
    }

    for result in results:
        for key in ("_counts", "_loci", "_observed_by_trace", "_detection"):
            result.pop(key)

    print(
        json.dumps(
            {
                "datasets": results,
                "conditions": condition_summaries,
                "combined": combined,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
