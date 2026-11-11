"""Datasets and labels.

axin2  Llimos et al. 2022: label 1 = leukemia-protective variant, 0 = reference (AXIN2 locus, chr17).
ctcf   Hafner et al. 2023: label 1 = CTCF-AID + auxin (CTCF degraded), 0 = untreated (chr3 and chr6 loci).

Each task has a manifest ``metadata/<task>_manifest.tsv`` (one row per 4DN file). Files live at
``data/<task>-data/<region>/<condition>/<experiment>/<file>.csv``; fetch them with ``scripts/download_data.sh``.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TASKS = ["axin2", "ctcf"]  # index = output head
DATASETS = ["axin2", "ctcf", "ctcf-chr3", "ctcf-chr6", "all"]


def load_sources(dataset):
    """Manifest rows of the core tables (spot coordinates) used by ``dataset``, plus task, locus and path."""
    tasks = TASKS if dataset == "all" else [dataset.split("-")[0]]
    m = pd.concat([pd.read_csv(ROOT / "metadata" / f"{t}_manifest.tsv", sep="\t", dtype={"replicate": str})
                   .assign(task=t) for t in tasks], ignore_index=True)
    m = m[m.file_type == "core"]
    if "-" in dataset:
        m = m[m.region == dataset.split("-")[1]]
    m = m.assign(locus=m.task + "_" + m.region,
                 path=[str(ROOT / "data" / f"{r.task}-data" / r.region / r.condition / r.experiment / f"{r.file}.csv")
                       for r in m.itertuples()])
    missing = [p for p in m.path if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} files missing, e.g. {missing[0]}; run scripts/download_data.sh")
    return m.reset_index(drop=True)
