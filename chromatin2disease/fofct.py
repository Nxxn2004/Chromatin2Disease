"""Reader for 4DN FOF-CT core tables: one row per spot (Spot_ID, Trace_ID, X, Y, Z, [Readout], Chrom, Start, End, ...).

The 4DN files vary: comment lines may be quoted or end in commas, some repeat a header row, and the ``##columns``
line is not always right. So columns are found from the data: X/Y/Z are fields 3-5 and the genomic interval follows
the first ``chr*`` field. Coordinates are in nm here, although the headers say micron.
"""
import pandas as pd


def read_core_table(path):
    with open(path) as f:
        rows = [line.strip().rstrip(",").split(",") for line in f
                if line.strip() and not line.lstrip('"').startswith("#")]
    if not rows[0][0].isdigit():  # header row such as "spotID,traceID,x,y,z,..."
        rows = rows[1:]
    c = next(i for i, v in enumerate(rows[0]) if v.startswith("chr"))
    df = pd.DataFrame([[r[1], r[2], r[3], r[4], r[c + 1], r[c + 2]] for r in rows],
                      columns=["trace_id", "x", "y", "z", "start", "end"])
    return df.astype({"x": float, "y": float, "z": float, "start": int, "end": int})
