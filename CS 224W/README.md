# Interpretable chromatin GAT

This project classifies individual chromosome-6 Hoxa-region traces as
untreated or CTCF-depleted using the Hafner et al. 4DN chromatin-tracing data.

## Preliminary model

- Retain traces containing at least 37 of the 73 canonical loci.
- Create an edge for a non-genomically-adjacent locus pair closer than 238 nm.
- Node features: paper CTCF-boundary-site indicator and normalized contact
  degree.
- Edge feature: whether the contact spans an intervening paper CTCF boundary
  site.
- Two single-head, edge-aware GAT layers.
- No virtual node during message passing.
- Final attention readout followed by one scalar logit. The logit is exactly
  the graph bias plus the sum of signed node contributions.

The current result uses a preliminary random 70/15/15 split stratified by
condition and replicate. It should not be presented as a final generalization
estimate; the final report should use replicate-held-out evaluation.

## Run

```bash
.venv/bin/python scripts/train_interpretable_gat.py \
  --epochs 60 \
  --seeds 11 22 33 \
  --batch-size 256 \
  --output results/interpretable_gat_no_distance_preliminary.json
```

The script saves one small checkpoint per seed in `models/` and a complete
metrics and locus-contribution report in `results/`.

