# Chromatin2Disease
Predicting cell states via GNN-based classifier on chromatin-tracing data. This project is initially built as a class project for Stanford CS224W (Fall 2026).

## Background

Inside the nucleus, DNA and its packaging histone proteins form chromatin, which folds into loops
and domains that help decide which genes are switched on. Chromatin tracing images this folding directly in single
cells. Methods such as [ORCA](https://doi.org/10.1038/s41586-019-1035-4) (optical reconstruction of chromatin architecture) label a run of consecutive genomic
segments, here 8 kb or 30 kb each, image them one after another, and record the 3D position of every segment
on every chromosome. Each traced chromosome is therefore a chain of 3D points, which differs strongly from cell to
cell and maps naturally onto a graph.

The datasets used in this project come from the [4D Nucleome Data Portal](https://doi.org/10.1038/s41467-022-29697-4). The [4D Nucleome Program](https://commonfund.nih.gov/4DNucleome) studies cellular nucleus organization in 3D space and its temporal changes, hence the fourth dimension. The data portal hosts Hi-C, imaging and chromatin tracing data in standardized formats. Tracing data use the FISH-Omics Format for Chromatin Tracing (FOF-CT), whose core table lists one imaged spot information including its trace, 3D coordinates and genomic interval in each row.

This project will classify FOF-CT chromatin traces from two genomic perturbations:

- **AXIN2**, which lies on human chromosome 17, is a negative regulator of Wnt/β-catenin signalling and thus acts as a tumor suppressor. Llimos et al. (2022) studied a
common germline 5-bp insertion/deletion near AXIN2 that is associated with a lower risk of chronic lymphocytic
leukemia (CLL), a form of blood cancer, which we will term the `leuk-protect` variant. The variant creates a binding site for the transcription factor MEF2, triggers a cluster of
enhancers spanning more than 150 kb, mildly compacts the gene locus in 3D and raises AXIN2 expression. Chromatin traces of AXIN2 locus in lymphoblastoid cell lines, with and without the variant, are used in our `axin2` task.

- **CTCF** is a DNA-binding protein that marks the boundaries of chromatin domains. It stops another ring-shaped protein complex called cohesin from extruding chromatin loops further, and such loops are anchored at CTCF sites. Hafner et al. (2023) tagged CTCF with
an auxin-inducible degron (AID) in mouse embryonic stem cells, so that adding auxin removes CTCF proteins within
hours. Chromatin traces around the *Sox2* (chromosome 3) and *Hoxa* (chromosome 6) domains in untreated and auxin-treated cells reveal that CTCF promotes chromatin loop stacking at the domain boundary. These tracing data are used in our `ctcf` task.

Our project asks whether we can classify theses cell variants based only on given chromatin tracing data.

## Overview

Each traced chromosome is turned into a graph, with imaged DNA segments as nodes and the
chromatin backbone plus 3D contacts as edges. A model based on Graph Attention Network then classifies the trace. Two tasks are supported, separately or jointly:

| task | data | label 1 | label 0 | locus |
|---|---|---|---|---|
| `axin2` | Llimos et al. 2022, *Nat Commun* ([doi](https://doi.org/10.1038/s41467-022-29625-6)) | leukemia-protective variant (`leuk-protect`) | reference (`non-protect`) | AXIN2, chr17, 25 × 8 kb |
| `ctcf` | Hafner et al. 2023, *Mol Cell* ([doi](https://doi.org/10.1016/j.molcel.2023.04.008)) | CTCF-AID + 4 h auxin (CTCF degraded) | CTCF-AID untreated | Sox2 domain chr3 (93 × 30 kb), Hoxa domain chr6 (73 × 30 kb) |

## Setup

```bash
conda env create -f environment.yml
conda activate chromatin2disease
```

## Data

```
data/                                                        # fetched by scripts/download_data.sh
  axin2-data/<region>/<condition>/<experiment>/<file>.csv
  ctcf-data/<region>/<condition>/<experiment>/<file>.csv
metadata/
  axin2_manifest.tsv    one row per 4DN file
  ctcf_manifest.tsv
```

Both manifests share the same columns: `region`, `condition`, `label`, `label_name`, `replicate`, `set`,
`experiment`, `file`, `file_type`, `md5`, `bytes`, `s3_key` and `description`. The loader in
`chromatin2disease/sources.py` reads only the rows whose `file_type` is `core`, which hold the spot coordinates.

The 4DN portal was unreachable, so both datasets are fetched the same way. The script reads each manifest, copies
the files from the public 4DN S3 bucket with the AWS CLI, and checks every md5 sum against the manifest. No AWS
account is needed. Files already present with the right md5 sum are skipped.

```bash
bash scripts/download_data.sh            # both datasets
bash scripts/download_data.sh axin2      # ~7 MB into data/axin2-data/
bash scripts/download_data.sh ctcf       # ~140 MB into data/ctcf-data/
```

**AXIN2 (Llimos et al., 2022)** has 4 core table files:

| condition | chromosome | label | replicate | experiment | core table |
|---|---|---|---|---|---|
| leuk-protect | chr17 | 1 | rep1 | 4DNEXU2FYC2S | 4DNFIG51R1QG |
| leuk-protect | chr17 | 1 | rep2 | 4DNEX4L5AR91 | 4DNFIIHHT16W |
| non-protect | chr17 | 0 | rep1 | 4DNEX1MX8Y7L | 4DNFIZ37CGLR |
| non-protect | chr17 | 0 | rep2 | 4DNEX5ICQCRY | 4DNFIXSTIVTW |

**CTCF (Hafner et al., 2023)** has 12 core table files:

| condition | chromosome | label | replicate | experiment | core table |
|---|---|---|---|---|---|
| auxin | chr3 | 1 | rep1 | 4DNEX5L4MPMG | 4DNFILD6TDR6 |
| auxin | chr3 | 1 | rep2 | 4DNEXOLS8BYS | 4DNFIKXR227H |
| auxin | chr3 | 1 | rep3 | 4DNEXJ215XWF | 4DNFIW3V9CRO |
| untreated | chr3 | 0 | rep1 | 4DNEXZU2N5SB | 4DNFI2OG3IPB |
| untreated | chr3 | 0 | rep2 | 4DNEX84R1VFO | 4DNFIU5X9MXT |
| untreated | chr3 | 0 | rep3 | 4DNEXRMBZE8V | 4DNFIIQ8YIG4 |
| auxin | chr6 | 1 | rep1 | 4DNEXZFUSZYX | 4DNFIIOVHG1I |
| auxin | chr6 | 1 | rep2 | 4DNEXGPCGGM9 | 4DNFIJYFR8FT |
| auxin | chr6 | 1 | rep3 | 4DNEXERBJAP8 | 4DNFI4VLY3WU |
| untreated | chr6 | 0 | rep1 | 4DNEX69R54RT | 4DNFIKVZTN9K |
| untreated | chr6 | 0 | rep2 | 4DNEXTZDK167 | 4DNFIZWUQTXF |
| untreated | chr6 | 0 | rep3 | 4DNEXMW5AM74 | 4DNFID7HZ8LX |

The CTCF replicate numbers are inferred, not official 4DN metadata. Each auxin file is paired with the untreated
file from the same multiplexed imaging run.

Coordinates in both datasets are inferred to be in nanometres.

## Graph construction (`chromatin2disease/graphs.py`)

* **Nodes** are the segments detected in one trace. Only segments imaged in every file of a locus are kept,
  so a replicate-specific probe grid cannot act as a batch signature.
* **Node features** are the genomic position in the locus (0 to 1) and the distance to the trace centroid. The
  centred xyz coordinates are kept separately and randomly rotated during training. The model adds a learned
  embedding of the segment identity.
* **Edges** connect consecutive observed segments (the backbone) and any pair closer than the contact
  threshold. As in Hafner et al., the threshold is the typical distance between genomically adjacent
  segments. The median gives 251 nm on chr3 and 222 nm on chr6 for untreated cells, close to the
  paper's 256 nm and 238 nm. Use `--contact-scale` to test other thresholds.
* **Edges carry no features.** Distances enter the model only through the edge set and the node coordinates.
* **Filters**: a trace needs at least 4 segments and at least 30% coverage. Spots more than 4 median radii of
  gyration from the trace median are dropped as outliers. These are fixed defaults in `build_graphs`.

## Model (`chromatin2disease/model.py`)



```
h_v^(k) = σ( MSG_self(h_v^(k-1)) + Σ_{u∈N(v)} α_vu MSG(h_u^(k-1)) ),   α from single-head GATConv
h_G     = σ( Σ_v softmax_v(Linear(h_v^(n))) · MSG(h_v^(n)) )            virtual-node readout
ŷ       = sigmoid(w_t^T h_G + b_t)                                        one head per task
```

The whole model is one short class. The node input is a linear encoding of the node features plus a
learned embedding of the segment identity. Each layer is a single-head `GATConv` without edge features,
added to a linear self-skip and passed through ELU. `--layers 1` gives the interpretable single-layer model,
and `--layers 0` a no-message-passing baseline. Training uses a class-balanced BCE loss and AdamW, with early
stopping on validation ROC-AUC.

Two flags give simpler variants for comparison:

* `--act {elu,leaky_relu,relu,sigmoid}` sets the non-linearity σ after each GAT layer. The readout keeps ELU.
* `--no-attn-leaky` removes the LeakyReLU from the attention score, α_vu = softmax_u(a · [W h_v ‖ W h_u]).
  The score is then linear, so the target node's term cancels in the softmax. A neighbour then gets the same
  score whichever node attends to it, before normalisation over that node's neighbours.

## Usage

```bash
# CTCF degraded vs untreated, both loci
python -m chromatin2disease.train --dataset ctcf

# leukemia-protective variant vs reference
python -m chromatin2disease.train --dataset axin2

# both tasks with one shared encoder
python -m chromatin2disease.train --dataset all

# single locus, single-layer interpretable model
python -m chromatin2disease.train --dataset ctcf-chr6 --layers 1

# simpler variants: ReLU after each layer, no LeakyReLU inside the attention score
python -m chromatin2disease.train --dataset ctcf --act relu --no-attn-leaky

# test on an imaging replicate never seen in training
python -m chromatin2disease.train --dataset ctcf --split replicate --test-replicate rep3

# non-graph reference: logistic regression on the pairwise distance map
python -m chromatin2disease.baseline --dataset ctcf --out runs/baseline_ctcf

# attention summaries per segment and class for a finished run
python -m chromatin2disease.interpret runs/<run_dir>

pytest -q tests
```

Each training run writes `config.json`, `best.pt`, `metrics.json` and `test_predictions.csv` to `runs/<name>/`.
The baseline writes `config.json` and `metrics.json`. `interpret` adds `attention_<locus>.csv` and `.png` to a
training run. Built graphs are cached in `cache/`, keyed by the input files (relative to the repo) and the contact
scale, so delete `cache/` after changing `graphs.py`.

## Results

All values are ROC-AUC per locus, from a single seed (0) on CPU. Each cell gives **train / val / test**, and the
test value is the one to compare. Training uses the default flags unless a row says otherwise.

| flag | default | meaning |
|---|---|---|
| `--layers` | 3 | number of GAT layers (1 = interpretable model) |
| `--hidden` | 64 | hidden size |
| `--act` | `elu` | non-linearity after each GAT layer |
| `--no-attn-leaky` | off | drop the LeakyReLU from the attention score |
| `--dropout` | 0.2 | dropout after each layer and before the heads |
| `--lr`, `--batch-size` | 1e-3, 64 | AdamW learning rate (weight decay 1e-4) and batch size |
| `--epochs`, `--patience` | 60, 12 | maximum epochs, and early-stopping patience on mean validation ROC-AUC |
| `--seed` | 0 | seed for the split and the weights |

### Splits

| split | flags | how traces are divided |
|---|---|---|
| random | `--split random` (default) | stratified by task, locus and label into 70 / 15 / 15% train / val / test |
| replicate, CTCF | `--split replicate --test-replicate rep3` | **rep3 is held out** as the test set. Train and val are drawn 85 / 15% from rep1 and rep2. |
| replicate, AXIN2 | `--split replicate --test-replicate rep1` | **rep1 is held out** as the test set. Train and val are drawn 85 / 15% from rep2. |

`--test-replicate` defaults to `rep2`, so always pass it with `--split replicate`. Graph counts per split:

| `--dataset` | trained on | split | train / val / test graphs |
|---|---|---|---|
| `ctcf` | CTCF chr3 + chr6, one CTCF head | random | 20,262 / 4,342 / 4,342 |
| `axin2` | AXIN2 chr17, one AXIN2 head | random | 3,252 / 697 / 698 |
| `all` | CTCF + AXIN2, shared encoder, one head per task | random | 23,515 / 5,039 / 5,039 |
| `ctcf` | CTCF chr3 + chr6 | replicate, test rep3 | 15,820 / 2,792 / 10,334 |
| `axin2` | AXIN2 chr17 | replicate, test rep1 | 3,243 / 573 / 831 |

### Random split

| model | command | CTCF chr3 | CTCF chr6 | AXIN2 |
|---|---|---|---|---|
| distance-map logistic regression | `baseline --dataset ctcf` / `--dataset axin2` | 0.968 / 0.617 / 0.620 | 0.833 / 0.596 / 0.576 | 0.684 / 0.567 / 0.518 |
| GAT, 1 layer | `train --dataset ctcf --layers 1` / `--dataset axin2 --layers 1` | 0.747 / 0.687 / 0.657 | 0.733 / 0.682 / 0.646 | 0.593 / 0.541 / 0.525 |
| GAT, 3 layers | `train --dataset ctcf` / `--dataset axin2` | 0.792 / 0.698 / 0.676 | 0.785 / 0.673 / 0.652 | 0.708 / 0.566 / 0.538 |
| GAT, 3 layers, joint | `train --dataset all` | 0.780 / 0.671 / **0.705** | 0.773 / 0.651 / **0.674** | 0.695 / 0.587 / **0.543** |

### Replicate split (CTCF tests on rep3, AXIN2 tests on rep1)

| model | command | CTCF chr3 | CTCF chr6 | AXIN2 |
|---|---|---|---|---|
| distance-map logistic regression | `baseline --dataset ctcf --split replicate --test-replicate rep3` / `--dataset axin2 ... rep1` | 0.999 / 0.648 / 0.569 | 0.865 / 0.616 / 0.527 | 0.682 / 0.544 / 0.515 |
| GAT, 3 layers | `train --dataset ctcf --split replicate --test-replicate rep3` / `--dataset axin2 ... rep1` | 0.854 / 0.766 / **0.586** | 0.821 / 0.713 / **0.535** | 0.696 / 0.616 / **0.524** |

### Simpler variants (3 layers, random split, each task trained on its own dataset)

| variant | command | CTCF chr3 | CTCF chr6 | AXIN2 |
|---|---|---|---|---|
| default: ELU, LeakyReLU in attention | `train --dataset ctcf` / `--dataset axin2` | 0.792 / 0.698 / 0.676 | 0.785 / 0.673 / 0.652 | 0.708 / 0.566 / 0.538 |
| LeakyReLU after each layer | add `--act leaky_relu` | 0.773 / 0.708 / **0.684** | 0.766 / 0.670 / 0.657 | 0.599 / 0.550 / 0.547 |
| ReLU after each layer | add `--act relu` | 0.740 / 0.675 / 0.653 | 0.743 / 0.665 / 0.651 | 0.636 / 0.563 / **0.559** |
| sigmoid after each layer | add `--act sigmoid` | 0.739 / 0.653 / 0.645 | 0.740 / 0.671 / 0.650 | 0.603 / 0.547 / 0.523 |
| no LeakyReLU in attention | add `--no-attn-leaky` | 0.768 / 0.683 / 0.668 | 0.759 / 0.673 / **0.659** | 0.603 / 0.564 / 0.528 |

Commands above omit `python -m chromatin2disease.`. To reproduce every run, with outputs named as in `runs/`:

```bash
T="python -m chromatin2disease.train"; B="python -m chromatin2disease.baseline"
$B --dataset ctcf  --out runs/baseline_ctcf;   $B --dataset ctcf  --split replicate --test-replicate rep3 --out runs/baseline_ctcf_rep3
$B --dataset axin2 --out runs/baseline_axin2;  $B --dataset axin2 --split replicate --test-replicate rep1 --out runs/baseline_axin2_rep1
$T --dataset ctcf  --out runs/ctcf_gat3;   $T --dataset ctcf  --layers 1 --out runs/ctcf_gat1
$T --dataset axin2 --out runs/axin2_gat3;  $T --dataset axin2 --layers 1 --out runs/axin2_gat1
$T --dataset all   --out runs/all_gat3
$T --dataset ctcf  --split replicate --test-replicate rep3 --out runs/ctcf_gat3_rep3
$T --dataset axin2 --split replicate --test-replicate rep1 --out runs/axin2_gat3_rep1
for d in ctcf axin2; do
  for a in leaky_relu relu sigmoid; do $T --dataset $d --act $a --out runs/${d}_gat3_${a/_relu/}; done
  $T --dataset $d --no-attn-leaky --out runs/${d}_gat3_noattnleaky
done
```

Every run folder in `runs/`, baselines included, holds `config.json` with the exact flags and `metrics.json` with the
train, val and test ROC-AUC per task and locus. The tables above are read from those files.

How to read these numbers:

* **The GAT beats the non-graph baseline under the random split.** On CTCF it gains about 0.06 to 0.08 test ROC-AUC.
* **Joint training helps CTCF.** Sharing the encoder with the AXIN2 task gives the best CTCF scores.
* **The variants differ by at most about 0.04, which is within what one seed can resolve.** Sigmoid is the
  weakest on every locus. Removing the attention LeakyReLU barely changes CTCF. Repeat with several seeds
  before drawing conclusions.
* **Performance drops on a held-out replicate,** from about 0.66 to 0.56 on CTCF, and AXIN2 is near chance. Validation
  traces come from the training replicates, so validation ROC-AUC overestimates the held-out score. Part of the
  random-split signal is therefore specific to each imaging run. Report the replicate split as the honest
  generalisation estimate, and consider per-replicate normalisation or training across more replicates.
* **The baseline overfits heavily.** Its train ROC-AUC reaches 0.97 to 1.00 on chr3, against about 0.6 on val and test.
* **Single-chromosome classification is hard.** CTCF loss changes loop frequencies at the population level, but individual
  chromosomes are very heterogeneous. AXIN2 traces also detect only about 10 of 25 segments on average.
* **Larger models gained little.** An earlier model with layer normalisation, residual connections, a scheduler and
  gradient clipping scored 0.005 to 0.03 above a 4-head model with edge features. That model in turn was within
  about 0.025 of the current single-head model, higher on most loci.

Attention maps for the 1-layer CTCF model are produced with `python -m chromatin2disease.interpret runs/ctcf_gat1`.
Readout attention concentrates on a few segments, for example chr6 around 51.4 Mb and 52.7 Mb, and is similar in both
classes. The classes differ mainly in what the model learns at those segments, not in where it looks.
