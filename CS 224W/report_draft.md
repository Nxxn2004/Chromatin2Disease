# Interpretable Graph Attention Network for CTCF Perturbation Classification

*Report draft*

## Problem description and motivation

The three-dimensional organization of chromatin affects which genomic regions can interact and is therefore an important component of gene regulation. Chromatin-tracing experiments measure the 3D positions of selected genomic loci in individual chromosome copies, providing a direct view of structural variation across cells. These measurements can be represented naturally as graphs: nodes correspond to genomic loci, and edges correspond to pairs of nonadjacent loci that are sufficiently close in 3D space to be considered in contact.

This project studies a binary graph-classification problem: given a single-chromosome contact graph from a mouse embryonic stem cell, predict whether the cell was untreated or underwent four hours of auxin treatment to deplete CTCF. This is a perturbation-classification task, not a healthy-versus-diseased classification task. CTCF is a DNA-binding protein involved in chromatin loop organization and topologically associating domain boundaries.

The project is motivated by the findings of Hafner et al. (2023), who showed that CTCF depletion does not simply eliminate all chromatin contacts. Instead, it redistributes contacts: some contacts within TADs are lost, some contacts between TADs are gained, and multiway contacts among CTCF boundary sites become less common. The researchers also observed transient, rosette-like structures in which several CTCF-associated loop anchors form a central hub. These results suggest that an effective classifier must learn where contacts occur and how they are organized, rather than relying only on the total number of edges.

The primary goal is interpretability rather than maximizing predictive performance. We ask whether a deliberately small Graph Attention Network can classify the perturbation while producing signed, locus-level explanations that can be compared with the mechanisms reported by the original study. In particular, we want to determine whether the model uses CTCF-associated loci and boundary-spanning contacts when distinguishing untreated from CTCF-depleted chromosomes.

## Dataset description and processing

We use processed Optical Reconstruction of Chromatin Architecture data deposited by Hafner et al. in the 4D Nucleome Data Portal. The selected experiment traces a chromosome 6 region containing the Hoxa locus in CTCF-AID mouse embryonic stem cells. Three untreated files and three four-hour auxin-treated files provide three paired experimental replicates. The public data use the GRCm38 mouse genome assembly and contain a trace identifier, genomic interval, and measured X, Y, and Z position for each detected locus.

The analysis focuses on 73 canonical loci spanning approximately 2.2 Mb of chromosome 6. Because not every locus is detected in every trace, graphs have variable numbers of nodes. Following the original paper's quality-control criterion, we retain traces with at least 37 observed loci, corresponding to at least 50% coverage. This produces 11,488 graphs: 5,283 untreated and 6,205 CTCF-depleted. The class ratio is approximately 1.17 to 1.

Each retained trace is converted into a contact graph as follows:

1. Create one node for every observed canonical locus. Missing loci are omitted rather than represented by dummy nodes.
2. Compute the Euclidean distance between every pair of observed loci.
3. Create an undirected contact edge when two loci are less than 238 nm apart, using the chromosome 6 threshold reported by Hafner et al.
4. Exclude pairs of genomically adjacent loci, even when they are spatially close. This ensures that an edge represents a nontrivial 3D contact rather than the ordinary continuity of the chromatin polymer.
5. Represent each undirected contact in both directions for message passing. Self-loops are added internally by the GAT layers but are not treated as biological contacts.

The current node feature vector contains two values. The first is an indicator for whether the locus is one of the five chromosome 6 CTCF boundary sites used in the paper's hub analysis, corresponding to original readouts 15, 23, 37, 46, and 57. The second is normalized contact degree,

\[
\widetilde d_i = \frac{d_i}{n-1},
\]

where \(d_i\) is the node's contact degree and \(n\) is the number of observed nodes in the graph.

Each edge has one binary feature indicating whether the contact spans at least one intervening paper CTCF boundary site. Continuous 3D distance is not included because the edge threshold already converts distance into a contact decision, and the paper's main CTCF findings concern contact organization rather than variation in distance among contacts. The boundary-spanning feature is a reproducible proxy based on the five reported CTCF boundary sites; it is not an authoritative cross-TAD label because the CSV files do not contain per-locus TAD assignments.

For the current preliminary experiment, graphs are randomly divided into 70% training, 15% validation, and 15% test sets while stratifying by condition and replicate. This gives 8,041 training graphs, 1,723 validation graphs, and 1,724 test graphs. Replicates are mixed across the three splits, so these results measure performance on held-out traces from familiar experiments rather than generalization to a new experimental replicate. The final evaluation should replace this split with replicate-held-out testing.

## Model design and metrics

The classifier contains two single-head, edge-aware Graph Attention Network layers with hidden dimension eight. Only biological contact edges participate in these message-passing layers. The final graph-level readout is applied after message passing, so a virtual readout does not create shortcuts between loci during the two contact-message-passing rounds.

For an edge from node \(j\) to node \(i\), the GAT constructs an attention logit from the endpoint representations and the edge feature:

\[
q_{ij} = a_s^\top W_s h_i + a_t^\top W_t h_j + a_e^\top W_e e_{ij}.
\]

The logits are normalized across the incoming contacts of node \(i\):

\[
\alpha_{ij} =
\frac{\exp(\operatorname{LeakyReLU}(q_{ij}))}
{\sum_{k\in N(i)}\exp(\operatorname{LeakyReLU}(q_{ik}))}.
\]

The updated representation is a weighted sum of neighboring messages. Two layers allow a locus to represent both its direct contacts and the contact environments of those neighbors. This is relevant to the paper's multiway-contact analysis, in which three loci can form a two-edge pattern through a shared anchor.

After the second GAT layer, the model assigns every node a readout-attention weight \(\gamma_i\) and a signed scalar score \(s_i\). The graph logit is

\[
z = b + \sum_i \gamma_i s_i.
\]

The CTCF-depleted condition is the positive class, and the predicted probability is \(\sigma(z)\). Each product \(\gamma_i s_i\) is therefore an exact signed node contribution: a positive contribution pushes the prediction toward CTCF depletion, while a negative contribution pushes it toward untreated. Keeping the final classifier linear preserves this additive decomposition.

We evaluate the model with ROC-AUC, average precision, accuracy, balanced accuracy, and F1 score. ROC-AUC is the primary metric because it evaluates ranking across thresholds and is less dependent on the selected decision threshold. Results are averaged across three random parameter initializations using the same data split.

| Model | Test ROC-AUC | Average precision | Accuracy | Balanced accuracy |
| --- | ---: | ---: | ---: | ---: |
| Class-prior baseline | 0.500 | 0.540 | 0.540 | 0.500 |
| Coverage and density logistic baseline | 0.563 | 0.582 | 0.571 | 0.546 |
| Two-layer interpretable GAT | 0.565 ± 0.005 | 0.593 ± 0.004 | 0.560 ± 0.0003 | 0.544 ± 0.005 |
| Mechanistic-summary logistic model | 0.589 | 0.604 | 0.565 | 0.552 |

The individual GAT test ROC-AUC values are 0.563, 0.561, and 0.570. These results show that the present GAT extracts a reproducible but weak signal. Its ROC-AUC is similar to the coverage-and-density baseline, so the current experiment does not yet demonstrate that the GAT captures substantially richer topology than simple graph statistics.

The mechanistic-summary model is currently more informative. It uses observed-node count, total contact count, mean degree, mean degree of the five paper CTCF sites, and the fraction of edges spanning an intervening CTCF boundary. Its strongest standardized coefficient is positive for the fraction of boundary-spanning edges, while the coefficient for mean CTCF-site degree is negative. Because the positive class is CTCF-depleted, these directions are consistent with the paper's reported redistribution toward cross-boundary contacts and reduction of CTCF-centered organization.

The GAT places substantially more final-readout attention on the five supplied CTCF sites than on other loci. This confirms that it uses the CTCF-site feature, but it is not an independent rediscovery of those sites because their identities were provided as input. Zeroing the boundary-spanning edge feature at test time lowers ROC-AUC only slightly, indicating that the current GAT relies primarily on node features and overall contact topology. A useful next step is to test whether a sum-based edge-aware message-passing layer can capture contact counts and boundary-spanning fractions more directly while retaining the same additive graph-level explanation.

## References

1. Hafner, A. et al. “Loop stacking organizes genome folding from TADs to chromosomes.” *Molecular Cell* 83, 1377–1392.e6 (2023). https://doi.org/10.1016/j.molcel.2023.04.008
2. 4D Nucleome Data Portal. “A. Hafner mESC loop stacking chromatin tracing.” https://data.4dnucleome.org/A_Hafner_mESC_loop_stacking_chromatin_tracing
3. Veličković, P. et al. “Graph Attention Networks.” *International Conference on Learning Representations* (2018). https://arxiv.org/abs/1710.10903
