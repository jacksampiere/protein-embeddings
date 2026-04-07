# ── research notes ────────────────────────────────────────────────────────────
# Can we use intermediate layer representations as features?
# Grab embeddings from all layers --> dimensionality reduction?
# Agent-based injection of domain-informed features? (A la DEFT)

# If using earlier/later layers is better/worse, can tell us whether lower-/higher-level features are helpful

# Guard against confirmation bias during pseudo-labeling; i.e., only add pseudo-labeled points when:
#     - predictive entropy is low
#     - two different TFMs agree, or when confidence is stable across multiple random context permutations

# Split by protein (or homology clusters if available) rather than variant

# Maybe look at delta embeddigns for a subset of assays per protein family

# North star: an ablation matrix where rows are tasks/splits and columns are tabularizations + TFMs
# Then, analyze which representation changes move the needle and when pseudo-label context expansion helps vs hurts
#    - Consider creating 2 axes (examples): one for local-to-global context and another from earlier --> later ESM-2 layers
#    - Then, plot a heatmap of performance metrics across those axes (will be coarse --> interpolate)
#    - Can then visualize: for xyz task, how does performance change as we get more/less global context and hgiher-/lower-level representations?
# Another axis could be psuedo-label leniency (i.e., label confidence thresholds)

# Consider incorporating ProteinNet later

# Tabular features:
#    - Pooled embedding vector of original DMS sequence (D,)
#    - Embedding vector of the residue at sequence index p (D,)
#        - Note that locations may be 1-indexed
#    - Location descriptor: p / L (1,)
#    - aa_from: what the original residue was (shape (20,) since there a 20 amino acids)
#    - aa_to: what the new residue is (20,)
#    - Optional further inclusions
#        - BLOSUM62 score
#        - Physchem deltas
#            - Each amino acid has a charge, hydrophobicity, size/volume, polarity, etc
#            - Consider, e.g., Δhydrophobicity = hydro(aa_to) - hydro(aa_from)

# Computational challenge of using the embedding vector of the new residue:
#    - We need to run ESM2 on the full sequence to get the embedding at position p
#    - Consider 217 DMS assays and 1000 mutants per assay
#    - We would need to run 217*1000 forward passes just to get those vectors

# Pooling methods + interpretations:
#    - Mean (baseline)
#        - Distributed signal across the sequence matters
#    - Segment/bin pooling: 8-16 windows --> concatenate
#        - Effect of coarse localization is significant
#    - Windowing around the mutation site
#        - Strong --> local context is the main driver; weak --> long-range/global context matters
# Additional:
#    - CLS/BOS token
#        - Strong --> the model’s learned global summary matters
#    - Attention-weighted pool (weights from a learned scorer on R_wt[i])
#        - Strong --> hints the task depends on a subset of positions rather than diffuse effects.
#    - Max pooling (per-dimension max)
#        - Strong --> rare/extreme motifs/features dominate (e.g., a short functional region)
#    - Motif/keyword pooling (pool only residues matching patterns, e.g., glyco motifs NXS/T, catalytic triads if annotated)
#        - Strong --> suggests dependence on known biochemical motifs.
#    - Any kind of domain-aware pooling
