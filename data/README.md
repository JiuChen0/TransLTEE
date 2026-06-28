# Dataset contract and provenance

The committed files are derived from the public data used by
[GitHubLuCheng/LTEE](https://github.com/GitHubLuCheng/LTEE), which in turn uses
the IHDP and News causal-inference benchmarks.

## IHDP

```text
IHDP/csv/ihdp_npci_<1..10>.csv
```

Each file has 747 rows. Columns are treatment, factual outcome,
counterfactual outcome, potential-outcome means, and 25 context features.

```text
IHDP/Series_y_<1..10>.txt
```

Shape: `747 × 101`. Column 0 is binary treatment and columns 1-100 are factual
longitudinal outcomes.

## News

`NEWS/topic_doc_mean_n5000_k3477_seed_<1..10>.csv.x` stores sparse
`document_id,word_id,count` triplets for a `5000 × 3477` document-term matrix.

`NEWS/topic_doc_mean_n5000_k3477_seed_<1..10>.csv.y` stores five numeric label
columns. The generation pipeline uses treatment, factual outcome, and
counterfactual outcome from its first three columns. The `.y` extension is
upstream naming, not the Yacc programming language.

`NEWS/Series_y_<1..10>.txt` has shape `5000 × 101`, with treatment followed by
100 factual longitudinal outcomes.

## Ground truth

`Series_groundtruth_<1..10>.txt` contains the mean
`potential_treated - potential_control` at each of 100 timesteps.

## Generated versus raw files

- The IHDP CSV and News `.csv.x/.csv.y` files are upstream raw benchmark
  inputs.
- `Series_y_*.txt` and `Series_groundtruth_*.txt` are generated snapshots.
- `data_generateIHDP.py` and `data_generate.py` deterministically regenerate
  the snapshots when supplied the same seed and dependency versions.
