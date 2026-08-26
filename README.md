# ARNLE-IAV

ARNLE-IAV is an influenza A virus (IAV) host-adaptation analysis framework adapted from the ARNLE model. This repository contains the influenza-specific sequence-processing code, ELMo representation export, the three-class Bi-LSTM host classifier, and the downstream PCA, transition-band, candidate-site, statistical-evidence, and HA-NA paired-isolate analyses used in the study.

Large-scale training/validation datasets, pretrained model weights, and the analysis-result tables associated with the manuscript are distributed through the companion Zenodo release.

## Relationship to ARNLE

ARNLE-IAV follows the original ARNLE framework for protein sequence representation learning and supervised host classification. The ELMo and Bi-LSTM framework is adapted to IAV by changing the sequence resources, tokenization/data preparation, host classification scheme, and downstream analyses. ELMo pretraining itself follows the original ARNLE implementation and is therefore not duplicated in this repository.


## Repository structure


```text
ARNLE-IAV/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── .gitignore
├── preprocessing/
├── model/
│   ├── embedding/
│   └── classifier/
└── downstream_analysis/
    ├── PCA_transition/
    ├── site_trajectory/
    ├── evidence/
    └── HA_NA_pairing/
```

## Script naming

Public script names are intentionally concise and function-oriented. Development labels, version suffixes, and environment-specific terms are omitted from filenames; the underlying analysis logic is unchanged.

## Analysis workflow

The public workflow is organized in the following order:

1. Extract IAV proteins and prepare ARNLE-compatible sequence input.
2. Generate ELMo sequence representations.
3. Train/evaluate the three-class Bi-LSTM host classifier.
4. Generate unified sequence-level PCA coordinates.
5. Perform pairwise source-to-target PCA transition analysis.
6. Generate protein/subtype/host-direction residue-frequency inputs.
7. Define the middle/transition band and rank candidate-site trajectories.
8. Map alignment columns to reference residue positions.
9. Run ordered-layer logistic trend tests and within-analysis BH-FDR correction.
10. Run permutation and threshold-sensitivity analyses supporting Figure 3.
11. Reproduce and validate the candidate-site evidence levels used by candidate-site evidence summary/Figure 5.
12. Perform HA-NA paired-isolate analysis.

## Installation

The base package versions in `requirements.txt` follow the public ARNLE repository. ARNLE-IAV adds only the extra packages required by the influenza-specific preprocessing and downstream analysis scripts.

```bash
pip install -r requirements.txt
```

ELMo representation export also requires **ELMoForManyLangs**, as in the original ARNLE workflow. Install ELMoForManyLangs separately before running `model/embedding/elmo_embedding_stream.py`. The original ARNLE repository uses the ELMoForManyLangs implementation available from:

```text
https://github.com/berkay-onder/ELMoForManyLangs
```

All commands below assume that the current directory is the root of the cloned `ARNLE-IAV` repository.

For readability, the examples use placeholders such as `<ZENODO_ROOT>`, `<WORK_DIR>`, and `<MASTER_TABLE.csv>`. Replace every placeholder with a real local path before running a command.

---

# Running ARNLE-IAV in analysis order

## Step 1. Extract IAV protein FASTA files

Script:

```text
preprocessing/extract_iav_proteins.py
```

Use an NCBI protein FASTA file as input. A GenPept file or directory can optionally be supplied to fill host, country, and collection-date metadata.

```bash
python preprocessing/extract_iav_proteins.py \
  --fasta <INPUT_NCBI_PROTEIN_FASTA> \
  --gp <GENPEPT_FILE_OR_DIRECTORY> \
  --outdir <WORK_DIR>/protein_fasta \
  --overwrite \
  --cache <WORK_DIR>/gp_metadata_cache.pkl
```

If no GenPept metadata is required, omit `--gp` and `--cache`.

---

## Step 2A. Generate 1-aa ARNLE RAW input

Script:

```text
preprocessing/fasta_to_raw.py
```

This is the 1-aa input preparation route. It cleans the FASTA file and writes an ARNLE-compatible tab-delimited RAW file.

```bash
python preprocessing/fasta_to_raw.py \
  --input_fasta <WORK_DIR>/protein_fasta/HA.fasta \
  --output_prefix <WORK_DIR>/elmo_raw/HA_token1 \
  --max_len 1500
```

Outputs include:

```text
HA_token1.cleaned.fasta
HA_token1.clean_report.txt
HA_token1.raw
```

Repeat for the protein FASTA files required for ELMo preparation.

---

## Step 2B. Generate k-mer RAW inputs for token comparison

Script:

```text
preprocessing/fasta_to_kmer_raw.py
```

This script generates the multi-token comparison inputs. The example below creates 2-aa through 10-aa token files.

```bash
python preprocessing/fasta_to_kmer_raw.py \
  --input_fasta <INPUT_PROTEIN_FASTA> \
  --ks 2 3 4 5 6 7 8 9 10 \
  --max_tokens_per_line 1500
```

Use `--drop_remainder` only if terminal fragments shorter than `k` should be discarded. The default behavior retains them.

**ELMo pretraining:** ELMo model training follows the original ARNLE implementation. The corresponding IAV training/validation datasets and pretrained ELMo weights are provided through Zenodo; this repository does not duplicate the original ARNLE ELMo-training program.

---

## Step 3. Export ELMo representations

Script:

```text
model/embedding/elmo_embedding_stream.py
```

The embedding sequence length must be kept consistent with the generated embedding dataset and the downstream classifier `--max_length`. When reusing a pretrained Bi-LSTM checkpoint, use the sequence length/configuration compatible with that checkpoint.

```bash
python model/embedding/elmo_embedding_stream.py \
  --file <INPUT_PROTEIN_FASTA> \
  --model_path <ZENODO_ROOT>/pretrained_models/ELMo/<ELMO_MODEL_DIR> \
  --output <WORK_DIR>/embeddings/token1_embedding.npy \
  --batchsize 256 \
  --max_length <EMBEDDING_MAX_LENGTH> \
  --split 1 \
  --chunk_size 20000 \
  --write_labels
```

The output is written as chunked NumPy arrays (`*_part000.npy`, `*_part001.npy`, ...), together with title/length files and optional labels.

---

## Step 4. Train and evaluate the three-class Bi-LSTM host classifier

Script:

```text
model/classifier/train_bilstm_host_classifier.py
```

Before running the Bi-LSTM classifier, generate the required ELMo embedding arrays and the accompanying `.labels.txt` and `.lengths.txt` files with `model/embedding/elmo_embedding_stream.py`. The commands below therefore use `<WORK_DIR>/embeddings/` rather than a precomputed embedding directory in Zenodo.

Bi-LSTM FASTA roles in the Zenodo release:

```text
training_validation_data/BiLSTM_train_validation/validation.fasta
    -> generate embeddings for --data_train

training_validation_data/BiLSTM_train_validation/test.fasta
    -> generate embeddings for --data_val
```


The classifier uses the three host classes `artiodactyla`, `primates`, and `aves`. The `--max_length` value must equal the second dimension of the ELMo embedding arrays supplied to this training run.

**Bi-LSTM dataset-role mapping:** generate the embeddings used for `--data_train` from `training_validation_data/BiLSTM_train_validation/validation.fasta`, and generate the embeddings used for `--data_val` from `training_validation_data/BiLSTM_train_validation/test.fasta`. These FASTA roles reflect the finalized release workflow.

```bash
python model/classifier/train_bilstm_host_classifier.py \
  --data_train <WORK_DIR>/embeddings/train_embedding_part000.npy \
  --label_train <WORK_DIR>/embeddings/<TRAIN_LABEL_OR_TITLE_FILE> \
  --length_train <WORK_DIR>/embeddings/train_embedding.lengths.txt \
  --data_val <WORK_DIR>/embeddings/validation_embedding_part000.npy \
  --label_val <WORK_DIR>/embeddings/<VAL_LABEL_OR_TITLE_FILE> \
  --length_val <WORK_DIR>/embeddings/validation_embedding.lengths.txt \
  --epoch 10 \
  --keepprob 0.8 \
  --num_class 3 \
  --hidden_size 256,128 \
  --lr 0.001 \
  --max_length <CLASSIFIER_MAX_LENGTH> \
  --batchsize 32 \
  --writer_path <WORK_DIR>/bilstm/tensorboard \
  --model_path <WORK_DIR>/bilstm/model.ckpt \
  --metrics_out_dir <WORK_DIR>/bilstm/metrics \
  --token_size 1 \
  --val_metadata_csv <VALIDATION_METADATA.csv> \
  --save_val_predictions
```

The script automatically discovers all matching `*_part*.npy` blocks when a base or `*_part000.npy` path is supplied. Validation outputs include global/host/protein/subtype metrics and confusion matrices.

**Downstream feature input:** the PCA scripts below use sequence-level master tables and attention-derived feature arrays. Large sequence datasets, pretrained models, checkpoints, and selected manuscript-supporting result tables are distributed through Zenodo. Intermediate embedding arrays and some analysis-stage inputs are regenerated locally from the provided resources using the public scripts. This repository does not fabricate or reconstruct missing inference outputs.

---

## Step 5. Generate unified sequence-level PCA coordinates

Script:

```text
downstream_analysis/PCA_transition/sequence_feature_3d_pca.py
```

One master table must correspond to one feature array. Multiple proteins can be passed in the same command.

```bash
python downstream_analysis/PCA_transition/sequence_feature_3d_pca.py \
  --master_tables <HA_MASTER_TABLE.csv> <NA_MASTER_TABLE.csv> <PB2_MASTER_TABLE.csv> <NP_MASTER_TABLE.csv> <NS1_MASTER_TABLE.csv> \
  --attn_npys <HA_FULL_ATTN.npy> <NA_FULL_ATTN.npy> <PB2_FULL_ATTN.npy> <NP_FULL_ATTN.npy> <NS1_FULL_ATTN.npy> \
  --out_root <WORK_DIR>/PCA \
  --subtype_mode auto \
  --min_subtype_samples 30 \
  --min_host_samples 5
```

If a supplied master table is a subset of the table used to generate the feature array, also provide the aligned full tables with `--feature_ref_tables`.

---

## Step 6. Perform pairwise PCA source-to-target transition analysis

Script:

```text
downstream_analysis/PCA_transition/pairwise_pca_transition.py
```

Example for the artiodactyla-to-primates direction:

```bash
python downstream_analysis/PCA_transition/pairwise_pca_transition.py \
  --pca_csv <WORK_DIR>/PCA/<PROTEIN>/subtypes/<SUBTYPE>/<PROTEIN>_<SUBTYPE>_pca_coordinates.csv \
  --host_pair artiodactyla:primates \
  --out_dir <WORK_DIR>/pairwise/<PROTEIN>_<SUBTYPE>_artiodactyla_to_primates \
  --color_by host
```

Replace the host pair for other directions, for example `aves:primates` or `aves:artiodactyla`.

---

## Step 7. Generate pairwise residue-frequency and sorted-difference tables

Script:

```text
downstream_analysis/PCA_transition/pairwise_residue_frequency.py
```

This analysis is run from master tables containing numeric residue-position columns and host prediction probabilities.

```bash
python downstream_analysis/PCA_transition/pairwise_residue_frequency.py \
  --master_tables <HA_MASTER_TABLE.csv> <NA_MASTER_TABLE.csv> <PB2_MASTER_TABLE.csv> <NP_MASTER_TABLE.csv> <NS1_MASTER_TABLE.csv> \
  --out_root <WORK_DIR>/pairwise_frequency \
  --compare_pairs artiodactyla:primates,aves:primates,aves:artiodactyla \
  --min_count 50 \
  --subtype_mode auto \
  --skip_monthly
```

The output directories contain the source/target residue-frequency (`*_bayes.csv`) tables and source-versus-target sorted-difference tables used by the trajectory analysis. Remove `--skip_monthly` only when monthly analysis is required.

---

## Step 8. Define the transition band and rank candidate-site trajectories

Script:

```text
downstream_analysis/site_trajectory/transition_band_site_trajectory.py
```

Example for one protein/subtype/direction:

```bash
python downstream_analysis/site_trajectory/transition_band_site_trajectory.py \
  --pairwise_csv <PAIRWISE_COORDINATES.csv> \
  --master_table_csv <MASTER_TABLE.csv> \
  --sorted_diff_csv <SOURCE_VS_TARGET_SORTED_DIFF.csv> \
  --source_bayes_csv <SOURCE_HOST_BAYES.csv> \
  --target_bayes_csv <TARGET_HOST_BAYES.csv> \
  --out_dir <WORK_DIR>/trajectory/<PROTEIN>_<SUBTYPE>_artiodactyla_to_primates \
  --source_host artiodactyla \
  --target_host primates \
  --top_n 40 \
  --source_upper_q 0.90 \
  --target_lower_q 0.10 \
  --core_keep_fraction 1.0 \
  --core_min_side_n 20 \
  --grouping_mode quantile_balanced \
  --min_delta 0.10 \
  --jump_ratio 1.5
```

Important outputs include `transition_band_samples.csv`, `site_trajectory_top40_ranked_informative.csv`, and `analysis_summary.json`.

---

## Step 9. Map MSA columns to reference residue positions

Script:

```text
downstream_analysis/site_trajectory/map_msa_to_reference.py
```

When candidate `position` values represent alignment columns:

```bash
python downstream_analysis/site_trajectory/map_msa_to_reference.py \
  --alignment_fasta <ALIGNED_FASTA> \
  --candidate_csv <WORK_DIR>/trajectory/<COMBINATION>/site_trajectory_top40_ranked_informative.csv \
  --out_dir <WORK_DIR>/trajectory/<COMBINATION>/reference_mapping \
  --position_col position \
  --position_mode alignment_column \
  --reference_contains <REFERENCE_ACCESSION_OR_UNIQUE_HEADER_TEXT> \
  --map_all_sequences \
  --strip_accession_version
```

If the candidate table is already reference-numbered, use `--position_mode reference_position` instead. Reference identifiers are protein-specific and should match the reference sequences used in the study.

---

## Step 10. Run ordered-layer logistic trend tests and BH-FDR correction

Script:

```text
downstream_analysis/site_trajectory/site_logistic_trend_fdr.py
```

For sequence-level long-format site-state input:

```bash
python downstream_analysis/site_trajectory/site_logistic_trend_fdr.py run \
  --site-state <TRANSITION_BAND_SITE_STATES_LONG.csv> \
  --format long \
  --output <WORK_DIR>/statistics/site_trajectory_logistic_trend_fdr.csv \
  --alpha 0.05 \
  --min-total 30 \
  --min-per-layer 30
```

For wide-format input, additionally provide the candidate-site table:

```bash
python downstream_analysis/site_trajectory/site_logistic_trend_fdr.py run \
  --site-state <TRANSITION_BAND_SITE_STATES_WIDE.csv> \
  --format wide \
  --candidate-sites <CANDIDATE_SITE_SUMMARY.csv> \
  --output <WORK_DIR>/statistics/site_trajectory_logistic_trend_fdr.csv \
  --alpha 0.05 \
  --min-total 30 \
  --min-per-layer 30
```

To verify that an existing final candidate-site table contains the expected within-analysis BH-adjusted q values:

```bash
python downstream_analysis/site_trajectory/site_logistic_trend_fdr.py validate-fdr \
  --table <TABLE_S8_CANDIDATE_SITE_EVIDENCE_SUMMARY.csv> \
  --output <WORK_DIR>/statistics/fdr_validation_report.csv \
  --alpha 0.05
```

The sequence-level site-state input is an explicit input to this script; use the staged site-state table associated with the corresponding analysis rather than inferring unavailable per-sequence states from a summary table.

---

## Step 11. Build Figure 3 permutation and threshold-sensitivity support tables

Script:

```text
downstream_analysis/evidence/transition_band_support.py
```

### Recommended batch mode

Prepare a batch root containing combination directories named in the form `Protein_Subtype_source_to_target`, then run:

```bash
python downstream_analysis/evidence/transition_band_support.py \
  --batch-root <FIGURE3_BATCH_INPUT_ROOT> \
  --batch-out-root <WORK_DIR>/Figure3_support \
  --n-perm 1000 \
  --seed 2026 \
  --default-q 0.15 \
  --default-trim 0.10 \
  --default-min-group-size 30
```

A `batch_run_summary.csv` file is written at the batch-output root. Per-combination outputs include permutation negative controls, the permutation null distribution, and transition-band sensitivity summaries.

### Single-combination mode

```bash
python downstream_analysis/evidence/transition_band_support.py \
  --pca <PCA_COORDINATES.csv> \
  --master <MASTER_TABLE.csv> \
  --transition <TRANSITION_BAND_SAMPLES.csv> \
  --outdir <WORK_DIR>/Figure3_support/<COMBINATION> \
  --source-host artiodactyla \
  --target-host primates \
  --protein NP \
  --subtype H1N1 \
  --direction artiodactyla_to_primates \
  --n-perm 1000 \
  --seed 2026 \
  --default-q 0.15 \
  --default-trim 0.10 \
  --default-min-group-size 30
```

---

## Step 12. Reproduce and validate candidate-site evidence summary candidate-site evidence levels

Script:

```text
downstream_analysis/evidence/candidate_site_evidence.py
```

The original candidate-site evidence summary evidence-level generator was not preserved. The public script is therefore a documented reconstruction of the evidence rule that was validated against all 543 exported candidate-site evidence summary records. The input table must already contain the columns required by that rule, including `delta_target_near_minus_source_near`, `trend_q_BH_within_analysis`, and `trajectory_score`.

```bash
python downstream_analysis/evidence/candidate_site_evidence.py \
  --input <CANDIDATE_SITE_SUMMARY_WITH_TREND_RESULTS.csv> \
  --output <WORK_DIR>/candidate_site_evidence_summary.csv \
  --validation-output <WORK_DIR>/Table_S8_evidence_validation.json
```

The script also validates reconstructed evidence labels against an existing `evidence_level` column when one is present.

---

## Step 13. Build the HA-NA paired-isolate model

Script:

```text
downstream_analysis/HA_NA_pairing/ha_na_paired_analysis.py
```

Example for H1N1, artiodactyla to primates:

```bash
python downstream_analysis/HA_NA_pairing/ha_na_paired_analysis.py \
  --ha-master <HA_MASTER_TABLE.csv> \
  --na-master <NA_MASTER_TABLE.csv> \
  --ha-pca <HA_PCA_COORDINATES.csv> \
  --na-pca <NA_PCA_COORDINATES.csv> \
  --ha-transition <HA_TRANSITION_BAND_SAMPLES.csv> \
  --na-transition <NA_TRANSITION_BAND_SAMPLES.csv> \
  --ha-trajectory <HA_SITE_TRAJECTORY.csv> \
  --na-trajectory <NA_SITE_TRAJECTORY.csv> \
  --pair-key auto \
  --min-pairs 20 \
  --top-k-sites 40 \
  --expected-subtype H1N1 \
  --source-host artiodactyla \
  --target-host primates \
  --joint-threshold 0.60 \
  --strict-high 0.80 \
  --strict-low 0.50 \
  --out-dir <WORK_DIR>/HA_NA_pairing/H1N1_artiodactyla_to_primates
```

Repeat with the appropriate subtype and host direction for the other manuscript combinations. The pairing code evaluates candidate isolate keys, requires unique one-to-one HA/NA keys for paired analysis, and writes pairing diagnostics together with the final paired tables.

---

## Script-to-analysis reference

| Script | Primary role |
|---|---|
| `extract_iav_proteins.py` | Extract IAV protein FASTA files and metadata |
| `fasta_to_raw.py` | Prepare 1-aa ARNLE RAW input |
| `fasta_to_kmer_raw.py` | Prepare multi-token RAW inputs |
| `elmo_embedding_stream.py` | Export ELMo protein representations |
| `train_bilstm_host_classifier.py` | Train/evaluate three-host Bi-LSTM classifier |
| `sequence_feature_3d_pca.py` | Generate unified sequence-level PCA coordinates |
| `pairwise_pca_transition.py` | Project source/target host pairs in PCA space |
| `pairwise_residue_frequency.py` | Generate pairwise residue-frequency and sorted-difference inputs |
| `transition_band_site_trajectory.py` | Define transition band and rank site trajectories |
| `map_msa_to_reference.py` | Map alignment columns to reference residue numbering |
| `site_logistic_trend_fdr.py` | Ordered logistic trend test and within-analysis BH-FDR |
| `transition_band_support.py` | Permutation negative controls and transition-band sensitivity |
| `candidate_site_evidence.py` | Reproduce/validate the documented candidate-site evidence-level rule |
| `ha_na_paired_analysis.py` | HA-NA same-isolate pairing and concordance analysis |

## Reproducibility notes

- The commands above are templates; paths must be replaced by actual local files.
- Do not change the row ordering between a master table and its corresponding feature array.
- The ELMo embedding length and Bi-LSTM `--max_length` must be consistent for a given classifier input dataset.
- Host directions are explicit (`source:target`) and should be kept consistent across pairwise PCA, trajectory, Figure 3 support, and HA-NA analyses.
- Large inputs, checkpoints, and manuscript result tables belong to the Zenodo release rather than the Git repository.
- Plotting scripts are intentionally not included in this public code release; the repository focuses on model adaptation and core analytical computations.

## Data and model availability

This GitHub repository contains source code only. Training/validation sequence resources, pretrained ELMo models, Bi-LSTM checkpoints, and selected manuscript-supporting analysis result tables are distributed through the associated Zenodo archive.

**Zenodo DOI:** `to be added`

After downloading the Zenodo archive, keep the data and model files outside the Git repository and replace the path placeholders used in the commands above (for example, `<ZENODO_ROOT>` and `<WORK_DIR>`) with the corresponding local paths. No fixed `data/` directory is required inside this repository.

Precomputed ELMo embedding arrays and dedicated Figure 2 source-data tables are not distributed separately. ELMo embeddings can be regenerated from the provided FASTA sequence resources using the pretrained ELMo models and `model/embedding/elmo_embedding_stream.py`; token-specific pretrained models and Bi-LSTM checkpoints are distributed through Zenodo.
## Citation

The ARNLE-IAV manuscript has not yet been published. Until a formal article citation is available, cite the archived software/data release identifiers provided with the public repositories and cite the original ARNLE framework where applicable. Software author: **Zhang Shenglong**. This section will be updated after publication.
