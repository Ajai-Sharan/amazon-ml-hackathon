# Business Entity Resolution — Amazon ML Challenge 2026

Blocking + gradient-boosted pairwise matcher that links every Source 2 / Source 3
record to (at most) one Source 1 entity.

## Layout

```
src/
  common.py           paths, vocabularies, text folding, phonetic skeleton
  build_translit.py   learns a Devanagari -> Latin token dictionary from training pairs
  preprocess.py       normalises names/addresses of all 6 source files -> parquet
  blocking.py         hashed multi-key blocking, IDF-weighted, top-K S1 candidates per S2/S3 record
  features.py         pairwise similarity features (rapidfuzz + token/number overlap + blocking context)
  model.py            2-fold LightGBM training, out-of-fold validation + threshold search, test scoring
  evaluate.py         ground-truth helpers and the macro F0.5 metric
  make_submission.py  writes output/matching_results.tsv and output/candidate_pairs.tsv
requirements.txt      pinned dependencies (Python 3.11)
```

## Reproduce end-to-end

One command (writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`):

```bash
pip install -r requirements.txt
ER_DATA_DIR=/path/to/student_resource/dataset ER_WORK_DIR=/path/to/scratch ./run_all.sh
```

Or step by step:

```bash
pip install -r requirements.txt
export ER_DATA_DIR=/path/to/student_resource/dataset   # contains train/ and test/
export ER_WORK_DIR=/path/to/scratch                     # ~6 GB of intermediate parquet
cd src
python3 build_translit.py            # Devanagari dictionary from training ground truth
python3 preprocess.py train test     # normalised parquet for all sources
python3 blocking.py train            # candidate pairs (train)
python3 blocking.py test             # candidate pairs (test)
python3 model.py train               # fold models, OOF macro F0.5, threshold
python3 model.py predict             # score test candidates
python3 make_submission.py ../output # write both submission files
python3 /path/to/student_resource/utils/validate_submission.py \
    --matching ../output/matching_results.tsv \
    --candidate ../output/candidate_pairs.tsv --test-dir $ER_DATA_DIR/test
```

Run the steps one at a time: the full pipeline fits in 16 GB RAM / 4 CPU cores
(no GPU) only when the steps run sequentially. Total runtime is about 2–3 hours.

No external data, APIs or pretrained models are used. The only model is a
LightGBM (MIT licence) classifier trained on the provided training data.
