#!/bin/sh
# End-to-end pipeline: data -> blocking -> matching -> output/.
# Steps run sequentially to stay within 16 GB RAM.
#   ER_DATA_DIR  dataset directory (contains train/ and test/)
#   ER_WORK_DIR  scratch directory for intermediate parquet files
#   OUT_DIR      where the two submission TSVs are written (default ./output)
set -e
cd "$(dirname "$0")/src"
OUT_DIR=${OUT_DIR:-../output}
export ER_TRAIN_Q=${ER_TRAIN_Q:-900000} ER_ROUNDS=${ER_ROUNDS:-500} ER_TOP_K=${ER_TOP_K:-10}

python3 build_translit.py
python3 preprocess.py train test
python3 blocking.py train
python3 blocking.py test
python3 model.py train        # stage 1, out-of-fold validation
python3 model.py predict
python3 stage2.py train       # stage 2 re-scoring, out-of-fold validation
python3 stage2.py predict
python3 make_submission.py "$OUT_DIR"
