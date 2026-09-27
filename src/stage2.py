"""Second-stage re-scoring with query-level and entity-level context.

Stage 1 scores every (query, Source 1 candidate) pair independently.  Stage 2
looks at the top-3 stage-1 links of each query and adds context:
  query side  : rank of the link, best competing probability, margin
  entity side : how many queries point at the Source 1 entity, their summed
                probability, this link's rank among them, same-source counts
Trained on out-of-fold stage-1 scores of the training set (2 folds split by
Source 1 entity), so the validation score is out-of-fold as well.
"""
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

from common import WORK_DIR
from evaluate import gt_pairs, id_maps, macro_f05

TOPN = 3
PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=63, min_data_in_leaf=200,
              feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1,
              num_threads=os.cpu_count(), verbose=-1)
ROUNDS = 300
TRAIN_ROWS = 6_000_000


def build(split):
    sc = pl.read_parquet(os.path.join(WORK_DIR, f"scores_{split}.parquet"))
    cand = pl.read_parquet(os.path.join(WORK_DIR, f"cand_{split}.parquet"), columns=["q", "s1", "bscore", "nkeys"])
    n2 = pl.scan_parquet(os.path.join(WORK_DIR, f"{split}_s2.parquet")).select(pl.len()).collect().item()
    sc = sc.with_columns(pl.col("p").rank("ordinal", descending=True).over("q").alias("prank"))
    d = sc.filter(pl.col("prank") <= TOPN).join(cand, on=["q", "s1"], how="left")
    del sc, cand
    d = d.with_columns((pl.col("q") >= n2).cast(pl.UInt8).alias("is_s3"))
    d = d.with_columns(
        pl.col("p").max().over("q").alias("q_pmax"),
        pl.col("p").sum().over("q").alias("q_psum"),
        pl.len().over("q").alias("q_n"),
    )
    d = d.with_columns(
        # best competing probability for this query (other candidates)
        pl.when(pl.col("prank") == 1)
        .then(pl.col("p").filter(pl.col("prank") == 2).first().over("q"))
        .otherwise(pl.col("q_pmax")).fill_null(0).alias("q_pother"),
    )
    d = d.with_columns((pl.col("p") - pl.col("q_pother")).alias("q_margin"))
    top = pl.col("prank") == 1
    d = d.with_columns(
        pl.len().over("s1").alias("e_n"),
        pl.col("p").sum().over("s1").alias("e_psum"),
        pl.col("p").max().over("s1").alias("e_pmax"),
        (top & (pl.col("p") > 0.5)).sum().over("s1").alias("e_ntop"),
        (top & (pl.col("p") > 0.5) & (pl.col("is_s3") == 1)).sum().over("s1").alias("e_ntop_s3"),
        (top & (pl.col("p") > 0.5) & (pl.col("is_s3") == 0)).sum().over("s1").alias("e_ntop_s2"),
        pl.col("p").rank("ordinal", descending=True).over("s1").alias("e_rank"),
        pl.col("p").rank("ordinal", descending=True).over(["s1", "is_s3"]).alias("e_rank_src"),
    )
    d = d.with_columns((pl.col("p") / pl.col("e_pmax")).alias("e_prel"))
    return d


FEATS = ["p", "prank", "bscore", "nkeys", "is_s3", "q_pmax", "q_psum", "q_n", "q_pother", "q_margin",
         "e_n", "e_psum", "e_pmax", "e_ntop", "e_ntop_s3", "e_ntop_s2", "e_rank", "e_rank_src", "e_prel"]


def X(d):
    return d.select([pl.col(c).cast(pl.Float32) for c in FEATS]).to_numpy()


def assign2(d, thr):
    best = d.sort(["q", "p2"], descending=[False, True]).group_by("q", maintain_order=True).first()
    return best.filter(pl.col("p2") >= thr).select("s1", "q")


def train():
    d = build("train")
    s1ids, qids = id_maps("train")
    gt = gt_pairs(s1ids, qids)
    d = d.join(gt.with_columns(pl.lit(1, pl.UInt8).alias("label")), on=["s1", "q"], how="left").with_columns(
        pl.col("label").fill_null(0))
    fold = (d["s1"].hash(seed=5) % 2).to_numpy()
    p2 = np.zeros(d.height, dtype=np.float32)
    Xa, y = X(d), d["label"].to_numpy()
    rng = np.random.default_rng(0)
    for k in (0, 1):
        tr = np.flatnonzero(fold == k)
        if len(tr) > TRAIN_ROWS:
            tr = rng.choice(tr, TRAIN_ROWS, replace=False)
        m = lgb.train(PARAMS, lgb.Dataset(Xa[tr], y[tr], feature_name=FEATS), ROUNDS)
        te = fold != k
        p2[te] = m.predict(Xa[te])
        print(f"fold {k} trained on {len(tr):,} rows", flush=True)
    d = d.with_columns(pl.Series("p2", p2))
    res = {}
    for thr in (0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8):
        f, p, r = macro_f05(assign2(d, thr), gt, s1ids["s1"])
        res[thr] = f
        print(f"stage2 thr {thr}: F0.5 {f:.4f} P {p:.4f} R {r:.4f}", flush=True)
    best = max(res, key=res.get)
    print("best", best, res[best])
    json.dump({"thr": best, "f05": res[best]}, open(os.path.join(WORK_DIR, "thr2.json"), "w"))
    # final model on all training rows (subsampled) for the test set
    idx = rng.choice(d.height, min(2 * TRAIN_ROWS, d.height), replace=False)
    m = lgb.train(PARAMS, lgb.Dataset(Xa[idx], y[idx], feature_name=FEATS), ROUNDS)
    m.save_model(os.path.join(WORK_DIR, "lgb_stage2.txt"))
    imp = sorted(zip(m.feature_importance("gain"), FEATS), reverse=True)
    print("stage2 importance:", [(n, int(g)) for g, n in imp[:10]])


def predict():
    d = build("test")
    m = lgb.Booster(model_file=os.path.join(WORK_DIR, "lgb_stage2.txt"))
    d = d.with_columns(pl.Series("p2", m.predict(X(d)).astype(np.float32)))
    d.select("q", "s1", "p", "p2").write_parquet(os.path.join(WORK_DIR, "scores2_test.parquet"))
    print("wrote stage-2 test scores", d.height)


if __name__ == "__main__":
    {"train": train, "predict": predict}[sys.argv[1]]()
