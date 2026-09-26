"""Train the pairwise matcher, score candidates, assign matches.

train  : 2-fold (split by query) LightGBM models on a sample of training
         candidates; out-of-fold scores for *all* training candidates are used
         to measure macro F0.5 and pick the decision threshold.
predict: average of the fold models on the test candidates.

Assignment: every Source 2/3 record matches at most one Source 1 entity
(true in the training ground truth), so each query is linked to its highest
scoring candidate if that score clears the threshold.
"""
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

import features as F
from common import WORK_DIR
from evaluate import gt_pairs, id_maps, macro_f05

CHUNK_Q = 400_000
TRAIN_Q_PER_FOLD = int(os.environ.get("ER_TRAIN_Q", 600_000))
PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=200,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              num_threads=os.cpu_count(), verbose=-1)
ROUNDS = int(os.environ.get("ER_ROUNDS", 500))


def fold_of(col="q"):
    return (pl.col(col).hash(seed=11) % 2).cast(pl.UInt8)


def score_chunks(cand, s1a, qa, models, fold_assign=True):
    """Yield (q, s1, p) for all candidates, chunked by query."""
    qs = cand["q"].unique().sort()
    t0 = time.time()
    out = []
    for i in range(0, len(qs), CHUNK_Q):
        lo, hi = qs[i], qs[min(i + CHUNK_Q, len(qs)) - 1]
        c = cand.filter(pl.col("q").is_between(lo, hi))
        f = F.compute(c, s1a, qa)
        X = f.select(FEATS).to_numpy()
        if fold_assign:
            fold = f.select(fold_of())["q"].to_numpy()
            p = np.empty(len(f), dtype=np.float32)
            for k, m in enumerate(models):
                sel = fold == (1 - k)          # model k was trained on fold k
                if sel.any():
                    p[sel] = m.predict(X[sel])
        else:
            p = np.mean([m.predict(X) for m in models], axis=0).astype(np.float32)
        out.append(f.select("q", "s1").with_columns(pl.Series("p", p)))
        print(f"  scored {i + len(c.unique('q')):,}/{len(qs):,} queries ({time.time()-t0:.0f}s)", flush=True)
    return pl.concat(out)


def assign(scores, thr):
    best = scores.sort(["q", "p"], descending=[False, True]).group_by("q", maintain_order=True).first()
    return best.filter(pl.col("p") >= thr).select("s1", "q")


FEATS = None


def train(split="train"):
    global FEATS
    cand = pl.read_parquet(os.path.join(WORK_DIR, f"cand_{split}.parquet"))
    s1ids, qids = id_maps(split)
    gt = gt_pairs(s1ids, qids).with_columns(pl.lit(1, pl.UInt8).alias("label"))
    s1a, qa = F.load_attrs(split)
    rec = cand.join(gt, on=["s1", "q"], how="semi").height / gt.height
    print(f"blocking recall {rec:.4f}  cand/query {cand.height / cand['q'].n_unique():.2f}", flush=True)

    uq = cand.select(pl.col("q").unique()).with_columns(fold_of().alias("fold"))
    models = []
    for k in (0, 1):
        sq = uq.filter(pl.col("fold") == k).sample(n=min(TRAIN_Q_PER_FOLD, uq.height // 2), seed=k)
        c = cand.join(sq.select("q"), on="q", how="semi")
        f = F.compute(c, s1a, qa).join(gt.select("s1", "q", "label"), on=["s1", "q"], how="left").with_columns(
            pl.col("label").fill_null(0))
        FEATS = F.feature_cols(f)
        ds = lgb.Dataset(f.select(FEATS).to_numpy(), f["label"].to_numpy(), feature_name=FEATS, free_raw_data=True)
        t0 = time.time()
        m = lgb.train(PARAMS, ds, ROUNDS)
        print(f"fold {k}: {f.height:,} rows, pos rate {f['label'].mean():.3f}, trained {time.time()-t0:.0f}s", flush=True)
        m.save_model(os.path.join(WORK_DIR, f"lgb_fold{k}.txt"))
        models.append(m)
        del f, ds
    imp = sorted(zip(models[0].feature_importance("gain"), FEATS), reverse=True)
    print("top features:", [(n, int(g)) for g, n in imp[:15]])
    json.dump(FEATS, open(os.path.join(WORK_DIR, "feats.json"), "w"))

    scores = score_chunks(cand, s1a, qa, models, fold_assign=True)
    scores.write_parquet(os.path.join(WORK_DIR, f"scores_{split}.parquet"))
    evaluate_scores(scores, gt.select("s1", "q"), s1ids["s1"])


def evaluate_scores(scores, gt, s1u):
    res = {}
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        f, p, r = macro_f05(assign(scores, thr), gt, s1u)
        res[thr] = f
        print(f"thr {thr:.2f}: F0.5 {f:.4f}  P {p:.4f}  R {r:.4f}", flush=True)
    best = max(res, key=res.get)
    json.dump({"thr": best, "f05": res[best]}, open(os.path.join(WORK_DIR, "thr.json"), "w"))
    print("best", best, res[best])


def predict(split="test"):
    global FEATS
    FEATS = json.load(open(os.path.join(WORK_DIR, "feats.json")))
    thr = json.load(open(os.path.join(WORK_DIR, "thr.json")))["thr"]
    models = [lgb.Booster(model_file=os.path.join(WORK_DIR, f"lgb_fold{k}.txt")) for k in (0, 1)]
    cand = pl.read_parquet(os.path.join(WORK_DIR, f"cand_{split}.parquet"))
    s1a, qa = F.load_attrs(split)
    scores = score_chunks(cand, s1a, qa, models, fold_assign=False)
    scores.write_parquet(os.path.join(WORK_DIR, f"scores_{split}.parquet"))
    print("threshold", thr)


if __name__ == "__main__":
    {"train": train, "predict": predict}[sys.argv[1]]()
