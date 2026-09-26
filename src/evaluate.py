"""Ground-truth helpers and the macro F0.5 metric used by the challenge."""
import os

import polars as pl

from common import DATA_DIR, WORK_DIR, read_tsv


def id_maps(split):
    """Return (s1 ids, query ids) where query id q indexes concat(S2, S3)."""
    s1 = pl.read_parquet(os.path.join(WORK_DIR, f"{split}_s1.parquet"), columns=["idx", "entity_id"])
    s2 = pl.read_parquet(os.path.join(WORK_DIR, f"{split}_s2.parquet"), columns=["idx", "entity_id"])
    s3 = pl.read_parquet(os.path.join(WORK_DIR, f"{split}_s3.parquet"), columns=["idx", "entity_id"])
    q = pl.concat([s2, s3.with_columns(pl.col("idx") + s2.height)])
    return s1.rename({"idx": "s1"}), q.rename({"idx": "q"})


def gt_pairs(s1ids, qids):
    gt = read_tsv(f"{DATA_DIR}/train/train_ground_truth.tsv")
    e = (gt.with_columns(pl.col("matched_entity_ids").str.split(","))
         .explode("matched_entity_ids").drop_nulls("matched_entity_ids")
         .filter(pl.col("matched_entity_ids") != ""))
    e = e.join(s1ids, left_on="source1_entity_id", right_on="entity_id").join(
        qids, left_on="matched_entity_ids", right_on="entity_id")
    return e.select("s1", "q")


def macro_f05(pred, truth, s1_universe):
    """pred/truth: DataFrames (s1, q).  s1_universe: Series of s1 ids scored."""
    u = pl.DataFrame({"s1": s1_universe})
    tp = pred.join(truth, on=["s1", "q"]).group_by("s1").agg(pl.len().alias("tp"))
    np_ = pred.group_by("s1").agg(pl.len().alias("np"))
    nt = truth.group_by("s1").agg(pl.len().alias("nt"))
    d = u.join(tp, on="s1", how="left").join(np_, on="s1", how="left").join(nt, on="s1", how="left").fill_null(0)
    p = pl.when(pl.col("np") > 0).then(pl.col("tp") / pl.col("np")).otherwise(0.0)
    r = pl.when(pl.col("nt") > 0).then(pl.col("tp") / pl.col("nt")).otherwise(0.0)
    f = pl.when((pl.col("nt") == 0) & (pl.col("np") == 0)).then(1.0).when(
        (pl.col("p") + pl.col("r")) > 0).then(1.25 * pl.col("p") * pl.col("r") / (0.25 * pl.col("p") + pl.col("r"))).otherwise(0.0)
    d = d.with_columns(p.alias("p"), r.alias("r")).with_columns(f.alias("f"))
    return d["f"].mean(), d["p"].mean(), d["r"].mean()
