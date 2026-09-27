"""Write output/matching_results.tsv and output/candidate_pairs.tsv."""
import json
import os
import sys

import polars as pl

from common import WORK_DIR
from evaluate import id_maps
from model import assign


def id_lists(pairs, s1ids, qids, col):
    lists = (pairs.join(qids, on="q").sort(["s1", "q"])
             .group_by("s1").agg(pl.col("entity_id").unique(maintain_order=True).str.join(",").alias(col)))
    out = s1ids.join(lists, on="s1", how="left").sort("s1")
    return out.select(pl.col("entity_id").alias("source1_entity_id"), pl.col(col).fill_null(""))


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    s1ids, qids = id_maps("test")
    # candidate set = every pair the stage-1 model scored
    cand = pl.read_parquet(os.path.join(WORK_DIR, "scores_test.parquet"), columns=["s1", "q"])
    s2_path = os.path.join(WORK_DIR, "scores2_test.parquet")
    if os.path.exists(s2_path):          # stage-2 re-scored links
        thr = float(os.environ.get("ER_THR", json.load(open(os.path.join(WORK_DIR, "thr2.json")))["thr"]))
        scores = pl.read_parquet(s2_path).select("q", "s1", pl.col("p2").alias("p"))
    else:
        thr = float(os.environ.get("ER_THR", json.load(open(os.path.join(WORK_DIR, "thr.json")))["thr"]))
        scores = pl.read_parquet(os.path.join(WORK_DIR, "scores_test.parquet"))
    matches = assign(scores, thr)
    id_lists(matches, s1ids, qids, "matched_entity_ids").write_csv(
        os.path.join(out_dir, "matching_results.tsv"), separator="\t", quote_style="never")
    id_lists(cand, s1ids, qids, "candidate_entity_ids").write_csv(
        os.path.join(out_dir, "candidate_pairs.tsv"), separator="\t", quote_style="never")
    print(f"threshold {thr}: {matches.height:,} matched pairs for {matches['s1'].n_unique():,} "
          f"of {s1ids.height:,} Source 1 entities")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "output"))
