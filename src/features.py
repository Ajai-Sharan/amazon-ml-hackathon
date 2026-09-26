"""Pairwise features for (query record, Source 1 candidate) pairs."""
import os

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from common import WORK_DIR

ATTR_COLS = ["idx", "name_f", "name_core", "name_alt", "name_ph", "addr_f", "addr_toks", "addr_nums"]


def load_attrs(split):
    s1 = pl.read_parquet(os.path.join(WORK_DIR, f"{split}_s1.parquet"), columns=ATTR_COLS)
    s2 = pl.read_parquet(os.path.join(WORK_DIR, f"{split}_s2.parquet"), columns=ATTR_COLS)
    s3 = pl.read_parquet(os.path.join(WORK_DIR, f"{split}_s3.parquet"), columns=ATTR_COLS)
    n2 = s2.height
    q = pl.concat([s2.with_columns(pl.lit(2, pl.UInt8).alias("qsrc")),
                   s3.with_columns(pl.col("idx") + n2, pl.lit(3, pl.UInt8).alias("qsrc"))])
    return s1.rename({"idx": "s1"}), q.rename({"idx": "q"})


def _pd(a, b, scorer, **kw):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32, **kw)


def add_group_feats(c):
    """Blocking-score context features within each query's candidate list."""
    return c.with_columns(
        pl.col("bscore").rank("ordinal", descending=True).over("q").cast(pl.Float32).alias("brank"),
        (pl.col("bscore") / pl.col("bscore").max().over("q")).alias("brel"),
        (pl.col("bscore").max().over("q") - pl.col("bscore")).alias("bgap"),
        pl.len().over("q").cast(pl.Float32).alias("ncand"),
    )


def compute(cand, s1a, qa):
    """cand: q, s1, bscore, nkeys  ->  DataFrame of features (plus q, s1)."""
    c = add_group_feats(cand)
    c = c.join(qa, on="q", how="left").join(s1a, on="s1", how="left", suffix="_1")
    qn, sn = c["name_core"].to_list(), c["name_core_1"].to_list()
    qf, sf = c["name_f"].to_list(), c["name_f_1"].to_list()
    qp, sp = c["name_ph"].to_list(), c["name_ph_1"].to_list()
    qalt, salt = c["name_alt"].to_list(), c["name_alt_1"].to_list()
    qa_, sa_ = c["addr_f"].to_list(), c["addr_f_1"].to_list()
    qns = [s.replace(" ", "") for s in qn]
    sns = [s.replace(" ", "") for s in sn]
    feats = {
        "n_ratio": _pd(qn, sn, fuzz.ratio),
        "n_tset": _pd(qn, sn, fuzz.token_set_ratio),
        "n_tsort": _pd(qn, sn, fuzz.token_sort_ratio),
        "n_partial": _pd(qn, sn, fuzz.partial_ratio),
        "n_jw": _pd(qn, sn, JaroWinkler.normalized_similarity),
        "f_tset": _pd(qf, sf, fuzz.token_set_ratio),
        "f_ratio": _pd(qf, sf, fuzz.ratio),
        "ns_ratio": _pd(qns, sns, fuzz.ratio),
        "ns_partial": _pd(qns, sns, fuzz.partial_ratio),
        "p_ratio": _pd(qp, sp, fuzz.ratio),
        "p_tset": _pd(qp, sp, fuzz.token_set_ratio),
        "alt_q": _pd(qalt, sn, fuzz.token_set_ratio),
        "alt_s": _pd(qn, salt, fuzz.token_set_ratio),
        "a_tset": _pd(qa_, sa_, fuzz.token_set_ratio),
        "a_tsort": _pd(qa_, sa_, fuzz.token_sort_ratio),
        "a_ratio": _pd(qa_, sa_, fuzz.ratio),
        "a_partial": _pd(qa_, sa_, fuzz.partial_ratio),
    }
    f = c.select(
        "q", "s1", "bscore", "nkeys", "brank", "brel", "bgap", "ncand", "qsrc",
        pl.col("name_core").str.len_chars().alias("qn_len"),
        pl.col("name_core_1").str.len_chars().alias("sn_len"),
        pl.col("name_core").str.count_matches(" ").alias("qn_ntok"),
        pl.col("name_core_1").str.count_matches(" ").alias("sn_ntok"),
        (pl.col("name_alt").str.len_chars() > 0).alias("q_has_alt"),
        pl.col("addr_f").str.len_chars().alias("qa_len"),
        pl.col("addr_f_1").str.len_chars().alias("sa_len"),
        pl.col("addr_nums").list.len().alias("q_nnum"),
        pl.col("addr_nums_1").list.len().alias("s_nnum"),
        pl.col("addr_nums").list.set_intersection("addr_nums_1").list.len().alias("num_inter"),
        (pl.col("addr_nums").list.first() == pl.col("addr_nums_1").list.first()).alias("num_first_eq"),
        pl.col("addr_nums").list.first().is_in(pl.col("addr_nums_1")).alias("num_first_in"),
        pl.col("addr_toks").list.set_intersection("addr_toks_1").list.len().alias("atok_inter"),
        pl.col("addr_toks").list.set_union("addr_toks_1").list.len().alias("atok_union"),
        pl.col("name_core").str.split(" ").list.set_intersection(pl.col("name_core_1").str.split(" ")).list.len().alias("ntok_inter"),
        pl.col("name_core").str.split(" ").list.set_union(pl.col("name_core_1").str.split(" ")).list.len().alias("ntok_union"),
    )
    f = f.with_columns(
        (pl.col("atok_inter") / pl.col("atok_union").clip(1)).alias("atok_jacc"),
        (pl.col("ntok_inter") / pl.col("ntok_union").clip(1)).alias("ntok_jacc"),
        pl.col("num_first_eq").fill_null(False), pl.col("num_first_in").fill_null(False),
        pl.DataFrame(feats),
    )
    f = f.with_columns([pl.col(x).cast(pl.Float32) for x in f.columns if x not in ("q", "s1")])
    # empty address -> similarity is meaningless; mark as missing
    empty = (pl.col("qa_len") == 0) | (pl.col("sa_len") == 0)
    f = f.with_columns([pl.when(empty).then(None).otherwise(pl.col(x)).alias(x)
                        for x in ("a_tset", "a_tsort", "a_ratio", "a_partial", "atok_jacc")])
    return f


FEATURES = None  # filled lazily: every column except the ids


def feature_cols(f):
    return [x for x in f.columns if x not in ("q", "s1", "label")]
