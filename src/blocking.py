"""Candidate generation (blocking).

Every record emits a set of hashed blocking keys (always prefixed with its
country, so blocks never cross countries):
  n  : core name token                 p : phonetic skeleton of a name token
  b  : adjacent core-token pair (order-insensitive)
  x  : whole core name without spaces (catches "dermatologycenter.com")
  nn : name token + house/plot number  a : adjacent address-token pair
Keys are weighted by inverse document frequency in Source 1; very frequent
keys (df > MAX_DF) are dropped.  For every Source 2/3 record ("query") the
Source 1 records sharing keys are scored by the summed key weight and the
top K are kept as candidates.
"""
import os
import subprocess
import sys
import time

import numpy as np
import polars as pl

from common import WORK_DIR

MAX_DF = int(os.environ.get("ER_MAX_DF", 300))
TOP_K = int(os.environ.get("ER_TOP_K", 12))
CHUNK = 250_000
MAX_DF_TYPE = {"a": 60}

KEY_WEIGHT = {"n": 1.0, "p": 0.6, "b": 1.0, "x": 1.0, "nn": 1.2, "a": 0.8, "al": 0.8, "xp": 0.8, "t": 1.0}


def _adjacent_pairs(tok_df, prefix, sort_pair):
    right = tok_df.select("rid", pl.col("pos") - 1, pl.col("tok").alias("tok2"))
    j = tok_df.join(right, on=["rid", "pos"])
    if sort_pair:
        a = pl.min_horizontal("tok", "tok2")
        b = pl.max_horizontal("tok", "tok2")
    else:
        a, b = pl.col("tok"), pl.col("tok2")
    return j.select("rid", "country", pl.concat_str([pl.lit(prefix + "|"), a, pl.lit(" "), b]).alias("key"))


def _explode_tokens(df, col, max_tokens=None):
    t = df.select("rid", "country", pl.col(col).alias("tok"))
    t = t.with_columns(pl.int_ranges(0, pl.col("tok").list.len()).alias("pos")).explode("tok", "pos")
    t = t.filter(pl.col("tok").is_not_null() & (pl.col("tok").str.len_chars() >= 2))
    if max_tokens is not None:
        t = t.filter(pl.col("pos") < max_tokens)
    return t


def make_keys(df):
    """df: rid, country, name_core, name_alt, name_ph, addr_toks, addr_nums -> (rid, h, kt)."""
    df = df.with_columns(
        pl.col("name_core").str.split(" ").alias("core"),
        pl.col("name_ph").str.split(" ").alias("ph"),
        pl.col("name_alt").str.split(" ").alias("alt"),
    )
    core = _explode_tokens(df, "core", 6)
    parts = [
        core.select("rid", "country", pl.concat_str([pl.lit("n|"), "tok"]).alias("key")),
        _explode_tokens(df, "alt", 4).select("rid", "country", pl.concat_str([pl.lit("al|"), "tok"]).alias("key")),
        _explode_tokens(df, "ph", 6).filter(pl.col("tok").str.len_chars() >= 3)
        .select("rid", "country", pl.concat_str([pl.lit("p|"), "tok"]).alias("key")),
        _adjacent_pairs(core, "b", True),
        df.filter(pl.col("name_core").str.len_chars() >= 5).select(
            "rid", "country", pl.concat_str([pl.lit("x|"), pl.col("name_core").str.replace_all(" ", "")]).alias("key")),
        df.with_columns(pl.col("name_core").str.replace_all(" ", "").alias("ns")).filter(pl.col("ns").str.len_chars() >= 8)
        .select("rid", "country", pl.concat_str([pl.lit("xp|"), pl.col("ns").str.slice(0, 8)]).alias("key")),
        df.filter(pl.col("core").list.len() >= 3).select(
            "rid", "country", pl.concat_str([pl.lit("t|"), pl.col("core").list.head(3).list.sort().list.join(" ")]).alias("key")),
    ]
    nums = _explode_tokens(df.with_columns(pl.col("addr_nums").list.head(2)), "addr_nums")
    nums = pl.concat([nums, df.select("rid", "country", pl.col("addr_nums").list.first().alias("tok"),
                                      pl.lit(0, pl.Int64).alias("pos"))
                      .filter(pl.col("tok").str.len_chars() == 1)], how="vertical_relaxed")
    parts.append(core.join(nums.select("rid", pl.col("tok").alias("num")), on="rid")
                 .select("rid", "country", pl.concat_str([pl.lit("nn|"), "tok", pl.lit("#"), "num"]).alias("key")))
    addr = df.select("rid", "country", pl.col("addr_toks").alias("tok"))
    addr = addr.with_columns(pl.int_ranges(0, pl.col("tok").list.len()).alias("pos")).explode("tok", "pos").drop_nulls("tok")
    parts.append(_adjacent_pairs(addr, "a", False))
    keys = pl.concat(parts)
    keys = keys.select(
        "rid",
        pl.concat_str(["country", pl.lit("|"), "key"]).hash(seed=7).alias("h"),
        pl.col("key").str.split("|").list.first().alias("kt"),
    ).unique(["rid", "h"])
    return keys


def load(split, src, cols=("idx", "country", "name_core", "name_alt", "name_ph", "addr_toks", "addr_nums")):
    return pl.read_parquet(os.path.join(WORK_DIR, f"{split}_s{src}.parquet"), columns=list(cols))


def build_index(split):
    """Hashed Source 1 keys with weights -> WORK_DIR/s1keys_{split}.parquet."""
    t0 = time.time()
    s1 = load(split, 1).rename({"idx": "rid"})
    n1 = s1.height
    k1 = make_keys(s1)
    del s1
    df = k1.group_by("h", "kt").agg(pl.len().alias("df"))
    df = df.filter(pl.col("df") <= pl.col("kt").replace_strict(MAX_DF_TYPE, default=MAX_DF, return_dtype=pl.UInt32)).drop("kt")
    df = df.with_columns((np.log(n1) - pl.col("df").log()).cast(pl.Float32).alias("idf"))
    k1 = k1.join(df, on="h").select("h", pl.col("rid").alias("s1"), "kt", "idf")
    k1 = k1.with_columns((pl.col("idf") * pl.col("kt").replace_strict(KEY_WEIGHT, return_dtype=pl.Float32)).alias("w")).drop("kt", "idf")
    k1.write_parquet(os.path.join(WORK_DIR, f"s1keys_{split}.parquet"))
    print(f"s1 keys {k1.height:,} ({time.time()-t0:.0f}s)", flush=True)


def query_frame(split):
    """Lazy concat of Source 2 and Source 3 with a global query id (S3 ids offset by len(S2))."""
    cols = ["idx", "country", "name_core", "name_alt", "name_ph", "addr_toks", "addr_nums"]
    p2 = os.path.join(WORK_DIR, f"{split}_s2.parquet")
    p3 = os.path.join(WORK_DIR, f"{split}_s3.parquet")
    n2 = pl.scan_parquet(p2).select(pl.len()).collect().item()
    s2 = pl.scan_parquet(p2).select(cols)
    s3 = pl.scan_parquet(p3).select(cols).with_columns(pl.col("idx") + n2)
    return pl.concat([s2, s3]).rename({"idx": "rid"})


def run_part(split, start, end):
    """Candidates for queries [start, end) -> WORK_DIR/cand_{split}_{start}.parquet."""
    t0 = time.time()
    k1 = pl.read_parquet(os.path.join(WORK_DIR, f"s1keys_{split}.parquet"))
    q = query_frame(split).slice(start, end - start).collect()
    out = []
    for st in range(0, q.height, CHUNK):
        kq = make_keys(q.slice(st, CHUNK)).select("rid", "h")
        j = kq.join(k1, on="h")
        sc = j.group_by("rid", "s1").agg(pl.col("w").sum().alias("bscore"), pl.len().cast(pl.UInt16).alias("nkeys"))
        sc = sc.sort(["rid", "bscore"], descending=[False, True]).group_by("rid", maintain_order=True).head(TOP_K)
        out.append(sc.with_columns(pl.col("bscore").cast(pl.Float32)))
    pl.concat(out).rename({"rid": "q"}).write_parquet(os.path.join(WORK_DIR, f"cand_{split}_{start}.parquet"))
    print(f"  part {start:,}-{end:,} done ({time.time()-t0:.0f}s)", flush=True)


def run(split, part_size=2_000_000):
    """Build the S1 index, then process queries in separate processes (bounded memory)."""
    build_index(split)
    nq = query_frame(split).select(pl.len()).collect().item()
    parts = []
    for st in range(0, nq, part_size):
        subprocess.run([sys.executable, __file__, "part", split, str(st), str(min(st + part_size, nq))], check=True)
        parts.append(os.path.join(WORK_DIR, f"cand_{split}_{st}.parquet"))
    cand = pl.concat([pl.read_parquet(p) for p in parts])
    cand.write_parquet(os.path.join(WORK_DIR, f"cand_{split}.parquet"))
    for p in parts:
        os.remove(p)
    print("wrote cand", split, cand.height, "queries with candidates", cand["q"].n_unique(), "of", nq, flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "part":
        run_part(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    else:
        run(sys.argv[1])
