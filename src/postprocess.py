"""Per-entity decision rule that maximises expected macro F0.5.

Each query is first attached to its best-scoring Source 1 candidate.  For a
Source 1 entity with attached queries sorted by probability p_1 >= p_2 >= ...
we choose k (possibly 0) maximising the expected F0.5:

  k = 0 : E[F] = P(no true match)      ~ prod_i (1 - p_i) over all candidate
                                          links pointing at this entity
  k > 0 : E[F] ~ 1.25 * sum_{i<=k} p_i / (k + 0.25 * E|T|)

where E|T| (expected number of true matches) is the sum of probabilities of
all candidate links pointing at the entity, scaled by `alpha` to account for
matches lost in blocking.  A small floor `pmin` drops very unlikely links.
"""
import numpy as np
import polars as pl


def select(scores, alpha=1.0, pmin=0.05, beta2=0.25):
    s = scores.filter(pl.col("p") >= 0.01)
    best = s.sort(["q", "p"], descending=[False, True]).group_by("q", maintain_order=True).first()
    # expected number of true matches and probability of none, from all links to the entity
    ent = s.group_by("s1").agg(
        (pl.col("p").sum() * alpha).alias("et"),
        (1 - pl.col("p").clip(0, 0.999999)).log().sum().exp().alias("p_none"),
    )
    b = best.filter(pl.col("p") >= pmin).sort(["s1", "p"], descending=[False, True])
    b = b.with_columns(
        pl.col("p").cum_sum().over("s1").alias("cp"),
        pl.int_range(1, pl.len() + 1).over("s1").alias("k"),
    ).join(ent, on="s1", how="left")
    b = b.with_columns(((1 + beta2) * pl.col("cp") / (pl.col("k") + beta2 * pl.max_horizontal("et", pl.col("cp")))).alias("ef"))
    kbest = b.group_by("s1").agg(
        pl.col("ef").max().alias("ef_max"),
        pl.col("k").get(pl.col("ef").arg_max()).alias("kstar"),
        pl.col("p_none").first(),
    )
    keep = kbest.filter(pl.col("ef_max") > pl.col("p_none")).select("s1", "kstar")
    return b.join(keep, on="s1").filter(pl.col("k") <= pl.col("kstar")).select("s1", "q")


def tune(scores, gt, s1u, macro_f05):
    res = {}
    for alpha in (0.9, 1.0, 1.1):
        for pmin in (0.05, 0.2, 0.35):
            f, p, r = macro_f05(select(scores, alpha, pmin), gt, s1u)
            res[(alpha, pmin)] = f
            print(f"alpha {alpha} pmin {pmin}: F0.5 {f:.4f} P {p:.4f} R {r:.4f}", flush=True)
    best = max(res, key=res.get)
    print("best", best, res[best])
    return best, res[best]
