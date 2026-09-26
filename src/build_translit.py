"""Learn a Devanagari -> Latin token dictionary from the training ground truth.

Some Source 2/3 names are written in Devanagari while the matching Source 1
name is Latin.  For matched pairs with the same number of tokens we align the
tokens by position and keep, for each Devanagari token, its most frequent
Latin counterpart.  Only training data is used.
"""
import json
import os
import re
from collections import Counter, defaultdict

import polars as pl

from common import DATA_DIR, DEVANAGARI_RE, WORK_DIR, read_tsv


MANUAL = {
    "हार्डवेयर": "hardware", "मोटर्स": "motors", "प्रोविजन": "provision",
    "टेक्सटाइल्स": "textiles", "फार्मेसी": "pharmacy", "बेकरी": "bakery",
    "जनरल": "general", "स्वीट्स": "sweets", "एजेंसीज": "agencies",
    "स्टील": "steel", "फर्नीचर": "furniture", "इलेक्ट्रॉनिक्स": "electronics",
    "रेस्टोरेंट": "restaurant", "स्टोर्स": "stores", "ऑटोमोबाइल्स": "automobiles",
    "गारमेंट्स": "garments", "मेडिकल्स": "medicals", "ट्रेडर्स": "traders",
    "ज्वेलर्स": "jewellers",
}


def clean_tok(t):
    return t.strip(".,()[]-")


def main():
    s1 = read_tsv(f"{DATA_DIR}/train/train_source1.tsv").select(
        "entity_id", pl.col("business_name").alias("n1"))
    other = pl.concat([
        read_tsv(f"{DATA_DIR}/train/train_source2.tsv"),
        read_tsv(f"{DATA_DIR}/train/train_source3.tsv"),
    ]).filter(pl.col("business_name").str.contains("[ऀ-ॿ]")).select(
        "entity_id", pl.col("business_name").alias("n2"))
    gt = read_tsv(f"{DATA_DIR}/train/train_ground_truth.tsv")
    pairs = (gt.with_columns(pl.col("matched_entity_ids").str.split(","))
             .explode("matched_entity_ids")
             .join(other, left_on="matched_entity_ids", right_on="entity_id")
             .join(s1, left_on="source1_entity_id", right_on="entity_id"))
    counts = defaultdict(Counter)
    for n2, n1 in pairs.select("n2", "n1").iter_rows():
        t2 = [clean_tok(t) for t in n2.split()]
        t1 = [re.sub(r"[^a-z0-9]", "", t.lower()) for t in n1.split()]
        t2 = [t for t in t2 if t]
        t1 = [t for t in t1 if t]
        if len(t1) != len(t2):
            continue
        for a, b in zip(t2, t1):
            if DEVANAGARI_RE.search(a) and b:
                counts[a][b] += 1
    mapping = {}
    for a, c in counts.items():
        b, n = c.most_common(1)[0]
        if n >= 2 and n / sum(c.values()) >= 0.5:
            mapping[a] = b
    # Common shop-type words that never occur in length-aligned training pairs
    # (plain transliterations, added by hand).
    for a, b in MANUAL.items():
        mapping.setdefault(a, b)
    with open(os.path.join(WORK_DIR, "dev_map.json"), "w") as f:
        json.dump(mapping, f, ensure_ascii=False)
    print("pairs", pairs.height, "dictionary size", len(mapping))


if __name__ == "__main__":
    main()
