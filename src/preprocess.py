"""Normalise names / addresses of every source file into parquet.

Output columns per record:
  idx, entity_id, country, name_f (folded full name), name_core (core tokens,
  legal words removed), name_alt (other side of a DBA, may be empty),
  name_ph (phonetic skeleton of core tokens), addr_f (normalised address),
  addr_toks (informative address tokens), addr_nums (digit groups).
"""
import json
import os
import re
import sys
from multiprocessing import Pool

import polars as pl

from common import (ADDR_CANON, ADDR_STOP, DATA_DIR, LEGAL_WORDS, NAME_CANON,
                    WORK_DIR, fold_text, phonetic, read_tsv)

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar",
    "california": "ca", "colorado": "co", "connecticut": "ct", "delaware": "de",
    "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
    "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne",
    "nevada": "nv", "new hampshire": "nh", "new jersey": "nj",
    "new mexico": "nm", "new york": "ny", "north carolina": "nc",
    "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or",
    "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut",
    "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
    "district of columbia": "dc",
}

DBA_RE = re.compile(r"\b(?:doing business as|d/b/a|dba|t/a|trading as|a/k/a)\b")
DOMAIN_RE = re.compile(r"\.(?:com|net|org|co\.in|in|fr|us|biz|info)\b")
NUM_RE = re.compile(r"\d+")
ORD_RE = re.compile(r"^(\d+)(?:st|nd|rd|th|er|e|eme)$")
NULL_RE = re.compile(r"<null>|\bn/a\b|\bnull\b|\bnone\b")

LEET = str.maketrans("013456789", "oleasgtbg")
LEET_RE = re.compile(r"(?=.*[a-z])(?=.*\d)")

DEV_MAP = None


def _init(dev_map):
    global DEV_MAP
    DEV_MAP = dev_map


def name_tokens(s):
    s = s.replace("&", " and ").replace("'", "").replace("`", "")
    s = DOMAIN_RE.sub(" ", s)
    s = s.replace("www.", " ").replace(".", "")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    toks = []
    for t in s.split():
        if t.isdigit() and len(t) >= 7:          # phone numbers appended to names
            continue
        if LEET_RE.search(t) and not t.isdigit():  # "antim0ny", "federa1", "6roup"
            t = t.translate(LEET)
        if len(t) > 8 and t.endswith("com"):      # "dermatologycentercom"
            t = t[:-3]
        t = NAME_CANON.get(t, t)
        if t:
            toks.extend(t.split())
    return toks


def norm_name(raw):
    s = fold_text(raw, DEV_MAP)
    alt = ""
    parts = [p for p in DBA_RE.split(s) if p.strip()]
    if len(parts) >= 2:
        s, alt = parts[-1], parts[0]
    toks = name_tokens(s)
    core = [t for t in toks if t not in LEGAL_WORDS]
    if not core:
        core = toks
    full = " ".join(toks)
    alt_core = [t for t in name_tokens(alt) if t not in LEGAL_WORDS] if alt else []
    ph = " ".join(phonetic(t) for t in core)
    return full, " ".join(core), " ".join(alt_core), ph


def norm_addr(raw):
    if raw is None:
        return "", [], []
    s = fold_text(raw)
    s = NULL_RE.sub(" ", s)
    for full, ab in US_STATES.items():
        if full in s:
            s = re.sub(r"\b" + full + r"\b", ab, s)
    s = s.replace("'", "").replace(".", " ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    toks = []
    for t in s.split():
        m = ORD_RE.match(t)
        if m:
            t = m.group(1)
        toks.append(ADDR_CANON.get(t, t))
    nums = []
    for t in toks:
        for n in NUM_RE.findall(t):
            n = n.lstrip("0") or "0"
            if n not in nums:
                nums.append(n)
    info = [t for t in toks if t not in ADDR_STOP]
    return " ".join(toks), info, nums


def process_rows(rows):
    out = []
    for eid, name, addr in rows:
        full, core, alt, ph = norm_name(name or "")
        af, atoks, nums = norm_addr(addr)
        out.append((full, core, alt, ph, af, atoks, nums))
    cols = list(zip(*out))
    return pl.DataFrame({
        "name_f": pl.Series(cols[0], dtype=pl.Utf8),
        "name_core": pl.Series(cols[1], dtype=pl.Utf8),
        "name_alt": pl.Series(cols[2], dtype=pl.Utf8),
        "name_ph": pl.Series(cols[3], dtype=pl.Utf8),
        "addr_f": pl.Series(cols[4], dtype=pl.Utf8),
        "addr_toks": pl.Series(cols[5], dtype=pl.List(pl.Utf8)),
        "addr_nums": pl.Series(cols[6], dtype=pl.List(pl.Utf8)),
    })


def process_file(split, src, pool):
    df = read_tsv(f"{DATA_DIR}/{split}/{split}_source{src}.tsv")
    chunk = 25000
    base = df.select("entity_id", "business_name", "business_address")
    tasks = (base.slice(i, chunk).rows() for i in range(0, df.height, chunk))
    parts = list(pool.imap(process_rows, tasks))
    res = pl.concat(parts)
    out = pl.concat([
        pl.DataFrame({"idx": pl.Series(range(df.height), dtype=pl.UInt32)}),
        df.select("entity_id", pl.col("country").fill_null(""),
                  pl.col("business_name").fill_null("").alias("name_raw"),
                  pl.col("business_address").fill_null("").alias("addr_raw")),
        res,
    ], how="horizontal")
    path = os.path.join(WORK_DIR, f"{split}_s{src}.parquet")
    out.write_parquet(path)
    print("wrote", path, out.height, flush=True)


def main():
    splits = sys.argv[1:] or ["train", "test"]
    with open(os.path.join(WORK_DIR, "dev_map.json")) as f:
        dev_map = json.load(f)
    with Pool(os.cpu_count(), initializer=_init, initargs=(dev_map,)) as pool:
        for split in splits:
            for src in (1, 2, 3):
                process_file(split, src, pool)


if __name__ == "__main__":
    main()
