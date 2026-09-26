"""Shared constants, paths and text-normalisation helpers."""
import os
import re

import polars as pl
from unidecode import unidecode

DATA_DIR = os.environ.get("ER_DATA_DIR", "/home/user/data/student_resource/dataset")
WORK_DIR = os.environ.get("ER_WORK_DIR", "/home/user/work")
os.makedirs(WORK_DIR, exist_ok=True)

DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
NON_ASCII_RE = re.compile(r"[^\x00-\x7F]")


def read_tsv(path):
    """Read a challenge TSV with every column as a string (no quoting)."""
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)


# ---------------------------------------------------------------------------
# Name vocabulary.  Legal forms / filler words are canonicalised, then removed
# from the "core" name that drives blocking and most similarity features.
# ---------------------------------------------------------------------------
NAME_CANON = {
    "pvt": "private", "prvt": "private", "pvtltd": "private limited",
    "ltd": "limited", "ltda": "limited", "lmtd": "limited",
    "corp": "corporation", "corpn": "corporation",
    "co": "company", "cos": "company", "coy": "company",
    "inc": "incorporated", "incorp": "incorporated",
    "l": "", "n": "", "&": "and", "et": "and",
    "intl": "international", "mfg": "manufacturing", "svcs": "services",
    "svc": "services", "mgmt": "management", "tech": "technology",
    "technologies": "technology", "ent": "enterprises", "bros": "brothers",
    "assoc": "associates", "dept": "department", "univ": "university",
    "ctr": "center", "centre": "center", "etablissements": "ets",
    "etablissement": "ets", "societe": "ste", "st": "saint",
}

LEGAL_WORDS = {
    "private", "limited", "llc", "incorporated", "corporation", "company",
    "llp", "pllc", "lp", "plc", "pc", "pa", "ltd", "the", "and", "of", "m", "s",
    "ms", "smt", "sarl", "sas", "sasu", "sa", "sci", "eurl", "snc", "ets", "ste",
    "dba", "aka", "com", "www", "net", "org", "in", "public", "opc", "de", "du",
    "des", "la", "le", "les", "et", "d", "a", "l", "fils", "cie", "group",
}

ADDR_CANON = {
    "street": "st", "str": "st", "road": "rd", "avenue": "ave", "av": "ave",
    "avn": "ave", "drive": "dr", "court": "ct", "place": "pl", "lane": "ln",
    "boulevard": "blvd", "bd": "blvd", "boul": "blvd", "turnpike": "tpke",
    "tpk": "tpke", "highway": "hwy", "parkway": "pkwy", "circle": "cir",
    "terrace": "ter", "square": "sq", "trail": "trl", "expressway": "expy",
    "suite": "ste", "apartment": "apt", "appartement": "apt", "building": "bldg",
    "floor": "fl", "north": "n", "south": "s", "east": "e", "west": "w",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
    "rue": "r", "allee": "all", "chemin": "ch", "impasse": "imp", "route": "rte",
    "quai": "qu", "cours": "crs", "faubourg": "fbg", "residence": "res",
    "near": "nr", "opposite": "opp", "opp": "opp", "number": "no",
    "nagar": "ngr", "sector": "sec", "marg": "mg", "colony": "col",
    "mount": "mt", "fort": "ft", "saint": "st", "sainte": "ste",
    "township": "twp", "townhip": "twp", "road.": "rd",
}

ADDR_STOP = {
    "no", "na", "null", "n", "a", "of", "the", "and", "de", "du", "des", "la",
    "le", "les", "d", "l", "h", "door", "plot", "house", "flat", "shop",
    "unit", "apt", "ste", "fl", "floor", "bldg", "nr", "opp", "india", "usa",
    "france", "us", "bis", "ter",
}


def fold_text(s, dev_map=None):
    """Fold a raw string to lowercase ASCII.

    Devanagari tokens are first looked up in the learned transliteration
    dictionary (built from training pairs); everything else goes through
    unidecode.
    """
    if s is None:
        return ""
    if NON_ASCII_RE.search(s) is None:
        return s.lower()
    if dev_map is not None and DEVANAGARI_RE.search(s):
        out = []
        for tok in s.split():
            core = tok.strip(".,()[]-")
            if core in dev_map:
                out.append(dev_map[core])
            else:
                out.append(unidecode(tok))
        s = " ".join(out)
    else:
        s = unidecode(s)
    return s.lower()


def phonetic(tok):
    """Crude consonant skeleton, robust to vowel typos and transliteration."""
    if not tok:
        return tok
    t = tok
    for a, b in (("ph", "f"), ("kh", "k"), ("gh", "g"), ("th", "t"),
                 ("dh", "d"), ("bh", "b"), ("sh", "s"), ("ch", "c"),
                 ("ck", "k"), ("q", "k"), ("z", "s"), ("w", "v"), ("x", "ks")):
        t = t.replace(a, b)
    t = t.replace("c", "k")
    first = t[0]
    rest = re.sub(r"[aeiouyh]", "", t[1:])
    t = first + rest
    return re.sub(r"(.)\1+", r"\1", t)
