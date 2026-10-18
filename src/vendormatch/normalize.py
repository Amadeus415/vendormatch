"""Stage 1: turn a raw vendor string into comparable tokens.

The goal is to remove differences that never mean "different company" (case, punctuation, legal form,
processor prefixes, store numbers, attention lines) and keep everything that might.
"""

from __future__ import annotations

import re
import unicodedata

from rapidfuzz.distance import OSA

# Legal-form words. Stripped from the end of a name, repeatedly ("Acme Holdings Co., Ltd." -> "acme hldg").
LEGAL_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "co", "company", "cos", "companies",
    "llc", "ltd", "limited", "plc", "lp", "llp", "pllc", "pc", "lc",
    "sa", "ag", "nv", "bv", "se", "spa", "sarl", "gmbh", "kk", "pte", "pty", "oyj", "asa", "ab",
    "com", "the",
}  # fmt: skip

# Spellings that mean the same word. Both sides map to one short canonical token.
CANONICAL = {
    "international": "intl", "intl": "intl", "internat": "intl",
    "national": "natl", "natl": "natl",
    "technologies": "tech", "technology": "tech", "tech": "tech", "techs": "tech",
    "systems": "sys", "system": "sys", "sys": "sys",
    "services": "svc", "service": "svc", "svcs": "svc", "svc": "svc", "srvcs": "svc",
    "manufacturing": "mfg", "mfg": "mfg",
    "laboratories": "lab", "laboratory": "lab", "labs": "lab", "lab": "lab",
    "pharmaceuticals": "pharma", "pharmaceutical": "pharma", "pharma": "pharma",
    "communications": "comm", "communication": "comm", "comms": "comm", "comm": "comm",
    "industries": "ind", "industry": "ind", "inds": "ind", "indus": "ind",
    "financial": "fin", "finl": "fin", "finance": "fin",
    "brothers": "bros", "bros": "bros",
    "associates": "assoc", "assoc": "assoc", "assocs": "assoc",
    "association": "assn", "assn": "assn",
    "management": "mgmt", "mgmt": "mgmt", "mgt": "mgmt",
    "holdings": "hldg", "holding": "hldg", "hldgs": "hldg", "hldg": "hldg",
    "group": "grp", "grp": "grp",
    "solutions": "soln", "solution": "soln", "solns": "soln",
    "products": "prod", "product": "prod", "prods": "prod",
    "properties": "prop", "property": "prop", "props": "prop",
    "entertainment": "ent", "entmt": "ent",
    "investments": "invest", "investment": "invest", "invt": "invest",
    "semiconductors": "semi", "semiconductor": "semi",
    "electric": "elec", "electrical": "elec", "elec": "elec",
    "airlines": "airline", "airline": "airline",
    "bancorporation": "bancorp",
    "resources": "rsrc", "resource": "rsrc",
    "energies": "energy",
    "therapeutics": "thera", "therapeutic": "thera",
    "biosciences": "bio", "bioscience": "bio",
    "partners": "ptnr", "partner": "ptnr", "ptnrs": "ptnr",
    "development": "dev", "dev": "dev",
    "and": "&",
}  # fmt: skip

# Words that carry little identity. Ignored when building initials and some features.
WEAK = {"&", "of", "the", "a", "an", "hldg", "grp", "intl", "co"}

SEC_TAG_END = re.compile(r"\s*[/\\][A-Z0-9 .]{0,6}[/\\]?\s*$", re.IGNORECASE)  # "INC /KY/", "CORP/"
SEC_TAG_MID = re.compile(r"\s+[/\\][A-Z]{2,3}[/\\]\s*", re.IGNORECASE)
ENDS_WITH_SUFFIX = re.compile(r"\b(inc|corp|co|ltd|llc|plc|lp)\.?$", re.IGNORECASE)
# Card processors that put their own name in front of the merchant's: "SQ *BLUE BOTTLE", "TST* JOES".
PROCESSORS = {"sq", "tst", "paypal", "pp", "sp", "in", "fs", "py", "cko", "dri", "2co", "paddle", "stripe", "squ"}
REF_CODE = re.compile(r"^(?=[a-z]*\d)[a-z0-9]{4,}$", re.IGNORECASE)  # "2K4TW1" after "AMAZON MKTP*"
STORE_NO = re.compile(r"(#\s*\d+|\b(store|str|loc|unit|no)\.?\s*#?\s*\d{2,}\b|\s\d{3,}\s*$)", re.IGNORECASE)
ATTENTION = re.compile(r"\b(attn|attention|c/o|care of|remit to|remittance)\b.*$", re.IGNORECASE)
LEADING_REMIT = re.compile(r"^\s*(remit to|remit|pay to)\s*:?\s*", re.IGNORECASE)
ALIAS_SPLIT = re.compile(r"\s*\b(?:d/b/a|dba|d\.b\.a\.|aka|a/k/a|fka|f/k/a|formerly)\b\.?\s*", re.IGNORECASE)
NON_ALNUM = re.compile(r"[^a-z0-9& ]+")
SPACES = re.compile(r"\s+")


def ascii_fold(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def strip_sec_tag(s: str) -> str:
    """SEC disambiguates same-named filers with tags: 'REPUBLIC BANCORP INC /KY/', 'SOUTHERN COPPER CORP/'.
    Only strip a slash tag after whitespace or a legal suffix, so names like 'AC/DC' survive."""
    m = SEC_TAG_END.search(s)
    if m and (m.group(0)[0].isspace() or ENDS_WITH_SUFFIX.search(s[: m.start()])):
        s = s[: m.start()]
    return SEC_TAG_MID.sub(" ", s)


def strip_star(s: str) -> str:
    """Card descriptors use '*' two ways: 'SQ *MERCHANT' (processor first) and 'MERCHANT*REFCODE'."""
    if "*" not in s:
        return s
    left, right = (part.strip() for part in s.split("*", 1))
    if left.casefold().replace(" ", "") in PROCESSORS:
        return strip_star(right)
    if REF_CODE.match(right) or not right:
        return left
    return strip_star(f"{left} {right}")


def clean_descriptor(raw: str) -> str:
    """Remove card-feed and AP-system wrapping around the vendor name (keeps case)."""
    s = ascii_fold(raw or "")
    s = LEADING_REMIT.sub("", s)
    s = strip_sec_tag(s)
    s = strip_star(s)
    s = ATTENTION.sub("", s)
    s = STORE_NO.sub(" ", s)
    return s.strip(" -,*")


def tokens(text: str) -> list[str]:
    """Casefold, drop punctuation, canonicalize abbreviations, strip legal suffixes."""
    s = text.casefold().replace("&", " & ").replace("'", "")
    s = NON_ALNUM.sub(" ", s)
    s = re.sub(r"\b([a-z])\s(?=[a-z]\b)", r"\1", s)  # "l l c" -> "llc", "i b m" -> "ibm"
    toks = [CANONICAL.get(t, t) for t in SPACES.sub(" ", s).strip().split(" ") if t]
    while len(toks) > 1 and (toks[-1] in LEGAL_SUFFIXES or toks[-1] == "&" or looks_like_suffix(toks[-1])):
        toks.pop()
    while len(toks) > 1 and toks[0] in ("the", "&"):
        toks.pop(0)
    return toks


LONG_SUFFIXES = ("incorporated", "corporation", "company", "limited")
TYPO_SUFFIXES = ("inc", "corp", "llc", "ltd", "plc", *LONG_SUFFIXES)
KNOWN_WORDS = set(CANONICAL.values())


def looks_like_suffix(tok: str) -> bool:
    """A legal suffix that was cut off or mistyped: 'INCOR' (22-char card limit), 'Corl', 'Nic'.
    Only tokens of 3+ letters that are not known words, so 'ind' (industries) is never mistaken for 'inc'."""
    if len(tok) < 3 or tok in KNOWN_WORDS or not tok.isalpha():
        return False
    if len(tok) >= 4 and any(s.startswith(tok) for s in LONG_SUFFIXES):
        return True
    return any(OSA.distance(tok, s) <= 1 for s in TYPO_SUFFIXES if abs(len(s) - len(tok)) <= 1)


def name_variants(raw: str) -> list[str]:
    """All normalized names a raw string stands for: the full name plus each side of a dba/fka."""
    cleaned = clean_descriptor(raw)
    parts = [p for p in ALIAS_SPLIT.split(cleaned) if p.strip()]
    out = []
    for part in parts:
        norm = " ".join(tokens(part))
        if norm and norm not in out:
            out.append(norm)
    return out or [""]


def normalize(raw: str) -> str:
    """The primary normalized name (the part before any dba)."""
    return name_variants(raw)[0]


def initials(norm: str) -> str:
    """'intl business machines' -> 'ibm'. Empty when there are fewer than two strong tokens."""
    strong = [t for t in norm.split() if t not in WEAK or t == "intl"]
    if len(strong) < 2:
        return ""
    expanded = {"intl": "i", "natl": "n", "tech": "t", "sys": "s", "svc": "s", "mfg": "m", "fin": "f"}
    return "".join(expanded.get(t, t[0]) for t in strong if t[0].isalpha())


STREET_WORDS = {
    "street": "st", "st": "st", "avenue": "ave", "ave": "ave", "av": "ave", "boulevard": "blvd", "blvd": "blvd",
    "drive": "dr", "dr": "dr", "road": "rd", "rd": "rd", "suite": "ste", "ste": "ste", "lane": "ln", "ln": "ln",
    "parkway": "pkwy", "pkwy": "pkwy", "highway": "hwy", "hwy": "hwy", "place": "pl", "pl": "pl",
    "court": "ct", "ct": "ct", "floor": "fl", "fl": "fl", "north": "n", "south": "s", "east": "e",
    "west": "w", "one": "1", "two": "2", "three": "3", "center": "ctr", "centre": "ctr", "ctr": "ctr",
}  # fmt: skip


def normalize_street(raw: str | None) -> str:
    if not raw:
        return ""
    s = NON_ALNUM.sub(" ", ascii_fold(raw).casefold())
    return " ".join(STREET_WORDS.get(t, t) for t in s.split())


def normalize_city(raw: str | None) -> str:
    if not raw:
        return ""
    return " ".join(NON_ALNUM.sub(" ", ascii_fold(raw).casefold()).split())


def normalize_zip(raw: str | None) -> str:
    digits = re.sub(r"\D", "", raw or "")
    return digits[:5] if len(digits) >= 5 else ""
