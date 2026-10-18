"""Generate the vendor benchmark: messy records for real companies, with ground truth by construction.

Each real company (an SEC registrant, identified by CIK) gets 1 to 8 records, each from one of four
"systems" that a finance team would actually have:

  erp   vendor master   legal name with inconsistent suffix/case/abbreviation, full address
  ap    AP invoices     payee name, typos, attention lines, dba or former names, sometimes a remit-to lockbox
  card  card feed       22-character uppercase descriptor, processor prefix, store number, charge city
  crm   procurement     common name or alias (from Wikidata), sometimes an acronym, city/state only

Every corruption applied to a record is logged in its `noise` column, so errors can be broken down by
cause. Different companies with confusable names (Delta Air Lines vs Delta Apparel, First Bancorp vs
First Busey) are not planted: they are already all over the SEC list, and records of both end up in
the same split.

    uv run python -m bench.generate            # data/dev.csv, data/test.csv
    uv run python -m bench.generate --scale    # also data/scale.csv (~50k records, companies not in dev)
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import string
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from vendormatch.normalize import initials, normalize, strip_sec_tag

SEED = 20261017
N_DEV, N_TEST = 2000, 1500
RECORDS_PER_ENTITY = {1: 0.22, 2: 0.25, 3: 0.20, 4: 0.13, 5: 0.08, 6: 0.06, 7: 0.04, 8: 0.02}
SOURCES = {"erp": 0.30, "ap": 0.30, "card": 0.25, "crm": 0.15}
FIELDS = ["record_id", "source", "name", "street", "city", "state", "zip", "entity_id", "noise"]

SUFFIX_FORMS = {
    "inc": ["Inc", "Inc.", ", Inc.", "Incorporated", "INC"],
    "corp": ["Corp", "Corp.", "Corporation", ", Corp."],
    "co": ["Co", "Co.", "Company", "& Co"],
    "llc": ["LLC", "L.L.C.", ", LLC"],
    "ltd": ["Ltd", "Ltd.", "Limited"],
    "plc": ["PLC", "plc", "Public Limited Company"],
    "lp": ["LP", "L.P."],
}
SUFFIX_FAMILY = {
    "inc": "inc", "incorporated": "inc", "corp": "corp", "corporation": "corp", "co": "co",
    "company": "co", "llc": "llc", "ltd": "ltd", "limited": "ltd", "plc": "plc", "lp": "lp",
}  # fmt: skip

# Abbreviations clerks actually type. Deliberately broader than the normalizer's dictionary: about half of
# these (Insurance->Ins, Medical->Med, Capital->Cap, ...) are unknown to it and must be caught by fuzzy scoring.
ABBREVIATIONS = {
    "international": "Intl", "corporation": "Corp", "company": "Co", "technologies": "Tech",
    "technology": "Tech", "systems": "Sys", "services": "Svcs", "manufacturing": "Mfg",
    "laboratories": "Labs", "pharmaceuticals": "Pharma", "communications": "Comm", "industries": "Inds",
    "financial": "Finl", "brothers": "Bros", "associates": "Assoc", "management": "Mgmt",
    "holdings": "Hldgs", "group": "Grp", "solutions": "Solns", "national": "Natl",
    "insurance": "Ins", "equipment": "Equip", "engineering": "Engrg", "pacific": "Pac", "medical": "Med",
    "capital": "Cap", "realty": "Rlty", "mortgage": "Mtg", "securities": "Secs", "america": "Amer",
    "american": "Amer", "bank": "Bk", "trust": "Tr", "energy": "Engy", "resources": "Res",
    "investors": "Invs", "products": "Prods", "properties": "Props", "development": "Dev",
    "partners": "Ptnrs", "therapeutics": "Thera", "acquisition": "Acq", "entertainment": "Ent",
    "electronics": "Elec", "electric": "Elec", "northern": "No", "southern": "So",
}  # fmt: skip

PROCESSORS = ["SQ *", "TST* ", "PAYPAL *", "SP * ", "IN *"]
ATTENTION = [" Attn: Accounts Receivable", " ATTN AR DEPT", " C/O Lockbox Services", " Attn Billing"]
LOCKBOXES = [
    ("CHICAGO", "IL", "606"), ("PITTSBURGH", "PA", "152"), ("ATLANTA", "GA", "303"), ("DALLAS", "TX", "752"),
    ("LOS ANGELES", "CA", "900"), ("PHILADELPHIA", "PA", "191"), ("CHARLOTTE", "NC", "282"),
    ("NEWARK", "NJ", "071"), ("BOSTON", "MA", "022"), ("ST LOUIS", "MO", "631"),
]  # fmt: skip
STREET_ABBR = {
    "STREET": "ST", "AVENUE": "AVE", "BOULEVARD": "BLVD", "DRIVE": "DR", "ROAD": "RD", "SUITE": "STE",
    "PARKWAY": "PKWY", "HIGHWAY": "HWY", "PLACE": "PL", "COURT": "CT", "FLOOR": "FL", "LANE": "LN",
    "NORTH": "N", "SOUTH": "S", "EAST": "E", "WEST": "W", "CENTER": "CTR",
}  # fmt: skip
KEYBOARD = {
    "q": "wa", "w": "qes", "e": "wrd", "r": "etf", "t": "ryg", "y": "tuh", "u": "yij", "i": "uok",
    "o": "ipl", "p": "ol", "a": "qsz", "s": "awdz", "d": "sefx", "f": "drgc", "g": "fthv", "h": "gyjb",
    "j": "hukn", "k": "jilm", "l": "kop", "z": "asx", "x": "zsdc", "c": "xdfv", "v": "cfgb", "b": "vghn",
    "n": "bhjm", "m": "njk",
}  # fmt: skip


@dataclass
class Entity:
    cik: int
    name: str
    street: str
    city: str
    state: str
    zip: str
    tickers: list[str] = field(default_factory=list)
    former: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)


def clean_sec_name(name: str) -> str:
    """'REPUBLIC BANCORP INC /KY/' -> 'REPUBLIC BANCORP INC'. SEC adds these tags; vendor records don't."""
    prev = None
    while prev != name:
        prev, name = name, strip_sec_tag(name).strip(" ,")
    return name


def load_entities(raw_dir: Path) -> list[Entity]:
    """Operating companies from SEC, with Wikidata aliases. Drops a company when another company has the
    same normalized name and the same address (nothing in a vendor record could tell them apart)."""
    alias_rows: dict[int, set[str]] = defaultdict(set)
    with (raw_dir / "wikidata_aliases.csv").open() as f:
        for r in csv.DictReader(f):
            for a in (r["label"], r["alias"]):
                a = re.sub(r"\s*\(.*?\)\s*", " ", a or "").strip()
                if a and a.isascii() and 2 <= len(a) <= 40 and ":" not in a:
                    alias_rows[int(r["cik"])].add(a)
    entities = []
    for line in (raw_dir / "sec_companies.jsonl").open():
        r = json.loads(line)
        if r["entity_type"] != "operating" or not r["name"]:
            continue
        addr = r["business_address"] if r["business_address"].get("city") else r["mailing_address"]
        name = clean_sec_name(r["name"])
        norm = normalize(name)
        if not norm:
            continue
        former = []
        for f in r["former_names"][:3]:  # the most recent few renames
            if f["to"] < "2005":  # a name retired 20 years ago is unlikely to sit in an AP system
                continue
            fn = clean_sec_name(f["name"])
            if normalize(fn) not in (norm, "") and fn not in former:
                former.append(fn)
        entities.append(
            Entity(
                cik=r["cik"],
                name=name,
                street=addr.get("street") or "",
                city=addr.get("city") or "",
                state=addr.get("state") or "",
                zip=addr.get("zip") or "",
                tickers=[t.upper() for t in r["tickers"]],
                former=former,
                aliases=sorted(alias_rows.get(r["cik"], ())),
            )
        )
    # Aliases must be useful (share a word with the name, or be an acronym of it) and unambiguous.
    by_norm = defaultdict(list)
    for e in entities:
        by_norm[normalize(e.name)].append(e)
    for e in entities:
        own = set(normalize(e.name).split()) | {t for f in e.former for t in normalize(f).split()}
        keep = []
        for a in e.aliases:
            na = normalize(a)
            if not na or na == normalize(e.name) or na in by_norm:
                continue
            if set(na.split()) & own or na.replace(" ", "") == initials(normalize(e.name)):
                keep.append(a)
        e.aliases = keep
    alias_owner = defaultdict(set)
    for e in entities:
        for a in e.aliases:
            alias_owner[normalize(a)].add(e.cik)
    for e in entities:
        e.aliases = [a for a in e.aliases if len(alias_owner[normalize(a)]) == 1]
    seen, out = set(), []
    for e in sorted(entities, key=lambda e: e.cik):
        key = (normalize(e.name), e.street.upper(), e.city.upper())
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


# ---- corruptions -----------------------------------------------------------------------------------------


def split_suffix(name: str) -> tuple[str, str | None]:
    m = re.search(r"[\s,]+(inc|incorporated|corp|corporation|co|company|llc|ltd|limited|plc|lp)\.?$", name, re.I)
    if not m:
        return name, None
    return name[: m.start()], SUFFIX_FAMILY[m.group(1).lower()]


def vary_suffix(rng: random.Random, name: str) -> str:
    base, fam = split_suffix(name)
    roll = rng.random()
    if fam is None:
        return base + " " + rng.choice(["Inc", "Corp", "LLC", "Co"]) if roll < 0.3 else base
    if roll < 0.35:
        return base  # dropped
    swappable = ("inc", "corp", "co", "llc")
    family = rng.choice(swappable) if roll > 0.85 and fam in swappable else fam  # Inc typed as Corp
    form = rng.choice(SUFFIX_FORMS[family])
    return base + form if form.startswith(",") else base + " " + form


def vary_case(rng: random.Random, name: str) -> str:
    return rng.choice([name.upper(), string.capwords(name.lower()), name, name])


def vary_punct(rng: random.Random, name: str) -> str:
    s = name.replace(".", "").replace(",", "")
    if "&" in s and rng.random() < 0.5:
        s = s.replace("&", "and")
    elif " and " in s.lower() and rng.random() < 0.5:
        s = re.sub(r"\band\b", "&", s, flags=re.I)
    if "-" in s:
        s = s.replace("-", rng.choice([" ", ""]))
    return re.sub(r"\s+", " ", s).strip()


def abbreviate(rng: random.Random, name: str) -> tuple[str, bool]:
    words = name.split()
    hits = [i for i, w in enumerate(words) if w.lower().strip(",.") in ABBREVIATIONS]
    if not hits:
        return name, False
    for i in rng.sample(hits, k=max(1, len(hits) // 2 + (rng.random() < 0.5))):
        abbr = ABBREVIATIONS[words[i].lower().strip(",.")]
        words[i] = abbr.upper() if words[i].isupper() else abbr
    return " ".join(words), True


def typo(rng: random.Random, name: str) -> str:
    letters = [i for i, c in enumerate(name) if c.isalpha() and i > 0]
    if not letters:
        return name
    i = rng.choice(letters)
    c = name[i]
    op = rng.choice(["delete", "insert", "substitute", "transpose"])
    near = KEYBOARD.get(c.lower(), "e")
    sub = rng.choice(near)
    sub = sub.upper() if c.isupper() else sub
    if op == "delete":
        return name[:i] + name[i + 1 :]
    if op == "insert":
        return name[:i] + sub + name[i:]
    if op == "substitute":
        return name[:i] + sub + name[i + 1 :]
    if i + 1 < len(name) and name[i + 1].isalpha():
        return name[:i] + name[i + 1] + c + name[i + 2 :]
    return name[:i] + name[i + 1 :]


def acronym(name: str) -> str:
    return initials(normalize(name)).upper()


def vary_street(rng: random.Random, street: str) -> str:
    s = street.upper()
    for long, short in STREET_ABBR.items():
        if rng.random() < 0.5:
            s = re.sub(rf"\b{long}\b", short, s)
        elif rng.random() < 0.3:
            s = re.sub(rf"\b{short}\b", long, s)
    if rng.random() < 0.3:
        s = re.sub(r",?\s*(SUITE|STE|FL|FLOOR|UNIT|#)\s*[\w-]+", "", s)
    return vary_case(rng, s.replace(".", "").strip())


def vary_zip(rng: random.Random, z: str) -> str:
    digits = re.sub(r"\D", "", z)
    if len(digits) >= 9 and rng.random() < 0.5:
        return f"{digits[:5]}-{digits[5:9]}"
    return digits[:5] if len(digits) >= 5 else z


# ---- records ---------------------------------------------------------------------------------------------


def base_name(rng: random.Random, e: Entity, source: str, noise: list[str]) -> str:
    """Pick which real name this record starts from: legal name, alias, or a former name."""
    roll = rng.random()
    if source == "crm":
        if e.aliases and roll < 0.45:
            noise.append("alias")
            return rng.choice(e.aliases)
        if roll < 0.60 and acronym(e.name) in e.tickers:  # IBM, AMD, ADP: initials that are also the ticker
            noise.append("acronym")
            return acronym(e.name)
        if e.former and roll < 0.62:
            noise.append("former_name")
            return e.former[0]
    if source == "card" and e.aliases and roll < 0.35:
        noise.append("alias")
        return rng.choice(e.aliases)
    if source == "ap" and e.former and roll < 0.10:
        noise.append("former_name")
        return e.former[0]
    return e.name


def make_record(rng: random.Random, e: Entity, source: str, pool: list[Entity]) -> dict:
    noise: list[str] = []
    name = base_name(rng, e, source, noise)
    rec = {"source": source, "street": "", "city": "", "state": "", "zip": ""}

    if source in ("erp", "ap", "crm"):
        if rng.random() < 0.6:
            new = vary_suffix(rng, name)
            if new != name:
                noise.append("suffix")
            name = new
        if rng.random() < 0.3:
            name, hit = abbreviate(rng, name)
            if hit:
                noise.append("abbreviation")
        if rng.random() < 0.4:
            name = vary_punct(rng, name)
            noise.append("punctuation")
        if rng.random() < (0.25 if source == "ap" else 0.12):
            name = typo(rng, name)
            noise.append("typo")
            if rng.random() < 0.25:
                name = typo(rng, name)
        if source == "ap" and rng.random() < 0.10:
            name += rng.choice(ATTENTION)
            noise.append("attention_line")
        if source == "ap" and e.aliases and rng.random() < 0.12:
            name = f"{name} DBA {rng.choice(e.aliases)}"
            noise.append("dba")
        if source == "ap" and rng.random() < 0.08:
            name = name[: rng.randint(24, 32)].rstrip()
            noise.append("truncated")
        name = vary_case(rng, name)

    if source == "card":
        name = vary_punct(rng, vary_suffix(rng, name) if rng.random() < 0.5 else split_suffix(name)[0]).upper()
        noise.append("card_descriptor")
        if rng.random() < 0.15:
            name = typo(rng, name)
            noise.append("typo")
        roll = rng.random()
        if roll < 0.30:  # small merchants: the processor's tag goes first
            name = rng.choice(PROCESSORS) + name
            noise.append("processor_prefix")
        elif roll < 0.55:  # chains: a location number goes last
            store = rng.choice([" #", " #", " ", " STORE "]) + str(rng.randint(10, 9999))
            name = name[: 22 - len(store)].rstrip() + store
            noise.append("store_number")
        if len(name) > 22:
            noise.append("truncated")
        name = name[:22].rstrip()
        if rng.random() < 0.5:
            rec["city"] = e.city.upper()[:13]
            rec["state"] = e.state
        else:
            other = rng.choice(pool)
            rec["city"] = other.city.upper()[:13]
            rec["state"] = other.state
            noise.append("charge_city")

    if source == "erp":
        rec.update(street=vary_street(rng, e.street) if rng.random() > 0.05 else "", city=e.city,
                   state=e.state, zip=vary_zip(rng, e.zip))  # fmt: skip
    elif source == "ap":
        if rng.random() < 0.3:
            city, state, z3 = rng.choice(LOCKBOXES)
            rec.update(street=f"PO BOX {rng.randint(1000, 99999)}", city=city, state=state,
                       zip=f"{z3}{rng.randint(0, 99):02d}")  # fmt: skip
            noise.append("remit_address")
        else:
            rec.update(street=vary_street(rng, e.street), city=e.city, state=e.state, zip=vary_zip(rng, e.zip))
    elif source == "crm":
        rec.update(city=e.city if rng.random() < 0.6 else "", state=e.state if rng.random() < 0.7 else "")

    if rec["city"] and source != "card":
        rec["city"] = vary_case(rng, rec["city"])
    rec["name"] = re.sub(r"\s+", " ", name).strip()
    rec["noise"] = ";".join(dict.fromkeys(noise))
    return rec


def generate(entities: list[Entity], rng: random.Random, prefix: str, mean_boost: int = 0) -> list[dict]:
    counts, weights = list(RECORDS_PER_ENTITY), list(RECORDS_PER_ENTITY.values())
    sources, src_w = list(SOURCES), list(SOURCES.values())
    rows = []
    for e in entities:
        n = rng.choices(counts, weights)[0] + mean_boost
        for _ in range(n):
            rec = make_record(rng, e, rng.choices(sources, src_w)[0], entities)
            rec["entity_id"] = f"cik{e.cik}"
            rows.append(rec)
    rng.shuffle(rows)  # records from one company should not sit next to each other
    for i, r in enumerate(rows):
        r["record_id"] = f"{prefix}{i:06d}"
    return rows


def write(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} records, {len({r['entity_id'] for r in rows})} companies -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--scale", action="store_true", help="also write data/scale.csv (~50k records)")
    args = ap.parse_args()

    entities = load_entities(args.raw)
    rng = random.Random(SEED)
    order = list(entities)
    rng.shuffle(order)
    dev, test = order[:N_DEV], order[N_DEV : N_DEV + N_TEST]
    print(f"{len(entities)} usable companies; dev {len(dev)}, test {len(test)}")
    write(generate(dev, random.Random(SEED + 1), "D"), args.out / "dev.csv")
    write(generate(test, random.Random(SEED + 2), "T"), args.out / "test.csv")
    if args.scale:  # every company not in dev, with more records each so that 50k records exist
        pool = order[N_DEV:]
        boost = max(0, round(52_000 / len(pool) - 2.9))
        write(generate(pool, random.Random(SEED + 3), "S", mean_boost=boost), args.out / "scale.csv")


if __name__ == "__main__":
    main()
