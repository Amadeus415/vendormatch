"""Download real company aliases from Wikidata, keyed by SEC CIK (property P5531).

Wikidata labels are the names people actually use ("IBM", "Bristol-Myers Squibb", "Dolby"), and its
aliases include acronyms and short forms ("BMS", "Dolby Labs"). Wikidata content is CC0.

    uv run python -m bench.fetch_wikidata
"""

from __future__ import annotations

import csv
import io
import urllib.parse
import urllib.request
from pathlib import Path

UA = "vendormatch research amadeus.colenbrander@gmail.com"
OUT = Path("data/raw/wikidata_aliases.csv")
QUERY = """
SELECT ?cik ?label ?alias WHERE {
  ?item wdt:P5531 ?cik .
  ?item rdfs:label ?label FILTER(lang(?label) = "en")
  OPTIONAL { ?item skos:altLabel ?alias FILTER(lang(?alias) = "en") }
}
"""


def main() -> None:
    url = "https://query.wikidata.org/sparql?" + urllib.parse.urlencode({"query": QUERY})
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "text/csv"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        text = resp.read().decode()
    rows = [r for r in csv.DictReader(io.StringIO(text)) if r["cik"].strip().isdigit()]
    rows.sort(key=lambda r: (int(r["cik"]), r["label"], r["alias"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cik", "label", "alias"])
        w.writeheader()
        for r in rows:
            w.writerow({"cik": int(r["cik"]), "label": r["label"], "alias": r["alias"]})
    print(f"wrote {len(rows)} rows for {len({r['cik'] for r in rows})} companies to {OUT}")


if __name__ == "__main__":
    main()
