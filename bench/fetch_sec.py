"""Download real company records from SEC EDGAR.

Two public endpoints, no key needed (SEC only asks for a User-Agent with contact info):
  https://www.sec.gov/files/company_tickers.json          every company with a ticker
  https://data.sec.gov/submissions/CIK##########.json     name, addresses, former names per company

Writes data/raw/sec_companies.jsonl, one company per line, with only the fields we use.
Resumable: companies already in the file are skipped. Stays under SEC's 10 requests/second limit.

    uv run python -m bench.fetch_sec
"""

from __future__ import annotations

import json
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

UA = "vendormatch research amadeus.colenbrander@gmail.com"
OUT = Path("data/raw/sec_companies.jsonl")
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
MAX_RPS = 8


def get_json(url: str, tries: int = 4) -> dict:
    for attempt in range(tries):
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Encoding": "identity"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.load(resp)
        except (urllib.error.URLError, TimeoutError) as err:
            if isinstance(err, urllib.error.HTTPError) and err.code == 404:
                raise
            time.sleep(2**attempt)
    raise RuntimeError(f"failed after {tries} tries: {url}")


class RateLimiter:
    def __init__(self, rps: float):
        self.interval = 1.0 / rps
        self.lock = threading.Lock()
        self.next_at = time.monotonic()

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            if self.next_at > now:
                time.sleep(self.next_at - now)
            self.next_at = max(self.next_at, now) + self.interval


def address(a: dict | None) -> dict:
    a = a or {}
    return {
        "street": " ".join(x for x in (a.get("street1"), a.get("street2")) if x) or None,
        "city": a.get("city"),
        "state": a.get("stateOrCountry"),
        "zip": a.get("zipCode"),
    }


def extract(sub: dict, tickers: list[str]) -> dict:
    return {
        "cik": int(sub["cik"]),
        "name": sub.get("name"),
        "tickers": tickers,
        "exchanges": sub.get("exchanges") or [],
        "entity_type": sub.get("entityType"),
        "sic": sub.get("sic"),
        "sic_description": sub.get("sicDescription"),
        "state_of_incorporation": sub.get("stateOfIncorporation"),
        "business_address": address(sub.get("addresses", {}).get("business")),
        "mailing_address": address(sub.get("addresses", {}).get("mailing")),
        "phone": sub.get("phone"),
        "website": sub.get("website") or None,
        "former_names": [
            {"name": f["name"], "from": f.get("from", "")[:10], "to": f.get("to", "")[:10]}
            for f in sub.get("formerNames") or []
        ],
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    raw = get_json(TICKERS_URL)
    tickers: dict[int, list[str]] = {}
    for row in raw.values():
        tickers.setdefault(int(row["cik_str"]), []).append(row["ticker"])
    (OUT.parent / "company_tickers.json").write_text(json.dumps(raw))

    done = set()
    if OUT.exists():
        done = {json.loads(line)["cik"] for line in OUT.open()}
    todo = [cik for cik in sorted(tickers) if cik not in done]
    print(f"{len(tickers)} companies, {len(done)} already fetched, {len(todo)} to go", flush=True)

    limiter = RateLimiter(MAX_RPS)
    write_lock = threading.Lock()
    out = OUT.open("a")
    count = 0

    def fetch(cik: int) -> None:
        nonlocal count
        limiter.wait()
        try:
            sub = get_json(SUBMISSIONS_URL.format(cik=cik))
        except Exception as err:  # noqa: BLE001 - log and keep going, the run is resumable
            print(f"skip {cik}: {err}", file=sys.stderr, flush=True)
            return
        line = json.dumps(extract(sub, tickers[cik]))
        with write_lock:
            out.write(line + "\n")
            count += 1
            if count % 250 == 0:
                out.flush()
                print(f"{count}/{len(todo)}", flush=True)

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(fetch, todo))
    out.close()

    # Rewrite sorted by CIK so the file is deterministic.
    rows = sorted((json.loads(line) for line in OUT.open()), key=lambda r: r["cik"])
    OUT.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"wrote {len(rows)} companies to {OUT}")


if __name__ == "__main__":
    main()
