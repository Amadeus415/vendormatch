import csv

from vendormatch import dedupe
from vendormatch.cli import main
from vendormatch.features import FEATURES, pair_features, prepare

RECORDS = [
    {"name": "Blue Bottle Coffee, Inc.", "city": "Oakland", "state": "CA", "source": "erp"},
    {"name": "SQ *BLUE BOTTLE COF", "city": "OAKLAND", "state": "CA", "source": "card"},
    {"name": "BLUE BOTTLE COFFEE INC", "city": "Oakland", "state": "CA", "source": "ap"},
    {"name": "Zebra Technologies Corp", "city": "Lincolnshire", "state": "IL", "source": "erp"},
    {"name": "ZEBRA TECHNOLOGIES", "city": "LINCOLNSHIRE", "state": "IL", "source": "card"},
    {"name": "Delta Air Lines, Inc.", "city": "Atlanta", "state": "GA", "source": "erp"},
    {"name": "Delta Apparel, Inc.", "city": "Duluth", "state": "GA", "source": "erp"},
]


def test_features_shape_and_range():
    prep = prepare(RECORDS)
    X = pair_features(prep, [(0, 1), (5, 6)])
    assert X.shape == (2, len(FEATURES))
    finite = X[~(X != X)]
    assert ((finite >= 0) & (finite <= 1.0001)).all()


def test_dedupe_groups_obvious_duplicates_and_keeps_hard_negative_apart():
    r = dedupe(RECORDS)
    c = r.clusters
    assert c[0] == c[1] == c[2]
    assert c[3] == c[4]
    assert c[5] != c[6]
    assert r.golden[int(c[0])] == "Blue Bottle Coffee, Inc."
    assert set(r.seconds) >= {"normalize", "block", "features", "score", "cluster"}


def test_cli_writes_clusters(tmp_path):
    src, out = tmp_path / "in.csv", tmp_path / "out.csv"
    with src.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["vendor", "city", "state"])
        w.writeheader()
        for r in RECORDS:
            w.writerow({"vendor": r["name"], "city": r["city"], "state": r["state"]})
    main(["dedupe", str(src), "--out", str(out), "--name-col", "vendor"])
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == len(RECORDS)
    assert {"cluster_id", "cluster_size", "golden_name"} <= set(rows[0])
    assert rows[0]["cluster_id"] == rows[1]["cluster_id"]
