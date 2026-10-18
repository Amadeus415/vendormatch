from vendormatch.cluster import average_link, golden_names, union_find


def test_union_find_is_transitive():
    labels = union_find(5, [(0, 1), (1, 2)])
    assert labels[0] == labels[1] == labels[2]
    assert len({labels[0], labels[3], labels[4]}) == 3


def test_average_link_refuses_a_single_bridge():
    # Two tight triangles joined by one strong edge: union-find glues them, average link does not.
    scores = {(0, 1): 0.95, (0, 2): 0.9, (1, 2): 0.9, (3, 4): 0.95, (3, 5): 0.9, (4, 5): 0.9, (2, 3): 0.8}
    uf = union_find(6, [p for p, s in scores.items() if s >= 0.5])
    assert len(set(uf)) == 1
    al = average_link(6, scores, threshold=0.5)
    assert al[0] == al[1] == al[2]
    assert al[3] == al[4] == al[5]
    assert al[0] != al[3]


def test_average_link_merges_when_evidence_is_dense():
    scores = {(0, 1): 0.9, (1, 2): 0.9, (0, 2): 0.6}
    labels = average_link(3, scores, threshold=0.5)
    assert len(set(labels)) == 1


def test_average_link_ignores_low_scores():
    labels = average_link(3, {(0, 1): 0.4, (1, 2): 0.1}, threshold=0.5)
    assert len(set(labels)) == 3


def test_golden_name_prefers_vendor_master():
    labels = [0, 0, 0]
    names = ["SQ *BLUE BOTTLE COF", "Blue Bottle Coffee Inc", "BLUE BOTTLE"]
    sources = ["card", "erp", "ap"]
    assert golden_names(labels, names, sources) == {0: "Blue Bottle Coffee Inc"}
