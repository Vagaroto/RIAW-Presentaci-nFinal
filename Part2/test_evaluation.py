import math

from part2.Evaluation import (precision_at_k, recall_at_k, f1_at_k, average_precision_at_k,
                              reciprocal_rank, ndcg_at_k, evaluate_query)

# Worked example: 6 results, 4 relevant products in total, one of them ("x") never retrieved
RANKED = ["a", "b", "c", "d", "e", "f"]
RELEVANT = {"a", "c", "e", "x"}


def close(value, expected):
    """Compare floats with a tolerance."""
    return math.isclose(value, expected, abs_tol=1e-9)

def test_precision_recall():
    assert close(precision_at_k(RANKED, RELEVANT, 3), 2 / 3)
    assert close(precision_at_k(RANKED, RELEVANT, 5), 3 / 5)
    assert close(recall_at_k(RANKED, RELEVANT, 3), 2 / 4)
    assert close(recall_at_k(RANKED, RELEVANT, 5), 3 / 4)

def test_f1():
    p, r = 2 / 3, 2 / 4
    assert close(f1_at_k(RANKED, RELEVANT, 3), 2 * p * r / (p + r))
    assert f1_at_k(["z"], RELEVANT, 1) == 0

def test_average_precision():
    # hits at ranks 1, 3, 5 -> P = 1, 2/3, 3/5, divided by min(k, 4 relevant)
    assert close(average_precision_at_k(RANKED, RELEVANT, 5), (1 + 2 / 3 + 3 / 5) / 4)
    assert close(average_precision_at_k(RANKED, RELEVANT, 3), (1 + 2 / 3) / 3)

def test_average_precision_whole_ranking():
    # relevant at ranks 1, 3, 5 of 6 results; 4 relevant in total (one never retrieved)
    assert close(evaluate_query(RANKED, RELEVANT)["AP"], (1 + 2 / 3 + 3 / 5) / 4)

def test_reciprocal_rank():
    assert reciprocal_rank(RANKED, RELEVANT) == 1
    assert close(reciprocal_rank(["z", "y", "c"], RELEVANT), 1 / 3)
    assert reciprocal_rank(["z", "y"], RELEVANT) == 0

def test_ndcg():
    dcg = 1 / math.log2(2) + 1 / math.log2(4) + 1 / math.log2(6)         # relevant at ranks 1, 3, 5
    ideal = sum(1 / math.log2(i + 1) for i in range(1, 5))               # 4 relevant at ranks 1 to 4
    assert close(ndcg_at_k(RANKED, RELEVANT, 5), dcg / ideal)
    assert close(ndcg_at_k(["a", "c"], {"a", "c"}, 5), 1)                # perfect ranking
    assert ndcg_at_k(["z", "y"], RELEVANT, 5) == 0

def test_perfect_and_empty_rankings():
    perfect = evaluate_query(["a", "c", "e", "x"], RELEVANT)
    assert close(perfect["P@5"], 4 / 5)                                   # only 4 results, k=5 is still divided by 5
    assert close(perfect["R@5"], 1) and close(perfect["RR"], 1) and close(perfect["NDCG@5"], 1)
    assert all(value == 0 for value in evaluate_query(["z"], RELEVANT).values())

def test_all_values_between_0_and_1():
    for value in evaluate_query(RANKED, RELEVANT).values():
        assert 0 <= value <= 1


if __name__ == "__main__":
    tests = [test for name, test in list(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print("ok", test.__name__)
