from paper.interpret.run import Prediction


def test_top_n() -> None:
    p = Prediction("1.0", {1, 2, 3}, [1, 2, 3, 4, 5])
    assert p.top_n(3) == {1, 2, 3}


def test_jaccard_1() -> None:
    p = Prediction("1.0", {1, 2, 3}, [1, 2, 3, 4, 5])
    assert p.jaccard_similarity(3) == 1.0
    assert p.target == p.top_n()


def test_jaccard_2() -> None:
    p = Prediction("1.0", {1, 2}, [1, 2, 3, 4, 5])
    assert p.jaccard_similarity(2) == 1.0
    assert p.target == p.top_n()


def test_jaccard_3() -> None:
    p = Prediction("1.0", {1, 2}, [1, 2, 3, 4, 5])
    assert p.jaccard_similarity() == 1.0
    assert p.target == p.top_n()


def test_jaccard_4() -> None:
    p = Prediction("1.0", {1, 2, 3, 4}, [1, 2, 3, 4, 5])
    assert p.jaccard_similarity() == 1.0
    assert p.target == p.top_n()


def test_jaccard_5() -> None:
    p = Prediction("1.0", {1}, [1, 2, 3, 4, 5])
    assert p.jaccard_similarity() == 1.0
    assert p.target == p.top_n()


def test_jaccard_6() -> None:
    p = Prediction("1.0", {1, 3}, [1, 2, 3, 4, 5])
    assert p.jaccard_similarity() == 1 / 3


def test_jaccard_7() -> None:
    p = Prediction("1.0", {1, 3, 5}, [1, 2, 3, 4, 5])
    assert p.jaccard_similarity() == 0.5
