import numpy as np

from TransLTEE.data import load_dataset, split_indices


def test_load_ihdp_and_split(project_root):
    bundle = load_dataset(
        "ihdp",
        project_root / "data",
        repetition=1,
        sequence_length=8,
    )
    assert bundle.features.shape == (747, 25)
    assert bundle.outcomes.shape == (747, 8)
    assert bundle.ground_truth_ate.shape == (8,)

    split = split_indices(bundle.treatment, seed=42)
    assert (split.train.size, split.validation.size, split.test.size) == (522, 150, 75)
    overall_rate = bundle.treatment.mean()
    for indices in (split.train, split.validation, split.test):
        assert abs(bundle.treatment[indices].mean() - overall_rate) < 0.02
    assert np.intersect1d(split.train, split.validation).size == 0


def test_load_news_contract(project_root):
    bundle = load_dataset(
        "news",
        project_root / "data",
        repetition=1,
        sequence_length=4,
    )
    assert bundle.features.shape == (5000, 3477)
    assert bundle.outcomes.shape == (5000, 4)
    assert bundle.ground_truth_ate.shape == (4,)
