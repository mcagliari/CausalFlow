''' tests for causalflow.split '''
import numpy as np
import pytest

from causalflow import split


def _groups(n_groups=1000, max_size=20, seed=0):
    sizes = np.random.default_rng(seed).integers(1, max_size + 1, n_groups)
    return np.repeat(np.arange(n_groups), sizes), sizes


def test_random_rows_exact_count():
    keep, hold = split.split_indices(1000, 0.1, seed=3)
    assert len(hold) == 100 and len(keep) == 900
    assert len(np.intersect1d(keep, hold)) == 0


@pytest.mark.parametrize('seed', range(10))
def test_groups_disjoint_and_overshoot_below_one_group(seed):
    groups, sizes = _groups()
    n = len(groups)
    keep, hold = split.split_indices(n, 0.1, groups=groups, seed=seed)
    assert len(keep) + len(hold) == n
    assert not set(groups[keep]) & set(groups[hold])
    assert 0.1 * n <= len(hold) < 0.1 * n + sizes.max()


def test_reproducible_and_global_rng_untouched():
    groups, _ = _groups()
    np.random.seed(123)
    before = np.random.get_state()[1].copy()
    a = split.split_indices(len(groups), 0.1, groups=groups, seed=7)
    b = split.split_indices(len(groups), 0.1, groups=groups, seed=7)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert np.array_equal(np.random.get_state()[1], before)


def test_train_val_test_partition():
    groups, _ = _groups()
    n = len(groups)
    tr, va, te = split.train_val_test(n, groups=groups, seed=1)
    assert len(tr) + len(va) + len(te) == n
    assert len(np.unique(np.concatenate([tr, va, te]))) == n
    for a, b in ((tr, va), (tr, te), (va, te)):
        assert not set(groups[a]) & set(groups[b])


def test_bad_inputs():
    with pytest.raises(ValueError): split.split_indices(10, 1.5)
    with pytest.raises(ValueError): split.split_indices(10, 0.1, groups=np.arange(9))
    with pytest.raises(ValueError):                 # holding out every group
        split.split_indices(4, 0.99, groups=np.array([0, 0, 1, 1]))
