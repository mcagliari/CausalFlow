'''

train / validation / test splits, by row or by group


'''
import numpy as np


def split_indices(n, fraction, groups=None, seed=None):
    ''' split the indices 0..n-1 into a kept part and a held-out part.

    args:
        n: int, number of rows.

        fraction: float in (0, 1), target fraction of rows held out.

    kwargs:
        groups: None or length-n array of group labels. None: random rows.
            Otherwise whole groups are held out, drawn in random order until
            the held-out part has at least fraction * n rows, so no group
            appears on both sides and the held-out fraction overshoots the
            target by less than one group. Use it when rows repeat per unit
            (e.g. several rows per place or per event), so that the held-out
            part measures generalization to unseen groups.

        seed: int or None, seed of a local random generator (the global numpy
            state is not touched).

    returns:
        keep, holdout: sorted int arrays of indices.
    '''
    if not 0. < fraction < 1.: raise ValueError('fraction must be in (0, 1)')
    rng = np.random.default_rng(seed)

    if groups is None:
        perm = rng.permutation(n)
        n_hold = int(np.ceil(fraction * n))
        return np.sort(perm[n_hold:]), np.sort(perm[:n_hold])

    groups = np.asarray(groups)
    if groups.shape != (n,): raise ValueError('groups must have length n')
    labels, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)

    order = rng.permutation(len(labels))
    # number of groups needed so that the held-out rows reach the target
    n_groups = np.searchsorted(np.cumsum(counts[order]), fraction * n) + 1
    if n_groups >= len(labels):
        raise ValueError('fraction too large: holding out the groups would leave none')
    is_hold = np.isin(inverse, order[:n_groups])
    return np.flatnonzero(~is_hold), np.flatnonzero(is_hold)


def train_val_test(n, val_fraction=0.1, test_fraction=0.1, groups=None, seed=None):
    ''' three-way split: test first, then validation out of the rest. Fractions
    are of all n rows. Same arguments as `split_indices`.

    returns:
        train, val, test: sorted int arrays of indices.
    '''
    rest, test = split_indices(n, test_fraction, groups=groups, seed=seed)
    sub = None if groups is None else np.asarray(groups)[rest]
    rng_seed = None if seed is None else seed + 1
    keep, hold = split_indices(len(rest), val_fraction / (1. - test_fraction),
                               groups=sub, seed=rng_seed)
    return rest[keep], rest[hold], test
