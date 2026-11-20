# `causalflow.split`

Train / validation / test splits, by row or by group. Used by applications before
training (the library's flows still make their own random early-stopping split).

## Public API
- `split_indices(n, fraction, groups=None, seed=None)` → `(keep, holdout)` sorted index
  arrays. `groups=None`: random rows, `ceil(fraction·n)` held out. With a length-n
  `groups` array: whole groups, drawn in random order until the holdout has ≥ fraction·n
  rows (overshoot < one group); no group on both sides.
- `train_val_test(n, val_fraction=0.1, test_fraction=0.1, groups=None, seed=None)` →
  `(train, val, test)`; test first (seed), then validation out of the rest (seed + 1).
  Fractions are of all n rows.

## Design notes
- Row fraction, not group fraction (decided 2026-11-20): the holdout has the intended size
  whatever the group sizes.
- Local `np.random.default_rng(seed)`: the global numpy state is not touched.
- Split before separating treatment arms, so a held-out group is unseen by every flow
  (floody `FEMA.prepare` does this).
- Tests: `tests/test_split.py` (disjoint groups, overshoot bound, reproducible, global
  RNG untouched, exact row count for `groups=None`, bad inputs).

## Cross-references
- [`causalflow.md`](causalflow.md) "Choosing the validation grouping"; workspace `Plan.md` §3b.
