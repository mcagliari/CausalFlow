# `causalflow.support`

Unconditional normalizing flow on p(X) (or p(X, T) in scenario C), used to decide whether
a covariate vector lies inside the training data's support before estimating an effect.
Used by: [`causalflow.md`](causalflow.md). Status: upstream `e73a4f6` + two 3a fixes (below).

## Public API

### `Support(device=None)`
`device` defaults to **CPU** (unlike `BaseCausalFlow`, which defaults to CUDA). The
CausalFlow classes construct `Support()` without passing their device.

| method | does | returns |
|---|---|---|
| `set_architecture(ndim, nhidden=128, nblock=5)` | `nflows` MAF: `nblock` × (MaskedAffineAutoregressive + RandomPermutation), standard-normal base; stored in `self._flow` | `None` |
| `train(X, inverse_cdf=False, batch_size=50, learning_rate=5e-4, num_iter=300, clip_max_norm=5, seed=0, verbose=False)` | wraps `_train`, sets `self.flow_support` | `None` |
| `_train(X, ..., seed=0)` | own loop: random 90/10 split (local `np.random.default_rng(seed)`), Adam, gradient clipping, early stopping with patience 20 on validation NLL; keeps the best epoch | `(best_flow, best_valid_loss)` |
| `validation_loss(flow, X, batch_size=10000)` | mean −log p(X) over held-out X; hyper-parameter objective on an external validation set | float, lower is better |
| `check_support(X, Nsample=10000, threshold=0.95, return_support=False)` | see below | bool array (`True` = in support), or the scores |
| `load(file_flow)` | `torch.load(weights_only=False)` one pickled flow | `None` |
| `load_optuna(study_name, study_dir, verbose=False)` | loads the single best trial (no ensemble); layout in [`study.md`](study.md) | `None` |

## Design notes

- **The support test.** For each X′: s = fraction of Nsample flow samples whose log p is
  higher than log p(X′). X′ is in support if s < threshold, i.e. X′ lies inside the
  `threshold` highest-density region of p(X). Higher threshold → accepts more → *less*
  conservative (the upstream README says the opposite). floody uses 0.99.
  `return_support=True` returns s itself.
- **Separate backend from the outcome flows** (`nflows` directly, not sbi): its own split,
  stopping rule and loss convention. Only outcome flows have a validation routine (TARP);
  support flows are checked visually (corner plots in floody notebooks).
- **No input standardization**: the MAF sees raw X (with the caller's log10 transforms).
  sbi standardizes the outcome flows' inputs; this flow does not.
- **Validation split**: the internal 90/10 split (early stopping) is random by row; it has
  the same leak concern as the outcome flows when rows repeat per zipcode, so
  hyper-parameters are chosen with `validation_loss` on an external, group-disjoint set
  (see `causalflow.md`, "Choosing the validation grouping"). Since 2026-11-20 the split
  uses a local generator (`seed=0` default); before, `np.random.seed(0)` reset numpy's
  global state. Weight initialization (in `set_architecture`) is not seeded here.
- Fixed in 3a (2026-11-18): `warnings` import; `_train`'s default
  `set_architecture(..., nblock=5)` kwarg name.
- `self._flow` is reused: calling `train` twice on the same `Support` continues from the
  previous weights (the CausalFlow classes always build a fresh `Support`).

## Cross-references
- Workspace `Plan.md` §3.
