# `causalflow.support`

Unconditional normalizing flow on p(X) (or p(X, T) in scenario C), used to decide whether
a covariate vector lies inside the training data's support before estimating an effect.
Used by: [`causalflow.md`](causalflow.md). Status: upstream code at `e73a4f6` (unmodified).

## Public API

### `Support(device=None)`
`device` defaults to **CPU** (unlike `BaseCausalFlow`, which defaults to CUDA). The
CausalFlow classes construct `Support()` without passing their device.

| method | does | returns |
|---|---|---|
| `set_architecture(ndim, nhidden=128, nblock=5)` | `nflows` MAF: `nblock` × (MaskedAffineAutoregressive + RandomPermutation), standard-normal base; stored in `self._flow` | `None` |
| `train(X, inverse_cdf=False, batch_size=50, learning_rate=5e-4, num_iter=300, clip_max_norm=5, verbose=False)` | wraps `_train`, sets `self.flow_support` | `None` |
| `_train(X, ...)` | own loop: random 90/10 split (`np.random.seed(0)`), Adam, gradient clipping, early stopping with patience 20 on validation NLL; keeps the best epoch | `(best_flow, best_valid_loss)` |
| `check_support(X, Nsample=10000, threshold=0.95, return_support=False)` | see below | bool array (`True` = in support), or the scores |
| `load(file_flow)` | `torch.load` one pickled flow | `None` |
| `load_optuna(study_name, study_dir, verbose=False)` | loads the single best trial (no ensemble) | `None` |

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
- **Validation split**: random 90/10 by row, same leak concern as the outcome flows when rows
  repeat per zipcode (see `causalflow.md`, "Choosing the validation grouping").
- `np.random.seed(0)` inside `_train` resets numpy's global random state as a side effect.
- Known bugs (workspace `Plan.md` §3): `warnings` not imported; `_train` calls
  `set_architecture(..., nblocks=5)` but the kwarg is `nblock`, so training without first
  calling `set_architecture` fails.

## Cross-references
- Workspace `Plan.md` §3.
