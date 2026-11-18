# `causalflow.causalflow`

Treatment-effect estimators built on conditional normalizing flows q(Y | X).
Reference: Cano & Hahn (2024), arXiv:2312.03843. Support checks: [`support.md`](support.md).
Status: upstream `e73a4f6` + step-3a port to sbi 0.25 (branch `work`, 2026-11-18; verified by
workspace `notebooks/01_causalflow_smoke.ipynb`). Open issues → workspace `Plan.md` §3.

## Public API

### `BaseCausalFlow(device=None)`
Shared machinery; `device` defaults to `'cuda'` if available, else `'cpu'`.

| method | does | returns |
|---|---|---|
| `set_architecture(arch='made', nhidden=64, ntransform=2, nblocks=2, num_mixture_components=2, batch_norm=True)` | stores an `sbi.neural_nets.posterior_nn` builder in `self._nde` (for `made`, only `nhidden` and `num_mixture_components` take effect, see notes) | `None` |
| `_train_flow(Y, X, outcome_range, inverse_cdf=False, training_batch_size=50, learning_rate=5e-4, verbose=False)` | trains q(Y\|X) with sbi SNPE; Y, X as N×D arrays (1-D promoted) | `(flow, best_valid_loss)`: sbi's `best_validation_loss` on its internal split, **lower is better** (was a log-prob before the port) |
| `_sample(flows, X, Nsample=10000, progress_bar=False)` | draws Y ~ q(Y\|X) at one X; `flows` may be a list (ensemble) | `(Nsample, D_y)` array |
| `_tarp_coverage(samples, Y, outcome_range)` (static) | rescales samples (Nsample×N×D_y) and Y (N×D_y) to [0,1] by `outcome_range`, then `tarp.get_tarp_coverage(norm=False)` | `(alpha, ecp)` |
| `_validate(flows, X_test, Y_test, outcome_range, Nsample=10000, verbose=False)` | samples q(Y\|X_test) and calls `_tarp_coverage`; `outcome_range` **required** (as in training) | `(alpha, ecp)` |
| `_torch_load_flow(path)` | `torch.load(map_location=device, weights_only=False)`, then `flow._device = device` | flow |
| `_load_flow(flow_name)` | one pickled flow via `_torch_load_flow` | flow |
| `_load_flows_optuna(study_name, study_dir, n_ensemble=5, verbose=False)` | loads the `n_ensemble` lowest-objective trials from `<dir>/<study>/<study>.db` (SQLite) + `<study>.<trial>.pt` | list of flows |

### `CausalFlowA(device=None)` — binary treatment, both arms well sampled
Holds `flow_treated`, `flow_control`, `support_treated`, `support_control`.
- `CATE(X, Nsample, Nsupport, support_threshold=0.95, progress_bar, transf=None)` →
  `mean(transf(Y_T)) − mean(transf(Y_C))` at one covariate vector X; warns if X is out of
  either support.
- Train/load: `train_flow_treated/control(Y, X, **_train_flow kwargs)`,
  `load_flow_treated/control(path)`, `load_flows_optuna_treated/control(study, dir, n_ensemble)`,
  `train_support_treated/control(X, nhidden, nblock, batch_size, learning_rate, num_iter, clip_max_norm, verbose)`,
  `load_support_treated/control(Support)`.

### `CausalFlowB(base, device=None)` — one well-sampled "base" arm (`'control'` or `'treated'`)
Holds `flow_base`, `support_base`.
- `CTE(X, Y, ...)` → per-unit effect: observed Y of the other arm minus the predicted base mean
  (sign flips when `base='treated'`).
- `CATE(Xs, Ys, ...)` → mean of CTEs over the units **inside** the base support; the others are
  dropped with a warning. Only scenario B drops; A and C just warn.
- `train_flow`, `load_flow`, `load_flows_optuna`, `train_support`, `load_support`.

### `CausalFlowC(device=None)` — continuous treatment
One flow q(Y | X, T) on the concatenation `[X, T]`, support on p(X, T).
- `CATE(X_fid, T_fid, dT, ...)` → `mean(Y | X_fid, T_fid+dT) − mean(Y | X_fid, T_fid)`.
- `train_flow(Y, X, T)`, `train_support(X, T, ...)`, plus the usual load methods.

Without a support flow, every estimator warns and skips the support check (B's `CATE` then
averages over all given units). All estimators return a **point estimate** (no uncertainty). `transf` is applied to samples
before averaging (e.g. `10**x` for flows trained on log10 Y → mean of 10**Y).

## Design notes

- **sbi as a conditional density estimator.** In sbi's terms Y is "θ" and X is "x", so the
  "posterior" is q(Y|X). The `BoxUniform(outcome_range)` prior is not a Bayesian prior: at
  sampling, sbi rejects draws outside its range, so `outcome_range` acts as a hard
  truncation of Y *(inference from sbi's behaviour — verify on the pinned version)*.
- **Decision (2026-11-17): keep sbi, port to sbi 0.25** (option (a)), instead of replacing
  it with a hand-written loop (option (b), reconsidered in restructuring if needed).
  Consequences:
  - summary key: only `best_validation_loss` (minimize) exists from sbi 0.23 on; the
    current `best_validation_log_prob(s)` lookup crashes.
  - sbi's internal `validation_fraction` split is a random row split that cannot be replaced.
    Hyper-parameter selection must use an **external validation set**, held out before
    `append_simulations`, scored with `density_estimator.loss(y_val, x_val).mean()`.
    The internal split then only governs early stopping.
- **Choosing the validation grouping.** The split should copy the generalization the result
  claims: by zipcode (new places), by community (CRS is assigned per community), by
  year/event (new storms), or by row (known places, new covariates). Rows repeated per
  zipcode (zipcode × month in FEMA) leak through a random split. Selection over many Optuna
  trials also overfits the validation set, so final checks need a separate, untouched test
  set. Data structure still to be measured in a notebook.
- **1-D outcome.** For D_y = 1, sbi's `maf` reduces to a Gaussian; `made` with
  `num_mixture_components > 1` (mixture of Gaussians) is the flexible option.
- **`made` in sbi 0.25 ignores most architecture kwargs.** `build_made`
  (`sbi/neural_nets/net_builders/flow.py`) hard-codes `num_blocks=5`, `use_batch_norm=False`
  and has no transforms: `ntransform`, `nblocks`, `batch_norm` are silent no-ops; only
  `nhidden` and `num_mixture_components` matter. `maf`/`nsf` do honour them. Whether old
  sbi did for `made` is unknown. Optuna search spaces must match the arch (Plan §3b).
- **Ensembles** are plain lists, equal-weight mixtures: each member contributes
  `int(Nsample / Nflow)` draws.
- **Saved models** are whole pickled sbi posteriors: they load only under matching
  library versions. `_torch_load_flow` handles GPU-trained → CPU (`map_location` moves the
  tensors but not the pickled `_device` string, which it resets) and passes
  `weights_only=False` (torch ≥ 2.6 defaults to True, which refuses whole objects).
- **TARP (`_validate`)** — checked against installed tarp 0.1.1 (`tarp/drp.py`):
  - `get_drp_coverage` is a deprecated wrapper (removal planned in 0.2.0) that returns
    `get_tarp_coverage(..., norm=True)`, i.e. **`(ecp, alpha)`**. The pre-port `_validate`
    unpacked it as `alpha, ecp` → outputs **swapped** under tarp 0.1.1. Fixed in 3a:
    `_tarp_coverage` calls `get_tarp_coverage` directly.
  - `references="random"` are **always** drawn from U[0,1]^D (hard-coded), whatever `norm`.
    `norm=True` maps samples and Y into that cube by the empirical min/max of the *test-set*
    Y; flow samples outside that range land outside [0,1] (tarp issue #14, an asymmetry that
    grows with dimension). `norm=False` on raw units leaves the references off the data.
  - **Implemented (choice from simbig2, measured there):** rescale samples and Y by the
    known bounds — here `outcome_range`, which sbi enforces on samples — then call with
    `norm=False`. simbig2 found this offset-invariant, while raw `norm=False` inflated a
    known bias ~3.3× and distorted its shape; it is not a no-op in 1-D (curves moved by up
    to 0.30). Not yet re-measured on CausalFlow outcomes (inference that it transfers).
  - Do not use `bootstrap=True`: each iteration overwrites one simulation with another,
    cumulatively and **in place** (mutates the caller's arrays). simbig2 measured its σ
    ~15% too small and replaced it with a proper resampling bootstrap.
  - tarp reseeds numpy's global RNG inside every call (`np.random.seed(seed)`, default
    `None` = OS entropy): results are not reproducible unless `seed=` is passed, and the
    caller's numpy state is reset.
  - Measured (workspace notebook 01, 1-D, 1000 units): a constant 0.5σ bias is at the
    noise level (max|ecp−α| ≈ 0.03–0.05); width errors give ≈ 0.25–0.30 (overconfidence
    crosses the diagonal, underconfidence lies above it). TARP cannot separate bias from
    width and can miss bias entirely; pair it with point-estimate checks.
- **Duplication.** Subclasses differ only in which models they hold and the estimator
  formula; `train_support_*` (×4) and `load_*` (×8) are copy-pasted, which is where the
  (fixed) `load_flow_control` bug came from.
- **Multi-dimensional Y.** `np.mean` in the estimators averages over all axes: correct
  only for D_y = 1.
- **Reproducibility.** No seeding of flows or of sbi's split (`torch.randperm`).

## Cross-references
- Workspace `Plan.md` §2 (versions), §3 (bugs, validation split, TARP).
- `support.md` — support flows used by every estimator.
