# `causalflow.study`

Optuna studies: storage and file layout, shared by training scripts and the loaders
(`BaseCausalFlow._load_flows_optuna`, `Support.load_optuna`).

    <study_dir>/<study_name>/<study_name>.journal       optuna journal storage
    <study_dir>/<study_name>/<study_name>.<trial>.pt    model of each trial

## Public API
- `storage(study_dir, study_name)` → `JournalStorage(JournalFileBackend(..., lock_obj=
  JournalFileOpenLock))`; creates the directory.
- `trial_file(study_dir, study_name, number)` → path of a trial's model.
- `best_trials(study_dir, study_name, n=1, verbose=False)` → numbers of the n completed
  trials with the lowest objective (error if fewer exist).

## Design notes
- Journal file instead of SQLite (2026-11-20): SQLite locking is unreliable on cluster
  file systems; the open-file lock works there (optuna docs). Hahn's SQLite studies are
  not readable any more — they point to old-sbi pickles anyway.
- One study name per setup: floody's `train_causalflow.py` stores the setup as study
  attributes and refuses to resume a study with a different one.

## Cross-references
- [`causalflow.md`](causalflow.md), [`support.md`](support.md); `tests/test_study.py`; workspace
  notebook 06 (V3, end to end with floody's training script).
