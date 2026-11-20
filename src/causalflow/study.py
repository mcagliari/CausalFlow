'''

optuna studies: where they are stored and which trials are best. One place for
the file layout shared by training scripts and the loaders:

    <study_dir>/<study_name>/<study_name>.journal       optuna journal storage
    <study_dir>/<study_name>/<study_name>.<trial>.pt    model of each trial


'''
import os
import numpy as np


def storage(study_dir, study_name):
    ''' optuna JournalStorage of a study (a plain file with an open-file lock,
    which works on cluster file systems where SQLite locking does not).
    Creates the study directory if needed.
    '''
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock

    os.makedirs(os.path.join(study_dir, study_name), exist_ok=True)
    path = os.path.join(study_dir, study_name, '%s.journal' % study_name)
    return JournalStorage(JournalFileBackend(path, lock_obj=JournalFileOpenLock(path)))


def trial_file(study_dir, study_name, number):
    ''' path of the model saved by trial `number` '''
    return os.path.join(study_dir, study_name, '%s.%i.pt' % (study_name, number))


def best_trials(study_dir, study_name, n=1, verbose=False):
    ''' numbers of the n completed trials with the lowest objective '''
    import optuna
    study = optuna.load_study(study_name=study_name, storage=storage(study_dir, study_name))
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(done) < n:
        raise ValueError('study %s has %i completed trials, %i requested' % (study_name, len(done), n))
    if verbose: print('select %i out of %i trials' % (n, len(done)))
    done.sort(key=lambda t: t.value)
    return [t.number for t in done[:n]]
