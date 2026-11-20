''' tests for causalflow.study (journal storage, best trials) '''
import optuna
import pytest

from causalflow import study as Study


def test_best_trials_order_and_layout(tmp_path):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    st = optuna.create_study(study_name='s', storage=Study.storage(str(tmp_path), 's'))
    for v in [3., 1., 2., 5.]:
        st.enqueue_trial({'x': v})
    st.optimize(lambda t: t.suggest_float('x', 0, 10), n_trials=4)
    assert (tmp_path / 's' / 's.journal').exists()
    assert Study.best_trials(str(tmp_path), 's', n=2) == [1, 2]
    assert Study.trial_file(str(tmp_path), 's', 3) == str(tmp_path / 's' / 's.3.pt')
    with pytest.raises(ValueError):
        Study.best_trials(str(tmp_path), 's', n=5)
