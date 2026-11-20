''' validation losses equal the losses the trainers minimize, on the trainers'
own validation rows (synthetic data, small models; ~seconds on CPU) '''
import numpy as np
import torch

from sbi import inference as Inference, utils as Ut

from causalflow import causalflow
from causalflow import support as Support


def _data(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    Y = (X[:, 0] + 0.5 * X[:, 1] ** 2 + 0.3 * rng.normal(size=n))[:, None]
    return Y, X


def test_flow_validation_loss_matches_sbi():
    ''' sbi averages its validation loss over (batches x batch size) and restores
    the best weights only on convergence: use a validation size that is a
    multiple of the batch size and let training converge. '''
    Y, X = _data()
    cf = causalflow.CausalFlowA(device='cpu')
    cf.set_architecture(arch='made', nhidden=16, num_mixture_components=2)
    torch.manual_seed(0)
    npe = Inference.SNPE(prior=Ut.BoxUniform(low=torch.tensor([-10.]), high=torch.tensor([10.])),
                         density_estimator=cf._nde, device='cpu')
    npe.append_simulations(torch.as_tensor(Y, dtype=torch.float32), torch.as_tensor(X, dtype=torch.float32))
    est = npe.train(training_batch_size=50, stop_after_epochs=3, max_num_epochs=500)
    assert npe._summary['epochs_trained'][-1] < 500, 'did not converge'
    post = npe.build_posterior(est)
    vi = np.asarray(npe.val_indices)
    assert len(vi) % 50 == 0
    ours = cf.validation_loss(post, Y[vi], X[vi])
    assert abs(ours - npe._summary['best_validation_loss'][-1]) < 1e-4 * abs(ours)


def test_support_validation_loss_matches_train():
    ''' Support._train: 90/10 split from default_rng(seed).permutation, mean of
    batch means; 100 validation rows = 2 full batches of 50 '''
    _, X = _data()
    sup = Support.Support()
    torch.manual_seed(0)
    sup.set_architecture(2, nhidden=16, nblock=2)
    flow, best = sup._train(X, batch_size=50, num_iter=30, seed=0)
    valid = np.random.default_rng(0).permutation(len(X))[int(0.9 * len(X)):]
    assert abs(sup.validation_loss(flow, X[valid]) - best) < 1e-4 * abs(best)


def test_train_flow_seed_reproducible():
    Y, X = _data(n=500)
    losses = []
    for _ in range(2):
        cf = causalflow.CausalFlowA(device='cpu')
        cf.set_architecture(arch='made', nhidden=16, num_mixture_components=2)
        flow, _ = cf._train_flow(Y, X, outcome_range=[[-10.], [10.]], seed=1,
                                 stop_after_epochs=2, max_num_epochs=50)
        losses.append(cf.validation_loss(flow, Y, X))
    assert losses[0] == losses[1]
