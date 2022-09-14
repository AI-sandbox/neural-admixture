import dask.array as da
import logging
import pytest
import random
import string

from neural_admixture.model.initializations import PCArchetypal, PCKMeansInitialization, SupervisedInitialization


@pytest.mark.parametrize(["n", "d", "K"], [[42, 1000, [2,3,4]], [500, 100, [99,100,101]], [30,100,[2,3,4]]])
def test_pckmeans(n, d, K, caplog):
    caplog.set_level(logging.CRITICAL)
    X = da.random.binomial(n=2, p=.1, size=(n,d))/2
    path = None
    run_name = "pckmeans_test"
    n_components = 4
    seed = 42
    batch_size = 200
    P_init = PCKMeansInitialization().get_decoder_init(X, K, path, run_name, n_components, seed, batch_size)
    assert P_init.shape[0] == sum(K) and P_init.shape[1] == d


@pytest.mark.parametrize(["n", "d", "K"], [[42, 1000, [2,3,4]], [500, 100, [3,4,5]], [30,100,[6,7,8]]])
def test_pcarchetypal(n, d, K, caplog):
    caplog.set_level(logging.CRITICAL)
    X = da.random.binomial(n=2, p=.1, size=(n,d))/2
    path = None
    run_name = "test_pcarchetypal"
    n_components = 4
    seed = 42
    batch_size = 200
    P_init = PCArchetypal().get_decoder_init(X, K, path, run_name, n_components, seed, batch_size)
    assert P_init.shape[0] == sum(K) and P_init.shape[1] == d

@pytest.mark.parametrize(["n", "d", "K"], [[42, 1000, [2]], [500, 100, [5]], [30,100,[10]]])
def test_supervised(n, d, K, caplog):
    caplog.set_level(logging.CRITICAL)
    X = da.random.binomial(n=2, p=.1, size=(n,d))/2
    ancs = random.sample(string.ascii_lowercase, K[0]) + ["-"]
    y = ancs+random.choices(ancs, k=n-len(ancs))
    P_init = SupervisedInitialization().get_decoder_init(X, y, K)
    assert P_init.shape[0] == K[0] and P_init.shape[1] == d

def test_supervised_fail(caplog):
    caplog.set_level(logging.CRITICAL)
    n, d = 100, 10
    K = [5,6,7]
    X = da.random.binomial(n=2, p=.1, size=(n,d))/2
    ancs = random.sample(string.ascii_lowercase, 5) + ["-"]
    y = random.choices(ancs, k=n)
    with pytest.raises(NotImplementedError):
        _ = SupervisedInitialization().get_decoder_init(X, y, K)
