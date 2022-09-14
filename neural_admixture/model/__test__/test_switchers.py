import pytest

from neural_admixture.model.switchers import Switchers


def test_initialization_switcher_nargs(mocker):
    mocker.patch("neural_admixture.model.initializations.PCKMeansInitialization.get_decoder_init", return_value=0)
    mocker.patch("neural_admixture.model.initializations.PCArchetypal.get_decoder_init", return_value=0)
    mocker.patch("neural_admixture.model.initializations.SupervisedInitialization.get_decoder_init", return_value=0)
    mocker.patch("neural_admixture.model.initializations.PretrainedInitialization.get_decoder_init", return_value=0)
    n_switcher_args = [None]*8
    assert Switchers().get_switchers()["initializations"]["pckmeans"](*n_switcher_args) == 0
    assert Switchers().get_switchers()["initializations"]["pcarchetypal"](*n_switcher_args) == 0
    assert Switchers().get_switchers()["initializations"]["supervised"](*n_switcher_args) == 0
    assert Switchers().get_switchers()["initializations"]["pretrained"](*n_switcher_args) == 0