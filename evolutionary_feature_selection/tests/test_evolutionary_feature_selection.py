import pytest
import numpy as np

from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from numpy.testing import assert_array_equal
from numpy.testing import assert_allclose

from evolutionary_feature_selection import EvolutionaryFeatureSelection


@pytest.fixture
def data():
    return load_iris(return_X_y=True)


@pytest.fixture()
def regressor():
    return LinearRegression

@pytest.fixture()
def scoring():
    return r2_score()

def nd_scoring(y1, y2):
    return -mean_squared_error(y1, y2)

def test_evolutionary_feature_selection(data, regressor):
    # fit on bullshit data
    est = EvolutionaryFeatureSelection(random_state=31337, n_features=2)
    est.fit(*data)
    assert_allclose(est.transform(data[0][0:5, :]), np.array([[3.5, 0.2],
                                                              [3.,  0.2],
                                                              [3.2, 0.2],
                                                              [3.1, 0.2],
                                                              [3.6, 0.2]]),
                    err_msg="Transformed features differ from reference")
    assert_allclose(est.fitness_values_, est.fitness_history_[-1],
                    err_msg="Fitness and fitness_history inconsistent")
    assert_allclose(est.fitness_values_, np.array([0.92173051, 0.92173051, 0.91498288, 0.91498288, 0.91498288,
                                                   0.91498288, 0.91498288, 0.91498288, 0.91498288, 0.91498288]),
                    err_msg="Fitted population fitness values differ from reference")
    assert_allclose(est.fitness_history_[0], np.array([0.92173051, 0.92173051, 0.91498288, 0.91498288, 0.91498288,
                                                       0.91498288, 0.9149828,  0.90115939, 0.90115939, 0.61240208]),
                    err_msg="Initial population fitness values differ from reference")
    # non-default regressor and scoring

    est2 = EvolutionaryFeatureSelection(predictor=regressor, scoring=nd_scoring, random_state=1337, n_features=2)
    est2.fit(*data)
    assert type(est2.predictor_) is regressor, "predictor not passed correctly from constructor"
    assert (est2.scoring_) is nd_scoring, "scoring not passed correctly from constructor"
    # initial population
    # same size initial population
    # we can only indirectly detect sameness of initial populations by comparing initial fitness values
    est3 = EvolutionaryFeatureSelection(initial_population=np.array(est.population_), random_state=1337, n_features=2)
    est3.fit(*data)
    assert_allclose(est3.fitness_history_[0], est.fitness_values_,
                    err_msg="Fitness mismatch on passed initial population")
