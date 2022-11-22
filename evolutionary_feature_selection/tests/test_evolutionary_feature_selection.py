import pytest
import numpy as np

from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from numpy.testing import assert_array_equal
from numpy.testing import assert_allclose

from evolutionary_feature_selection import EvolutionaryFeatureSelection


@pytest.fixture
def data():
    return load_iris(return_X_y=True)


@pytest.fixture()
def regressor():
    return LinearRegression


def test_evolutionary_feature_selection(data, regressor):
    est = EvolutionaryFeatureSelection(regressor, random_state=31337)
    est.fit(*data)
    assert_allclose(est.transform(data[0][0:5, :]), np.array([[3.5], [3.], [3.2], [3.1], [3.6]]))
    assert_allclose(est.fitness_values_, est.fitness_history_[-1])
    assert_allclose(est.fitness_values_, np.array([0.18203667, 0.18203667, 0.18203667, 0.18203667, 0.18203667,
                                                   0.18203667, 0.18203667, 0.18203667, 0.18203667, 0.18203667]))
    assert_allclose(est.fitness_history_[0], np.array([0.18203667, 0.18203667, 0.18203667, 0.61240208, 0.90066686,
                                                       0.9149828,  0.9149828,  0.9149828,  0.9149828,  0.9149828]))



