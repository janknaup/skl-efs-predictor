import pytest
import numpy as np
from numpy.testing import assert_array_equal

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression

from evolutionary_feature_selection import EvolutionaryFeatureSelection


@pytest.fixture()
def data():
    rand_stat = np.random.RandomState(31337)
    X = rand_stat.random((20, 20))
    y_true = 0.5 + (8.0 * X[:, 0]) + (6.0 * X[:, 1]) + (4.0 * X[:, 2]) + (2.0 * X[:, 3]) + X[:, 4]
    return X, y_true


@pytest.fixture()
def efs():
    return EvolutionaryFeatureSelection(generations=50, n_features=5, population_size=20, n_breeders=8,
                                        mutation_rate=0.1, random_state=1337, n_jobs=1)


def test_efs_pipeline(data, efs):
    pipe = Pipeline([
        ('efs', efs),
    ])
    pipe.fit(*data)
    assert_array_equal(pipe.get_feature_names_out(), np.array(['x0', 'x1', 'x2', 'x3', 'x4']),
                       err_msg="Wrong features selected in pipeline")


def test_efs_gridsearch(data, efs):
    pipe = Pipeline([
        ('efs', efs),
        ('predictor', LinearRegression())
    ])
    param_grid = [{'efs__n_features': np.arange(4, 8)}]
    gs = GridSearchCV(pipe, param_grid, cv=5, return_train_score=True)
    gs.fit(*data)
    assert gs.best_params_ == {'efs__n_features': 7}, "wrong optimal number of features"
    all_param_grid = [{
        'efs__n_features': [4],
        'efs__n_breeders': [4],
        'efs__population_size': [5],
        'efs__n_mutation_features': [1],
        'efs__generations': [2],
        'efs__mutation_rate': [0.1],
    }]
    gs2 = GridSearchCV(pipe, all_param_grid, cv=2, return_train_score=True)
