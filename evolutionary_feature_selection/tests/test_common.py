import pytest

from sklearn.utils.estimator_checks import check_estimator
from evolutionary_feature_selection import EvolutionaryFeatureSelection
from sklearn.linear_model import LinearRegression

@pytest.mark.parametrize(
    "estimator",
    [EvolutionaryFeatureSelection(LinearRegression)]
)
def test_all_estimators(estimator):
    return check_estimator(estimator)
