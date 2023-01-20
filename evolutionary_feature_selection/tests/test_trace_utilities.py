import pytest
import numpy as np
from numpy.testing import assert_array_equal

from evolutionary_feature_selection import EvolutionaryFeatureSelection, feature_trace, fitness_trace, \
    fitness_diversity_trace

@pytest.fixture()
def data():
    rand_stat = np.random.RandomState(31337)
    X = rand_stat.random((20, 20))
    y_true = 0.5 + (8.0 * X[:, 0]) + (6.0 * X[:, 1]) + (4.0 * X[:, 2]) + (2.0 * X[:,  3]) + X[:, 4]
    return X, y_true

@pytest.fixture()
def efs():
    return EvolutionaryFeatureSelection(generations=50, n_features=5, population_size=20, n_breeders=8,
                                        mutation_rate=0.1, random_state=1337)

@pytest.mark.mpl_image_compare(remove_text=True, tolerance=8)
def test_fitness_trace(data, efs):
    efs.fitness_trace = True
    efs.fit(*data)
    trace_fig = fitness_trace(efs.fitness_history_, top_specimen=True, all_specimens=True, mean=True, range_area=True)
    return trace_fig

@pytest.mark.mpl_image_compare(remove_text=True, tolerance=8)
def test_feature_trace(data, efs):
    efs.population_trace = True
    efs.fit(*data)
    top_specimen_trace = feature_trace(efs.population_history_, specimen='top')
    all_specimen_trace = feature_trace(efs.population_history_, specimen='all',
                                       feature_names=['X{0:02d}'.format(i) for i in range(len(data[1]))])
    return all_specimen_trace

@pytest.mark.mpl_image_compare(remove_text=True, tolerance=8)
def test_fitness_diversity_trace(data, efs):
    efs.fitness_trace = True
    efs.fit(*data)
    trace_fig = fitness_diversity_trace(efs.fitness_history_, std=True, range=True)
    return trace_fig
