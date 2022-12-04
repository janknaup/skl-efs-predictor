import pytest
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from numpy.testing import assert_array_equal
from numpy.testing import assert_allclose

from evolutionary_feature_selection import EvolutionaryFeatureSelection


@pytest.fixture
def data():
    return load_iris(return_X_y=True)


@pytest.fixture
def regressor():
    return LinearRegression


@pytest.fixture
def scoring():
    return r2_score()


def nd_scoring(y1, y2):
    return -mean_squared_error(y1, y2)


@pytest.fixture
def synthetic_dataset():
    """
    Returns
    -------
    X : ndarray, float shape(20,20)
        random numbers
    Y_true : 0.5 + X[:,0] + (2.0 * X[:,1]) + (4.0 * X[:,2]) + (6.0 * X[:,3]) + (8.0 * X[:,4])
        a linear function of the 1st 5 elements of X
    Y_fit : Y_true + np.random.normal(scale=np.var(Y_true)*0.1, size=Y_true.shape)
        Y_true with some normal distributed noise
    """
    X = np.array(
        [[0.80137, 0.72391, 0.60732, 0.73443, 0.09881, 0.54520, 0.69290, 0.93452, 0.79411, 0.49018, 0.34381, 0.76212,
          0.22538, 0.33014, 0.25895, 0.93673, 0.81686, 0.19273, 0.09593, 0.61784],
         [0.83005, 0.41436, 0.51144, 0.14464, 0.06673, 0.23759, 0.79393, 0.15183, 0.48951, 0.82814, 0.13327, 0.90146,
          0.01373, 0.45276, 0.23481, 0.91586, 0.70961, 0.87237, 0.51975, 0.60809],
         [0.96390, 0.03900, 0.10980, 0.45746, 0.40082, 0.93653, 0.34810, 0.38781, 0.02690, 0.41445, 0.03016, 0.92082,
          0.95693, 0.36446, 0.35043, 0.87621, 0.42398, 0.56390, 0.42304, 0.47191],
         [0.03961, 0.32912, 0.65414, 0.57216, 0.69425, 0.21957, 0.38606, 0.13243, 0.21100, 0.85884, 0.58128, 0.00700,
          0.56567, 0.19933, 0.06938, 0.29934, 0.70919, 0.83511, 0.64443, 0.58391],
         [0.78917, 0.85907, 0.85717, 0.61840, 0.27239, 0.07927, 0.13968, 0.31514, 0.71946, 0.34410, 0.33951, 0.01508,
          0.97620, 0.22193, 0.95409, 0.30118, 0.50626, 0.97602, 0.42920, 0.30206],
         [0.70039, 0.27355, 0.10390, 0.88195, 0.59273, 0.54032, 0.61879, 0.23052, 0.22973, 0.55234, 0.89494, 0.41565,
          0.88915, 0.02219, 0.66380, 0.37647, 0.16625, 0.97185, 0.44439, 0.21690],
         [0.68653, 0.21699, 0.78011, 0.36668, 0.25246, 0.52910, 0.45910, 0.33039, 0.54962, 0.47008, 0.32642, 0.68525,
          0.73092, 0.08418, 0.21559, 0.12048, 0.86418, 0.16281, 0.70666, 0.44776],
         [0.03907, 0.02877, 0.58390, 0.65541, 0.34840, 0.02032, 0.98649, 0.54299, 0.30484, 0.41928, 0.16826, 0.23659,
          0.36868, 0.11280, 0.90247, 0.92737, 0.57269, 0.90643, 0.58467, 0.37325],
         [0.33571, 0.56166, 0.99835, 0.71396, 0.59501, 0.93244, 0.38599, 0.01835, 0.05954, 0.78744, 0.76053, 0.48920,
          0.67954, 0.85783, 0.72453, 0.81003, 0.47943, 0.76547, 0.30040, 0.16407],
         [0.14837, 0.71181, 0.00153, 0.29445, 0.11795, 0.78590, 0.19318, 0.74584, 0.23949, 0.75022, 0.10379, 0.96772,
          0.73061, 0.68969, 0.77211, 0.90114, 0.74025, 0.72178, 0.45157, 0.96021],
         [0.53532, 0.86542, 0.00800, 0.34083, 0.72879, 0.72793, 0.33708, 0.89182, 0.58632, 0.91492, 0.31333, 0.00340,
          0.08711, 0.10384, 0.89820, 0.23836, 0.69862, 0.12074, 0.01226, 0.99084],
         [0.27116, 0.38629, 0.89641, 0.19935, 0.95614, 0.14205, 0.98326, 0.65688, 0.76412, 0.10330, 0.17495, 0.04045,
          0.21596, 0.41213, 0.90190, 0.68773, 0.30942, 0.01571, 0.72228, 0.56419],
         [0.65593, 0.58923, 0.16732, 0.41696, 0.67558, 0.18070, 0.81363, 0.91857, 0.42699, 0.00047, 0.00520, 0.89068,
          0.02178, 0.88992, 0.30571, 0.77229, 0.78262, 0.97645, 0.78683, 0.19610],
         [0.12995, 0.77744, 0.75203, 0.65017, 0.36600, 0.52477, 0.72683, 0.97449, 0.68050, 0.70987, 0.90008, 0.20979,
          0.88389, 0.32863, 0.81828, 0.51616, 0.12354, 0.27127, 0.88084, 0.06849],
         [0.62480, 0.61747, 0.75166, 0.15317, 0.43502, 0.22613, 0.90412, 0.90823, 0.71930, 0.62386, 0.27402, 0.81831,
          0.69587, 0.91607, 0.48021, 0.84686, 0.43908, 0.96148, 0.87785, 0.42359],
         [0.91381, 0.74182, 0.48260, 0.70399, 0.38430, 0.63318, 0.85410, 0.69679, 0.86428, 0.95554, 0.77500, 0.10197,
          0.16941, 0.43649, 0.60242, 0.16425, 0.25089, 0.48433, 0.75074, 0.12957],
         [0.51484, 0.80035, 0.31703, 0.05262, 0.22468, 0.19152, 0.95901, 0.26500, 0.39138, 0.54799, 0.86046, 0.03572,
          0.53637, 0.97833, 0.64290, 0.62725, 0.12480, 0.97565, 0.29386, 0.60190],
         [0.39331, 0.94188, 0.29797, 0.62912, 0.15720, 0.81326, 0.12892, 0.65237, 0.73652, 0.96239, 0.48915, 0.33423,
          0.80844, 0.45654, 0.82198, 0.60356, 0.81259, 0.52657, 0.06561, 0.71042],
         [0.44960, 0.78332, 0.19905, 0.18862, 0.35294, 0.98888, 0.72283, 0.71446, 0.73216, 0.14122, 0.18353, 0.47805,
          0.85675, 0.44683, 0.64424, 0.30170, 0.37098, 0.06220, 0.54310, 0.70363],
         [0.84614, 0.98510, 0.30790, 0.64153, 0.28578, 0.62497, 0.74349, 0.53000, 0.29905, 0.75485, 0.10080, 0.65703,
          0.57013, 0.10693, 0.48604, 0.53441, 0.04254, 0.99562, 0.66300, 0.82241]]
    )
    y_fit = np.array(
        [9.97276831, 5.20347703, 8.44838987, 13.40445604, 12.87295258,
         12.85158101, 9.54494144, 9.75856414, 14.63767055, 4.94163855,
         9.59647977, 13.1030849, 11.29800941, 11.82067791, 11.06054368,
         12.09024811, 5.59730675, 9.96535479, 7.82585093, 10.74349242]
    )
    y_true = np.array(
        [10.37553193, 5.60622391, 7.9324083, 12.80136197, 12.32556865,
         12.19662639, 8.96070533, 9.651885, 14.9962888, 4.78846267,
         10.67343369, 13.97464298, 10.91001649, 12.02196237, 9.76558201,
         12.12622737, 5.99678822, 9.00129322, 7.26769625, 10.68338695]
    )
    X_df = pd.DataFrame(X, columns=["D{0:02d}".format(i) for i in range(X.shape[1])])
    return X, y_fit, y_true, X_df


def test_evolutionary_feature_selection(data, regressor, synthetic_dataset):
    # fit on bullshit data
    est = EvolutionaryFeatureSelection(random_state=31337, n_features=2)
    est.fit(*data)
    assert_allclose(est.transform(data[0][0:5, :]), np.array([[1.4, 0.2],
                                                              [1.4, 0.2],
                                                              [1.3, 0.2],
                                                              [1.5, 0.2],
                                                              [1.4, 0.2]]),
                 err_msg="Transformed features differ from reference")
    assert_allclose(est.fitness_values_, est.fitness_history_[-1],
                    err_msg="Fitness and fitness_history inconsistent")
    assert_allclose(est.fitness_values_, np.array([0.92574512, 0.92574512, 0.92574512, 0.92574512, 0.92574512,
                                                   0.92574512, 0.92574512, 0.92574512, 0.92574512, 0.92574512]),
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
    # test successful feature selection
    est4 = EvolutionaryFeatureSelection(generations=50, n_features=5, population_size=20, n_breeders=8,
                                        mutation_rate=0.1, random_state=1337)
    est4.fit(synthetic_dataset[0], synthetic_dataset[2])
    assert_array_equal(est4.get_support(), np.array([True, True, True, True, True, False, False, False, False,
                                                     False, False, False, False, False, False, False, False, False,
                                                     False, False]),
                       err_msg="Wrong features selected from synthetic data set")
    assert_array_equal(est4.get_feature_names_out(), np.array(['x0', 'x1', 'x2', 'x3', 'x4']),
                       err_msg="Generated feature names differ")
    est5 = EvolutionaryFeatureSelection(generations=50, n_features=5, population_size=20, n_breeders=8,
                                        mutation_rate=0.1, random_state=1337)
    est5.fit(synthetic_dataset[3], synthetic_dataset[2])
    assert_array_equal(est5.get_feature_names_out(), np.array(['D00', 'D01', 'D02', 'D03', 'D04']))

