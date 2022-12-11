"""
==========================================
Evolutionary Feature Selection Transformer
==========================================

An example plot of :class:`evolutionary_feature_selection.EvolutionaryFeatureSelection` fitness and population traces.

The data set has 20 features, all of which are random in the range [0..1]. The response depends only on the first 5
variables by the formula:

    0.5 + (8.0 * X[:, 0]) + (6.0 * X[:, 1]) + (4.0 * X[:, 2]) + (2.0 * X[:,  3]) + X[:, 4]

"""

import numpy as np
from matplotlib import pyplot as plt
from evolutionary_feature_selection import EvolutionaryFeatureSelection

rand_stat = np.random.RandomState(31337)

X = rand_stat.random((20, 20))

y_true = 0.5 + (8.0 * X[:, 0]) + (6.0 * X[:, 1]) + (4.0 * X[:, 2]) + (2.0 * X[:,  3]) + X[:, 4]

efs = EvolutionaryFeatureSelection(generations=45, population_size=20, n_breeders=10, n_features=5,
                                   random_state=rand_stat, population_trace=True)
efs.fit(X, y_true)

fig, ax = plt.subplots(2, 1, squeeze=False, sharex='all')

# Full population fitness
ax[0][0].plot(efs.fitness_history_)
ax[0][0].set_ylabel('R² Score')
ax[0][0].set_ylim((0.0, 1.05))

ax[1][0].pcolormesh(np.mean(efs.population_history_, axis=1).transpose())
ax[1][0].set_ylabel('Feature')
ax[1][0].set_xlabel('Iteration')

fig.suptitle('Specimen Fitness and Feature Frequency')
fig.show()

