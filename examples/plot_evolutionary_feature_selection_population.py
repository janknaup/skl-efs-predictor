"""
===================================================================
Evolutionary Feature Selection Transformer - Features in Population
===================================================================

An example plot of :class:`evolutionary_feature_selection.EvolutionaryFeatureSelection` population trace.

The data set has 20 features, all of which are random in the range [0..1]. The response depends only on the first 5
variables by the formula:

    0.5 + (8.0 * X[:, 0]) + (6.0 * X[:, 1]) + (4.0 * X[:, 2]) + (2.0 * X[:,  3]) + X[:, 4]

The heatmap color shows the fraction of population specimens containing each of the features on the y-axis. The
x-axis shows the generations (iterations) of the evolution.

The algorithm is set up to contain 20 specimens, the top 50% of which (10) are allowed to breed. Each specimen is
set to contain 5 features.
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

plt.pcolormesh(np.mean(efs.population_history_, axis=1).transpose())
plt.ylabel('feature')
plt.xlabel('generation')
colbar = plt.colorbar()
colbar.ax.set_ylabel('fraction of population')
plt.title('Fraction of Specimens Containing Feature')

plt.show()

