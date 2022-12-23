"""
=================================================================
Evolutionary Feature Selection Transformer - Fitness of Specimens
=================================================================

An example plot of :class:`evolutionary_feature_selection.EvolutionaryFeatureSelection` fitness trace.

The data set has 20 features, all of which are random in the range [0..1]. The response depends only on the first 5
variables by the formula:

    0.5 + (8.0 * X[:, 0]) + (6.0 * X[:, 1]) + (4.0 * X[:, 2]) + (2.0 * X[:,  3]) + X[:, 4]

Each line in the plot shows the fitness of the specimen occupying one place in the population, ranked by the fitness
value.

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
                                   random_state=rand_stat, fitness_trace=True)
efs.fit(X, y_true)

# Full population fitness
plt.plot(efs.fitness_history_)
plt.ylabel('R² score')
plt.xlabel('generation')
plt.ylim((0.0, 1.05))
plt.grid(axis='y')
plt.title('Fitnesses of all Specimens')

plt.show()
