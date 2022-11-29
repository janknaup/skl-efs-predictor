.. title:: User guide : contents

.. _user_guide:

==========================================
User guide: Evolutionary Feature Selection
==========================================

Theory
------

For highly under-determined data sets, i.e. systems with a far larger number of
feature columns than data points, an evolutionary feature selection can be more efficient at
selecting a predictive feature set than generalized regression techniques. An evolutionary
algorithm features a loop of recombination - mutation - selection acting on a population of
candidate solutions.

Population
~~~~~~~~~~
Each specimen in the population represents a mask for selecting features from the input set.
Internally it is represented as a boolean array of shape (population_size, n_features_in).
Each row of this array should have n_features True values. If an initial population with a
differing number of selected features is passed, this will be corrected by the recombination
algorithm over time, but deviating specimen may endure if their fitness is greater that that
of offspring with the specified number of features.

Recombination
~~~~~~~~~~~~~
Recombination starts with random selection of 2 different parent specimen fromm the breeders.
These are the n_breeders specimen with the highest fitness values. The parent masks are then
combined by bitwise OR, creating the union of selected features from both parents. Random
features are then deselected until the number of selected features is less or equal to
n_features. This procedure will lead to offspring with the correct number of features as long
as the union of parent features is larger than n_features. Recombined feature selections with
less features will be tolerated.
In each cycle, one new member will be added to the temporary population for each specimen that
is not a breeder, i.e. population_size - n_breeders offspring specimen are generated.

Mutation
~~~~~~~~
Each offspring specimen may be mutated directly after recombination with probability
mutation_rate. Mutation is performed by deselecting n_mutation_features features and selecting
the same number of formerly unselected features. I.e. a mutation element will always replace
n_mutation_features features, a null mutation event in which the same feature is reselected is
prevented.

Selection
~~~~~~~~~
Selection is based on fitting the provided predictor to the training data and calculating the
score via the provided scoring function. The scoring function must be defined such that a higher
score marks a fitter specimen. The default predictor is LinerRegression and the default
scoring function is r2_score.

In each generation the fitness calculation is only done for new specimens. The whole population
is then ranked by fitness value and the n_features fittest specimens for the new population,
while the rest are discarded. The fittest specimen of the population is used as the feature
selection mask for transform() calls. The fitness values of each generation are stored in
in the fitness_history attribute for tracing the progress of the feature set optimization.

Usage
-----
not yet

