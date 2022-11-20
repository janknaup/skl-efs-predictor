"""
Select Features by Evolutionary Algorithm
"""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_X_y, check_is_fitted, check_random_state
from sklearn.metrics import r2_score


class EvolutionaryFeatureSelection(BaseEstimator, SelectorMixin):
    """ A Transformer that

    Parameters
    ----------
    predictor : predictor to train on the reduced feature set
    score : scoring function to use as fitness function for specimen selection
    n_features : int number of features to select per specimen
    population_size : int number of specimen in the
    breeder_fraction : float fraction of the population from which breeders are selected
    mutation_rate : float fraction of offspring per generation that are mutated
    n_mutation_features : int number of features to swap upon mutation. default 0.1 * n_features
    generations : number of generations
    initial_population : array-like initial feature selections. shape(N, n_features). If N < population_size.
        random specimens will be generated to reach population_size, if N > population_size, the first
        population_size specimens from initial_population will be used. Useful for continuing previous
        evolutionary feature selection runs.
    random_state : random state for repeatability in testing

    """

    def __int__(self, predictor, scoring=None, n_features=None,
                population_size=10, breeder_fraction=0.5, mutation_rate=0.5, n_mutation_features=None,
                generations=10, initial_population=None, random_state=None):
        self.predictor = predictor
        if scoring is not None:
            self.scoring = scoring
        else:
            self.scoring = r2_score
        self.n_features = n_features
        self.population_size = population_size
        self.breeder_fraction = breeder_fraction
        self._n_breeders = round(self.n_features * breeder_fraction)
        self.mutation_rate = mutation_rate
        if n_mutation_features is None:
            self.n_mutation_features = round(self.n_features * 0.1)
        else:
            self.n_mutation_features = n_mutation_features
        self.generations = generations
        self.initial_population = initial_population
        self.random_state = random_state
        # ---------
        self._population = None
        self._fitness_values = None
        self._current_specimen = None
        self._fitness_history = []


    def _mutate(self, specimen):
        """
        Mutates a specimen by replacing a number of feature indices from the self._features_seen
        The method assures that the removed features are replaced by features not present in the original specimen

        Parameters
        ----------
        specimen : array-like of feature indices or column names

        Returns
        -------
        modified array-like of feature indices or column names

        """
        self.random_state_ = check_random_state(self.random_state)
        new_specimen = np.array(specimen)
        for i in range(self.n_mutation_features):
            new_specimen[self.random_state_.choice(np.argwhere(specimen))] = False
            new_specimen[self.random_state_.choice(np.argwhere(np.bitwise_not(specimen)))] = True
        return new_specimen

    def _procreate(self, father, mother):
        """
        Create a new specimen by randomly combining features from parent specimens

        Parameters
        ----------
        father : first parent mask array
        mother : second parent mask array

        Returns
        -------
        mask array
        """
        self.random_state_ = check_random_state(self.random_state)
        offspring = father + mother
        while np.count_nonzero(offspring) > self.n_features:
            offspring[self.random_state_.choice(np.argwhere(offspring))] = False
        return offspring

    def _populate(self, n_features_in):
        """
        Creates the population from initial population if specified, plus the required number of random specimen

        Parameters
        ----------
        n_features_in : int number of features in the input data set
        """
        self.random_state_ = check_random_state(self.random_state)
        self._population = np.zeros(self.population_size, n_features_in, dtype=bool)
        if self.initial_population is not None:
            row = self.initial_population.shape[1]
            self._population[0:min(self.initial_population.shape[1], self.population_size)] = self.initial_population
        else:
            row = 0
        while row < self.population_size:
            self._population[row, self.random_state_.randint(0, n_features_in, self.n_features)] = True
            row += 1

    def _calculate_fitness_values(self, X, y):
        """
        Calculate Fitnesses for the population

        Parameters
        ----------
        y : array-line true values

        Returns
        -------
        array of float fitness values
        """
        for specimen_index in range(self._population.shape[0]):
            self._current_specimen = self._population[specimen_index]
            y_pred = self.predictor.fit(self.transform(X))
            self._fitness_values[specimen_index] = self.scoring(y, y_pred)
        return self._fitness_values

    def _get_support_mask(self):
        return self._current_specimen

    def fit(self, X, y):
        X, y = check_X_y(X, y, ensure_min_features=self.n_features)
        self.random_state_ = check_random_state(self.random_state)
        if self._population is None:
            self._populate(X)
        # set initial fitness values and sort initial speciments by fitness
        print("Initiating evolutionary feature selection")
        self._fitness_values = np.zeros(self.population_size, dtype=float)
        self._calculate_fitness_values(X, y)
        fitness_order = np.argsort(self._fitness_values)
        self._population = self._population[fitness_order]
        self._fitness_values = self._fitness_values[fitness_order]
        print("Initial population fitness scores ({0})".format(self.scoring.__name__))
        print(self._fitness_values)
        # iterate generations
        converged = False
        generation = 1
        while not converged:

            generation += 1
            if generation > self.generations:
                converged = True
        return self

