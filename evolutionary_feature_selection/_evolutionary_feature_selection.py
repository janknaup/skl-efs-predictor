"""
Select Features by Evolutionary Algorithm
"""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_X_y, check_is_fitted, check_random_state, check_scalar
from sklearn.metrics import r2_score


class EvolutionaryFeatureSelection(BaseEstimator, SelectorMixin):
    """ A feature selection transformer that selects a set of features of given size by evolutionary optimization

    The optimization is performed in a procreation-mutation-selection cycle with the fitness calculated as the
    validation score of the predictor fitted using a particular selection of features.

    Parameters
    ----------
    predictor : predictor
        predictor to train on the reduced feature set
    score : callable
        scoring function to use as fitness function for specimen selection
    n_features : int
        number of features to select per specimen
    population_size : int
        number of specimens in the population
    n_breeders : int
        number of specimens from the population from which breeders are selected. must be < population_size
    mutation_rate : float 0 <= mutation_rate <= 1
        fraction of offspring per generation that are mutated
    n_mutation_features : int >0, <n_features, default = 0.1 * n_features
        number of features to swap upon mutation.
    generations : int > 0
        number of generations
    initial_population : array-like shape(N, n_features)
        initial feature selections. If N < population_size, random specimens will be generated to reach population_size,
        if N > population_size, the first population_size specimens from initial_population will be used. Useful for
        continuing previous evolutionary feature selection runs.
    random_state : numpy.random.random_state
        random state for repeatability in testing

    """

    def __int__(self, predictor, scoring=None, n_features=None,
                population_size=10, n_breeders=None, mutation_rate=0.5, n_mutation_features=None,
                generations=10, initial_population=None, random_state=None):
        self.predictor = predictor
        if scoring is not None:
            self.scoring = scoring
        else:
            self.scoring = r2_score
        self.n_features = n_features
        self.population_size = population_size
        self.breeder_fraction = n_breeders
        if n_breeders is None:
            self.n_breeders = round(self.n_features * 0.5)
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
        self._fitness_history = None

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
            self._fitness_values[specimen_index] = self._calculate_fitness_specimen(X, y,
                                                                                    self._population[specimen_index])
        return self._fitness_values

    def _calculate_fitness_specimen(self, X, y, specimen):
        self._current_specimen = specimen
        y_pred = self.predictor.fit(self.transform(X))
        fitness = self.scoring(y, y_pred)
        self._current_specimen = self._population[0]
        return fitness

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
        self._fitness_history = np.zeros((self.generations + 1, self.population_size))
        self._fitness_history[0] = self._fitness_values
        temp_pop_size = (2 * self.population_size) - self.n_breeders
        converged = False
        generation = 0
        while not converged:
            temp_pop = np.zeros((temp_pop_size, self._population[0].shape[1]))
            temp_pop[0:self.population_size] = self._population
            temp_fitness_values = np.zeros(temp_pop_size)
            temp_fitness_values[0:self.population_size] = self._fitness_values
            for offspring_idx in range(self.population_size, temp_pop_size + 1):
                dad, mom = self.random_state_.randint(self.n_breeders, size=2)
                temp_pop[offspring_idx] = self._procreate(self._population[dad], self._population[mom])
                if self.random_state_.rand() <= self.mutation_rate:
                    temp_pop[offspring_idx] = self._mutate(temp_pop[offspring_idx])
                temp_fitness_values[offspring_idx] = self._calculate_fitness_specimen(temp_pop[offspring_idx])
            fitness_order = np.argsort(temp_fitness_values)[0:self.population_size]
            self._population = temp_pop[fitness_order]
            self._fitness_values = temp_fitness_values[fitness_order]
            self._fitness_history[generation + 1] = self._fitness_values
            generation += 1
            if generation >= self.generations:
                converged = True
        return self

