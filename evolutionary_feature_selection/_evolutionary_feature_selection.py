"""
Select Features by Evolutionary Algorithm
"""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_X_y, check_is_fitted, check_random_state, check_scalar
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression


class EvolutionaryFeatureSelection(BaseEstimator, SelectorMixin):
    """ A feature selection transformer that selects a set of features of given size by evolutionary optimization

    The optimization is performed in a procreation-mutation-selection cycle with the fitness calculated as the
    validation score of the predictor fitted using a particular selection of features.

    Parameters
    ----------
    predictor : predictor
        predictor to train on the reduced feature set
    scoring : callable
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

    Attributes
    ----------
    random_state_ : Numpy random state
        For testing and consistent parallel processing
    population_ : Ndarray shape(population_size, n_features_in_)
        The full population of feature masks in the current iteration/generation
    fitness_values_ : Ndarray, shape(population_size)
        The fitness/score values of the population
    current_specimen_ : Ndarray, shape(n_features_in_)
        The specimen with the highest fitness value, same as population_[0]
    fitness_history_ : Ndarray, shape(generations, population_size)
        Trace of the population fitness values over along all generations for fit debugging and quality assessment
    """

    def __init__(self, predictor=None, scoring=None, n_features=1,
                 population_size=10, n_breeders=5, mutation_rate=0.5, n_mutation_features=1,
                 generations=10, initial_population=None, random_state=None):
        self.predictor = predictor
        self.scoring = scoring
        self.n_features = check_scalar(n_features, "n_features", int, min_val=1)
        self.population_size = check_scalar(population_size, "population_size", int, min_val=1)
        self.n_breeders = check_scalar(n_breeders, "n_breeders", int, min_val=1)
        self.mutation_rate = check_scalar(mutation_rate, "mutation_rate", float, min_val=0, max_val=1)
        self.n_mutation_features = check_scalar(n_mutation_features, "n_mutation_features", int, min_val=0,
                                                max_val=self.n_features)
        self.generations = check_scalar(generations, "generations", int, min_val=1)
        self.initial_population = initial_population
        self.random_state = random_state

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
        self.random_state_ = check_random_state(self.random_state_)
        new_specimen = np.array(specimen)
        for i in range(self.n_mutation_features):
            new_specimen[self.random_state_.choice(np.argwhere(specimen).ravel())] = False
            new_specimen[self.random_state_.choice(np.argwhere(np.bitwise_not(specimen)).ravel())] = True
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
        self.random_state_ = check_random_state(self.random_state_)
        offspring = father + mother
        while np.count_nonzero(offspring) > self.n_features:
            offspring[self.random_state_.choice(np.argwhere(offspring).ravel())] = False
        return offspring

    def _populate(self, n_features_in):
        """
        Creates the population from initial population if specified, plus the required number of random specimen

        Parameters
        ----------
        n_features_in : int number of features in the input data set
        """
        self.random_state_ = check_random_state(self.random_state_)
        self.population_ = np.zeros((self.population_size, n_features_in), dtype=bool)
        if self.initial_population is not None:
            row = min(self.initial_population.shape[0], self.population_size)
            self.population_[0:row] = self.initial_population[0:row]
        else:
            row = 0
        while row < self.population_size:
            self.population_[row, self.random_state_.randint(0, n_features_in, self.n_features)] = True
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
        for specimen_index in range(self.population_.shape[0]):
            self.fitness_values_[specimen_index] = self._calculate_fitness_specimen(X, y,
                                                                                    self.population_[specimen_index])
        return self.fitness_values_

    def _calculate_fitness_specimen(self, X, y, specimen):
        local_X = X[:, specimen]
        y_pred = self.predictor_.fit(local_X, y).predict(local_X)
        fitness = self.scoring_(y, y_pred)
        return fitness

    def _get_support_mask(self):
        check_is_fitted(self, ("population_", "current_specimen_", "fitness_values_"))
        return self.current_specimen_

    def fit(self, X, y):
        self._validate_data(X=X, y=y, reset=True)
        X, y = check_X_y(X, y, ensure_min_features=2)
        self._check_n_features(X, reset=True)
        # self._check_feature_names(X, reset=True) This has no meaningful function yet
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        else:
            self.feature_names_in_ = np.array(["X{0:d}".format(i) for i in range(self.n_features_in_)])
        if self.predictor is None:
            self.predictor_ = LinearRegression()
        else:
            self.predictor_ = self.predictor()
        if self.scoring is None:
            self.scoring_ = r2_score
        else:
            self.scoring_ = self.scoring
        self.random_state_ = check_random_state(self.random_state)
        self._populate(X.shape[1])
        # set initial fitness values and sort initial specimens by fitness
        self.fitness_values_ = np.zeros(self.population_size, dtype=float)
        self._calculate_fitness_values(X, y)
        fitness_order = np.argsort(- self.fitness_values_)
        self.population_ = self.population_[fitness_order]
        self.fitness_values_ = self.fitness_values_[fitness_order]
        self.current_specimen_ = self.population_[0]
        # iterate generations
        self.fitness_history_ = np.zeros((self.generations + 1, self.population_size))
        self.fitness_history_[0] = self.fitness_values_
        temp_pop_size = (2 * self.population_size) - self.n_breeders
        converged = False
        generation = 0
        while not converged:
            temp_pop = np.zeros((temp_pop_size, self.n_features_in_), dtype=bool)
            temp_pop[0:self.population_size] = self.population_
            temp_fitness_values = np.zeros(temp_pop_size)
            temp_fitness_values[0:self.population_size] = self.fitness_values_
            for offspring_idx in range(self.population_size, temp_pop_size):
                dad, mom = self.random_state_.randint(self.n_breeders, size=2)
                temp_pop[offspring_idx] = self._procreate(self.population_[dad], self.population_[mom])
                if self.random_state_.rand() <= self.mutation_rate:
                    temp_pop[offspring_idx] = self._mutate(temp_pop[offspring_idx])
                temp_fitness_values[offspring_idx] = self._calculate_fitness_specimen(X, y, temp_pop[offspring_idx])
            fitness_order = np.argsort(-temp_fitness_values)[0:self.population_size]
            self.population_ = temp_pop[fitness_order]
            self.fitness_values_ = temp_fitness_values[fitness_order]
            self.current_specimen_ = self.population_[0]
            generation += 1
            self.fitness_history_[generation] = self.fitness_values_
            if generation >= self.generations:
                converged = True
        return self

