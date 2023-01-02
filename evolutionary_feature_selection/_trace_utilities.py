""" Utilities for plotting evolutionary feature selection diagnostic traces """

from matplotlib import pyplot as plt
import matplotlib
import numpy as np
def fitness_trace(fitness_trace, top_specimen=True, all_specimens=False, mean=True, range_area=True):
    """
    Plots a trace of the population fitnesses for

    Parameters
    ----------
    fitness_trace : Ndarray, float, shape(generations, population_size)
        The population trace to be plotted
    top_specimen : bool, default True
        Plot a line representing the highest fitness in the population
    all_specimens : bool, default False
        Plot lines representing each specimen rank in the population
    mean : bool, default True
        Plot a line representing the mean fitness in the population
    range_area : bool, default True
        Plot a colored area between the top and bottom fitness values

    Returns
    -------
    figure object
    """

    x = np.arange(fitness_trace.shape[0])
    fig = plt.figure()

    if top_specimen or all_specimens:
        top_line = plt.plot(fitness_trace[:, 0], label='top fitness', c='blue')

    if all_specimens:
        all_lines = plt.plot(fitness_trace[:, 1:], c='black', lw=0.1)

    if mean:
        mean_line = plt.plot(np.mean(fitness_trace, axis=1), label='mean fitness', c='red')

    if range_area:
        range_area_trace = plt.fill_between(x=x, y1=fitness_trace[:, 0], y2=fitness_trace[:, -1],
                                            facecolor='red', alpha=0.25)
    plt.legend()
    plt.xlabel('generation')
    plt.ylabel('fitness')

    return fig

def feature_trace(population_trace, specimen='top', feature_names=None):
    """
    Plots a trace which features are contained in the specimens of each generation

    Parameters
    ----------
    population_trace : Ndarray, float, shape(generations, population_size, n_features_in)
        The population trace to be diagnosed
    specimen : str, from ['top', 'all']
        If *top*, plot the trace of the support mask (i.e. features in the fittest specimen),
        if *all* plot a heatmap of the fraction of the specimens containing each feature.
    feature_names : array-like, str, shape(n_features) or None
        If not none, label y-axis with the feature names in the passed array.

    Returns
    -------
    matplotlib figure object

    """
    if specimen not in ['top', 'all']:
        raise ValueError("Unknown specimen selection ", specimen)
    else:
        fig = plt.figure()
        if specimen == 'top':
            data = np.transpose(population_trace[:, 0, :])
            title = "Feature Support Mask"
            color_scheme = 'Greys'
        elif specimen == 'all':
            data = np.mean(population_trace, axis=1).transpose()
            title = "Fraction of Population Containing Feature"
            color_scheme = 'viridis'
        plt.imshow(data, cmap=color_scheme)
        plt.title(title)
        plt.xlabel('generation')
        plt.ylabel('feature')
        #if feature_names is not None:
        #    plt.yticks(labels=feature_names)
        if specimen == 'all':
            plt.colorbar()
    return fig

def fitness_diversity_trace(fitness_trace, std=True, range=False):
    """
    Plots a trace of population fitness diversity measures.

    Parameters
    ----------
    fitness_trace : Ndarray, float, shape(generations, population_size)
        The population trace to be analyzed
    std : bool, default=True
        Plot the standard deviation of the population fitness per generation
    range : bool, default=False
        Plot the range of population fitness values per generation

    Returns
    -------
    matplotlib figure object

    """
    x = np.arange(fitness_trace.shape[0])
    fig = plt.figure()

    if std:
        plt.plot(np.std(fitness_trace, axis=1), label=r"$\sigma$")
    if range:
        plt.plot(np.max(fitness_trace, axis=1) - np.min(fitness_trace, axis=1), label="range")
    plt.title('Fitness Diversity')
    plt.xlabel('generation')
    plt.ylabel('fitness')
    plt.legend()
    return fig
