# given a list of points create a regression fit (polynomial, n = 2)(flowability, % composition)
# Polynomial Regression Tutorial: https://link.medium.com/eDOWMkix4cb
# Normalized Inverse Flowability (0 - 1) (y value)
# Percent Composition (0-100) (x value)
from statistics import mean

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures


def regression_fit(x, y) -> LinearRegression:
    """
    given a list of x and y values return a regression model of polynomial n = 2
    :param x: x-coordinates
    :param y: y-coordinates
    :return: regression model
    """
    polynomial_features = PolynomialFeatures(degree=2)
    x_poly = polynomial_features.fit_transform(x)

    model = LinearRegression()
    model.fit(x_poly, y)

    return model


def calculate_r2(x, y, regression_model: LinearRegression):
    """
    given a list of x, y values and a regression model find the r-squared value of the fit
    :param x: x-coordinates
    :param y: y-coordinates
    :param regression_model: regression fit of points
    :return: r2 value
    """
    polynomial_features = PolynomialFeatures(degree=2)
    x_poly = polynomial_features.fit_transform(x)
    y_poly_pred = regression_model.predict(x_poly)

    return r2_score(y, y_poly_pred)


def graph_scatter(x, y, zorder=1, color=None, alpha=None):
    """
    given a set of points generate a scatter plot of the points
    :param x: x-coordinates
    :param y: y-coordinates
    :param zorder: the z order in which the scatter plot should appear
    :return: scatterplot graph
    """
    if color is not None and alpha is not None:
        return plt.scatter(x, y, zorder=zorder, color=color, alpha=alpha)
    else:
        return plt.scatter(x, y, zorder=zorder)


def add_regression_model(regression_model: LinearRegression, zorder=2, color=None, alpha=None):
    """
    given a scatter graph and a regression model, add the regression model to the graph, return a copy of graph
    :param regression_model:
    :param zorder: the order in which the regression model should appear when plotted against other values
    :return: regression/scatter graph
    """
    axes = plt.gca()
    _, x_max = axes.get_xlim()
    x = np.arange(0, 101, 10)
    poly_reg = PolynomialFeatures(degree=2)
    x_poly = poly_reg.fit_transform(x.reshape(-1, 1))
    # m = regression_model.coef_[0][0]
    # b = regression_model.intercept_[0]
    y = regression_model.predict(x_poly)
    axes.set_ylim([0, 1])
    if color is not None and alpha is not None:
        return plt.plot(x, y, zorder=zorder, color=color, alpha=alpha)
    else:
        return plt.plot(x, y, zorder=zorder)


def generate_all_regression_fits(x, y, subset_size=2):
    """
    given a set of points, and a subset size, generate all of the possible regression fits
    TODO add restriction saying subset_size >= 2?
    :param x: x-coordinates
    :param y: y-coordinates
    :param subset_size: default 2, size of regression subsets
    :return: list of regression models
    """
    result = []
    # generate indices of all subsets of each axis based on subset size
    x_subsets = list(ind_combinations(x.tolist(), subset_size))
    y_subsets = list(ind_combinations(y.tolist(), subset_size))

    for i in range(len(x_subsets)):
        # create temporary lists that act as subsets of x or y
        x_temp = np.array(get_list_subset(x, x_subsets[i])).reshape(-1, 1)
        y_temp = np.array(get_list_subset(y, y_subsets[i])).reshape(-1, 1)

        # get the fit for that subset, and add it to our result
        result.append(regression_fit(x_temp, y_temp))

    return result


def ind_combinations(iterable, r):
    """
    a version of itertools.combinations that returns indices of subsets, rather than the values for those indices
    taken from: https://docs.python.org/3/library/itertools.html#itertools.combinations
    :param iterable: something to iterate over, i.e. a list or string
    :param r: the length of each tuple subset
    :return: a list of all subsets of size r
    """
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = list(range(r))
    yield tuple(i for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i+1, r):
            indices[j] = indices[j-1] + 1
        yield tuple(i for i in indices)


def get_list_subset(base_list, index_list):
    """
    creates a subset of base_list based on the indices listed in index_list
    :param base_list: a list that we're taking a subset of
    :param index_list: a list containing the indices we want in our subset
    :return: a subset of base_list based on the indices listed in index_list
    """
    result = []

    for i in index_list:
        result.append(base_list[i])

    return result


def calculate_all_r2(x, y, regression_models):
    """
    given a set of points, a list of regression models return a list of r2 values
    :param x: x-coordinates
    :param y: y-coordinates
    :param regression_models: a list of regression models
    :return: list of r2_values
    """
    return [calculate_r2(x, y, model) for model in regression_models]


def create_graph_animation_frame(x, y, regression_model, r2_value, r2_value_all_points):
    """
    create one frame of the regression fit animation
    :param x: x-coordinates
    :param y: y-coordinates
    :param regression_model: regression fit model
    :param r2_value: r2_value fit on just the regression points
    :param r2_value_all_points: r2_value for all points
    :return: annotated regression/scatter graph
    """
    plt.clf()
    add_regression_model(regression_model)
    graph_scatter(x, y)
    axes = plt.gca()

    plt.text(2, 0.01, f"R2: {r2_value}\nR2 (average): {r2_value_all_points}")
    axes.set_xlim([0, 100])
    axes.set_ylim([0, 1])
    plt.ylabel("Inverse Flowability")
    plt.xlabel("% Mixture")


def next_frame(index, x, y, all_models, all_r2s, r2_value_all_points):
    print(index)
    create_graph_animation_frame(x, y, all_models[index], all_r2s[index], r2_value_all_points)


# https://www.c-sharpcorner.com/article/create-animated-gif-using-python-matplotlib/
def create_graph_animation(x, y, all_models, all_r2s):
    """
    given a list of the graph animation frames, return a gif/matplotlib graph iterator
    :param graph_frames: a list of graphs
    :return: return a gif
    """

    ani = FuncAnimation(fig=plt.figure(), func=next_frame, fargs=(x, y, all_models, all_r2s, mean(all_r2s)))

    writer = PillowWriter(fps=2)
    ani.save("demo.gif", writer=writer)

    plt.show()


def create_overlapped_graph(x, y, all_models, mixture_title):
    """
    given a list of all models, overlap them over the general scatter-plot
    :param x: x-coordinates of points on the scatter plot
    :param y: y-coordinates of points on the scatter plot
    :param all_models: a list of models
    :return: none
    """
    for model in all_models:
        add_regression_model(model, zorder=1, color='lightskyblue', alpha=.7)

    graph_scatter(x, y, zorder=2, color='skyblue')
    plt.xlabel("% Composition of Mixture")
    plt.ylabel("Normalized Inverse Flowability")
    plt.title("Normalized Inverse Flowability of " + mixture_title)
    plt.show()
