""" SplitXY

This module is responbile for splitting microtrac data pandas dataframe into x and y numpy arrays
"""

import pandas


def get_numpy_data(x: pandas.DataFrame,  y: pandas.DataFrame):
    """
    given the pandas dataframe of x and y convert to numpy array
    :param x: microtrac features
    :param y: flowability data
    :return:
    """
    x = x.to_numpy()
    y = y.to_numpy()
    return x, y
