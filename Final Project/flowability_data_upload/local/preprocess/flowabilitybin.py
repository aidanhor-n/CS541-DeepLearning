"""Flowability Bin

Bins the flowability target variable data into binary data
"""

import pandas


def binary_bin(data: pandas.DataFrame, split_value=30) -> pandas.DataFrame:
    """
    label the flowability values given a split value, flowability is the time taken to flow.
    lower flowability is a fast flowability. Ex: (fast-flow = 0, slow-flow = 1, no-flow = 1)
    :param data: a pandas dataframe of Microtrac data
    :param split_value: value to split between high and low flowability.
    :return: pandas dataframe of Microtrac data
    """
    data = data['flowability'].apply(binary_split_helper, split_value=split_value)
    return data


def binary_split_helper(flowability, split_value):
    # No flowability
    if flowability == 0:
        return 1

    # slow flowability
    if flowability >= split_value:
        return 1

    # fast flowability
    if flowability < split_value:
        return 0


def multiclass_bin(data: pandas.DataFrame, split_values) -> pandas.DataFrame:
    """
    label the flowability values given a split value, flowability is the time taken to flow.
    lower flowability is a fast flowability. Ex: (fast-flow = 0, slow-flow = 1, no-flow = 1)
    :param data: a pandas dataframe of Microtrac data
    :param split_values: value to split between high and low flowability.
    :return: pandas dataframe of Microtrac data
    """
    data = data['flowability'].apply(multiclass_split_helper, split_values=split_values)
    return data


def multiclass_split_helper(flowability, split_values):
    # No flowability
    if flowability == 0:
        return len(split_values) - 1

    for i, value in enumerate(split_values, start=1):
        if flowability < value:
            return i - 1
        else:
            return len(split_values)
