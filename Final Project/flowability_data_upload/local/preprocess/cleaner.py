"""Cleaner
This class cleans the data a step further than what happens on the cloud. 

Removes several columns with almost all null values, (Filter1, Filter2..., ect) and drops NA values.
"""

import pandas


def clean(data: pandas.DataFrame) -> pandas.DataFrame:
    """

    :param data: a pandas dataframe of Microtrac data

    :return: pandas dataframe of Microtrac data

    given data as pandas dataframe drop problematic columns and drop NA values in place

    """
    # Remove Filter columns and microtrac img id columns
    drop_cols = [col for col in data.columns if 'Filter' in col]
    drop_cols.append('hash')
    drop_cols.append('Id')
    drop_cols.append('Img Id')
    data = data.drop(columns=drop_cols)
    data = data.dropna()
    return data


def remove_identifying_columns(data: pandas.DataFrame) -> pandas.DataFrame:
    """

    :param data: # TODO
    :return: # TODO

    Remove remaining columns not used for model training. some columns are needed for preprocessing.
    Ex: group by (sample_id) to balance to particles

    """

    data = data.drop(columns=['sample_id', 'name'])
    return data


def remove_duplicate_powders(data: pandas.DataFrame):
    """

    This method was built after we found duplicated powder files in our raw dataset. These files
    have the same name, and their sample ids are different. We check for duplicity initially by the same
    name. Then, we check the lengths of the dataframe and go one by one through the rows to find any that
    are different. If there are no such rows, then we assume duplicity.
    *** This method may need reworking if our definition of duplicity changes. ***
    Returns data with the removed duplicate columns

    """

    duplicate_names = check_for_duplicate_names(data)

    for name in duplicate_names:
        data = remove_duplicates(data, name)

    return data


def remove_duplicates(data, name):
    named_data = data[data.name == name]
    ids = named_data["sample_id"].unique()
    df_unique = named_data[named_data.sample_id == ids[0]]
    for i in range(1, len(ids)):
        df_dup = named_data[named_data.sample_id == ids[i]]

        if (check_for_same_length):
            if (check_for_duplicate_rows):
                data = data[data.sample_id != ids[i]]
    return data.reset_index(drop=True)


def check_for_duplicate_names(data):
    duplicate_names = []
    sample_names = data["name"].unique()
    for name in sample_names:
        ids = data[data.name == name]["sample_id"].unique()
        if (len(ids) > 1):
            duplicate_names.append(name)
    return duplicate_names


def check_for_same_length(df1, df2):
    return len(df1) == len(df2)


def check_for_duplicate_rows(df1, df2):
    df1 = df1.drop(columns=["sample_id", "name"])
    df2 = df2.drop(columns=["sample_id", "name"])
    for i in range(0, len(df1)):
        for col in df1.columns:
            if (df1[col].iloc[i] != df2[col].iloc[i]):
                return False
    return True
