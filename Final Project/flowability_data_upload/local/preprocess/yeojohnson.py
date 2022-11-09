"""Yeo Johnson

Performs the Yeo Johnson transformation in order to make the columns follow a more Gaussian distribution
to improve machine learning model performance
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer


def yeo_transform(data: pd.DataFrame) -> pd.DataFrame:
    """
    given the microtrac dataframe, transform using Yeo johnson
    :param data: microtrac dataframe
    :return:
    """
    # make data have Gaussian distribution, with 0 mean, unit variance
    pt = PowerTransformer(standardize=True)
    transform_data = pd.DataFrame()
    feature_columns = data.columns
    for col in feature_columns:
        col_data = data[col].tolist()
        data_arr = np.asarray(col_data).reshape(-1, 1)
        fit_pt = pt.fit(data_arr)
        norm_ = fit_pt.transform(data_arr).reshape(-1, )
        transform_data[col] = norm_

    return transform_data
