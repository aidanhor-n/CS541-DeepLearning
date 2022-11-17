"""ParticleBalance

This class is responsible for balancing the number of particles in each dataset in order to
"""

import pandas
import random


def balance_particles(data: pandas.DataFrame, particle_limit=None, seed=None):
    """

    :param data: pandas dataframe of microtrac data
    :param particle_limit: optional manual particle limit
    :param seed: optional seed to get consistent results

    balance the particles per powder to the minimum # of particles of the smallest sample or the given #.

    """

    if particle_limit is None:
        particle_limit = min(data.groupby("sample_id").size())
    if seed is None:
        seed = random.randint(1, 1000)
    balanced_data = data.groupby("sample_id").sample(n=particle_limit, random_state=seed)
    return balanced_data
