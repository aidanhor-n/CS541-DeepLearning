def separate_holdout_set(data, holdout_id=['c1adc55e-0821-4408-b379-2d51b8e5c9ef', '2fe1c429-aae7-406a-9178-4cae6716cc96', 'f28c1c6a-43d5-41cd-bd8b-10df1e639381', '7a1edf53-720b-4c9c-8516-d8ebdd038571']):
    holdout_set = data[data.sample_id.isin(holdout_id)]
    train_set = data[~data.sample_id.isin(holdout_id)]
    return holdout_set, train_set

def separate_train_test_powders_index_reverse(data, train_length=18):
    data_length = len(data["sample_id"].unique())
    test_ids = data["sample_id"].unique()[0:(data_length - train_length)]
    test_set = data[data.sample_id.isin(test_ids)]
    train_set = data[~data.sample_id.isin(test_ids)]
    return test_set, train_set

def generate_particle_level_train_and_test_sets(data, mass_composition=0.2):
    '''
    The goal of this method is the separate the full dataset into a train
    and test set. The test set will contain 20% of the particles from each
    powder, and the train set will remove these chosen particles. These 
    train and test sets will then be virtually mixed.
    '''
    sample_ids = data['sample_id'].unique()
    train_set = []
    test_set = []
    for sample_id in sample_ids:
        powder_particles = data[data['sample_id'] == sample_id]
        count_sample_particles = int(len(powder_particles) * mass_composition)
        test_powder = powder_particles.sample(count_sample_particles)
        train_powder = powder_particles[~powder_particles.isin(test_powder)]
        if train_set is None:
            train_set = train_powder
            test_set = test_powder
        else:
            train_set = train_set.append(train_powder)
            test_set = test_set.append(test_powder)
    
    return train_set, test_set