from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split, StratifiedShuffleSplit, StratifiedKFold, KFold

import numpy as np
import secrets
import pickle as pkl
import graphviz
#import xgboost as xgb
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score, recall_score, precision_score
from sklearn.linear_model import LinearRegression, LogisticRegression, BayesianRidge, SGDRegressor, SGDClassifier
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.decomposition import PCA, KernelPCA
from sklearn.svm import SVR
import seaborn as sns
from collections import Counter
import os


def get_good_split(x_data, samples_arr, num_test_samples):
    unique_samples = all_samples.unique()
    heldout = np.random.choice(all_samples.unique(),num_heldout, replace=False)

    total_test = 0
    for targ in heldout:
        inds_ = np.where(samples_arr == targ)
        x_ = x_data[inds_]
        total_test += x_.shape[0]
    prop = total_test/len(x_data)
    return (True, heldout)
    # if prop < 0.4 and prop>=0.2:
    #     print(total_test)
    #     return (True, heldout)
    # else:
    #     return (False, None)
# leaves 3 samples untouched for testing, does an 80/20 split with the rest 
# all samples: serie
# train on 5 samples, test on all 8 (testing on training samples and completely new ones)
# heldout is one of list of samples, 'random', 'balanced_random'
def holdout_train_test(x_data, y_data, all_samples, heldout = 'random', num_heldout = 3):
    unique_samples = all_samples.unique()
    samples_arr = all_samples.to_numpy()
    if heldout == 'balanced_random':
        balanced, heldout = get_good_split(x_data, samples_arr, num_heldout)
        while not balanced:
            balanced, heldout = get_good_split(x_data, samples_arr, num_heldout)
    elif heldout == 'random':
        heldout = np.random.choice(all_samples.unique(),num_heldout, replace=False).tolist()

    # otherwise, a heldout set was specified. 

    
    train_x = None
    train_y = []
    test_x = None
    test_y = []
    train_samples = []
    test_samples = []
    
    for sample in unique_samples:
        inds_ = np.where(samples_arr == sample)
        x_ = x_data[inds_[0]]
        y_ = y_data[inds_[0]]
        inds_list = inds_[0].tolist()
        if sample in heldout:
            # if this is a heldout sample, leave it out for the test set
            if test_x is not None: 
                test_x = np.concatenate([test_x, x_])
                test_y = np.concatenate([test_y, y_])
                test_samples = np.concatenate([test_samples, samples_arr[inds_[0]]])
            else:
                test_x = x_
                test_y = y_
                test_samples = samples_arr[inds_[0]]
            
        else: 
            # if this is a regular sample, keep it for cross/val training 
            if train_x is not None: 
                train_x = np.concatenate([train_x, x_])
                train_y = np.concatenate([train_y, y_])
                train_samples = np.concatenate([train_samples, samples_arr[inds_[0]]])
            else:
                train_x = x_
                train_y = y_
                train_samples = samples_arr[inds_[0]]
            
    
    train_inds = [i for i in range(len(train_x))]
    test_inds = [i for i in range(len(test_x))]
    np.random.shuffle(train_inds)
    np.random.shuffle(test_inds)
    train_x_shuffled = train_x[train_inds]
    train_y_shuffled = train_y[train_inds]
    train_samples_shuffled = train_samples[train_inds]

    test_x_shuffled = test_x[test_inds]
    test_y_shuffled = test_y[test_inds]
    test_samples_shuffled = test_samples[test_inds]
    
    
    return {'train': [train_x_shuffled, train_y_shuffled, train_samples_shuffled], 
    'test': [test_x_shuffled, test_y_shuffled, test_samples_shuffled]}



def is_classification(in_algo):
    # returns true if it's a classification algorithm
    # false if it's regression
    models = {'DecisionTreeRegressor': False,
             'SupportVectorRegressor': False, 
               'RandomForestRegressor': False, 'LogisticRegression': True,
              'kNeighborsRegressor': False,
              'DecisionTreeClassifier': True,
             'RandomForestClassifier': True, 
              'KNeighborsClassifier': True,
              'QDAAnalysis': True
              }         
def build_algorithm(algo_name):
    regressors = {'DecisionTreeRegressor': DecisionTreeRegressor(max_depth = 15),
             'SupportVectorRegressor': SVR(), 
               'RandomForestRegressor': RandomForestRegressor(), 'LogisticRegression': LogisticRegression(solver='sag'),
              'kNeighborsRegressor': KNeighborsRegressor(),
              'DecisionTreeClassifier': DecisionTreeClassifier(),
             'RandomForestClassifier': RandomForestClassifier(), 
              'KNeighborsClassifier': KNeighborsClassifier(),
               'QDAAnalysis': QuadraticDiscriminantAnalysis()
              }
    return regressors[algo_name]

# just trains a model on the whole training set without doing any kind of cross-validation
def basic_train(algo_name, x_data, y_data, all_samples):
    algo = build_algorithm(algo_name)
    is_classify = is_classification(algo_name)
    
    trained = algo.fit(x_data, y_data)    
    predicts = algo.predict(x_data) 
    
    return {'scores': scores, 'y_test': y_data, 'predicted': np.asarray(predicts), 
            'train_samples':all_samples, 'test_samples': all_samples, 'model':trained}

# do kfold train + validation 
# make sure to leave a held out test set on which to test
# all_samples is a 
# perform kfold on the train/val set
# test data should never be used with this function!!! 
def leave_one_out_train_val(algo_name, x_data, y_data, all_samples):
    ##skf = StratifiedKFold(n_splits)
    #kf = KFold(n_splits)
    #stratified_ss = StratifiedShuffleSplit(n_splits)
    '''
    STRATIFIED K FOLD DOES NOT WORK! only for binary/multiclass classification 
    '''
    algo = build_algorithm(algo_name)
    # are we doing classification or regression?
    classify = is_classification(algo_name)
    
    unique_samples = np.unique(all_samples)
    np.random.shuffle(unique_samples)
    scores = []
    
    true_vals = None
    predicts = []
    all_tr_samples = None
    all_test_samples = None
    # split the data 5 different ways (each fold is is a new sample)
    # kind of like "leave one out"
    for sample in unique_samples:
        tr_idxs = np.where(all_samples!=sample)[0]
        test_idxs = np.where(all_samples==sample)[0]
        X_tr, y_tr = x_data[tr_idxs], y_data[tr_idxs]
        X_test, y_test = x_data[test_idxs], y_data[test_idxs]
        train_samples = all_samples[tr_idxs]
        test_samples = all_samples[test_idxs]

        if true_vals is not None: 
            true_vals = np.concatenate([true_vals, y_test])
            all_tr_samples = np.concatenate([all_tr_samples, train_samples])
            all_test_samples = np.concatenate([all_test_samples, test_samples])
        else:
            true_vals = y_test
            all_tr_samples =  train_samples
            all_test_samples =  test_samples

        #train &test
        print("Training on {}".format(np.unique(train_samples)))
        print("Validating on {}".format(sample))
        # classifier = build_algorithm(algo_name)
        trained = algo.fit(X_tr, y_tr)
        preds = trained.predict(X_test)
        predicts.append(preds)
        if classify:
            f1 = f1_score(y_test, preds)
            print("F1 Validation: %.2f"%(f1))
            scores.append(f1)
        else:
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            print("RMSE Validation: %.2f" %(rmse))
            scores.append(rmse)
    trained = algo.fit(x_data, y_data)     
    print("Average Validation Score: %.2f" %np.mean(scores))

    return {'scores': scores, 'y_test': true_vals, 'predicted': np.asarray(predicts), 
            'train_samples':all_tr_samples, 'test_samples': all_test_samples, 'model':trained}