from training_utils import leave_one_out_train_val, holdout_train_test, basic_train, is_classification
from testing_utils import get_predictions_and_score, get_performance_stats, visualize_regression_performance, visualize_classification_performance, visualize_tree
import numpy as np
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score, recall_score, precision_score
from itertools import combinations 
import pdb
import os
import pandas as pd
class TrainTestPipeline: 

    def __init__(self, x_data, y_data, all_samples, model_name, heldout_samples = 'random'):
        # pass a preprocessed data csv output by the data preprocessor 
        # all transformations, pca, feature selection should have been done at this point 

        self.x_data = x_data
        self.y_data = y_data
        self.all_samples = all_samples
        self.model_name = model_name 
        self.heldout_samples = heldout_samples
        
        self.data_split = None
        self.model = None

   

    def split_data(self):
        train_test_dict = holdout_train_test(self.x_data, self.y_data, self.all_samples, heldout = self.heldout_samples)
        return train_test_dict
    
    # trains a model using the train/test splits (uses only the train data)
    # option to do LOO cross validation (cv = True) or just basic validation set (cv=False)
    # training on whole test set (to avoid comp cost of cross-val)
    # default: LOO c/v 
    # returns dict of training results, and with model trained on full test data 
    # {'scores': scores, 'y_test': true_vals, 'predicted': np.asarray(predicts), 
         #   'train_samples':all_tr_samples, 'test_samples': all_test_samples, 'model':trained}
    def train_model(self, train_test_dict, cv=True):
        tr_x, tr_y, tr_samples = train_test_dict['train']
        held_out_x, held_out_y, held_out_test_samples = train_test_dict['test']
        print("Train Set Proportion: %f" %(tr_x.shape[0]/len(self.all_samples)))
        print("Test Set Proportion: %f" %(held_out_x.shape[0]/len(self.all_samples)))
        print("Train Set Samples: {}".format(np.unique(tr_samples)))
        print("Test Set Samples: {}".format(np.unique(held_out_test_samples)))
        if cv:
            k_dict = leave_one_out_train_val(self.model_name, tr_x, tr_y, tr_samples)
        else:
            k_dict = basic_train(self.model_name, tr_x, tr_y, tr_samples)

        return k_dict 


    def test_model(self, train_test_dict, trained_model):
        held_out_x, held_out_y, held_out_test_samples = train_test_dict['test']
        preds = trained_model.predict(held_out_x)
        classif = is_classification(self.model_name)
        if classif:
            score_ = f1_score(held_out_y, preds)
            print("Test F1-Score %f"% score_)
            print("Test Accuracy %f"% accuracy_score(held_out_y, preds))
            print("Test Recall %f"% recall_score(held_out_y, preds))
            print("Test Precision %f"% precision_score(held_out_y, preds))
        else:
            score_ = np.sqrt(mean_squared_error(held_out_y, preds))
            print("Test RMSE %f"% (score_))
        return preds, score_



    def visualize_performance(self, model, train_test_split, output_file, class_labels=None):
        held_out_x, held_out_y, held_out_test_samples = train_test_split['test']
        classif = is_classification(self.model_name)
        predictions = model.predict(held_out_x)
        if classif:
            if class_labels is None:
                class_labels = np.sort(np.unique(held_out_y))
            visualize_classification_performance(held_out_y, predictions, held_out_test_samples, output_file, class_labels)
        else: 
            visualize_regression_performance(held_out_y, predictions, held_out_test_samples, output_file)
        

    def do_train_test(self, cv=True):
        data_split = self.split_data()
        kdict = self.train_model(data_split, cv)
        preds, score_ = self.test_model(data_split, kdict['model'])
        return {'model': kdict['model'], 'train_test_split': data_split, 'train_performance': kdict, 'test_predictions': preds, 'test_score': score_}




    def exhaustive_train(self, outfile_name):
        i = 0
        unique_targs = self.all_samples.unique()
        combs = [c for c in combinations(unique_targs, 3)]
        is_classif = is_classification(self.model_name)
        if is_classif:
            score_metric = 'F1-Score'
        else:
            score_metric = 'RMSE'
        total_combos = len(combs)
        stats_cols  = ['TrainProp', 'TestProp', 'TestSample1', 'TestSample2', 'TestSample3', 'CrossVal {}'.format(score_metric), 'Test {}'.format(score_metric)]
        all_tr_stats_data = []
        for combo in combs:
            combo = list(combo)
            i+=1
            print("Combo {}, {} of {}".format(combo, i, total_combos))
            tr_round_stats = []
            self.heldout_samples = combo
            train_test_dict = self.split_data()
            tr_test_results = self.do_train_test()

            tr_x, tr_y, tr_samples = tr_test_results['train_test_split']['train']
            held_out_x, held_out_y, held_out_test_samples = tr_test_results['train_test_split']['test']

            tr_round_stats.append(tr_x.shape[0]/len(self.all_samples))
            tr_round_stats.append(held_out_x.shape[0]/len(self.all_samples))
            for h in self.heldout_samples:
                tr_round_stats.append(h)
            
            tr_round_stats.append(np.mean(tr_test_results['train_performance']['scores']))
            tr_round_stats.append(tr_test_results['test_score'])
            all_tr_stats_data.append(tr_round_stats)
        tr_stats_df = pd.DataFrame(all_tr_stats_data, columns = stats_cols)
        tr_stats_df = tr_stats_df.sort_values('Test {}'.format(score_metric))
        tr_stats_df.to_csv(outfile_name, index=False)


    # save tree visualization as a png
    def trained_tree_to_png(self, model, feature_names, all_y_data, outfile_name):
        if 'Tree' in self.model_name:
            name, ext = os.path.splitext(outfile_name)
            if ext == '.png':
                visualize_tree(model, feature_names, all_y_data, outfile_name = outfile_name)
            else:
                print("Specify a .png file type for the file name!")
            ## save the feature importances to a file 
            feature_df = pd.DataFrame()
            feature_df['feature_name'] = feature_names
            feature_df['score'] = [float(g) for g in model.feature_importances_]
            feature_df = feature_df.sort_values('score', ascending =False)
            print("Feature Importances")
            print(feature_df)

            feature_df.to_csv('.'.join([name + "_feature_importances", 'csv']), index=False)
        else:
            print("Error! This isn't a tree model.")