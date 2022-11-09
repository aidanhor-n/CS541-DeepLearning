from sklearn.tree import DecisionTreeRegressor
import pandas as pd
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split, StratifiedShuffleSplit, StratifiedKFold, KFold
from itertools import cycle
import numpy as np
import secrets
import seaborn as sns
import graphviz
from sklearn import tree 
import pdb
from sklearn.metrics import mean_squared_error, f1_score, recall_score, accuracy_score, precision_score, confusion_matrix
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LinearRegression, BayesianRidge, SGDRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA, KernelPCA
from sklearn.svm import SVR
from collections import Counter
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
# testing a model and visualizing performance 
# make sure to leave a held out test set using StratifiedShuffleSplit!!! 




'''
inputs a classifier object, x test values, y test values 
prints performance summary to stdout
returns dictionary of scores
'''
def get_predictions_and_score(classifier, test_x, test_y, classify = False):
    predictions = classifier.predict(test_x)
    if classify:
        print("Classification")
        r2_score = classifier.score(test_x, test_y)
        diffs = abs(test_y-predictions)
        rmse = np.sqrt(mean_squared_error(test_y, predictions))
        print("Regression Performance:")
        print("R^2 score: %f" % (r2_score))
        print("Average Absolute Difference: %f" % (np.mean(diffs)))
        print("Maximum Difference: %f" % (np.max(diffs)))
        print("Minimum Difference: %f" % (np.min(diffs)))
        print("Standard Dev. of Difference: %f" % (np.std(diffs)))
        print("Mean-Scaled RMSE: %f" % (rmse/np.mean(test_y)))
        return predictions, {'r2': r2_score, 'differences': {'mean': np.mean(diffs), 'max': np.max(diffs), 'min': np.min(diffs), 
                                            'standard_deviation': np.std(diffs)}, 
            'rmse': rmse}
    else: 
        f1_ = f1_score(test_y, predictions)
        recall_ = recall_score(test_y, predictions)
        precision_ = precision_score(test_y, predictions)
        accuracy_ = accuracy_score(test_y, predictions)
        print("Classification Performance")
        print("F1 Score: {}".format(f1_))
        print("Recall: {}".format(recall_))
        print("Precision: {}".format(precision_))
        print("Accuracy: {}".format(accuracy_))
        return predictions, {'f1': f1_, 'recall': recall_, 'precision': precision_, 'accuracy': accuracy_}


'''
returns a dictionary of the performance summary 
for each sample, shows the predicted & actual values 
used for visualizing the performance in matplotlib
'''
def get_performance_stats(test_y, predictions, test_sample_array, classify=False):
    performance_summary = {}
    for idx in range(len(test_y)):
        #idx_original = test_indices[idx]
        pred = predictions[idx]
        real = test_y[idx]
        sample_ = test_sample_array[idx]
        if sample_ in performance_summary:
            tmp_ = performance_summary[sample_]
            
            if classify:
                print("Classification")

                tmp_[idx] = {'f1':f1_score(real, pred), 'recall': recall_score(real, pred), 
                'precision': precision_score(real, pred), 'accuracy': accuracy_score(real, pred),
                                                            'predicted':pred, 'actual':real}
            else:
                
                
                abs_diff = abs(pred-real)
                tmp_[idx] = {'abs_difference':abs_diff, 'sample': sample_,
                                                            'predicted':pred, 'actual':real}
            performance_summary[sample_] = tmp_
        else:
            if classify:
                performance_summary[sample_]={local_idx: {'predicted':pred, 'actual':real, }}
            else:
                performance_summary[sample_]={local_idx: {'abs_difference':abs_diff, 
                                                        'predicted':pred, 'actual':real}}
    return performance_summary


'''
visualizes performance given predictions and true flow values 
saves result to a pdf file 
'''
def visualize_regression_performance(test_y, predictions, test_sample_array, output_file):
    figs = []
    pdf = matplotlib.backends.backend_pdf.PdfPages(output_file)
    unique_test_samples = np.unique(test_sample_array)
    fig, ax = plt.subplots(figsize=(11,9))
    cycol = cycle('grcmbk')
    mark = cycle('o*<>xD')
    annotated_y = []
    for i in range(len(unique_test_samples)):
        sample = unique_test_samples[i]
        idxs_ = np.where(test_sample_array == sample)[0]
        preds_ = predictions[idxs_]
        reals_ = test_y[idxs_]
        #xs_ = [i+1 for i in range(len(data_dict))]
        color = next(cycol)
        plt.plot([i for i in range(len(test_y))], [reals_[0]]*len(test_y), '--', c = color, label='{} Actual'.format(sample))
        if len(annotated_y) > 0:
            for y in annotated_y:
                if abs((reals_[0]-y))<1:
                    flow_true_label_x = len(test_sample_array)/4
                else:
                    flow_true_label_x = 0
        else:
            flow_true_label_x = 0
        bbox_props = dict(boxstyle="round", fc=color,ec=None, alpha=0.65)
        
        
        plt.scatter(idxs_, preds_, marker = next(mark), c=color,alpha = 0.7, label='{} Predicted'.format(sample))
        ax.text(flow_true_label_x, reals_[0]+0.05, "True Flow: %.2f" % (reals_[0]), ha="center", va="center",bbox = bbox_props)
        annotated_y.append(reals_[0])
        #plt.show()
        # plt.axis([xs_[0], xs_[-1], min(0,min(reals_)-5, min(preds_)-5), max(max(reals_)+5, max(preds_) +5)])
        # #plt.plot([i for i in range(len(preds_))], [np.mean(np.asarray(reals_))]*len(preds_), color = 'b', alpha = 0.04)
        # plt.hlines(np.mean(np.asarray(reals_)), xmin=xs_[0], xmax =xs_[-1], linestyles = 'dashdot', label = 'Actual Flow Value')
        # plt.scatter(xs_, preds_, color = 'c', marker = 'o', alpha = 0.2)
        # plt.title("{} Predicted (Cyan) vs Actual (Dash/Dot Line)".format(sample))
    plt.axis([0, len(test_y), min(0,min(test_y)-5, min(predictions-5)), max(max(test_y)+5, max(predictions) +5)])
    plt.title("Regression Performance for Test Samples")    
    ax.legend(loc='upper right')
    #plt.show()
    #pdb.set_trace()
    pdf.savefig(fig)
    pdf.close()

'''
visualizes performance given predictions and true flow values 
saves result to a pdf file 
make sure class-Labels are in ascending numerical order 
first label is taken as class 0 
'''
def visualize_classification_performance(test_y, predictions, test_sample_array, output_file, class_labels):
    pdf = matplotlib.backends.backend_pdf.PdfPages(output_file)
    unique_test_samples = np.unique(test_sample_array)
    fig, axes = plt.subplots(3,1, figsize = (11,11))
    for idx in range(len(unique_test_samples)):
        sample = unique_test_samples[idx]
        idxs_ = np.where(test_sample_array == sample)[0]
        preds_ = predictions[idxs_]
        reals_ = test_y[idxs_]

        labels = class_labels
        cm = confusion_matrix(reals_, preds_, labels)
        ax = axes[idx]
        cax = ax.matshow(cm)
        ax.set_title('Confusion Matrix for {}'.format(sample))
        fig.colorbar(cax, ax = ax)
        ax.set_xticklabels([''] + labels)
        ax.set_yticklabels([''] + labels)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')        
    pdf.savefig(fig)
    pdf.close()


# outfile name is a png!!!
def visualize_tree(model, feature_names, all_y_data, outfile_name):
    
    dot_data = tree.export_graphviz(model, out_file=None, 
                                    feature_names=feature_names, class_names=list(set(all_y_data)), filled=True, rounded=True, 
                                    special_characters=True)  
    graph = graphviz.Source(dot_data) 
    png_bytes = graph.pipe(format='png')
    with open(outfile_name,'wb') as f:
        f.write(png_bytes)