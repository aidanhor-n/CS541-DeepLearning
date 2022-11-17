# data preprocessing toolbox 
from io import StringIO
import os
import csv
import pickle as pkl

from sklearn.preprocessing import PowerTransformer
import numpy as np
import matplotlib.pyplot as plt
from minepy import MINE

import pandas as pd 
import re 
'''
NOTE: the expected Excel data filename has the format 
<MaterialName> (<SizeRange>) Particles.xls
for example: 
SS-17-4 (10-45) Particles.xls

make sure that the MaterialName and SizeRange correspond in the raw data Excel filename AND 
                                                        in the true flow value Excel file 
'''
'''
inputs raw data file (.xls)
deletes 2nd row of file which has micron symbol, which throws the csv decoder
saves a clean (un corrupted) file 
'''

## NEW -- DENSITIES 

densities = {'Ti-5Al-5V-5Mo-3Cr': 4.657, 
'Ti-Nb-Zr': 6.115, 'Ti-Nb': 5.936, 'SS-17-4':7.615, 'Inconel 718': 8.085}


def preproc_csv(filename):
    # have to remove the 2nd line of the file because it has symbols that pandas can't parse
    fname = filename.split('.xls')[0]
    clean_fname = "_".join([fname, 'clean'])
    clean_fname += '.xls'
    print("Hello")
    print(fname)
    with open(filename, 'rb') as raw_file, open(clean_fname, 'wt') as clean_file:
        writer = csv.writer(clean_file)
        idx = 0
        for row in raw_file:
            if idx!=1:
                # MUST DELETE ROW with 'micron' symbols, throws parse errors
                row = str(row, 'utf-8')
                data = StringIO(row)
                clean_file.write(data.read())
            idx += 1
'''
reads input (preprocessed) csv 
if not cleaned, then returns None (otherwise pandas will throw a decode error!)
returns dataframe with dropped columns (Filter columns and ID columns)
'''
def read_materials_csv(csv_filename):
    if 'clean' in csv_filename:
        data_ = pd.read_csv(csv_filename, sep='\t')
        drop_cols = [col for col in data_.columns if 'Filter' in col]
        drop_cols.append('hash')
        data_ = data_.drop(columns = drop_cols)
        data_ = data_.drop(columns=['Id', 'Img Id'])
        return data_
    else:
        return None


'''
inputs the samples data dictionary (already read)
inputs a dictionary of flow values extracted from Excel using get_flow_values 
inputs a method for labeling each particle 
    'mean' will average possible flow values to produce one target value per particle in a sample
    'random' will randomly take one of possible flow values for each particle in a sample
returns:  True if df saved successfuly 
            False if there was no data! 
'''
def get_data_and_labels(data_foldername):
    # choose whether random or take mean of flows 
    # used only if we are using a held out set

    total_len = 0
    x_data = None
    y_data_flow = None
    y_data_class = None
    
    # used only if we are not using a held out set
    all_samples = []
    data_all = None
    y_all = None
    x_data_cols = None
    
    for fname in os.listdir(data_foldername):
        sample_name, ext = os.path.splitext(fname)
        if ext == '.csv':
            # read raw  excel file 
            df = pd.read_csv(os.path.join(data_foldername, fname))
            if x_data_cols is None:
                x_data_cols = [col for col in df.columns if col not in ['Flow', 'FlowClass']]
            

            
            # split up the y and x data 
            df_as_array = df[x_data_cols].to_numpy()
            y1 = np.asarray(df['Flow'].tolist())
            y2 = np.asarray(df['FlowClass'].tolist())
            
            
        
            total_len+=len(df)
            if data_all is not None: 
                # add new sample's Microtrac measurements to those already tracked
                tmp = np.concatenate((data_all, df_as_array))
                data_all = tmp
            else:
                # start tracking the Microtrac measurements of the first sample
                data_all = df_as_array
            if y_data_flow is not None: 
                # add the sample's flow data to the flow data already seen 
                tmp_y = np.concatenate((y_data_flow, y1))
                y_data_flow = tmp_y
            if y_data_class is not None:
                # add the sample's class data to the class data already seen
                tmp_y = np.concatenate((y_data_class, y2))
                y_data_class = tmp_y
            else:
                # starting track the flowability + class values of the first sample
                y_data_flow = y1
                y_data_class = y2
            # track datapoint's sample and its local index in its own df 
            sample_ = [sample_name for i in range(len(df))]
            all_samples += sample_
    if len(data_all) > 0:
        inds = [i for i in range(len(data_all))]
        np.random.shuffle(inds)
        data_all_shuffled = data_all[inds]
        y_flow_shuffled = y_data_flow[inds]
        y_class_shuffled = y_data_class[inds]
        all_samples_shuffled = [all_samples[i] for i in inds]
        df = pd.DataFrame(data_all_shuffled, columns = x_data_cols)
        df['Flow'] = y_flow_shuffled
        df['FlowClass'] = y_class_shuffled
        df['SampleName'] = all_samples_shuffled
                
        df.to_csv('AllData.csv', index=False)
        return (True, df)
    return (False,None)

'''
inputs a Data Folder name (where all the excel files of raw particle data are)
returns a dictionary of clean_data_sample_file: dataframe 
if transform = True, applies Yeo-Johnson transformation to the data (with standardizing)
so it has a more normal distribution (less skew)
feature_columns takes a list of features to keep or None (if keeping all features)

writes each clean + scaled filename to a csv file; flow value, flowclass is also included in this file 
'''
def preprocess_data(data_folder, clean_data_folder, flow_val_file, transform = True, flows = 'mean'):
    samples_data = {}
    all_files = os.listdir(data_folder)
    all_data = 0
    true_flow_vals = get_true_flow_values(flow_val_file)
    for data_file in all_files:
        # ignore stupid .DS_Store!
        fname, ext = os.path.splitext(data_file)
        if ext == '.xls':
            if 'clean' not in data_file:
                sample_name = fname
                clean_fname = "_".join([sample_name, 'clean'])
                clean_fname += '.xls'
                if clean_fname not in all_files:
                    preproc_csv(os.path.join(data_folder, data_file))
            else:
                clean_fname = data_file
            sample_tag = clean_fname.split(' Particles')[0]
            pandas_df = read_materials_csv(os.path.join(data_folder,clean_fname))
            if transform:
                # make data have Gaussian distribution, with 0 mean, unit variance 
                pt = PowerTransformer(standardize=True)
                transform_data = pd.DataFrame()
            feature_columns = pandas_df.columns
            for col in feature_columns:
                data = pandas_df[col].tolist()
                data_arr = np.asarray(data).reshape(-1,1)
                if transform:
                    fit_pt = pt.fit(data_arr)

                
                    norm_ = fit_pt.transform(data_arr).reshape(-1,)
                    # min_ = np.asarray([np.min(norm_)]*(norm_.shape[0]))
                    # max_ = np.asarray([np.max(norm_)]*(norm_.shape[0]))
                    # norm_ = (norm_-min_)/(max_-min_)
                    transform_data[col] = norm_

                    all_data += len(transform_data)
                    #samples_data[sample_tag] = transform_data
                    tmp = transform_data

                else:
                    #samples_data[sample_tag] = pandas_df[feature_columns]
                    tmp = pandas_df[feature_columns]
                    all_data += len(pandas_df)
            
            flow_vals, density, flow_class = true_flow_vals[sample_tag]
            size = re.search("\([1-9]?[1-9]?[0-9]\-[1-9]?[1-9]?[0-9]\)", sample_tag)
            min_sz, max_sz = size[0].split('-')
            min_sz = min_sz.strip('(')
            max_sz = max_sz.strip(')')
            

            
            
            if flows == 'mean':
                y = [np.mean(flow_vals)]*len(tmp)
            else:
                y = [secrets.choice(flow_vals) for i in range(len(tmp))]
            num_rows = len(tmp)
            tmp['Flow'] = y
            tmp['FlowClass'] = [flow_class]*num_rows
            tmp['MinSize'] = [min_sz]*num_rows
            tmp['MaxSize'] = [max_sz]*num_rows
            tmp['Density'] = [density]*num_rows
            samples_data[sample_tag] = tmp
    if not os.path.isdir(clean_data_folder):
        os.mkdir(clean_data_folder)
    for sample, df in samples_data.items():
        sample_filename = "{}.csv".format(sample)
        if sample_filename not in os.listdir(clean_data_folder):
            df.to_csv(os.path.join(clean_data_folder, sample_filename), index=False)
    return samples_data

def transform_data(data_frame):
    # get unique values of samples
    # for_each through all of the unique sample names: |sample|
        # sample_data = [all of the data where sample name is equal to sample
        # for loop through the columns of the dataframe
            # Perform transform on each column
            # Add this to a new dataframe (we need to update the values in the file with this new data
    
    
    
            # make data have Gaussian distribution, with 0 mean, unit variance 
            pt = PowerTransformer(standardize=True)
            transform_data = pd.DataFrame()
            feature_columns = pandas_df.columns
            for col in feature_columns:
                data = pandas_df[col].tolist()
                data_arr = np.asarray(data).reshape(-1,1)
                fit_pt = pt.fit(data_arr)

                
                norm_ = fit_pt.transform(data_arr).reshape(-1,)
                # min_ = np.asarray([np.min(norm_)]*(norm_.shape[0]))
                # max_ = np.asarray([np.max(norm_)]*(norm_.shape[0]))
                # norm_ = (norm_-min_)/(max_-min_)
                transform_data[col] = norm_

                #samples_data[sample_tag] = transform_data
                tmp = transform_data
            
            flow_vals, density, flow_class = true_flow_vals[sample_tag]
            
            if flows == 'mean':
                y = [np.mean(flow_vals)]*len(tmp)
            else:
                y = [secrets.choice(flow_vals) for i in range(len(tmp))]
            num_rows = len(tmp)
            tmp['Flow'] = y
            tmp['FlowClass'] = [flow_class]*num_rows
            tmp['MinSize'] = [min_sz]*num_rows
            tmp['MaxSize'] = [max_sz]*num_rows
            tmp['Density'] = [density]*num_rows
            samples_data[sample_tag] = tmp
    if not os.path.isdir(clean_data_folder):
        os.mkdir(clean_data_folder)
    for sample, df in samples_data.items():
        sample_filename = "{}.csv".format(sample)
        if sample_filename not in os.listdir(clean_data_folder):
            df.to_csv(os.path.join(clean_data_folder, sample_filename), index=False)
    return samples_data
    
'''
read excel file with ground truth flow values 
note: had to perform minor edits to the given flow value file on OneDrive -- differences between namings
returns (flow value, density) dictionary key'd by sample name.  "<Material> (<size>): (list of possible flow values, density)
'''
def get_true_flow_values(flow_file):
    flow_excel = pd.read_excel(flow_file)
    materials = flow_excel['Material']
    sizeranges = flow_excel['SizeRange']
    all_densities = flow_excel['AugDensity']
    densities_unique = flow_excel['AugDensity'].unique()
    d_mean = np.mean(densities_unique)
    std_dev = np.std(densities_unique)


    flow_ = flow_excel['Flow']
    flow_class = flow_excel['FlowClass']

    true_flow_vals = {}
    for m, sz, val, fl_cl, raw_dens in zip(materials, sizeranges, flow_, flow_class, all_densities):
        sample_name = ' '.join([m, "({})".format(sz)])
        if sample_name in true_flow_vals:
            # if we already have a flow measurement, add another one to the list 
            tmp, dens_, fl_ = true_flow_vals[sample_name]
            tmp.append(val)
            true_flow_vals[sample_name] = (tmp, dens_, fl_)
        else:
            # we don't have a flow measurement for this sample 
            # add it to our dictionary -- initialize list of flow values 
            # add the density, z-score transformed
            # add the flowclass **NEW** -- extension for classification 

            density = (raw_dens-d_mean)/std_dev
            true_flow_vals[sample_name] = ([val], density, fl_cl)
    return true_flow_vals

def compute_correlation(df, method, threshold, heldout_cols = None):
        x_cols = [col for col in df.columns if col not in ['SampleName', 'Flow', 'FlowClass']]
        if heldout_cols:
            for col in heldout_cols:
                x_cols.remove(col)
        x_data = df[x_cols]
        files = os.listdir('.')
        if method == 'pearson':
            # linear correlation
            corr_matrix = x_data.corr('pearson')
        else: 
            if 'nonlinear_correlation_matrix.pkl'  not in files:
                # nonlinear correlation
                # subset = np.arange(2000)
                # np.random.shuffle(subset)
                x_data = x_data.sample(n=2000, replace = False)
                mine = MINE(alpha=0.6, c=15, est="mic_approx")
                corr_matrix = [[0]*len(x_cols) for i in range(len(x_cols))]
                for i in range(len(x_cols)):
                    i_col = x_cols[i]
                    for j in range(len(x_cols)):
                        if i == j:
                            corr_matrix[i][j] = 1
                        else: 
                            j_col = x_cols[j]
                            i_data = x_data[i_col]
                            j_data = x_data[j_col]
                            mine.compute_score(i_data, j_data)
                            corr_matrix[i][j] = mine.mic()
                
                corr_matrix = np.asarray(corr_matrix)
                with open('nonlinear_correlation_matrix.pkl', 'wb') as f:
                    pkl.dump(corr_matrix, f)
            else:
                with open('nonlinear_correlation_matrix.pkl', 'rb') as f:
                    corr_matrix = pkl.load(f)
        return corr_matrix
def update_flow_classes(df, flow_file):
    og_flow_classes = df['FlowClass'].tolist()
    og_flow_vals = df['Flow'].tolist()
    true_flow_vals = get_true_flow_values(flow_file)
    modified = False
    for sample in true_flow_vals.keys():
        flows, new_density, new_flow_class = true_flow_vals[sample]
        new_flow = np.mean(flows)
        if new_flow!= df.loc[df['SampleName']==sample, 'Flow'].iloc[0]:
            df.loc[df['SampleName']==sample, 'Flow'] = new_flow
            modified = True
        if new_density != df.loc[df['SampleName']==sample, 'Density'].iloc[0]:
            df.loc[df['SampleName']==sample, 'Density'] = new_density
            modified = True
        if new_flow_class != df.loc[df['SampleName']==sample, 'FlowClass'].iloc[0]:
            df.loc[df['SampleName']==sample, 'FlowClass'] = new_flow_class
            
            modified = True
    return df, modified


