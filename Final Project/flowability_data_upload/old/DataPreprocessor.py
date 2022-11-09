from data_preprocessing import preprocess_data, get_data_and_labels, update_flow_classes, compute_correlation
import pandas as pd
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
import pickle as pkl
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.backends.backend_pdf
import os
import seaborn as sns
import pdb
class DataPreprocessor:

    def __init__(self, data_folder, flow_values_excel):
        self.data_folder = data_folder # name of folder with excel microtrac data
        self.flow_values = flow_values_excel # name of file with flow values + flow classes 
        files = os.listdir('.')
        if ('AllData.csv') not in files:
            self.data_csv = None
        else:
            # update the target values if needed
            # to update the target class, make sure the first entry of that sample has the target class label you want 
            tmp = pd.read_csv('AllData.csv')  
            print("Before Balancing Size: ", len(tmp))
            tmp = tmp.groupby("SampleName").apply(pd.DataFrame.sample, n=10000, random_state=1, replace=True)
            print("After Balancing Size: ", len(tmp))
            updated_, mod = update_flow_classes(tmp, flow_values_excel)
            self.data_csv = updated_
            self.x_data_cols = [col for col in self.data_csv.columns if col not in ['Flow', 'SampleName', 'FlowClass']]
            if mod:
                updated_.to_csv('AllData.csv', index=False)
            
    
    # loads or prepares the data csv with flow values 
    
    def prepare_df(self):
        files = os.listdir('.')

        if self.data_csv is None:
            # data_folder, clean_data_folder, flow_val_file, transform = True, flows = 'mean', feature_columns = None
            clean_data_folder = '_'.join([self.data_folder, 'transformed'])
            samples_data = preprocess_data(self.data_folder, clean_data_folder, self.flow_values, transform=False)
            success, df = get_data_and_labels(clean_data_folder)
            if success: 
                df = df.groupby("SampleName").apply(pd.DataFrame.sample, n=10000, random_state=1, replace=True)
                df = transform_data(df)
                self.data_csv = df
                self.x_data_cols = [col for col in self.data_csv.columns if col not in ['Flow', 'SampleName', 'FlowClass']]
                return True
            else:
                return False
        else:
            # make sure the flow class labels correspond 

            return True
            
     # returns the data csv upon request       
    def get_data_csv(self):
        return self.data_csv
    

    # returns a filtered x_data with uncorrelated features only 
    # returns numpy matrix
    def get_feature_selection_x(self,method='pearson', threshold = 0.8, heldout_cols=None):
        df = self.data_csv
        corr_matrix = compute_correlation(df, method, threshold, heldout_cols)
        x_cols = [col for col in df.columns if col not in ['SampleName', 'Flow', 'FlowClass']]
        if heldout_cols:
            for col in heldout_cols:
                x_cols.remove(col)
        cols = [col for col in x_cols]
        for i in range(corr_matrix.shape[0]):
            
            col = x_cols[i]
            
            if col in cols:
                for j in range(i+1,corr_matrix.shape[0]):
                    
                    if method == 'pearson':
                        corr_val = corr_matrix[col][j]
                    else:
                        corr_val = corr_matrix[i][j]
                    if corr_val >= threshold: 
                        col_to_drop = x_cols[j]
                        if col_to_drop in cols:
                            cols.remove(col_to_drop)
        x_data_filt = np.asarray(df[cols])
        print("Feature columns used: {}".format(cols))
        return x_data_filt, cols
    
    # returns the raw (transformed) data for all columns, no filtering
    def get_raw_x(self, heldout_cols = None):
        if self.data_csv is not None:
            if heldout_cols:
                x_cols = [col for col in self.x_data_cols if col not in heldout_cols]
            else:
                x_cols = self.x_data_cols
            return self.data_csv[x_cols].to_numpy(), cols
        else:
            return None, None

    # returns a Series of samples corresponding to each row in the dataset
    def get_samples(self):
        if self.data_csv is not None:
            return self.data_csv['SampleName']
        else:
            return None

    # returns numpy array of target values, for classification
    def get_classification_y(self):
        if self.data_csv is not None:
            return self.data_csv['FlowClass'].to_numpy()

    # returns numpy array of target values, for regression 
    def get_regression_y(self):
        if self.data_csv is not None:
            return self.data_csv['Flow'].to_numpy()


    # runs PCA and transforms the data 
    # optionally specify extra columns to drop out of the PCA
    # returns numpy matrix
    def get_pca_x(self,pca_type='linear', heldout_cols = None):
        #x_data_cols = [col for col in self.data_csv.columns if col not in ['Flow', 'SampleName', 'FlowClass']]
        if heldout_cols:
            x_data_cols = [col for col in self.x_data_cols if col not in heldout_cols]
        else:
            x_data_cols = self.x_data_cols
        x_data = self.data_csv[x_data_cols]
        if pca_type == 'linear':
            pca = PCA().fit(x_data)
        else:
            sample = x_data.sample(2000)
            pca = KernelPCA(kernel=pca_type).fit(sample)
        transformed = pca.transform(x_data)
        return transformed


    # run LDA to reduce dimensions of the data
    # allows the user to change solver and shrinkage values for the LDA
    def get_lda_x(self, solver='eigen', shrinkage='auto', heldout_cols = None):
        lda = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)
        if heldout_cols:
            x_data_cols = [col for col in self.x_data_cols if col not in heldout_cols]
        else:
            x_data_cols = self.x_data_cols
        x_data = self.data_csv[x_data_cols]
        x_data = x_data.replace(np.nan, 0)
        result = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage).fit(x_data, self.get_classification_y())
        transformed = result.transform(x_data)
        return transformed
    
    # run LDA to reduce dimensions of the data
    # allows the user to change solver and shrinkage values for the LDA
    def get_lda_x_fit(self, solver='eigen', shrinkage='auto', heldout_cols = None):
        lda = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)
        if heldout_cols:
            x_data_cols = [col for col in self.x_data_cols if col not in heldout_cols]
        else:
            x_data_cols = self.x_data_cols
        x_data = self.data_csv[x_data_cols]
        x_data = x_data.replace(np.nan, 0)
        result = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage).fit(x_data, self.get_classification_y())
        return result
    
    # run QDA to reduce dimensions of the data
    def get_qda_x(self, heldout_cols = None):
        if heldout_cols:
            x_data_cols = [col for col in self.x_data_cols if col not in heldout_cols]
        else:
            x_data_cols = self.x_data_cols
        x_data = self.data_csv[x_data_cols]
        x_data = x_data.replace(np.nan, 0)
        result = QuadraticDiscriminantAnalysis().fit(x_data, self.get_classification_y())
        return result

    def print_invalid_x(self):
        x_data = self.data_csv[self.x_data_cols]
        return x_data
        results = []
        for x in x_data:
            results.append(x)
        return results

    def get_data(self, heldout_cols = None):
        if heldout_cols:
            x_data_cols = [col for col in self.x_data_cols if col not in heldout_cols]
        else:
            x_data_cols = self.x_data_cols
        x_data = self.data_csv[x_data_cols]
        return x_data
    
    def make_histograms(self):
        x_data = self.data_csv[self.x_data_cols]
        for col in x_data.columns:
            plt.figure()
            plt.title(col)
            plt.hist(x_data.loc[:,col])
        return
    
    # CHANGE DOCUMENTATION LATER!!!!!
    # visualize the first two principal components, colored by flowability 
    # visualize the cumulative explained variance of the transformed data 
    # works for both linear and nonlinear pca 
    # out_file_name is some 'outfile.pdf' string
    def visualize_lda(self,transformed_data, targs, out_file_name):
        #fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11,11)) 
        plt.figure()
        colors = ['navy', 'turquoise', 'darkorange']
        target_names = ['<= 15', '30 > 15', '> 30']
        
        for color, i, target_name in zip(colors, [0, 1, 2], target_names):
            plt.scatter(transformed_data[targs == i, 0], transformed_data[targs == i, 1], alpha=.8, color=color,
                        label=target_name)
        plt.legend(loc='best', shadow=False, scatterpoints=1)
        plt.title('LDA Components')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.show()
        
        pdf = matplotlib.backends.backend_pdf.PdfPages(out_file_name)
        #pdf.savefig(fig)
        pdf.close()
    
    # visualize the first two principal components, colored by flowability 
    # visualize the cumulative explained variance of the transformed data 
    # works for both linear and nonlinear pca 
    # out_file_name is some 'outfile.pdf' string
    def visualize_pca(self,transformed_data, targs, out_file_name):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11,11)) 
        ax1.grid(color='lightgrey', lw=0.3)
        ax1.set_title('Principle Components 1 vs 2, Colored by Flowability')
        ax1.set_ylabel('Principle Component 2')
        ax1.set_xlabel('Principle Component 1')

        im = ax1.scatter(transformed_data[:,0], transformed_data[:,1], c=targs, cmap = 'Reds')
        
        # for x, y in zip(x_pca[:,0], x_pca[:,1]):
        #     targs_seen.append(targs[i])
        #     y = targs[i]
        #     plt.annotate("{}".format(y), (x,y))
        fig.colorbar(im, ax=ax1)
        ax1.patch.set_facecolor('whitesmoke')
        variance_ = np.var(transformed_data, axis=0)
        explained_variance_ratio_ = variance_/np.sum(variance_)
        ax2.plot(np.cumsum(explained_variance_ratio_))
        ax2.set_xlabel('Number of components')
        ax2.set_ylabel('Cumulative Explained Variance (0-1)');
        #plt.show()
        pdf = matplotlib.backends.backend_pdf.PdfPages(out_file_name)
        pdf.savefig(fig)
        pdf.close()
    
    # method = pearson for pearson correlation 
    # method = mic for MIC-based correlation
    def visualize_correlation(self, output_file, method='pearson', threshold=0.8):
        df = self.data_csv
        files = os.listdir('.')
        method_names = {'pearson': "Pearson", "mic": "MIC"}
        f, ax = plt.subplots(figsize=(11, 9))
        corr_matrix = compute_correlation(df, method, threshold)
        x_cols = [col for col in df.columns if col not in ['SampleName', 'Flow', 'FlowClass']]


        sns.heatmap(corr_matrix, xticklabels=x_cols, yticklabels=x_cols)
        
        plt.title("{} Correlation for Full Data Matrix".format(method_names[method]))
        pdf = matplotlib.backends.backend_pdf.PdfPages(output_file)
        pdf.savefig(f)
        pdf.close()