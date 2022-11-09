import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import matplotlib.pyplot as plt
import matplotlib as mat
import colordict as cd
import seaborn as sns

# Make numpcy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
module_path = module_path + "\\flowability_data_upload\\PSD_TurboSync"


class Data:

    def __init__(self):
        # these lists are lists of powder specific dataframes
        self.alldatalist = []
        self.datalist = []
        self.flowpowders = []
        self.noflowpowders = []

        # below dataframes are not separated by powder
        self.alldata = pd.DataFrame()  # all of the data
        self.flowall = pd.DataFrame()  # powders with flow
        self.noflowall = pd.DataFrame()  # powders with no flow
        self.data = pd.DataFrame()  # all of the data
        self.flow = pd.DataFrame()  # powders with flow
        self.noflow = pd.DataFrame()  # powders with no flow

    def importpowders(self):
        cdd = []
        icdd = []
        cdd = ["Id", "Img Id", "Filter0", "Filter1", "Filter2", "Filter3", "Filter4", "Filter5",
               "Filter6", "hash"]  # general columns to drop
        icdd = cdd.copy()
        icdd.remove("Id")
        datanamelist = ["July15B.csv", "JulyB1.csv", "JulyB2.csv", "JulyB4.csv", "JulyB5.csv", "JulyB6.csv",
                        "JulyB10.csv",
                        "JulyB12.csv", "JulyB13.csv", "JulyB14.csv", "JulyB16.csv", "JulyB17.csv", "JulyB18.csv",
                        "JulyB19.csv", "JulyB20.csv"]
        for i in datanamelist:
            idata = pd.read_csv(i);
            pdata = idata.drop(columns=cdd)  # powder data minus general unwanted columns

            if idata["Flow"][0] != 0:
                pdata["Flow Class"] = pdata["Flow Class"] * 0 + 1
                idata["Flow Class"] = idata["Flow Class"] * 0 + 1
                self.flowall = self.flowall.append(idata)
                self.flowpowders.append(pdata)
            else:
                self.noflowall = self.noflowall.append(idata)
                self.noflowpowders.append(pdata)
            self.datalist.append(pdata)  # flow class might be broke
            self.alldatalist.append(idata)
            self.alldata = self.alldata.append(idata)

        self.flow = self.flowall.drop(columns=cdd)
        self.noflow = self.noflowall.drop(columns=cdd)
        self.data = self.alldata.drop(columns=cdd)
        print(self.data)
        
        
    def importnewpowders(self):
        cdd = []
        icdd = []
        cdd = ["Id", "Img Id", "Filter0", "Filter1", "Filter2", "Filter3", "Filter4", "Filter5", "Filter6", "hash"]  # general columns to drop
        icdd = cdd.copy()
        icdd.remove("Id")
        
        datanamelist = os.listdir(module_path)
        datanamelist.pop(0)
        flowvaldf = pd.read_excel(module_path + "\\" + datanamelist.pop(0), index_col=9)
        #print(datanamelist)
        #print(flowvaldf)
        for i in datanamelist:
            #print(i)
            if i[:2] == "~$":
                break
       
            #print(module_path)
            #idata = pd.read_csv(i);
            idata = pd.read_csv(module_path + "\\" + i, sep = '\t', header=(0), skiprows = [1])
            #print(idata)
            
            if i[:-4] != "PA_Ni-914-3":
                flowval = flowvaldf.loc[i[:-4], "Average Hall Flow"]
                #print(flowval)
                if flowval == "No Flow":
                    flowval = 0
                    flowcval = 0
                elif float(flowval) > 30:
                    flowcval = 1
                else:
                    flowcval = 2
                    
            
                idata["Flow"] = [flowval] * len(idata.index)
                idata["Flow Class"] = [flowcval] * len(idata.index)
                idata["Powder"] = [i] * len(idata.index)
                if len(idata.columns) > 40:
                    idata = idata.drop(columns = ["Fiber Length", "Fiber Width"])

                #print(idata)
                pdata = idata.drop(columns=cdd)  # powder data minus general unwanted columns
              
                if idata["Flow"][0] != 0:
                    pdata["Flow Class"] = pdata["Flow Class"] * 0 + flowcval
                    idata["Flow Class"] = idata["Flow Class"] * 0 + flowcval
                    self.flowall = self.flowall.append(idata)
                    self.flowpowders.append(pdata)
                else:
                    self.noflowall = self.noflowall.append(idata)
                    self.noflowpowders.append(pdata)
                self.datalist.append(pdata)  # flow class might be broke
                self.alldatalist.append(idata)
                self.alldata = self.alldata.append(idata)

        self.flow = self.flowall.drop(columns=cdd)
        self.noflow = self.noflowall.drop(columns=cdd)
        self.data = self.alldata.drop(columns=cdd)
        print(self.data)

    def coltest(self, cols, ctd=[]):
        plist = []
        for i in self.datalist:
            plist.append(i.sample(n=2842))
        df = pd.concat(plist)
        df = df.sample(frac=1)
        flowdf = df[df["Flow Class"] == 1]
        results = []
        datar = []
        dfr = []
        results.append(datar)
        results.append(dfr)
        for i in ctd:
            if i in cols:
                cols.remove(i)
        for i in cols:
            cttd = ctd.copy()
            cttd.append(i)
            print("Feature dropped: ", i)
            results[0].append([i, "Data", trymodel(df, flowdf, self.data, ctd=cttd)])
            results[1].append([i, "Df", trymodel(df, flowdf, df, ctd=cttd)])
            print("\n\n\n\n\n\n\n\n\n")
        return results


def makemodel(df, y="Flow"):
    # print(df)

    df_features = df.copy()
    df_labels = df_features.pop(y)
    # print(df_labels)
    # df_features = df.drop(columns = y)
    df_features = np.array(df_features)
    # print(len(df_features))

    df_model = tf.keras.Sequential([
        layers.Dense(64),
        layers.Dense(1)
    ])

    df_model.compile(loss=tf.losses.MeanSquaredError(),
                     optimizer=tf.optimizers.Adam())
    df_model.fit(df_features, df_labels, epochs=1)

    normalize = preprocessing.Normalization()

    normalize.adapt(df_features)

    norm_df_model = tf.keras.Sequential([
        normalize,
        layers.Dense(64),
        layers.Dense(1)
    ])

    norm_df_model.compile(loss=tf.losses.MeanSquaredError(),
                          optimizer=tf.optimizers.Adam())

    norm_df_model.fit(df_features, df_labels, epochs=10)

    return norm_df_model


def modeltest(c, r, t):  # c = classify model, r = regression model, t = test set
    def pfix(arr):  # it works (flips axes or something)
        fppp = np.array(arr)
        fppp = np.absolute(fppp)
        fpp = fppp > 0
        newfp = fppp[fpp]
        return newfp

    rs = []  # r^2 values to return
    cps = c.predict(t.drop(columns=["Flow", "Flow Class"]))  # classify predictions
    rps = r.predict(t.drop(columns=["Flow", "Flow Class"]))  # regression predictions

    cps = pfix(cps)  # fixes cps and rps into 1d arrays of floats, instead of 2d arrays of floats
    rps = pfix(rps)

    cps = np.where(cps < .5, 0,
                   1)  # forces predictions into classification of 1 or 0 (flow or no flow) (to be revised later)

    fps = cps * rps  # final predictions (0 if no flow and regression prediction otherwise)

    exp = t["Flow"].to_numpy()  # expected values
    cexp = t["Flow Class"].to_numpy()  # classify expected values

    b = exp != 0
    rexp = exp[b]  # regression expected values

    cstats = (cps * cexp) * 2
    cstats = cstats + (cexp - cps)  # breaks results into one array of values 0, 1, -1, and 2 (specified below)
    cstats = np.where(cstats < 0, 3, cstats)

    csum = np.bincount(cstats.astype(int), minlength=4)
    tn = csum[0]  # true negative, red
    fn = csum[1]  # false negative, yellow
    fp = csum[3]  # false positive, orange
    tp = csum[2]  # true positive, green

    cp = tn + tp  # correct predictions
    tot = tn + fn + fp + tp

    print("Classify results:")
    print("\nNegative Predictions: ", (tn + fn) / tot * 100, "%\tActual Negative: ", (tn + fp) / tot * 100, "%")
    print("Positive Predicitons: ", (tp + fp) / tot * 100, "%\tActual Positive: ", (tp + fn) / tot * 100, "%")
    print("\nTrue Negatives: ", tn, "\t", tn / tot * 100, "%")
    print("False Negatives: ", fn, "  ", fn / tot * 100, "%")
    print("False Positives: ", fp, "\t", fp / tot * 100, "%")
    print("True Positives: ", tp, "\t", tp / tot * 100, "%")
    print("\nNo Flow Predicitons: ", (tn / (tn + fn)) * 100, "% Correct")
    print("Flow Predictions: ", (tp / (tp + fp)) * 100, "% Correct")
    print("\nCorrect Predictions:", cp, "  ", cp / tot * 100, "%")
    R = np.corrcoef(cps, cexp)[1, 0]
    R2 = R ** 2
    R = R * 100
    R2 = R2 * 100
    R = round(R, 4)
    R2 = round(R2, 4)
    rs.append(R2)
    print("R: ", R, "%  R^2: ", R2, "%")

    print("\n\nRegression Visualization and Statistics: ")

    colors = np.where(cstats == 0, "red", np.where(cstats == 1, "yellow", np.where(cstats == 3, "orange", "green")))
    rcb = colors != "orange"  # regression color boolean
    rcolors = colors[rcb]  # colors for flow powders only
    rcb = rcolors != "red"
    rcolors = rcolors[rcb]

    # scatter = plt.scatter(rps[b], rexp, c = rcolors, label = rcolors)

    cdict = {"yellow": "False Negative", "green": "True Positive", "red": "True Negative",
             "orange": "False Positive"}

    cax = np.empty(0)
    cay = np.empty(0)
    for g in ["yellow", "green"]:
        ix = np.where(colors == g)
        plt.scatter(rps[ix], exp[ix], c=colors[ix], label=cdict[g])
        cax = np.append(cax, rps[ix])
        cay = np.append(cay, exp[ix])
    R = np.corrcoef(cax, cay)[1, 0]
    plt.legend()
    plt.xlabel("Predicted Flow")
    plt.ylabel("Expected Flow")
    plt.title("Flow Powders Regression Prediction")
    plt.show()
    R2 = R ** 2
    R = R * 100
    R2 = R2 * 100
    R = round(R, 4)
    R2 = round(R2, 4)
    rs.append(R2)
    print("R: ", R, "%  R^2: ", R2, "%")

    cax = np.empty(0)
    cay = np.empty(0)
    for g in ["green", "orange"]:
        ix = np.where(colors == g)
        plt.scatter(fps[ix], exp[ix], c=colors[ix],
                    label=cdict[g])  # some of this can be made more efficient or modularized or something
        cax = np.append(cax, fps[ix])
        cay = np.append(cay, exp[ix])

    plt.xlabel("Predicted Flow")
    plt.ylabel("Expected Flow")
    plt.title("Predicted Flow Powder Regression Predictions vs Expected Flow")
    plt.legend()
    plt.show()
    R = np.corrcoef(cax, cay)[1, 0]
    R2 = R ** 2
    R = R * 100
    R2 = R2 * 100
    R = round(R, 4)
    R2 = round(R2, 4)
    rs.append(R2)
    print("R: ", R, "%  R^2: ", R2, "%")

    cax = np.empty(0)
    cay = np.empty(0)
    for g in ["green", "orange", "yellow", "red"]:
        ix = np.where(colors == g)
        plt.scatter(fps[ix], exp[ix], c=colors[ix], label=cdict[g])
        cax = np.append(cax, fps[ix])
        cay = np.append(cay, exp[ix])
    R = np.corrcoef(cax, cay)[1, 0]
    plt.xlabel("Predicted Flow")
    plt.ylabel("Expected Flow")
    plt.title("Final Prediction vs Final Expected")
    plt.legend()
    plt.show()
    R2 = R ** 2
    R = R * 100
    R2 = R2 * 100
    R = round(R, 4)
    R2 = round(R2, 4)
    rs.append(R2)
    print("R: ", R, "%  R^2: ", R2, "%")
    return rs


def trymodel(c, r, t, visualize=True,
             ctd=[]):  # returns 1x4 array containing 4 R^2 values. Classification, Regression, Predicted Regression,
    # and Overall
    if len(ctd) != 0:
        c = c.drop(columns=ctd)
        r = r.drop(columns=ctd)
        t = t.drop(columns=ctd)
    cmodel = makemodel(c.drop(columns="Flow"), "Flow Class")
    rmodel = makemodel(r.drop(columns="Flow Class"))
    if visualize:
        return modeltest(cmodel, rmodel, t)
