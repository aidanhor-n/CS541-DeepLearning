import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from flowability_data_upload.local.generate_datasets.loaddata import load_data
import flowability_data_upload.local.preprocess as preprocess
from flowability_data_upload.local.generate_datasets import TrainAndTestSetGenerator as testSetGenerator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Load normal data
data = load_data()

#Clean and preprocess data
data = preprocess.clean(data)
data = preprocess.remove_duplicate_powders(data)

# One testing and one training set
test_set, train_set = testSetGenerator.separate_train_test_powders_index_reverse(data)

ids = train_set["sample_id"].unique()
flowability = []
for id in ids:
    flowability.append(train_set[train_set.sample_id == id].iloc[0]["flowability"])

flowability = np.array(flowability)

flowability[flowability == 0] = 60

flowability[flowability == 0] = 60
plt.hist(flowability, bins=50)
plt.title("Flowability Distribution for Raw Powders in Training Set")
plt.xlabel("Flowability")
plt.ylabel("Number of Powders")
plt.savefig("training_powder_dist.png")
plt.close()

flowability[flowability < 30] = 0
flowability[flowability >= 30] = 1

flowability_amounts = [len(flowability[flowability == 0]), len(flowability[flowability == 1])]

plt.bar(["High Flow","Low Flow"], height=flowability_amounts)
plt.title("Class Distribution for Raw Powders in Training Set")
plt.xlabel("Flowability Class")
plt.ylabel("Number of Powders")
plt.savefig("train_raw_distribution.png")
plt.close()

ids = test_set["sample_id"].unique()
flowability = []
for id in ids:
    flowability.append(test_set[test_set.sample_id == id].iloc[0]["flowability"])

flowability = np.array(flowability)

flowability[flowability == 0] = 60
plt.hist(flowability, bins=50)
plt.title("Flowability Distribution for Raw Powders in Testing Set")
plt.xlabel("Flowability")
plt.ylabel("Number of Powders")
plt.savefig("testing_powder_dist.png")
plt.close()

flowability[flowability < 30] = 0
flowability[flowability >= 30] = 1

flowability_amounts = [len(flowability[flowability == 0]), len(flowability[flowability == 1])]

plt.bar(["High Flow","Low Flow"], height=flowability_amounts)
plt.title("Class Distribution for Raw Powders in Testing Set")
plt.xlabel("Flowability Class")
plt.ylabel("Number of Powders")
plt.savefig("test_raw_distribution.png")
plt.close()

