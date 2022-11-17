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

data_path = "./data/"
train_average_data = pd.read_csv(data_path+"averaged_data_train.csv")
test_average_data = pd.read_csv(data_path+"averaged_data_test.csv")

n_bins = 2

flowability = train_average_data["flowability"]
flowability = np.array(flowability)
flowability[flowability == 0] = 60
flowability[flowability < 30] = 0
flowability[flowability >= 30] = 1

flowability_amounts = [len(flowability[flowability == 0]), len(flowability[flowability == 1])]

plt.bar(["High Flow","Low Flow"], height=flowability_amounts)
plt.title("Class Distribution for Virtual Training Set")
plt.xlabel("Flowability Class")
plt.ylabel("Number of Powders")
plt.savefig("train_virtual_distribution.png")
plt.close()

flowability = test_average_data["flowability"]
flowability = np.array(flowability)
flowability[flowability == 0] = 60
flowability[flowability < 30] = 0
flowability[flowability >= 30] = 1

flowability_amounts = [len(flowability[flowability == 0]), len(flowability[flowability == 1])]

plt.bar(["High Flow","Low Flow"], height=flowability_amounts)
plt.title("Class Distribution for Virtual Testing Set")
plt.xlabel("Flowability Class")
plt.ylabel("Number of Powders")
plt.savefig("test_virtual_distribution.png")
plt.close()