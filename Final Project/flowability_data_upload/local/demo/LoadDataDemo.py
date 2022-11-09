import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from generate_datasets.loaddata import load_data
import matplotlib.pyplot as plt
from analytics import visualize_pdf
import numpy as np

# Load the Data (load)
data = load_data()

# Print Value Names
print(data.columns)
print(data['name'].unique())


# Variables
powder_sample_name = ''
feature_value_name = ''

# Filter Data (get data of a specific name, then filter out everything but the specified powder,
# then get the correct feature)


# Plotting
