import os
import sys
import time
# If the run_job.sh file is outside of the project folder, then
# this following code is not needed (this moves the working directory
# up by one so that everything is consistent in terms of directories).
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from flowability_data_upload.local.generate_datasets.loaddata import load_data
import flowability_data_upload.local.preprocess as preprocess
from flowability_data_upload.local.generate_datasets import VirtualPowderMixing as powderMixer
from flowability_data_upload.local.generate_datasets import TrainAndTestSetGenerator as testSetGenerator
import pandas

set_type = "train"
start_index = 0
end_index = None
if len(sys.argv) > 1:
    set_type = sys.argv[1]
    if len(sys.argv) > 2:
        start_index = int(sys.argv[2])
        if len(sys.argv) > 3:
            end_index = int(sys.argv[3])


start_time = time.perf_counter()

#Load normal data
data = load_data()

load_end_time = time.perf_counter()
print(f"Loading data took {(load_end_time - start_time)/60.0:0.4f} minutes")

#Clean and preprocess data
data = preprocess.clean(data)
data = preprocess.remove_duplicate_powders(data)

clean_end_time = time.perf_counter()
print(f"Cleaning data took {(clean_end_time - start_time)/60.0:0.4f} minutes")

# One testing and one training set
test_set, train_set = testSetGenerator.separate_train_test_powders_index_reverse(data)

generate_end_time = time.perf_counter()
print(f"Generating test and train sets took {(generate_end_time - start_time)/60.0:0.4f} minutes")

data_subset = train_set
if set_type == "test":
    data_subset = test_set

#Mix powders
print("Starting to mix " + set_type + "...")
print("Start index: " + str(start_index))
if end_index != None:
    print("End index: " + str(end_index))
powderMixer.generate_virtually_mixed_powders_batch_jobs(data_subset, set_type, start_index, end_index)

virtual_end_time = time.perf_counter()
print(f"Generating virtual data took {(virtual_end_time - start_time)/60.0:0.4f} minutes")
