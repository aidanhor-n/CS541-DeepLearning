import flowability_data_upload.local.troubleshoot as troubleshoot
import pandas as pd
import numpy as np
import glob

def average_features_in_powder(data):
    '''
    This method takes in the particles of a powder and derives an average value for each
    feature
    return summarization of particles in the form of averages
    '''
    numeric_data = data.drop(columns=["sample_id", "name"])
    data_averages = numeric_data.mean()
    data_averages = data_averages.values.reshape((1,30))
    new_data_frame = pd.DataFrame(data_averages, columns=numeric_data.columns)
    return new_data_frame

def average_powders(set_type):
    if troubleshoot.check_gen_average_data(set_type):
        print("Averaging data has already been generated.")
        return
    else:
        if troubleshoot.check_gen_virtual_data():
            if troubleshoot.check_gen_dataset_type(set_type):
                average_all_powders(set_type)
            else:
                print("Data of this set type has not been created. Cannot average non-existent data.")
        else:
            print("Virtual powders have not been created yet. Cannot average non-existent data.")

def average_all_powders(set_type):
    print("Averaging all powders...")
    path = './virtual_data/' + set_type + "_data"
    filenames = glob.glob(path + "/*.csv")
    averaged_powders = pd.DataFrame()
    sampled_ids = []
    for filename in filenames:
        powder_data = pd.read_csv(filename)
        sample_id = powder_data["sample_id"].iloc[0]
        sampled_ids.append(sample_id)
        averaged_powder = average_features_in_powder(powder_data)
        if(averaged_powders.empty):
            averaged_powders = averaged_powder
        else:
            averaged_powders = averaged_powders.append(averaged_powder)
    averaged_powders["Sample_id"] = sampled_ids
    averaged_powders.to_csv("./data/averaged_data_" + set_type + ".csv",index=False)