import os

def check_gen_average_data(set_type):
    path_name = "./data/averaged_data_"+set_type+".csv"
    if os.path.isfile(path_name):
            return True
    return False