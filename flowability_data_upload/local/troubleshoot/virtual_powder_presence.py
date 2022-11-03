import os

def check_gen_virtual_data():
    if os.path.isdir("./virtual_data"):
        path, dirs, files = next(os.walk("./virtual_data"))
        if len(dirs) != 0:
            return True
    return False

def check_gen_dataset_type(set_type):
    path_name = "./virtual_data/"+set_type+"_data"
    if os.path.isdir(path_name):
        path, dirs, files = next(os.walk(path_name))
        if len(files) != 0:
            return True
    return False