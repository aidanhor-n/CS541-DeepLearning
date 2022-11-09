import os

import flowability_data_upload.local.generate_datasets.virtualpowder as VirtualPowder
import flowability_data_upload.local.troubleshoot as troubleshoot
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor, wait

def generate_virtually_mixed_powders_batch_jobs(data, set_type, batch_start_index=0, batch_end_index=None, increment=0.1, threads=16):
    '''
    Same as the method below, but this skips the check to skip over making virtual powders to allow us to make virtual powders
    in an existent folder.
    '''
    multi_thread_data_generation(data, set_type, batch_start_index, batch_end_index, increment, threads)

def generate_virtually_mixed_powders(data, set_type, batch_start_index=0, batch_end_index=None, increment=0.1, threads=16):
    """
    virtually mix the powders from the dataset, this will generate the data into the virtual_data folder

    :param increment: what increments should the powders be mixed in, .1 -> 10%, 20%, 30% ect. [0-1]. The lower this,
        the larger the generated dataset. The generated virtual dataset size
        equals [N_powders * N_powders * (1 / increment)

    :param threads: the number of threads to use in order to generate the virtual dataset. generating the dataset is
        computationally expensive and has been implemented using multithreading to speed up processing
    """
    if not(troubleshoot.check_gen_virtual_data()):
        os.mkdir("./virtual_data")
    
    if troubleshoot.check_gen_dataset_type(set_type):
        return True
    else:
        path_name = "./virtual_data/"+set_type+"_data"
        os.mkdir(path_name)
        print("No virtual data")
    #multi_thread_data_generation(data, set_type, batch_start_index, batch_end_index, increment, threads)
    

def multi_thread_data_generation(data, set_type, batch_start_index=0, batch_end_index=None, increment=0.1, threads=16):
    def generate_powder_multi(powder_info):
        dictionary = powder_info[0]
        name = powder_info[1]
        set_type = powder_info[2]
        print("___________________________________")
        # print("Powder #" + str(count) + ":")
        print(dictionary)
        #print("(" + str(mix_1) + ")" + " " + sample_id_1 + "- " + " (" + str(mix_2) + ")" + " " + sample_id_2)
        virtual_powder = VirtualPowder.VirtualPowder(dictionary,
                                                data,
                                                name=name)
        name = name + ".csv"
        gen_virtual_powder = virtual_powder.generate_sample()
        gen_virtual_powder.to_csv("./virtual_data/"+set_type+"_data/" + name,index=False)
    
    sample_ids = data['sample_id'].unique()
    #sample_ids = sample_ids[0:3] # Only making virtual powders out of the first 2-3 powders
    print(sample_ids)
    # print(len(sample_ids))
    virtual_powder_generation_data = list()

    if batch_end_index == None:
        batch_end_index = len(sample_ids)

    # generate generation_powder_data_list
    for powder_index1 in range(batch_start_index,batch_end_index):
        sample_id_1 = sample_ids[powder_index1]
        name = sample_id_1 + "_" +str(1.0)
        info = [{sample_id_1: 1.0}, name, set_type]
        virtual_powder_generation_data.append(info)

        for portion1 in range(1, 10, 1):
            # mix with entire rest of list
            for powder_index2 in range(powder_index1+1,len(sample_ids)):
            # second sample to mix with
                sample_id_2 = sample_ids[powder_index2]

                # Yes, I realize that we could just use portion1 as is.
                # However, the multithreading gets weird, so this ensures that I'm
                # producing the value I expect and my mix_1 is not changing in the midst
                # of me trying to use the variable.
                mix_1 = portion1 / 10.0
                mix_2 = 1 - mix_1
                dictionary = dict()
                dictionary[sample_id_1] = mix_1
                dictionary[sample_id_2] = mix_2
                name = sample_id_1 + "_" + sample_id_2 + "_" + str(mix_1) + "_" + str(mix_2)
                info = [dictionary, name, set_type]
                virtual_powder_generation_data.append(info)

                for portion2 in range(1, (10-portion1), 1):
                    for powder_index3 in range(powder_index2+1,len(sample_ids)):
                        sample_id_3 = sample_ids[powder_index3]

                        mix_1 = portion1 / 10.0
                        mix_2 = portion2 / 10.0
                        mix_3 = 1 - mix_1 - mix_2
                        dictionary = dict()
                        dictionary[sample_id_1] = mix_1
                        dictionary[sample_id_2] = mix_2
                        dictionary[sample_id_3] = mix_3
                        name = sample_id_1 + "_" + sample_id_2 + "_" + sample_id_3 + "_" + str(mix_1) + "_" + str(mix_2) + "_" + str(mix_3)
                        info = [dictionary, name, set_type]
                        virtual_powder_generation_data.append(info)


    with ThreadPoolExecutor(max_workers=32) as executor:
        executor.map(generate_powder_multi, virtual_powder_generation_data)