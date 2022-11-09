from flowability_data_upload.local.generate_datasets.loaddata import load_data
from flowability_data_upload.local.generate_datasets.loaddata import load_data
import flowability_data_upload.local.preprocess as preprocess
import flowability_data_upload.local.generate_datasets.virtualpowder as VirtualPowder
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor, wait


def print_powders_helper(data):
    sample_ids = data['sample_id'].unique()
    for sample_id in sample_ids:
        flowability = data[data.sample_id == sample_id].iloc[0].flowability
        name = data[data.sample_id == sample_id].iloc[0][data.columns[1]]
        print("NAME: ", name, "FLOW:", flowability, "ID:", sample_id)


# generate 50/50 Mix
data = load_data()

data = preprocess.clean(data)

print_powders_helper(data)
mix_50_50 = "a532cffa-1e9e-44e4-acbf-30d59f3a080c"
pure_cu = "b4512fec-0088-4a46-bed3-35b431b9293c"
pure_ti = "ec20d06a-9f8a-412a-b166-5e66797e4f20"

mix_90_10 = "b8d9ec23-2615-4b73-89e2-2ac13a7ef6ea"
mix_30_70 = "94d6d959-cb8e-4cad-8006-5a8f806a342a"

# print_powders_helper(data)

def single_thread_data_gen():
    sample_ids = data['sample_id'].unique()
    print(sample_ids)
    count = 0
    for sample_id_1 in sample_ids:
        # mix with entire rest of list
        for sample_id_2 in sample_ids:
            #second sample to mix with
            for x in np.arange(.1, 1, 0.1):
                count += 1
                mix_1 = x
                mix_2 = 1 - mix_1
                print("___________________________________")
                print("Powder #" + str(count) + ":")
                print("(" + str(mix_1) + ")" + " " + sample_id_1 + "- " + " (" + str(mix_2) + ")" + " " + sample_id_2)
                virtual_powder = VirtualPowder.VirtualPowder({sample_id_1: mix_1,
                                                           sample_id_2: mix_2},
                                                          data,
                                                          name=sample_id_1 + "_" + sample_id_2 + "_" + str(mix_1) + "_" + str(mix_2))
                name = sample_id_1 + "_" + sample_id_2 + "_" + str(mix_1) + "_" + str(mix_2) + ".csv"
                gen_virtual_powder = virtual_powder.generate_sample()
                gen_virtual_powder.to_csv("./virtual_data/" + name)


def multi_thread_data_generation(increment=0.1, threads=16):
    sample_ids = data['sample_id'].unique()
    print(sample_ids)
    # print(len(sample_ids))
    count = 0
    virtual_powder_generation_data = list()

    # generate generation_powder_data_list
    for sample_id_1 in sample_ids:
        # mix with entire rest of list
        for sample_id_2 in sample_ids:
            # second sample to mix with
            for x in np.arange(.1, 1, 0.1):
                count += 1
                mix_1 = x
                mix_2 = 1 - mix_1
                info = [sample_id_1, sample_id_2, mix_1, mix_2, data]
                virtual_powder_generation_data.append(info)

    with ThreadPoolExecutor(max_workers=32) as executor:
        executor.map(generate_powder_multi, virtual_powder_generation_data)




def generate_powder_multi(powder_info):
    sample_id_1 = powder_info[0]
    sample_id_2 = powder_info[1]
    mix_1 = powder_info[2]
    mix_2 = powder_info[3]
    data = powder_info[4]
    print("___________________________________")
    # print("Powder #" + str(count) + ":")
    print("(" + str(mix_1) + ")" + " " + sample_id_1 + "- " + " (" + str(mix_2) + ")" + " " + sample_id_2)
    virtual_powder = VirtualPowder.VirtualPowder({sample_id_1: mix_1,
                                               sample_id_2: mix_2},
                                              data,
                                              name=sample_id_1 + "_" + sample_id_2 + "_" + str(
                                                  mix_1) + "_" + str(mix_2))
    name = sample_id_1 + "_" + sample_id_2 + "_" + str(mix_1) + "_" + str(mix_2) + ".csv"
    gen_virtual_powder = virtual_powder.generate_sample()
    gen_virtual_powder.to_csv("./virtual_data/" + name)


# batched #2
def multi_thread_data_generation_b2(increment=0.1, threads=16):
    sample_ids = data['sample_id'].unique()
    print(sample_ids)
    count = 0
    virtual_powder_generation_data = list()

    # generate generation_powder_data_list
    for sample_id_1 in sample_ids:
        # mix with entire rest of list
        for sample_id_2 in sample_ids:
            # second sample to mix with
            for x in np.arange(.1, 1, 0.1):
                count += 1
                mix_1 = x
                mix_2 = 1 - mix_1
                info = [sample_id_1, sample_id_2, mix_1, mix_2, data]
                virtual_powder_generation_data.append(info)

    chunks = [data[x:x + 100] for x in range(0, len(data), 100)]
    count = 0
    for chunk in chunks:
        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = []
            futures.append(executor.map(generate_powder_multi_b2, virtual_powder_generation_data))
            print("100 chunk")
            for result in futures:
                result.result()
                count += 1
                print("Count: " + str(count))
                print(result)
                result.to_csv("./virtual_data/" + str(count) + ".csv")


def generate_powder_multi_b2(powder_info):
    sample_id_1 = powder_info[0]
    sample_id_2 = powder_info[1]
    mix_1 = powder_info[2]
    mix_2 = powder_info[3]
    data = powder_info[4]
    print("___________________________________")
    # print("Powder #" + str(count) + ":")
    print("(" + str(mix_1) + ")" + " " + sample_id_1 + "- " + " (" + str(mix_2) + ")" + " " + sample_id_2)
    virtual_powder = VirtualPowder.VirtualPowder({sample_id_1: mix_1,
                                               sample_id_2: mix_2},
                                              data,
                                              name=sample_id_1 + "_" + sample_id_2 + "_" + str(
                                                  mix_1) + "_" + str(mix_2))
    name = sample_id_1 + "_" + sample_id_2 + "_" + str(mix_1) + "_" + str(mix_2) + ".csv"
    gen_virtual_powder = virtual_powder.generate_sample()
    return gen_virtual_powder
    # gen_virtual_powder.to_csv("./virtual_data/" + name)


# Batched Data
# def multi_thread_data_generation(increment=0.1, threads=16):
#     sample_ids = data['sample_id'].unique()
#     print(sample_ids)
#     count = 0
#     virtual_powder_generation_data = list()
#
#     # generate generation_powder_data_list
#     for sample_id_1 in sample_ids:
#         # mix with entire rest of list
#         for sample_id_2 in sample_ids:
#             # second sample to mix with
#             for x in np.arange(.1, 1, 0.1):
#                 count += 1
#                 mix_1 = x
#                 mix_2 = 1 - mix_1
#                 info = [sample_id_1, sample_id_2, mix_1, mix_2, data]
#                 virtual_powder_generation_data.append(info)
#
#     with ThreadPoolExecutor(max_workers=32) as executor:
#         futures = []
#         futures.append(executor.submit(generate_powder_multi, virtual_powder_generation_data))
#         count = 0
#         count_cache = 0
#         cache = []
#         for x in as_completed(futures):
#             print("WRITING BATCH")
#             cache.append(x.result())
#             count += 1
#             count_cache += 1
#             if count_cache >= 100:
#                 for file in cache:
#                     file.to_csv("./virtual_data/" + str(count) + ".csv")
#                 cache = []
#                 count_cache = 0




multi_thread_data_generation(0.1, 16)

# virtual_powder = VirtualPowder.VirtualPowder({mix_90_10: 0.50,
#                                 mix_50_50: 0.50},
#                                 data,
#                                 name="virtual_50_50")
#
# gen_virtual_powder = virtual_powder.generate_sample()
# gen_virtual_powder.to_csv("./virtual_50_50.csv")

# # get pure cu
# pure_cu_sample = data[data.sample_id == pure_cu]
# pure_cu_sample.to_csv("./pure_cu.csv")
#
# # get pure ti
# pure_ti_sample = data[data.sample_id == pure_ti]
# pure_ti_sample.to_csv("./pure_ti.csv")
#
# # get 50/50 Mix
# mix_50_50_sample = data[data.sample_id == mix_50_50]
# mix_50_50_sample.to_csv("./mix_50_50.csv")
#
# combined = gen_virtual_powder.append(pure_cu_sample).append(pure_ti_sample).append(mix_50_50_sample)
# combined.to_csv("./combined.csv")
