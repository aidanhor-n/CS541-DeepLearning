from flowability_data_upload.local.generate_datasets.loaddata import load_data
from flowability_data_upload.local.generate_datasets.loaddata import load_data
import flowability_data_upload.local.preprocess as preprocess
import flowability_data_upload.local.generate_datasets.virtualpowder as VirtualPowder


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

virtual_powder = VirtualPowder.VirtualPowder({mix_90_10: 0.50,
                                mix_50_50: 0.50},
                                data,
                                name="virtual_50_50")

gen_virtual_powder = virtual_powder.generate_sample()
gen_virtual_powder.to_csv("./virtual_50_50.csv")

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
