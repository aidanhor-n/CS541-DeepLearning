from flowability_data_upload.local.generate_datasets.loaddata import load_data
import flowability_data_upload.local.preprocess as preprocess
import flowability_data_upload.local.generate_datasets.virtualpowder as VirtualPowder

def print_powders_helper(data):
    sample_ids = data['sample_id'].unique()
    for sample_id in sample_ids:
        flowability = data[data.sample_id == sample_id].iloc[0].flowability
        name = data[data.sample_id == sample_id].iloc[0][data.columns[1]]
        print("NAME: ", name, "FLOW:", flowability, "ID:", sample_id)


# Load the Data (load)
data = load_data()

# Call Preprocessing Steps (preprocess)

data = preprocess.clean(data)

# Show Samples and Flowability Values
# print_powders_helper(data)

print("Flow 1: 2.54 Flow 2: 34.0")
print("Generating Sample")
virtual_powder = VirtualPowder.VirtualPowderMass({"4d9d565d-98b4-4d5c-bb60-82a8d0c45abf": 0.50,
                                "e14b6e01-c678-4a9e-ac78-46870c8f94bb": 0.50},
                                data,
                                name="test sample")

gen_virtual_powder = virtual_powder.generate_sample()
print(gen_virtual_powder.head(5))
