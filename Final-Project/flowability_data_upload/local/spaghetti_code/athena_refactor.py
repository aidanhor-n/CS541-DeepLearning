from flowability_data_upload.local.generate_datasets.loaddata import load_data
import flowability_data_upload.local.preprocess as preprocess
import pandas


def get_powder_data():
    data = load_data()

    data = preprocess.clean(data)

    data = data.drop_duplicates(subset="sample_id", keep="first")

    data = data[["sample_id", "name", "flowability"]]
    data['powder_type'] = 'NA'
    print(data)
    print(data.columns)
    data.to_csv("./powder.csv", index=False)


def create_mixture_table():
    powders = pandas.read_csv("./powder.csv")
    mixed_powders = powders['powder_type'] == "Mixture"
    print(mixed_powders)
    mixed_powders.to_csv("./powder.csv", index=False)

