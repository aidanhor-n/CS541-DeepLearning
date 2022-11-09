import base64
import requests
import os


def upload_data(sample_name, flowability, filename=None):
    """
    upload a powder sample, given the sample_name, flowability and optional filename
    :param sample_name: name of sample
    :param flowability: flowability of sample
    :param filename: file location, or use sample_name automatically ./data/[sample_name].xls
    """
    if filename is None:
        filename = "./data/" + sample_name + ".xls"
    api_endpoint = "https://5dtv22kdli.execute-api.us-east-1.amazonaws.com/prod/upload-file"
    query_params = {"flowability": flowability, "sample_name": sample_name}

    # Get Presigned URL/Upload Flowability/Name
    response = requests.post(api_endpoint, json=query_params)
    print(response.json())
    response = response.json()

    # Upload File using Presigned URL
    with open(filename, 'rb') as f:
        files = {'file': (filename, f)}
        http_response = requests.post(response['url'], data=response['fields'], files=files)
        print(http_response)


# upload_data("S5B2_4120_Excel", 49.46)
upload_data("S14B3_4420_Excel", 35.51)
# upload_data("S3B2_4120_Excel", 21.42)


#
# upload_data(sample_name="S6B1_3720_Excel", flowability=0)
