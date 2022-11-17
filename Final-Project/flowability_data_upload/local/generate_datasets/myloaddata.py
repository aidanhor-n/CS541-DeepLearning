"""Load Data

Use to load the data. Using the local data and updating the local dataset if it has been changed since last call on
local machine. This class is responsible for maintaining synchronization of the dataset with AWS S3 bucket.
"""
import os
import csv
import json
import requests
import logging
import pandas
from datetime import datetime


def load_data():
    """
    load the flowability dataset, check online api if in sync,
    download it if it does not in sync, use what exists if internet fails
    :return:
    """
    if not local_data_exists():
        initialize_data()

    local_sync_time = load_recent_sync_time()
    response = check_api_sync(local_sync_time)
    sync_data(response)
    print("__________________________________________________________________")
    print("Loading Local Data...")
    return load_with_pandas()


def load_with_pandas():
    """
    load the dataset into a dataframe with pandas
    :return:
    """
    dataframe = pandas.read_csv("./data/data.csv")
    return dataframe


def load_recent_sync_time():
    """
    load sync time of most recently added file
    :return:
    """
    with open("./data/last_sync.json", "r") as f:
        time_info = json.load(f)
    last_sync = datetime.strptime(time_info["last_file_upload"], "%Y-%m-%d %H:%M:%S.%f")
    last_check = datetime.strptime(time_info["last_server_sync"], "%Y-%m-%d %H:%M:%S.%f")
    print("Sync Info:\t\nLast Sync Time (Server Time): " + str(last_check) +
          "\t\nUpload Time of Last Added File: " + str(last_sync))
    return last_sync


def local_data_exists():
    """
    return if data folder exists
    :return: bool if data folder already exists
    :rtype: bool
    """
    return os.path.exists("./data")


def initialize_data():
    """
    initialize the contents of the data folder
    """
    print("Initializing Data Folder...")
    os.mkdir("./data")
    initialize_data_csv()
    initialize_sync_time()


def initialize_data_csv():
    """
    create the empty data csv
    """
    columns = "sample_id,name,Id,Img Id,Da,Dp,FWidth,FLength,ELength,EWidth,Volume,Area,Perimeter," \
        "CHull  Area,CHull Perimeter," \
        "Sphericity,Compactness,Roundness,Ellipse Ratio,Circularity,Solidity,Concavity,Convexity,Extent,hash," \
        "Transparency,Curvature,Surface Area,Filter0,Filter1,Filter2,Filter3,Filter4,Filter5,Filter6," \
        "L/W Ratio,W/L Aspect Ratio,CHull Surface Area,Ellipticity,Fiber Length,Fiber Width,flowability".split(",")
    columns = [columns]
    with open("./data/data.csv", "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerows(columns)


def initialize_sync_time():
    """
    create the sync time
    """
    new_sync_timestamp = {"last_file_upload": "1970-01-01 00:00:00.000", "last_server_sync": "1970-01-01 00:00:00.000"}
    with open("./data/last_sync.json", "w") as f:
        json.dump(new_sync_timestamp, f)


def check_api_sync(local_sync_time):
    """
    call API Gateway endpoint to check if this sync time is up to date
    :param local_sync_time: epoch timestamp to check against server sync time
    :return: api response
    """
    local_sync_time = local_sync_time.strftime("%Y%m%d%H%M%S%f")
    try:
        api_endpoint = "https://5dtv22kdli.execute-api.us-east-1.amazonaws.com/prod/sync-data"
        query_params = {"local_sync_time": local_sync_time}
        response = requests.get(url=api_endpoint, params=query_params)
        response = response.json()
    except Exception as e:
        logging.exception("Error connecting to API ... using local data")
        response = {"in_sync": True}

    print("Data Sync Status: ", response["in_sync"])
    return response


def download_sample_data_file(presigned_url):
    """
    download csv file from presigned_url
    :param presigned_url:
    :return:
    """
    try:
        response = requests.get(presigned_url)
        if response.status_code == 200:
            response = response.content.decode("utf-8")
        else:
            raise PreSignedURLException
    except Exception as e:
        raise PreSignedURLException(e)
    return response


def add_to_local_dataset(data):
    """
    append this powder sample data to entire local dataset main file
    :param data: microtrac data for one sample
    """
    with open("./data/data.csv", "a", newline='') as f:
        # f.write(data)
        data_list = list(csv.reader(data.splitlines(), delimiter=","))
        writer = csv.writer(f)
        writer.writerows(data_list[1:])


def update_timestamp(time_uploaded):
    """
    write new recent upload time to timestamp
    :param time_uploaded: date timestamp in ISO 8601 standard format
    """
    with open("./data/last_sync.json", "r+") as f:
        sync_info = json.load(f)
        sync_info["last_file_upload"] = str(datetime.strptime(time_uploaded, "%Y-%m-%d %H:%M:%S.%f"))
        f.seek(0)
        json.dump(sync_info, f)
        f.truncate()


def update_server_check(time_checked):
    """
    write new recent upload time to timestamp
    :param time_checked: date timestamp in ISO 8601 standard format
    """
    with open("./data/last_sync.json", "r+") as f:
        sync_info = json.load(f)
        sync_info["last_server_sync"] = str(datetime.strptime(time_checked, "%Y-%m-%d %H:%M:%S.%f"))
        f.seek(0)
        json.dump(sync_info, f)
        f.truncate()


def sync_data(response):
    """
    using the information from api, decide if sync is needed then for each sample, download, append to dataset and
    update synchronization timestamp
    :param response: dict with in_sync info and new_files list of new sampel metadat
    """
    if response["in_sync"]:
        pass
    else:
        for count, sample in enumerate(response["new_files"]):
            print("Downloading File " + str(count + 1) + " of " + str(len(response["new_files"])))
            try:
                data = download_sample_data_file(sample["presigned_url"])
                add_to_local_dataset(data)
                update_timestamp(sample["time_uploaded"])
            except PreSignedURLException as e:
                logging.exception("Error downloading data from S3 ... halted synchronization, using local data")
        update_server_check(response["server_time"])


class PreSignedURLException(Exception):
    """Occurs when the presigned url fails to download from S3 on AWS"""
    pass


class SyncAPIException(Exception):
    """Occurs when the Sync API fails to respond in API Gateway on AWS"""
    pass
