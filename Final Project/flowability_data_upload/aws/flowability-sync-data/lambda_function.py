"""Flowability - Sync Data
This lambda function is the endpoint data scientists will ping when they are using the data loader. It will check
the timestamp of their last sync and check if new files have been added since then. Returning pre-signed download URLs
of the new data if it exists.
"""

import json
import uuid
import boto3
import time
from datetime import datetime


def lambda_handler(event, context):
    """
    interface for lambda function that pulls parameters from request body
    :param event: APIGateway Proxy Lambda request event
    :param context:
    :return: json with in_sync information and urls
    """
    # load parameters from request body
    local_sync_time = json.loads(event['queryStringParameters']['local_sync_time'])
    local_sync_time = datetime.fromtime

    stamp(local_sync_time).strftime('%Y-%m-%d %H:%M:%S.%f')
    response = synchronize(local_sync_time)
    return {
        'statusCode': 200,
        'body': json.dumps(response)
    }


def search_new_data(timestamp):
    """
    this function searches the metadata s3 folder using amazon athena for sample_IDs, timestamps uploaded after
    the given timestamp
    :param timestamp: the time of the last sync with server
    :return: list(str), a list of data file metadata with IDs and timestamps that were uploaded after the given timestamp
    :rtype: list(dict{"sample_id": UUID, "time_uploaded": timestamp})
    """
    athena = boto3.client('athena')
    # Setup and perform query
    prep_string = 'SELECT sample_id, time_uploaded FROM metadata WHERE time_uploaded > TIMESTAMP ' + "'" + timestamp[:-3] + "';"
    print(prep_string)
    # prep_string = "SELECT sample_id, time_uploaded FROM metadata;"
    query = athena.start_query_execution(
        QueryString=prep_string,
        QueryExecutionContext={
            'Database': 'flowability'
        },
        ResultConfiguration={
            'OutputLocation': 's3://flowability-data/query/'
        }
    )

    # Observe results
    query_id = query['QueryExecutionId']
    results = ping_athena_until_result(query_id, athena)

    return results


def ping_athena_until_result(query_id, athena):
    """
    ping the query results every second until it finishes
    :param query_id: query_id from the executed query
    :param athena: boto client athena connection
    :return: list(str), a list of data file metadata with IDs and timestamps that were uploaded after the given timestamp
    :rtype: list(dict{"sample_id": UUID, "time_uploaded": timestamp})
    """
    new_datas = list()
    success = False
    while not success:
        try:
            time.sleep(1)
            # Query Result every second until it is successful
            results = athena.get_query_results(QueryExecutionId=query_id)
            for count, row in enumerate(results['ResultSet']['Rows']):

                if count == 0:
                    continue
                new_data = {"sample_id": row['Data'][0]["VarCharValue"],
                            "time_uploaded": row['Data'][1]["VarCharValue"]}
                new_datas.append(new_data)
            success = True
        except Exception as e:
            print("HI")
            print(e)
            # not found
            pass
    return new_datas


def generate_presigned_url(sample_id: str):
    """
    given a sample_id, generate a presigned URL to download that file.
    :param sample_id: UUID of a microtrac data sample
    :return: presigned URL to download the specified file ID
    :rtype: str
    """
    key_name = 'data/' + sample_id + '.csv'
    s3 = boto3.client('s3')
    upload_url = s3.generate_presigned_url(ClientMethod="get_object", Params={'Bucket': 'flowability-data',
                                                                              'Key': key_name})
    return upload_url


def build_new_sample_info(sample_metadata):
    """
    build the json response with sample_id, presigned download url and upload timestamp
    :param sample_metadata: metadata of a new sample file list(dict{"sample_id": UUID, "upload_time": timestamp})
    :return: dict{"sample_id": UUID, "presigned_url": url, "time_uploaded": timestamp}
    """
    presigned_url = generate_presigned_url(sample_metadata["sample_id"])
    sample_id = sample_metadata["sample_id"]
    time_uploaded = sample_metadata["time_uploaded"]
    sample_info = {"sample_id": sample_id, "presigned_url": presigned_url, "time_uploaded": time_uploaded}
    return sample_info


def check_insync(local_sync_time):
    """
    checks if the data is in sync with the server
    :return: if the data is in sync
    :rtype: bool
    """
    s3_client = boto3.client('s3')
    response = s3_client.get_object(Bucket="flowability-data", Key="last_data_upload_timestamp.txt")
    last_upload = datetime.strptime(json.loads(response["Body"].read())["last_upload"], "%Y-%m-%d %H:%M:%S.%f")
    local_sync_time = datetime.strptime(local_sync_time, "%Y-%m-%d %H:%M:%S.%f")
    in_sync = True
    if local_sync_time < last_upload:
        in_sync = False


def synchronize(local_sync_time):
    """
    given a local sync time, check if any new data has been added to the main server after that time respond with
    presigned urls to download the new files if they exist
    :param local_sync_time: epoch timestamp of the last upload time of the newest file (essentially last sync time)
    :return: json response
    """
    server_time = datetime.now()
    response = dict()

    in_sync = check_insync(local_sync_time)
    if not in_sync:
        download_info = list()
        new_files_metadata = search_new_data(local_sync_time)
        for file_metadata in new_files_metadata:
            download_info.append(build_new_sample_info(file_metadata))

        response["in_sync"] = False
        response["new_files"] = download_info
        response["server_time"] = server_time

    else:
        response["in_sync"] = True
        response["server_time"] = server_time
    return response
