"""Flowability - Upload File
This lambda function is responsible for uploading new data files to s3.
When called it generates a UUID for the new data and uploads the metadata of that
file. Additionally if creates a presigned_url in s3 to directly upload the data
file
"""

import json
import os
import uuid
import boto3
import csv
import time
from datetime import datetime


def lambda_handler(event, context):
    # load parameters from request body
    print(event)
    params = json.loads(event['body'])
    print(params)
    flowability = params['flowability']
    name = params['sample_name']
    
    # generate UUID and create presigned url for file upload
    sample_id = str(uuid.uuid4())
    upload_url = create_presigned_url(sample_id)
    
    # upload metadata from parameters
    upload_metadata(sample_id, flowability, name)
    
    
    return {
        'statusCode': 200,
        'body': json.dumps(upload_url)
    }


def upload_metadata(sample_id, flowability, name):
    """
    upload the metadata json file into s3 containing:
    flowability, name, sample_id, time_added, time_uploaded
    
    :param sample_id: unique id (UUID) for the data
    :param flowability: inputed flowability value of the material powder
    :param name: name of the sample
    """
    ts = time.time()
    ts = str(datetime.fromtimestamp(ts))
    metadata = {"sample_id": sample_id, "name": name, "flowability": flowability, "time_added": ts, "time_uploaded": None}
    
    # upload name and flow values to folders
    upload_file(sample_id, "metadata", metadata)

    
def upload_file(sample_id, folder, metadata):
    """
    helper function to upload metadata
    given sample_id a folder, upload the metadata to s3
    
    :param sample_id: unique id (UUID) for the data
    :param folder: folder location in s3 to upload file
    :param metadata: data to upload
    """
    # Generate Local and S3 file names using UUID
    filename = sample_id + ".json"
    bucket_key = folder + "/" + filename
    tmp_file = '/tmp/' + filename
    with open(tmp_file, 'w') as f:
        json.dump(metadata, f)

    # Upload to S3 Bucket
    s3 = boto3.resource('s3')
    bucket = s3.Bucket('flowability-data')
    bucket.upload_file(tmp_file, bucket_key)
    
    # Clean up Lambda tmp folder
    os.remove(tmp_file)
    
    
def create_presigned_url(sample_id):
    """
    create presigned url for raw_data upload. this is used instead of a direct
    upload because API Gateway has a limit of 10MB data payload and microtrac
    datafiles have been collected that are over 100 MB.
    
    :param sample_id: unique id (UUID) for the data
    """
    key_name = 'raw/' + sample_id + '.xls'
    
    s3 = boto3.client('s3')
    upload_url = s3.generate_presigned_post('flowability-data', key_name)
    return upload_url
