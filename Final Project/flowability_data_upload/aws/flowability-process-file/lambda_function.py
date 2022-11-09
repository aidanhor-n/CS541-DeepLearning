import json
import os
import boto3
from s3integrity import S3Integrity
from dataclean import DataCleaner


def lambda_handler(event, context):
    """
    This lambda function cleans newly uploaded raw_data files and updates
    metadata synchronization files so the data science team can access the clean_data
    data.
    
    """
    # get bucket name
    bucket = event['Records'][0]['s3']['bucket']['name']
    # get file/key name
    key = event['Records'][0]['s3']['object']['key']
    
    # object initialization
    integrity = S3Integrity(bucket, key)
    data_cleaner = DataCleaner(integrity)
    
    # data cleaning, metadata updates
    metadata = integrity.load_metadata()
    clean_data_location = data_cleaner.clean_data(metadata)
    updated_metadata = integrity.update_metadata_timestamp(metadata)
    
    updates = {"clean_data_location": clean_data_location, 
               "updated_metadata": updated_metadata}
    
    return {
        'statusCode': 200,
        'body': json.dumps(updates)
    }