"""Maintain S3 Integrity

These functions are used to update the metadata files important to maintain the
integrity and sychonization with user access of the data.

"""
import json
import time
from datetime import datetime
import os
import boto3
import botocore


class S3Integrity:

    def __init__(self, bucket, key):
        self.bucket = bucket
        self.key = key

        self.s3_client = boto3.client('s3')
        self.s3_res = boto3.resource('s3')

        self.sync_filename = "last_data_upload_timestamp.txt"
        self.sync_local_filename = "/tmp/last_data_upload_timestamp.txt"

        self.metadata_filename = None
        self.metadata_local_filename = None

        self.clean_data_filename = None
        self.clean_data_local_filename = None

        self.sample_id = self.parse_sample_id()

        self.parse_clean_data_filenames()
        self.parse_metadata_filenames()

    def parse_clean_data_filenames(self):
        """
        Parse clean data filepath names by parsing bucket and key
        """
        key = self.sample_id
        self.clean_data_filename = "data/" + key + ".csv"
        self.clean_data_local_filename = "/tmp/" + key + ".csv"

    def parse_metadata_filenames(self):
        """
        Parse bucket and key in order to generate filenames for metadata file
        """
        key = self.sample_id
        self.metadata_filename = "metadata/" + key + ".json"
        self.metadata_local_filename = "/tmp/" + key + ".json"

    def parse_sample_id(self):
        return self.key.split("/")[1].split(".")[0]

    def update_syncfile_timestamp(self, ts):
        try:
            response = self.s3_client.get_object(Bucket=self.bucket, Key=self.sync_filename)
            last_ts = datetime.strptime(json.loads(response["Body"].read())["last_upload"])
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "NoSuchKey":
                # object does not exist
                last_ts = datetime.fromtimestamp(0.0)
            else:
                # other error
                raise
        if ts > last_ts:
            new_upload = {"last_upload": str(ts)}
            # save to local tmp folder
            with open(self.sync_local_filename, 'w') as f:
                json.dump(new_upload, f)

            # upload to s3 bucket
            bucket = self.s3_res.Bucket(self.bucket)
            bucket.upload_file(self.sync_local_filename, self.sync_filename)

            # clean up lambda tmp folder
            os.remove(self.sync_local_filename)

    def load_metadata(self):
        """
        Given raw data object, load the metadata json file

        """
        print(self.bucket)
        print(self.key)
        response = self.s3_client.get_object(Bucket=self.bucket, Key=self.metadata_filename)
        content = response["Body"].read()
        metadata = json.loads(content)
        return metadata

    def save_metadata(self, metadata: dict):
        """
        Save new metadata of the given raw data key

        :param metadata: metadata file to save
        """
        # key = key.split("/")[1]
        # s3_filename = "metadata/" + key
        # local_filename = "/tmp/" + key

        # save to local tmp folder
        with open(self.metadata_local_filename, 'w') as f:
            json.dump(metadata, f)

        # upload to s3 bucket
        bucket = self.s3_res.Bucket(self.bucket)
        bucket.upload_file(self.metadata_local_filename, self.metadata_filename)

        # clean up lambda tmp folder
        os.remove(self.metadata_local_filename)

    def update_metadata_timestamp(self, metadata: dict, ts=time.time()):
        """
        Update the time_uploaded parameter is the metadata json as well
        as update the synchronization file with newest data upload timestamp

        :param metadata: loaded dictionary of metadata
        :param ts: timestamp, default to current time or can be specified
        """
        ts = datetime.fromtimestamp(ts)
        metadata["time_uploaded"] = str(ts)
        self.save_metadata(metadata)
        self.update_syncfile_timestamp(ts)
        return metadata
