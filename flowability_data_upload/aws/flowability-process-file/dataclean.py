"""Clean Data

These functions clean the raw Microtrac data. This includes removing non utf-8
charcters to allow compatability with Pandas, additionally the metadata
(UUID, Name, Flowability) are added to the dataset at the particle level.

"""
import csv
import os
from s3integrity import S3Integrity


class DataCleaner:

    def __init__(self, integrity: S3Integrity):
        self.s3i = integrity

    def clean_data(self, metadata: dict):
        """
        load raw data, clean it, label with metadata
        :param metadata: loaded metadata from s3
        :return: location in s3 where file is located
        """
        data = self.load_clean_raw_data()
        labeled_data = self.label_data(data, metadata)
        clean_data_location = self.save_clean_data(labeled_data)
        return clean_data_location

    def load_clean_raw_data(self):
        """
        Load the uploaded raw_data and remove the non utf-8 second row.
        """
        # fetch file from s3
        response = self.s3i.s3_client.get_object(Bucket=self.s3i.bucket, Key=self.s3i.key)
        content = response["Body"].iter_lines()

        data = list()
        for line_count, row in enumerate(content):
            if line_count != 1:
                data.append(row.decode("utf-8"))

        # skip the second line( non utf-8 symbol)
        reader = csv.reader(data, delimiter="\t")
        for line_count, row in enumerate(reader):
                data[line_count] = row

        return data

    def label_data(self, data: list, metadata: dict):
        """Given the data and the metadata append the metadata info to each line
        of the data

        :param data: microtrac data
        :param metadata: metadata to append to each row of the microtrac data
        """
        flowability = metadata['flowability']
        name = metadata['name']
        sample_id = self.s3i.sample_id

        for line, row in enumerate(data):
            if line == 0:
                data[line] = ["sample_id", "name"] + row + ["flowability"]
            else:
                data[line] = [sample_id, name] + row + [flowability]
        return data

    def save_clean_data(self, data: list):

        # save to local tmp folder
        with open(self.s3i.clean_data_local_filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(data)

        # upload to s3 bucket
        bucket = self.s3i.s3_res.Bucket(self.s3i.bucket)
        bucket.upload_file(self.s3i.clean_data_local_filename, self.s3i.clean_data_filename)

        # clean up lambda tmp folder
        os.remove(self.s3i.clean_data_local_filename)
        return self.s3i.clean_data_filename
