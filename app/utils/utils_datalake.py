import os
from urllib import request
from xml.etree import ElementTree

from dotenv import load_dotenv
import requests
from minio import Minio, error

from scipy import stats
import numpy as np

# Create client with access key and secret key.
load_dotenv()

# new_client = Minio(
#     "storage.mescobrad.digital-enabler.eng.it",
#     access_key=os.getenv("MINIO_ACCESS_KEY"),
#     secret_key=os.getenv("MINIO_SECRET_KEY"),
#     # session_token=session_token
# )


def chk_bucket_exists(bucket_name: str, session_token: str):
    """ NOT USED """
    # Create new client
    new_client = Minio(
        "storage.mescobrad.digital-enabler.eng.it",
        access_key=os.getenv("MINIO_ACCESS_KEY"),
        secret_key=os.getenv("MINIO_SECRET_KEY"),
        session_token= session_token
    )
    if new_client.bucket_exists(bucket_name):
        return 1
    else:
        return 0


def create_new_bucket(bucket_name: str, session_token: str):
    """ NOT USED """
    # Create new client
    new_client = Minio(
        "storage.mescobrad.digital-enabler.eng.it",
        access_key=os.getenv("MINIO_ACCESS_KEY"),
        secret_key=os.getenv("MINIO_SECRET_KEY"),
        session_token=session_token
    )

    # Create bucket.
    if new_client.bucket_exists(bucket_name):
        print(bucket_name + " exists")
        return 0
    else:
        new_client.make_bucket(bucket_name)
        print(bucket_name + " created")
        return 1

    # if chk_bucket_exists(bucket_name) == 0:
    #     new_client.make_bucket(bucket_name)
    #     print(bucket_name + " created")
    #     return 1
    # else:
    #
    #     return 0


def list_existing_buckets(session_token: str):
    """ NOT USED """
    # Create new client
    new_client = Minio(
        "storage.mescobrad.digital-enabler.eng.it",
        access_key=os.getenv("MINIO_ACCESS_KEY"),
        secret_key=os.getenv("MINIO_SECRET_KEY"),
        session_token=session_token
    )

    buckets = new_client.list_buckets()
    for bucket in buckets:
        print(bucket.name, bucket.creation_date)


def list_all_objects(bucket_name: str, session_token: str):
    """ NOT USED """
    # Create new client
    new_client = Minio(
        "storage.mescobrad.digital-enabler.eng.it",
        access_key=os.getenv("MINIO_ACCESS_KEY"),
        secret_key=os.getenv("MINIO_SECRET_KEY"),
        session_token=session_token
    )

    objects = new_client.list_objects(bucket_name, recursive=True)
    for obj in objects:
        print(obj.content_type, obj.etag, obj.metadata)


def fget_object(bucket_name: str, object_name: str, file_location: str, session_token: str):
    # Create new client
    # session_token = ""
    # print("Info about request download data")
    # print(session_token)
    # print(os.getenv("MINIO_ACCESS_KEY"),)
    # print(os.getenv("MINIO_SECRET_KEY"))
    minio_url = "https://storage.mescobrad.digital-enabler.eng.it"
    minio_data = {
        "Action": "AssumeRoleWithWebIdentity",
        "Version": "2011-06-15",
        "WebIdentityToken":  session_token }
    # print("Request user data")
    try:
        response = requests.post(minio_url, data=minio_data)
    except Exception as e:
        print(e)
    xml_data = ElementTree.fromstring(response.text)
    # print("Request user data END")
    # print(response.content)
    # Step 2: Parse the output to extract the credentials
    access_key = xml_data.find('.//{https://sts.amazonaws.com/doc/2011-06-15/}AccessKeyId').text
    secret_access_key = xml_data.find('.//{https://sts.amazonaws.com/doc/2011-06-15/}SecretAccessKey').text
    session_token_send = xml_data.find('.//{https://sts.amazonaws.com/doc/2011-06-15/}SessionToken').text

    # print("USER INFORMATION")
    # print(access_key)
    # print(secret_access_key)
    # print(session_token_send)
    new_client_1 = Minio(
        "storage.mescobrad.digital-enabler.eng.it",
        access_key=access_key,
        # access_key=os.getenv("MINIO_ACCESS_KEY"),
        secret_key=secret_access_key,
        # secret_key=os.getenv("MINIO_SECRET_KEY"),
        session_token=session_token_send
    )
    # Download data of an object.
    # try:
    new_client_1.fget_object(bucket_name, object_name, file_location)
    #     print(response)
    # finally:
    #     response.close()
    #     response.release_conn()
    # File location needs also a name for the file to be downloaded



def get_data_of_object(bucket_name: str, object_name: str, session_token: str):
    """ NOT USED """
    # Create new client
    new_client = Minio(
        "storage.mescobrad.digital-enabler.eng.it",
        access_key=os.getenv("MINIO_ACCESS_KEY"),
        secret_key=os.getenv("MINIO_SECRET_KEY"),
        session_token=session_token
    )
    # Get data of an object.
    try:
        response = new_client.get_object(bucket_name, object_name)
        print(response)
        # Read data from response.
    finally:
        response.close()
        response.release_conn()


def upload_object(bucket_name: str, object_name: str, file: str, session_token: str):
    print("Starting upload")
    print(session_token)
    # Create new client
    minio_url = "https://storage.mescobrad.digital-enabler.eng.it"
    minio_data = {
        "Action": "AssumeRoleWithWebIdentity",
        "Version": "2011-06-15",
        "WebIdentityToken": session_token}
    print("Request user data")
    try:
        response = requests.post(minio_url, data=minio_data)
    except Exception as e:
        print(e)
    xml_data = ElementTree.fromstring(response.text)
    print("Request user data END")
    print(response.content)
    # Step 2: Parse the output to extract the credentials
    access_key = xml_data.find('.//{https://sts.amazonaws.com/doc/2011-06-15/}AccessKeyId').text
    secret_access_key = xml_data.find('.//{https://sts.amazonaws.com/doc/2011-06-15/}SecretAccessKey').text
    session_token_send = xml_data.find('.//{https://sts.amazonaws.com/doc/2011-06-15/}SessionToken').text

    print("USER INFORMATION")
    print(access_key)
    print(secret_access_key)
    print(session_token_send)
    new_client_1 = Minio(
        "storage.mescobrad.digital-enabler.eng.it",
        access_key=access_key,
        # access_key=os.getenv("MINIO_ACCESS_KEY"),
        secret_key=secret_access_key,
        # secret_key=os.getenv("MINIO_SECRET_KEY"),
        session_token=session_token_send
    )

    # Upload data .
    try:
        # print("Result DATA PRINT HERE NOW")
        result = new_client_1.fput_object(bucket_name, object_name, file)
        # print("Result DATA PRINT HERE NOW")
        print(result)

        print(
            "created {0} object; etag: {1}, version-id: {2}".format(
                result.object_name, result.etag, result.version_id))
    except Exception as exc:
        print(exc)
        print("error")

def object_stat(bucket_name: str, object_name: str, session_token: str):
    """ NOT USED """
    # Create new client
    new_client = Minio(
        "storage.mescobrad.digital-enabler.eng.it",
        access_key=os.getenv("MINIO_ACCESS_KEY"),
        secret_key=os.getenv("MINIO_SECRET_KEY"),
        session_token=session_token
    )

    result = new_client.stat_object(bucket_name, object_name)
    print(
        "last-modified: {0}, size: {1}".format(
            result.last_modified, result.size,
        ),
    )

def get_saved_dataset_for_Hypothesis(bucket_name: str, object_name: str, file_location: str, session_token: str):
    try:
        fget_object(bucket_name, object_name, file_location, session_token)
        print("file has been downloaded")
    except Exception as exc:
        print(exc)
        print("error")

def get_saved_mri_files(bucket_name: str, object_name: str, file_location: str):
    """ NOT USED """
    try:
        fget_object(bucket_name, object_name, file_location)
        print("file has been downloaded")
    except Exception as exc:
        print(exc)
        print("error")

# fget_object('saved', f"{'folder01'}/test-object", 'gd_test_data/Downloaded_object.json')

# fget_object('demo', "expertsystem/workflow/3fa85f64-5717-4562-b3fc-2c963f66afa6/3fa85f64-5717-4562-b3fc-2c963f66afa6/3fa85f64-5717-4562-b3fc-2c963f66afa6/mescobrad_dataset.csv", 'gd_test_data/Downloaded_object.json')
# list_all_objects("demo")
# get_saved_dataset_for_Hypothesis('saved', 'FriSep302022182125.csv', 'runtime_config/FriSep302022182125.csv')
# fget_object("demo","expertsystem/workflow/3fa85f64-5717-4562-b3fc-2c963f66afa6/3fa85f64-5717-4562-b3fc-2c963f66afa6/3fa85f64-5717-4562-b3fc-2c963f66afa6/mescobrad_dataset.csv", 'gd_test_data/Downloaded_object.json')
# def normal_val():
#     rng = np.random.default_rng()
#     print(rng)
#     x = stats.norm.rvs(loc=0, scale=2, size=10, random_state=rng)
#     print(x)
#
#
# a = normal_val()
