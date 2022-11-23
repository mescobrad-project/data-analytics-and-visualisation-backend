from minio import Minio, error

from scipy import stats
import numpy as np

# Create client with access key and secret key.
new_client = Minio(
    "storage.mescobrad.digital-enabler.eng.it",
    access_key="mescobrad-user",
    secret_key="DWnqw7Bp"
)


def chk_bucket_exists(bucket_name: str):
    if new_client.bucket_exists(bucket_name):
        return 1
    else:
        return 0


def create_new_bucket(bucket_name: str):
    # Create bucket.
    if chk_bucket_exists(bucket_name) == 0:
        new_client.make_bucket(bucket_name)
        print(bucket_name + " created")
        return 1
    else:
        print(bucket_name + " exists")
        return 0


def list_existing_buckets():
    buckets = new_client.list_buckets()
    for bucket in buckets:
        print(bucket.name, bucket.creation_date)


def list_all_objects(bucket_name: str):
    objects = new_client.list_objects(bucket_name, recursive=True)
    for obj in objects:
        print(obj.content_type, obj.etag, obj.metadata)


def fget_object(bucket_name: str, object_name: str, file_location: str):
    # Download data of an object.
    # try:
        new_client.fget_object(bucket_name, object_name, file_location)
    #     print(response)
    # finally:
    #     response.close()
    #     response.release_conn()
    # File location needs also a name for the file to be downloaded



def get_data_of_object(bucket_name: str, object_name: str):
    # Get data of an object.
    try:
        response = new_client.get_object(bucket_name, object_name)
        print(response)
        # Read data from response.
    finally:
        response.close()
        response.release_conn()


def upload_object(bucket_name: str, object_name: str, file: str):
    # Upload data .
    result = new_client.fput_object(
        bucket_name, object_name, file)
    print(
        "created {0} object; etag: {1}, version-id: {2}".format(
            result.object_name, result.etag, result.version_id))


def object_stat(bucket_name: str, object_name: str):
    result = new_client.stat_object(bucket_name, object_name)
    print(
        "last-modified: {0}, size: {1}".format(
            result.last_modified, result.size,
        ),
    )

def get_saved_dataset_for_Hypothesis(bucket_name: str, object_name: str, file_location: str):
    try:
        fget_object(bucket_name, object_name, file_location)
        print("file has been downloaded")
    except Exception as exc:
        print(exc)
        print("error")

# fget_object('saved', f"{'folder01'}/test-object", 'gd_test_data/Downloaded_object.json')

# get_saved_dataset_for_Hypothesis('saved', 'FriSep302022182125.csv', 'runtime_config/FriSep302022182125.csv')

# def normal_val():
#     rng = np.random.default_rng()
#     print(rng)
#     x = stats.norm.rvs(loc=0, scale=2, size=10, random_state=rng)
#     print(x)
#
#
# a = normal_val()
