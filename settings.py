import os
import tensorflow as tf

GOOGLE_CLOUD_PROJECT = 'ml-insider-test'
GOOGLE_CLOUD_REGION = 'us-central1'
GCS_BUCKET_NAME = 'insider-test'
SERVICE_ACCOUNT = "913507232607-compute@developer.gserviceaccount.com"


EPOCHS = 10

OUTPUT_PREFIX   = f"gs://{GCS_BUCKET_NAME}/vertex-training/{GOOGLE_CLOUD_PROJECT }"

VERTEX_TENSORBOARD = (
    f"projects/{GOOGLE_CLOUD_PROJECT}/locations/{GOOGLE_CLOUD_REGION}"
    "/tensorboards/6450522458859503616"
)

PIPELINE_NAME = 'insider-vertex-training'

PIPELINE_ROOT = 'gs://{}/pipeline_root/{}'.format(GCS_BUCKET_NAME, PIPELINE_NAME)

MODULE_ROOT = 'gs://{}/pipeline_module/{}'.format(GCS_BUCKET_NAME, PIPELINE_NAME)

DATA_ROOT = 'gs://insider-test/data/'

ENDPOINT_NAME = 'prediction-' + PIPELINE_NAME



