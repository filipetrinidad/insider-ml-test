from google.cloud import aiplatform
import numpy as np

from settings.settings import (
    GOOGLE_CLOUD_REGION,
    GOOGLE_CLOUD_PROJECT,
    ENDPOINT_ID
)


client_options = {
    'api_endpoint': GOOGLE_CLOUD_REGION + '-aiplatform.googleapis.com'
    }

client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)

instances = [{
    'culmen_length_mm':[0.71],
    'culmen_depth_mm':[0.38],
    'flipper_length_mm':[0.98],
    'body_mass_g': [0.78],
}]

endpoint = client.endpoint_path(
    project=GOOGLE_CLOUD_PROJECT,
    location=GOOGLE_CLOUD_REGION,
    endpoint=ENDPOINT_ID,
)

response = client.predict(endpoint=endpoint, instances=instances)

print('species:', np.argmax(response.predictions[0]))