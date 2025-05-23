from google.cloud import aiplatform
import numpy as np

from settings import (
    GOOGLE_CLOUD_REGION,
    GOOGLE_CLOUD_PROJECT,
)

ENDPOINT_ID = "3133393734394183680"

print(f"--- Informações de Depuração ---")
print(f"GOOGLE_CLOUD_PROJECT: {GOOGLE_CLOUD_PROJECT}")
print(f"GOOGLE_CLOUD_REGION: {GOOGLE_CLOUD_REGION}")
print(f"ENDPOINT_ID: {ENDPOINT_ID}")

client_options = {
    'api_endpoint': f'{GOOGLE_CLOUD_REGION}-aiplatform.googleapis.com'
}
client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)

instances = [
    {
        'pclass': [3.0],
        'age': [22.0],
        'parch': [0.0],
        'fare': [7.25],
        'sex': [1],
    },
]

endpoint_path_str = client.endpoint_path(
    project=GOOGLE_CLOUD_PROJECT,
    location=GOOGLE_CLOUD_REGION,
    endpoint=ENDPOINT_ID,
)
print(f"Caminho do Endpoint Construído: {endpoint_path_str}")
print(f"--- Fim das Informações de Depuração ---")


response = client.predict(endpoint=endpoint_path_str, instances=instances)

for i, prediction in enumerate(response.predictions):
    probs = np.exp(prediction) / np.sum(np.exp(prediction))
    pred_class = int(np.argmax(probs))
    print(f'Instância {i}:')
    print(f'  logits  = {prediction}')
    print(f'  prob. 0 (não sobreviveu) = {probs[0]:.3f}')
    print(f'  prob. 1 (sobreviveu)     = {probs[1]:.3f}')
    print(f'  previsão final (0/1)     = {pred_class}')
    print('---')