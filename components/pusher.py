import tfx.v1 as tfx
from tfx.v1.extensions.google_cloud_ai_platform import Pusher

def create_pusher(
    *,
    model_artifact,
    model_blessing_artifact,
    region: str,
    container_image_uri: str,
    serving_args: dict,
) -> Pusher:
    """
    Creates and returns a TFX Pusher component configured for Vertex AI Prediction.

    Args:
        model_artifact: The trained model artifact (e.g. trainer.outputs['model']).
        model_blessing_artifact: The blessing artifact from the Evaluator
                                 (e.g. evaluator.outputs['blessing']).
        region: GCP region where Vertex AI is running.
        container_image_uri: URI of the serving container image.
        serving_args: Dict of serving arguments (e.g. endpoint name, machine type,
                      accelerator settings).

    Returns:
        A configured Pusher component.
    """
    return Pusher(
        model=model_artifact,
        model_blessing=model_blessing_artifact,
        custom_config={
            tfx.extensions.google_cloud_ai_platform.ENABLE_VERTEX_KEY: True,
            tfx.extensions.google_cloud_ai_platform.VERTEX_REGION_KEY: region,
            tfx.extensions.google_cloud_ai_platform.VERTEX_CONTAINER_IMAGE_URI_KEY: container_image_uri,
            tfx.extensions.google_cloud_ai_platform.SERVING_ARGS_KEY: serving_args,
        }
    )
