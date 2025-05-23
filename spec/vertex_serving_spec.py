import json

def build_vertex_serving_spec(
    *,
    project_id: str,
    endpoint_name: str,
    machine_type: str = "n1-standard-4",
    use_gpu: bool = False,
    gpu_type: str = "NVIDIA_TESLA_K80",
    gpu_count: int = 1,
    as_json: bool = False,
):
    """
    Builds and returns the Vertex AI serving spec (and selects the correct container image)
    required by the TFX Pusher.

    Args:
      project_id:     GCP project ID.
      endpoint_name:  Name of the Vertex AI Endpoint to deploy to.
      machine_type:   VM machine type for serving (default: "n1-standard-4").
      use_gpu:        Whether to attach GPU accelerators.
      gpu_type:       Type of GPU to attach (default: "NVIDIA_TESLA_K80").
      gpu_count:      Number of GPUs to attach (default: 1).
      as_json:        If True, return the spec as a formatted JSON string; otherwise a dict.

    Returns:
      If as_json=False (default), returns a tuple:
        (serving_spec: dict, serving_image: str)
      If as_json=True, returns a JSON string with two top-level keys:
        {
          "serving_spec": { … },
          "serving_image": "<container image URI>"
        }
    """
    spec = {
        "project_id": project_id,
        "endpoint_name": endpoint_name,
        "machine_type": machine_type,
    }

    serving_image = "us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-6:latest"

    if use_gpu:
        spec.update({
            "accelerator_type": gpu_type,
            "accelerator_count": gpu_count,
        })
        serving_image = "us-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-6:latest"

    if as_json:
        return json.dumps(
            {"serving_spec": spec, "serving_image": serving_image},
            indent=2
        )
    else:
        return spec, serving_image
