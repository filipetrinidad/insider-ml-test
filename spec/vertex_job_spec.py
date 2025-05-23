import json
import tfx.v1 as tfx

def build_vertex_job_spec(
    *,
    project_id: str,
    tensorboard_uri: str,
    service_account: str,
    output_prefix: str,
    machine_type: str = "n1-standard-4",
    use_gpu: bool = False,
    gpu_type: str = "NVIDIA_TESLA_K80",
    gpu_count: int = 1,
    as_json: bool = False,
):
    """
    Builds and returns the vertex_job_spec required by the TFX Trainer.

    Args:
      project_id:        GCP project ID.
      tensorboard_uri:   URI of the Vertex TensorBoard instance.
      service_account:   Service account to run the training job.
      output_prefix:     GCS path prefix where job outputs will be stored.
      machine_type:      VM machine type (default: "n1-standard-4").
      use_gpu:           If True, adds GPU accelerators to the machine spec.
      gpu_type:          Type of GPU to attach (default: "NVIDIA_TESLA_K80").
      gpu_count:         Number of GPUs to attach (default: 1).
      as_json:           If True, returns the spec as a formatted JSON string;
                         otherwise returns a Python dict.

    Returns:
      A Python dict (or JSON string, if as_json=True) containing the
      Vertex AI training job configuration.
    """
    spec = {
        "project": project_id,
        "tensorboard": tensorboard_uri,
        "service_account": service_account,
        "base_output_directory": {"output_uri_prefix": output_prefix},
        "worker_pool_specs": [{
            "machine_spec": {"machine_type": machine_type},
            "replica_count": 1,
            "container_spec": {
                "image_uri": f"gcr.io/tfx-oss-public/tfx:{tfx.__version__}"
            },
        }],
    }

    if use_gpu:
        spec["worker_pool_specs"][0]["machine_spec"].update({
            "accelerator_type": gpu_type,
            "accelerator_count": gpu_count,
        })

    return json.dumps(spec, indent=2) if as_json else spec
