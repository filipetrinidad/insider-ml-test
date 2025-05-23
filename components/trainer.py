from tfx.v1.extensions.google_cloud_ai_platform import Trainer
from tfx.v1.proto import TrainArgs, EvalArgs
import tfx.v1 as tfx

def create_trainer(
    module_file: str,
    examples,
    schema,
    vertex_job_spec: dict,
    region: str,
    epochs: int,
    use_gpu: bool,
    train_steps: int = 100,
    eval_steps: int = 5,
) -> Trainer:
    """
    Creates and returns a Vertex AI Trainer component for TFX.

    Args:
      module_file: Path to the Python module containing your `run_fn`.
      examples: The `examples` output artifact from CsvExampleGen.
      schema: The `schema` output artifact from SchemaGen.
      vertex_job_spec: A worker pool spec dict as returned by `build_vertex_job_spec`.
      region: GCP region in which to launch the training job.
      epochs: Number of epochs to pass through the dataset.
      use_gpu: Whether to enable GPU training.
      train_steps: Number of training steps per epoch.
      eval_steps: Number of evaluation steps.

    Returns:
      A configured `Trainer` component ready to be added to your pipeline.
    """
    custom_config = {
        tfx.extensions.google_cloud_ai_platform.ENABLE_VERTEX_KEY: True,
        tfx.extensions.google_cloud_ai_platform.VERTEX_REGION_KEY: region,
        tfx.extensions.google_cloud_ai_platform.TRAINING_ARGS_KEY: vertex_job_spec,
        "epochs": epochs,
        "use_gpu": use_gpu,
    }

    return Trainer(
        module_file=module_file,
        examples=examples,
        schema=schema,
        train_args=TrainArgs(num_steps=train_steps),
        eval_args=EvalArgs(num_steps=eval_steps),
        custom_config=custom_config,
    )
