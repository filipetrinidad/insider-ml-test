import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tfx.v1.dsl import Pipeline
import tensorflow_model_analysis as tfma

from components.csv_example_gen import create_csv_example_gen
from components.statistics_gen import create_statistics_gen
from components.schema_gen import create_schema
from spec.vertex_job_spec import build_vertex_job_spec
from components.trainer import create_trainer
from spec.vertex_serving_spec import build_vertex_serving_spec
from components.pusher import create_pusher
from components.evaluator import create_evaluator

eval_config = tfma.EvalConfig(
    model_specs=[tfma.ModelSpec(label_key='survived')],
    metrics_specs=[
        tfma.MetricsSpec(
            metrics=[
                tfma.MetricConfig(
                    class_name='SparseCategoricalAccuracy',
                    threshold=tfma.MetricThreshold(
                        value_threshold=tfma.GenericValueThreshold(
                            lower_bound={'value': 0.75}),
                    )
                )
            ]
        )
    ],
    slicing_specs=[tfma.SlicingSpec()]
)

def create_pipeline(
    pipeline_name: str,
    pipeline_root: str,
    data_root: str,
    module_file: str,
    endpoint_name: str,
    project_id: str,
    region: str,
    tensorboard_vertex: str,
    service_account: str,
    output_tb: str,
    epochs: int,
    use_gpu: bool,
) -> Pipeline:
    """
    Constructs and returns a TFX Pipeline with all core components wired up:
    CSV ingestion, statistics, schema inference, training, evaluation, and push.

    Args:
      pipeline_name:         The name to assign to the pipeline.
      pipeline_root:         GCS or local root directory for pipeline outputs.
      data_root:             Path to the input CSV data.
      module_file:           Path to the Trainer module file.
      endpoint_name:         Vertex AI endpoint for serving.
      project_id:            GCP project ID for Vertex AI.
      region:                GCP region for Vertex AI services.
      tensorboard_vertex:    URI of the Vertex TensorBoard instance.
      service_account:       Service account to run training and serving jobs.
      output_tb:             GCS prefix where TensorBoard logs are written.
      use_gpu:               Whether to enable GPU acceleration.

    Returns:
      A fully configured TFX Pipeline object.
    """
    example_gen = create_csv_example_gen(input_base=data_root)

    statistics = create_statistics_gen(examples=example_gen.outputs['examples'])

    schema = create_schema(statistics=statistics.outputs['statistics'])

    vertex_job_spec = build_vertex_job_spec(
        project_id=project_id,
        tensorboard_uri=tensorboard_vertex,
        service_account=service_account,
        output_prefix=output_tb,
        machine_type="n1-standard-4",
        use_gpu=use_gpu,
    )

    trainer = create_trainer(
        module_file=module_file,
        examples=example_gen.outputs['examples'],
        schema=schema.outputs['schema'],
        vertex_job_spec=vertex_job_spec,
        region=region,
        epochs=epochs,
        use_gpu=use_gpu,
    )

    evaluator = create_evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        eval_config=eval_config
    )

    vertex_serving_spec, serving_image = build_vertex_serving_spec(
        project_id=project_id,
        endpoint_name=endpoint_name,
        use_gpu=use_gpu,
        gpu_type="NVIDIA_TESLA_K80",
        gpu_count=1
    )

    pusher = create_pusher(
        model_artifact=trainer.outputs['model'],
        model_blessing_artifact=evaluator.outputs['blessing'],
        region=region,
        container_image_uri=serving_image,
        serving_args=vertex_serving_spec,
    )

    return Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=[
            example_gen,
            statistics,
            schema,
            trainer,
            evaluator,
            pusher,
        ],
    )