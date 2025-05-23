import sys
import os

from tfx.v1.orchestration.experimental import (
    KubeflowV2DagRunner,
    KubeflowV2DagRunnerConfig,
)
import tfx.v1 as tfx
from google.cloud import aiplatform
from google.cloud.aiplatform import pipeline_jobs

from settings import (
    GOOGLE_CLOUD_REGION,
    GOOGLE_CLOUD_PROJECT,
    PIPELINE_ROOT,
    DATA_ROOT,
    PIPELINE_NAME,
    ENDPOINT_NAME,
    VERTEX_TENSORBOARD,
    SERVICE_ACCOUNT,
    OUTPUT_PREFIX,
    EPOCHS
)

from pipeline.pipeline import create_pipeline

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)


PIPELINE_DEFINITION_FILE = PIPELINE_NAME + '_pipeline.json'
LOCAL_MODULE_FILE = os.path.join(os.path.dirname(__file__), 'src', 'insider_trainer.py')


if __name__ == '__main__':
    runner_config = KubeflowV2DagRunnerConfig(
        default_image=f"gcr.io/tfx-oss-public/tfx:{tfx.__version__}"
    )
    runner = KubeflowV2DagRunner(
        config=runner_config,
        output_filename=PIPELINE_DEFINITION_FILE,
    )
    runner.run(
        create_pipeline(
            pipeline_name=PIPELINE_NAME,
            pipeline_root=PIPELINE_ROOT,
            data_root=DATA_ROOT,
            module_file=LOCAL_MODULE_FILE,
            endpoint_name=ENDPOINT_NAME,
            project_id=GOOGLE_CLOUD_PROJECT,
            region=GOOGLE_CLOUD_REGION,
            tensorboard_vertex=VERTEX_TENSORBOARD,
            service_account=SERVICE_ACCOUNT,
            output_tb=OUTPUT_PREFIX,
            epochs=EPOCHS,
            use_gpu=False,
        )
    )

    aiplatform.init(
        project=GOOGLE_CLOUD_PROJECT,
        location=GOOGLE_CLOUD_REGION,
    )
    job = pipeline_jobs.PipelineJob(
        template_path=PIPELINE_DEFINITION_FILE,
        display_name=PIPELINE_NAME,
    )
    job.submit()