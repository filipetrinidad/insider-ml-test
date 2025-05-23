# utils/pipeline_components.py
from tfx.v1.components import SchemaGen

def create_schema(statistics):
    """
    Creates and returns a SchemaGen component using the provided statistics.

    Args:
      statistics_artifact: The statistics artifact produced by StatisticsGen,
                           typically `statistics.outputs['statistics']`.

    Returns:
      A SchemaGen component instance ready to be added to the pipeline.
    """
    return SchemaGen(statistics=statistics)
