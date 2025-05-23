from tfx.v1.components import StatisticsGen

def create_statistics_gen(
    examples
) -> StatisticsGen:
    """
    Creates and returns a StatisticsGen component using the given ExampleGen output.
    
    Args:
      examples_artifact: The artifact produced by an ExampleGen component,
                         typically `example_gen.outputs['examples']`.
    
    Returns:
      A StatisticsGen component ready to be added to your TFX pipeline.
    """
    return StatisticsGen(examples=examples)

