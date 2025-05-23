from tfx.v1.components import Evaluator
import tensorflow_model_analysis as tfma


def create_evaluator(
    examples,
    model,
    eval_config
) -> Evaluator:
    """
    Creates and returns a TFX Evaluator component.

    Args:
      examples_artifact:   The Examples channel, e.g. example_gen.outputs['examples'].
      model_artifact:      The trained model channel, e.g. trainer.outputs['model'].
      eval_config:         A tfma.EvalConfig instance defining
                           model_specs, metrics_specs, slicing_specs, etc.

    Returns:
      A configured Evaluator component ready to plug into your pipeline.
    """
    return Evaluator(
        examples=examples,
        model=model,
        eval_config=eval_config,
    )
