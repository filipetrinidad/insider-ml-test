import os
from typing import List

import tensorflow as tf
from tensorflow import keras
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_transform.tf_metadata import schema_utils
from tfx import v1 as tfx
from tfx_bsl.public import tfxio
from absl import logging


_FLOAT_FEATURE_KEYS = ['pclass', 'age', 'parch', 'fare']
_INT_FEATURE_KEYS   = ['sex']
_LABEL_KEY          = 'survived'

_TRAIN_BATCH_SIZE = 20
_EVAL_BATCH_SIZE  = 10

_FEATURE_SPEC = {
    **{f: tf.io.FixedLenFeature([1], tf.float32) for f in _FLOAT_FEATURE_KEYS},
    **{f: tf.io.FixedLenFeature([1], tf.int64)   for f in _INT_FEATURE_KEYS},
    _LABEL_KEY: tf.io.FixedLenFeature([1], tf.int64),
}


def _input_fn(
    file_pattern: List[str],
    data_accessor: tfx.components.DataAccessor,
    schema: schema_pb2.Schema,
    batch_size: int
) -> tf.data.Dataset:
    """
    Generates a tf.data.Dataset for training or evaluation.

    Args:
      file_pattern:    List of TFRecord filenames or patterns.
      data_accessor:   TFX DataAccessor used to read the records.
      schema:          A schema_pb2.Schema describing the features.
      batch_size:      Number of examples per batch.

    Returns:
      A Dataset of (features_dict, label_tensor) tuples, repeated indefinitely.
    """
    return data_accessor.tf_dataset_factory(
        file_pattern,
        tfxio.TensorFlowDatasetOptions(
            batch_size=batch_size,
            label_key=_LABEL_KEY
        ),
        schema=schema
    ).repeat()


def _make_keras_model() -> tf.keras.Model:
    """
    Builds and compiles the Keras model.

    • Inputs: one scalar tensor per feature in _FLOAT_FEATURE_KEYS and _INT_FEATURE_KEYS.
    • Casts any int features to float32.
    • Two hidden Dense layers of size 8 (ReLU), then a Dense(2) for logits.

    Returns:
      A compiled tf.keras.Model ready for train/eval.
    """
    raw_inputs, encoded = [], []

    for f in _FLOAT_FEATURE_KEYS:
        inp = keras.layers.Input(shape=(1,), name=f, dtype='float32')
        raw_inputs.append(inp)
        encoded.append(inp)

    sex_in = keras.layers.Input(shape=(1,), name='sex', dtype='int64')
    raw_inputs.append(sex_in)
    encoded.append(
        keras.layers.Lambda(lambda x: tf.cast(x, tf.float32))(sex_in)
    )

    x = keras.layers.concatenate(encoded)
    x = keras.layers.Dense(8, activation='relu')(x)
    x = keras.layers.Dense(8, activation='relu')(x)
    outputs = keras.layers.Dense(2)(x)

    model = keras.Model(inputs=raw_inputs, outputs=outputs)
    model.compile(
        optimizer='adam',
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['sparse_categorical_accuracy'],
    )
    return model


def _get_distribution_strategy(fn_args: tfx.components.FnArgs):
    """
    Chooses a TF distribution strategy based on custom_config.

    Args:
      fn_args:  The FnArgs object passed to run_fn, which includes custom_config.

    Returns:
      A tf.distribute.Strategy (MirroredStrategy) if fn_args.custom_config['use_gpu']
      is True, otherwise None.
    """
    if fn_args.custom_config.get('use_gpu', False):
        logging.info('Using MirroredStrategy with one GPU.')
        return tf.distribute.MirroredStrategy(devices=['/device:GPU:0'])
    return None


def run_fn(fn_args: tfx.components.FnArgs):
    """
    Entry point for TFX Trainer component. Builds, trains, and exports the model.

    Args:
      fn_args.train_files:       List of training file patterns.
      fn_args.eval_files:        List of eval file patterns.
      fn_args.data_accessor:     DataAccessor for reading input.
      fn_args.train_steps:       Number of steps per epoch.
      fn_args.eval_steps:        Number of eval steps.
      fn_args.serving_model_dir: Directory to export the SavedModel.
      fn_args.custom_config:     Dict; supports:
         - 'epochs' (int): number of training epochs.
         - 'use_gpu' (bool): whether to enable GPU strategy.

    Behavior:
      1. Builds train and eval Datasets via _input_fn.
      2. Creates the model in a strategy scope if needed.
      3. Trains for the given number of epochs/steps.
      4. Writes the SavedModel to fn_args.serving_model_dir.
    """
    epochs = fn_args.custom_config.get('epochs', 1)
    schema = schema_utils.schema_from_feature_spec(_FEATURE_SPEC)

    train_ds = _input_fn(
        fn_args.train_files,
        fn_args.data_accessor,
        schema,
        _TRAIN_BATCH_SIZE
    )
    eval_ds = _input_fn(
        fn_args.eval_files,
        fn_args.data_accessor,
        schema,
        _EVAL_BATCH_SIZE
    )

    strategy = _get_distribution_strategy(fn_args)
    if strategy:
        with strategy.scope():
            model = _make_keras_model()
    else:
        model = _make_keras_model()

    tb_callback = tf.keras.callbacks.TensorBoard(
        log_dir=os.environ.get('AIP_TENSORBOARD_LOG_DIR', '/tmp/tb'),
        histogram_freq=1,
    )

    model.fit(
        train_ds,
        epochs=epochs,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_ds,
        validation_steps=fn_args.eval_steps,
        callbacks=[tb_callback],
    )

    model.export(fn_args.serving_model_dir)