import json
import os

import tensorflow as tf
from model.pgdl_data.pgdl_structure import get_task_data_location
from tensorflow.keras import Sequential


def get_model_names(data_location: str):
    model_names = [name for name in os.listdir(data_location) if os.path.isdir(os.path.join(data_location, name))
                   and name[:5] == 'model']
    return model_names


def get_model_names_by_task(pgdl_folderpath: str, task_number: int):
    data_location = get_task_data_location(pgdl_folderpath, task_number)
    return get_model_names(data_location)


def get_model_names_by_task_and_model(pgdl_folderpath: str, task_number: int, model_name: str):
    data_location = get_task_data_location(pgdl_folderpath, task_number)
    model_folder = os.path.join(data_location, model_name)
    return get_model_names(model_folder)


def get_model_by_task_and_model_name(pgdl_folderpath: str, task_number: int, model_name: str):
    data_location = get_task_data_location(pgdl_folderpath, task_number)
    model_folder = os.path.join(data_location, model_name)
    return get_model(model_folder)


def get_model(model_location: str):
    absolute_model_path = os.path.abspath(model_location)
    config_path = os.path.join(absolute_model_path, 'config.json')
    weights_path = os.path.join(model_location, 'weights.hdf5')
    initial_weights_path = os.path.join(model_location, 'weights_init.hdf5')
    model_instance = _create_model_instance(config_path)
    try:
        _load_initial_weights_if_exist(model_instance, initial_weights_path)
    except ValueError as e:
        print('Error while loading initial weights of {} from {}'.format(model_instance, initial_weights_path))
        exit(1)
    model_instance.load_weights(weights_path)
    return model_instance


def _load_model(config):
    model_instance = _model_def_to_keras_sequential(config['model_config'])
    model_instance.build([0] + config['input_shape'])
    return model_instance


def _create_model_instance(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return _load_model(config)


def _load_initial_weights_if_exist(model_instance, initial_weights_path):
    if os.path.exists(initial_weights_path):
        model_instance.load_weights(initial_weights_path)
        model_instance.initial_weights = model_instance.get_weights()


def _create_model_instance(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return _load_model(config)


def _model_def_to_keras_sequential(model_def):
    """Convert a model json to a Keras Sequential model.

    Args:
        model_def: A list of dictionaries, where each dict describes a layer to add
            to the model.

    Returns:
        A Keras Sequential model with the required architecture.
    """

    def _cast_to_integer_if_possible(dct):
        dct = dict(dct)
        for k, v in dct.items():
            if isinstance(v, float) and v.is_integer():
                dct[k] = int(v)
        return dct

    def parse_layer(layer_def):
        layer_cls = getattr(tf.keras.layers, layer_def['layer_name'])
        # layer_cls = wrap_layer(layer_cls)
        kwargs = dict(layer_def)
        del kwargs['layer_name']
        return _wrap_layer(layer_cls, **_cast_to_integer_if_possible(kwargs))
        # return layer_cls(**_cast_to_integer_if_possible(kwargs))

    return Sequential([parse_layer(l) for l in model_def])


def _wrap_layer(layer_cls, *args, **kwargs):
    """Wraps a layer for computing the jacobian wrt to intermediate layers."""

    class wrapped_layer(layer_cls):
        def __call__(self, x, *args, **kwargs):
            self._last_seen_input = x
            return super(wrapped_layer, self).__call__(x, *args, **kwargs)
    return wrapped_layer(*args, **kwargs)
