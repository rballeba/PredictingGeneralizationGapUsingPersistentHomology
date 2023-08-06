import numpy as np
import tensorflow as tf

from model.mapped_matrix import MappedMatrix


def compute_activations(model: tf.keras.Model, train_sample: tf.Tensor,
                        num_skipped_layers_from_start: int = 0):
    """
    Compute the activations of a model for neurons with trainable parameters. It considers neurons from all layers
    except the first num_skipped_layers_from_start layers.

    It must return a tensor of shape (number_of_neurons_model, len(train_sample)).
    :param model: tf.keras.Model Model to sample the activations from.
    :param train_sample: tf.Tensor A tensor containing the train sample.
    :param num_skipped_layers_from_start: int Number of layers to skip from the start.
    :return: MappedMatrix of shape (number_of_neurons_considered, len(train_sample))
    containing the activations of considered neurons in the train sample.
    """
    examples_x_activations = _examples_x_activations_for_input(model, train_sample, num_skipped_layers_from_start)
    activations_x_examples = examples_x_activations.transpose()
    return activations_x_examples


def _examples_x_activations_for_input(model: tf.keras.Model, x: tf.Tensor, num_skipped_layers_from_start: int):
    first_layer = True
    skipped_iterations = 0
    for layer in model.layers:
        x = layer(x)
        if skipped_iterations < num_skipped_layers_from_start:
            skipped_iterations += 1
        else:
            if len(layer.get_weights()) > 0:
                examples_x_neurons = np.reshape(np.copy(x.numpy()), newshape=(x.shape[0], -1))
                if first_layer:
                    activations_bd = MappedMatrix(array=examples_x_neurons)
                    first_layer = False
                else:
                    activations_bd.concatenate(examples_x_neurons)
    return activations_bd
