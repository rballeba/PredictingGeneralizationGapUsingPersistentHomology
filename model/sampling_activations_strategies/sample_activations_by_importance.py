import collections
from collections import Counter

import numpy as np
import tensorflow as tf

from model.conf import num_skipped_layers
from model.mapped_matrix import MappedMatrix
from model.neural_networks import compute_activations
from model.sampling_activations_strategies.activation_sampling_interface import ActivationSamplingInterface


class SampleActivationsByImportance(ActivationSamplingInterface):
    @classmethod
    def sample_activations(cls, model: tf.keras.Model, train_sample: tf.Tensor, neurons_sampled: int,
                           number_of_samples: int):
        activations = compute_activations(model, train_sample, num_skipped_layers_from_start=num_skipped_layers)
        max_activation_neuron_indices = [np.argmax(np.abs(activations.array[:, idx_col]))
                                         for idx_col in range(activations.array.shape[1])]  # We only consider the
        # first occurrence of the maximum activation. However, it is almost impossible that two neurons have the same
        # activation value.
        neuron_indices_counter = Counter(max_activation_neuron_indices)
        sampled_neurons = [_sample_neurons(activations, neuron_indices_counter, neurons_sampled)
                           for _ in range(number_of_samples)]
        sampled_activations = [np.asarray(np.take(activations.array, sample, 0)) for sample in sampled_neurons]
        # Free disk memory from activations using MappedMatrix
        activations.delete_matrix()
        return sampled_activations


def _sample_neurons(activations: MappedMatrix, neuron_indices_counter: collections.Counter, neurons_sampled: int):
    total_neurons = activations.array.shape[0]
    total_counter = activations.array.shape[1]
    neurons_indices = list(range(total_neurons))
    total_counter_neurons = len(list(neuron_indices_counter))
    difference_neurons_counter = total_neurons - total_counter_neurons
    probs = np.array(list(map(
        lambda neuron_idx: _compute_probs(neuron_idx, neuron_indices_counter, difference_neurons_counter,
                                          total_counter), neurons_indices)))
    return np.random.choice(neurons_indices, size=neurons_sampled, replace=False, p=probs)


def _compute_probs(neuron_idx: int, neuron_idxs_counter: collections.Counter, difference_neurons_counter: int,
                   total_counter: int):
    if neuron_idx in neuron_idxs_counter:
        return neuron_idxs_counter[neuron_idx] / (total_counter + 1)
    else:
        return 1 / (difference_neurons_counter * (total_counter + 1))
