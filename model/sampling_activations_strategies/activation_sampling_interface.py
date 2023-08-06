import abc
from typing import List

import tensorflow as tf


class ActivationSamplingInterface(abc.ABC):
    """
    Abstract class for sampling activation functions.
    """

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'sample_activations') and
                callable(subclass.sample_activations))

    @classmethod
    @abc.abstractmethod
    def sample_activations(cls, model: tf.keras.Model, train_sample: List[tf.Tensor], neurons_sampled: int,
                           number_of_samples: int):
        """
        Sample the activations of a model. It must return a tensor of shape (neurons_sampled, len(train_sample)).
        :param model: tf.keras.Model Model to sample the activations from.
        :param train_sample: list of tf.Tensor A list of tensors containing the train sample.
        :param neurons_sampled: int Number of neurons to sample.
        :param number_of_samples: int Number of samples to take.
        :return: A list of tf.Tensor of shape (neurons_sampled, len(train_sample)) containing the activations
        for the different samples of the neurons in the train set of examples.
        """
        raise NotImplementedError
