import abc

import numpy as np


class DistanceFunctionInterface(abc.ABC):
    """
    Abstract class for sampling activation functions.
    """

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'compute_distance_matrix') and
                callable(subclass.compute_distance_matrix))

    @classmethod
    @abc.abstractmethod
    def compute_distance_matrix(cls, activations:np.array):
        """
        Compute the distance matrix of a set of activations.
        :param activations: np.array of shape (number_of_neurons, number_of_examples) containing the activations of a
        set of neurons.
        """
        raise NotImplementedError
