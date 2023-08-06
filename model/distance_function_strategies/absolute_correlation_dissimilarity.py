import numpy as np

from model.distance_function_strategies.distance_function_interface import DistanceFunctionInterface


class AbsoluteCorrelationDissimilarity(DistanceFunctionInterface):
    @classmethod
    def compute_distance_matrix(cls, activations: np.array):
        absolute_correlations = np.abs(np.corrcoef(activations))
        # Due to floating point errors, some correlations may be slightly greater than 1 or slightly less than 0.
        # We clip them to the interval [0, 1].
        absolute_correlations = np.clip(np.nan_to_num(absolute_correlations, copy=False, nan=0.0), 0, 1)
        d = 1 - absolute_correlations
        # Now we have a dissimilarity matrix, where the diagonal is 0. We need to set the diagonal to zero
        # to avoid floating point errors.
        np.fill_diagonal(d, 0)
        assert d.shape == (activations.shape[0], activations.shape[0])
        # Assert there are no NaNs.
        assert not np.any(np.isnan(d))
        return d
