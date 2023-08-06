from typing import List
import numpy as np
import numpy.typing as npt
from gtda.diagrams import ComplexPolynomial


def get_finite_persistence_diagrams(persistence_diagrams: List[npt.NDArray]):
    """We assume that points (b,d) in the persistence diagram satisfy b <= d."""
    return [persistence_diagram[np.isfinite(persistence_diagram[:, 1])]
            for persistence_diagram in persistence_diagrams]


def get_average_lives(persistence_diagrams: List[npt.NDArray]):
    finite_persistence_diagrams = get_finite_persistence_diagrams(persistence_diagrams)
    return [np.mean(persistence_diagram[:, 1] - persistence_diagram[:, 0])
            for persistence_diagram in finite_persistence_diagrams]


def get_average_midlives(persistence_diagrams: List[npt.NDArray]):
    finite_persistence_diagrams = get_finite_persistence_diagrams(persistence_diagrams)
    return [np.mean(persistence_diagram[:, 0] + persistence_diagram[:, 1]) / 2
            for persistence_diagram in finite_persistence_diagrams]


def get_average_births(persistence_diagrams: List[npt.NDArray]):
    finite_persistence_diagrams = get_finite_persistence_diagrams(persistence_diagrams)
    return [np.mean(persistence_diagram[:, 0]) for persistence_diagram in finite_persistence_diagrams]


def get_average_deaths(persistence_diagrams: List[npt.NDArray]):
    finite_persistence_diagrams = get_finite_persistence_diagrams(persistence_diagrams)
    return [np.mean(persistence_diagram[:, 1]) for persistence_diagram in finite_persistence_diagrams]


def get_standard_deviations_lives(persistence_diagrams: List[npt.NDArray]):
    finite_persistence_diagrams = get_finite_persistence_diagrams(persistence_diagrams)
    return [np.std(persistence_diagram[:, 1] - persistence_diagram[:, 0])
            for persistence_diagram in finite_persistence_diagrams]


def get_standard_deviations_midlives(persistence_diagrams: List[npt.NDArray]):
    finite_persistence_diagrams = get_finite_persistence_diagrams(persistence_diagrams)
    return [(np.std(persistence_diagram[:, 0] + persistence_diagram[:, 1]) / 2)
            for persistence_diagram in finite_persistence_diagrams]


def get_standard_deviations_births(persistence_diagrams: List[npt.NDArray]):
    finite_persistence_diagrams = get_finite_persistence_diagrams(persistence_diagrams)
    return [np.std(persistence_diagram[:, 0]) for persistence_diagram in finite_persistence_diagrams]


def get_standard_deviations_deaths(persistence_diagrams: List[npt.NDArray]):
    finite_persistence_diagrams = get_finite_persistence_diagrams(persistence_diagrams)
    return [np.std(persistence_diagram[:, 1]) for persistence_diagram in finite_persistence_diagrams]


def get_persistent_entropies(persistence_diagrams: List[npt.NDArray]):
    finite_persistence_diagrams = get_finite_persistence_diagrams(persistence_diagrams)
    total_sums = [np.sum(persistence_diagram[:, 1] - persistence_diagram[:, 0])
                  for persistence_diagram in finite_persistence_diagrams]
    normalized_differences = [(1.0 / total_sums[dim]) * (finite_persistence_diagrams[dim][:, 1] -
                                                         finite_persistence_diagrams[dim][:, 0])
                              for dim in range(len(finite_persistence_diagrams))]
    return [-np.sum(normalized_differences[dim] * np.log(normalized_differences[dim]))
            for dim in range(len(finite_persistence_diagrams))]


def get_pooling_vectors(persistence_diagrams: List[npt.NDArray], n: int = 10):
    finite_persistence_diagrams = get_finite_persistence_diagrams(persistence_diagrams)
    lives = [persistence_diagram[:, 1] - persistence_diagram[:, 0]
             for persistence_diagram in finite_persistence_diagrams]
    ordered_lives_descending_order = [-np.sort(-lives_dim) for lives_dim in lives]  # Descending order
    # Take the first n elements of the ordered lives, if there are less than n elements, pad with zeros
    elements_to_take = [min(n, ordered_lives_dim.shape[0]) for ordered_lives_dim in ordered_lives_descending_order]
    pooling_vectors = [np.concatenate((ordered_lives_descending_order[dim][:elements_to_take[dim]],
                                       np.zeros(n - elements_to_take[dim]))) for dim in
                       range(len(finite_persistence_diagrams))]
    return pooling_vectors


def get_complex_polynomials(persistence_diagrams: List[npt.NDArray], n_coefficients=10, pol_type='T'):
    finite_persistence_diagrams = get_finite_persistence_diagrams(persistence_diagrams)
    cp_fn = ComplexPolynomial(n_coefficients=n_coefficients, polynomial_type=pol_type)
    augmented_persistence_diagrams = [np.expand_dims(np.concatenate((finite_persistence_diagrams[dim],
                                                                     dim * np.ones((finite_persistence_diagrams[
                                                                                        dim].shape[0], 1))),
                                                                    axis=1), axis=0)
                                      for dim in range(len(finite_persistence_diagrams))]
    complex_polynomials = [cp_fn.fit_transform(augmented_pd)[0] for augmented_pd in augmented_persistence_diagrams]
    return complex_polynomials
