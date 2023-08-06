from typing import List

import numpy as np
import numpy.typing as npt
import os


def save_persistence_diagrams(filepath_to_save: str, ordered_persistence_diagrams: List[npt.NDArray]):
    """
    It saves a list of persistence diagrams computed from the same point cloud for different dimensions. Each
    persistence diagram is a numpy array with two columns, the first one with the birth time and the second one with the
    death time. It is assumed that the i-th element of the list corresponds to the i-th dimensional persistence diagram.
    :param filepath_to_save: Filepath where the persistence diagrams will be saved.
    :param ordered_persistence_diagrams: List where the ith element represents the i-th dimensional persistence diagram
    for the same point cloud.
    :return:
    """
    enumerated_persistence_diagrams = {str(i): ordered_persistence_diagrams[i]
                                       for i in range(len(ordered_persistence_diagrams))}
    np.savez(filepath_to_save, **enumerated_persistence_diagrams)


def load_persistence_diagrams(filepath: str):
    """
    It loads a list of persistence diagrams computed from the same point cloud for different dimensions. It is assumed
    that the list of persistence diagrams is saved using the method save_persistence_diagrams.
    :param filepath: Filepath where the persistence diagrams are saved.
    :return: A list of persistence diagrams where the i-th element represents the i-th dimensional persistence diagram
    for the same point cloud.
    """
    persistence_diagrams = np.load(filepath)
    return [persistence_diagrams[str(i)] for i in range(len(persistence_diagrams))]


def load_persistence_diagrams_task(results_base_filepath: str, task: int):
    def get_persistence_diagrams_from_npz(pd_npz):
        return [pd_npz[str(i)] for i in range(len(pd_npz))]
    """
    It loads the persistence diagrams for a given task. It is assumed that the persistence diagrams are saved using the
    method save_persistence_diagrams. We assume that results_base_filepath is the same value entered by parameter
    as output filepath when computing the persistence diagrams.
    :param results_base_filepath: str: Filepath where the persistence diagrams are saved.
    :param task: int: Task for which we want to load the persistence diagrams.
    :return:
    """
    results_folderpath = os.path.join(results_base_filepath, f"task_{task}")
    assert os.path.isdir(results_folderpath)
    model_names = os.listdir(results_folderpath)
    assert len(model_names) > 0, f"No models found in {results_folderpath}"
    persistence_diagrams = {}
    for model_name in model_names:
        model_folderpath = os.path.join(results_folderpath, model_name)
        persistence_diagrams[model_name] = [get_persistence_diagrams_from_npz(
            np.load(os.path.join(model_folderpath, persistence_diagram_filepath)))
                                                for persistence_diagram_filepath in os.listdir(model_folderpath)
                                                if persistence_diagram_filepath.endswith(".npz")]
    return persistence_diagrams
