import json
import os
import time
from typing import Union, List

import numpy as np
import numpy.typing as npt
from gph import ripser_parallel

from model.conf import activation_sampling_method, distance_function_method
from model.dataset import take_random_sample_dataset
from model.distance_function_strategies.distance_function_interface import DistanceFunctionInterface
from model.io_persistence import save_persistence_diagrams
from model.pgdl_data.pgdl_datasets import load_google_dataset_by_task
from model.pgdl_data.pgdl_models import get_model_names_by_task, get_model_by_task_and_model_name
from model.pgdl_data.pgdl_structure import pgdl_folder_names
from model.sampling_activations_strategies.activation_sampling_interface import ActivationSamplingInterface

# Do not change the following includes. They contain the implementations of the sampling methods and distance functions.

from model.sampling_activations_strategies.sample_activations_by_importance import SampleActivationsByImportance
from model.distance_function_strategies.absolute_correlation_dissimilarity import AbsoluteCorrelationDissimilarity

# Finishing the includes.


def compute_persistence_diagrams(pgdl_folderpath: str, task: Union[int, str], neurons_sampled: int,
                                 inputs_sampled: int, number_of_persistence_diagrams_per_model: int, max_ph_dim: int,
                                 output_folderpath: str):
    assert task in pgdl_folder_names.keys() or task == 'all', 'task must be either "all" or a task number'
    if task == 'all':
        for task_number in pgdl_folder_names.keys():
            compute_persistence_diagrams_for_specific_pgdl_task(pgdl_folderpath, task_number, neurons_sampled,
                                                                inputs_sampled,
                                                                number_of_persistence_diagrams_per_model, max_ph_dim,
                                                                output_folderpath)
    else:
        compute_persistence_diagrams_for_specific_pgdl_task(pgdl_folderpath, task, neurons_sampled,
                                                            inputs_sampled,
                                                            number_of_persistence_diagrams_per_model, max_ph_dim,
                                                            output_folderpath)


def compute_persistence_diagrams_for_specific_pgdl_task(pgdl_folderpath: str, task_number: int, neurons_sampled: int,
                                                        inputs_sampled: int,
                                                        number_of_persistence_diagrams_per_model: int, max_ph_dim: int,
                                                        output_folderpath: str,
                                                        verbose: bool = True):

    if verbose:
        start_time = time.time()
        print(f"Computing persistence diagrams for task {task_number}")
    output_folderpath = os.path.join(output_folderpath, f"task_{task_number}")
    if not os.path.isdir(output_folderpath):
        os.makedirs(output_folderpath)
    train_dataset, test_dataset = load_google_dataset_by_task(pgdl_folderpath, task_number)
    train_x_sample, train_y_sample = take_random_sample_dataset(train_dataset, inputs_sampled)
    task_models = get_model_names_by_task(pgdl_folderpath, task_number)
    for model_name in task_models:
        compute_persistence_diagrams_for_specific_pgdl_model(pgdl_folderpath, task_number, model_name, train_x_sample,
                                                             neurons_sampled, inputs_sampled,
                                                             number_of_persistence_diagrams_per_model, max_ph_dim,
                                                             output_folderpath)
    if verbose:
        print(f"Task {task_number} finished in {time.time() - start_time} seconds")


def compute_persistence_diagrams_for_specific_pgdl_model(pgdl_folderpath, task_number: int,
                                                         model_name: str,
                                                         train_x_sample: List[npt.NDArray],
                                                         neurons_sampled: int, inputs_sampled: int,
                                                         number_of_persistence_diagrams_per_model: int, max_ph_dim: int,
                                                         output_folderpath: str,
                                                         verbose: bool = True):
    if verbose:
        start_time_all_diagrams = time.time()
        print(f"Computing persistence diagrams for model {model_name}")
    output_folderpath_model = os.path.join(output_folderpath, model_name)
    if not os.path.isdir(output_folderpath_model):
        os.makedirs(output_folderpath_model)
    else:
        if verbose:
            print(f"Model {model_name} already computed. Skipping.")
        return
    # Setting the distance and sampling functions configured in the config file.
    distance_function = eval(distance_function_method)
    activation_sampling_strategy = eval(activation_sampling_method)
    assert issubclass(distance_function, DistanceFunctionInterface) \
           and issubclass(activation_sampling_strategy, ActivationSamplingInterface)
    # Computing persistence diagrams
    if not os.path.isdir(output_folderpath_model):
        os.makedirs(output_folderpath_model)
    model = get_model_by_task_and_model_name(pgdl_folderpath, task_number, model_name)
    if verbose:
        start_time_activations = time.time()
        print(f"Computing activations for model {model_name}")
    all_neuron_activations = activation_sampling_strategy.sample_activations(model, train_x_sample, neurons_sampled,
                                                                             number_of_persistence_diagrams_per_model)
    if verbose:
        print(f"Activations for model {model_name} computed in {time.time() - start_time_activations} seconds")
    for i, neuron_activations in enumerate(all_neuron_activations):
        if verbose:
            start_pd_time = time.time()
            print(f"Computing persistence diagram {i} for model {model_name}")
        # Compute distance matrix
        distance_matrix = distance_function.compute_distance_matrix(neuron_activations)
        # Compute the persistence diagrams for the sample i and enumerate them by their dimension.
        persistence_diagrams = ripser_parallel(distance_matrix, metric="precomputed", maxdim=max_ph_dim)['dgms']
        # Save the persistence diagrams
        if verbose:
            print(f"Persistence diagrams for the sample {i} for model {model_name} computed in "
                  f"{time.time() - start_pd_time} seconds")
        save_persistence_diagrams(os.path.join(output_folderpath_model, f"persistence_diagrams_{i}.npz"),
                                  persistence_diagrams)
        # Save a json file with the parameters used to compute the persistence diagram
        with open(os.path.join(output_folderpath_model, f"persistence_diagrams_{i}.json"), "w") as f:
            json.dump({"neurons_sampled": neurons_sampled, "inputs_sampled": inputs_sampled,
                       "max_ph_dim": max_ph_dim, 'distance_function': distance_function_method,
                       "sampling_strategy": activation_sampling_method}, f)
    if verbose:
        print(f"Model {model_name} finished in {time.time() - start_time_all_diagrams} seconds")
