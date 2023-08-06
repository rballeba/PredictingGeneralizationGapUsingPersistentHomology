import os
import pickle
import time
from typing import Union

import numpy as np

from model.pgdl_data.pgdl_datasets import load_google_dataset_by_task
from model.pgdl_data.pgdl_models import get_model_names_by_task, get_model_by_task_and_model_name
from model.pgdl_data.pgdl_structure import pgdl_folder_names

from model.sota_methods.interpex import complexity as interpex_fn
from model.sota_methods.brain import complexity as brain_fn
from model.sota_methods.always_generalize import complexity as always_generalize_fn


def compute_sota_measures(pgdl_folderpath: str, task: Union[int, str], output_folderpath: str):
    assert task in pgdl_folder_names.keys() or task == 'all', 'task must be either "all" or a task number'
    if task == 'all':
        for task_number in pgdl_folder_names.keys():
            compute_sota_measures_for_specific_pgdl_task(pgdl_folderpath, task_number, output_folderpath)
    else:
        compute_sota_measures_for_specific_pgdl_task(pgdl_folderpath, task, output_folderpath)


def compute_sota_measures_for_specific_pgdl_task(pgdl_folderpath: str, task_number: int, output_folderpath: str,
                                                 verbose: bool = True):
    # Create the output folder if it does not exist.
    if not os.path.isdir(output_folderpath):
        os.makedirs(output_folderpath)
    sota_results = dict()
    train_dataset, _ = load_google_dataset_by_task(pgdl_folderpath, task_number)
    task_models = get_model_names_by_task(pgdl_folderpath, task_number)
    if verbose:
        start_time = time.time()
        print(f'Computing SOTA measures for task {task_number}...')
    for model_name in task_models:
        model = get_model_by_task_and_model_name(pgdl_folderpath, task_number, model_name)
        interpex_score = interpex_fn(model, train_dataset, None)
        brain_score = brain_fn(model, train_dataset)
        always_generalize_score = always_generalize_fn(model, train_dataset)
        sota_results[model_name] = {
            'interpex': np.atleast_1d(interpex_score),
            'brain': np.atleast_1d(brain_score),
            'always_generalize': np.atleast_1d(always_generalize_score)
        }
    if verbose:
        end_time = time.time()
        print(f'Finished computing SOTA measures for task {task_number} in {end_time - start_time}s')
    output_filename = os.path.join(output_folderpath, f"sota_task_{task_number}.pkl")
    with open(output_filename, "wb") as results_file:
        pickle.dump(sota_results, results_file)
