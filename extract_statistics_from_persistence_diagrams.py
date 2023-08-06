import copy
import itertools
import os.path
import pickle
import time
from itertools import combinations
from typing import Union, List, Dict, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold
from scipy import stats

from model.conf import random_seed_cross_validation, features_to_compare
from model.diagram_vectorizations import get_pooling_vectors, get_average_lives, get_average_midlives, \
    get_average_births, get_average_deaths, get_persistent_entropies, get_complex_polynomials, \
    get_standard_deviations_births, get_standard_deviations_deaths
from model.io_persistence import load_persistence_diagrams_task
from model.mutual_information import conditional_mutual_information
from model.pgdl_data.pgdl_datasets import load_task_generalization_gaps, load_task_metadata
from model.pgdl_data.pgdl_structure import pgdl_folder_names


def compute_statistics_from_persistence_diagrams(pgdl_folderpath: str, results_folderpath: str, task: Union[int, str],
                                                 output_folderpath: str):
    assert task in pgdl_folder_names.keys() or task == 'all', 'task must be either "all" or a task number'
    if task == 'all':
        for task_number in pgdl_folder_names.keys():
            compute_statistics_from_persistence_diagrams_task(pgdl_folderpath, results_folderpath, task_number,
                                                              output_folderpath)
    else:
        compute_statistics_from_persistence_diagrams_task(pgdl_folderpath, results_folderpath, task, output_folderpath)


def get_sota_models_and_measures(results_folderpath: str, task: int):
    sota_results_filename = f'{results_folderpath}/sota_task_{task}.pkl'
    if os.path.isfile(sota_results_filename):
        with open(sota_results_filename, 'rb') as f:
            sota_results = pickle.load(f)
        return sota_results
    else:
        raise ValueError(f'SOTA results for task {task} not found')


def compute_scalars_for_exploratory_analysis_coming_from_persisence_diagrams(
        bootstrapped_vectorizations_and_generalizations: Dict[str, Tuple[Dict[str, npt.NDArray],
        int]]):
    """
    Compute the scalars that will be used to compute the mutual information. These scalars are all the scalars contained
    in the vectorizations of the persistence diagrams for all dimensions (in the paper 0 and 1). These are:
    - Average lives
    - Average midlives
    - Average births
    - Average deaths
    - Persistent entropies
    - Standard deviations of births
    - Standard deviations of deaths
    :param bootstrapped_vectorizations_and_generalizations: For each model, we have its list of vectorized persistence
    diagrams for each dimension combinations and its generalization gap.
    :return: A dict with keys the model names and values a dict with keys the names of the scalars and values the
    actual scalars.
    """
    ppdd_scalars_and_generalizations = {}
    for model_name, vectorizations_and_generalizations in bootstrapped_vectorizations_and_generalizations.items():
        ppdd_scalars_and_generalizations[model_name] = (dict(), vectorizations_and_generalizations[1])
        bootstrapped_vectorizations = vectorizations_and_generalizations[0]
        ppdd_scalars_and_generalizations[model_name][0]['average_lives_dim_0'] = \
            bootstrapped_vectorizations['average_lives_midlives_dim_0'][0]
        ppdd_scalars_and_generalizations[model_name][0]['average_midlives_dim_0'] = \
            bootstrapped_vectorizations['average_lives_midlives_dim_0'][1]
        ppdd_scalars_and_generalizations[model_name][0]['average_lives_dim_1'] = \
            bootstrapped_vectorizations['average_lives_midlives_dim_1'][0]
        ppdd_scalars_and_generalizations[model_name][0]['average_midlives_dim_1'] = \
            bootstrapped_vectorizations['average_lives_midlives_dim_1'][1]
        ppdd_scalars_and_generalizations[model_name][0]['average_births_dim_0'] = \
            bootstrapped_vectorizations['average_births_deaths_dim_0'][0]
        ppdd_scalars_and_generalizations[model_name][0]['average_deaths_dim_0'] = \
            bootstrapped_vectorizations['average_births_deaths_dim_0'][1]
        ppdd_scalars_and_generalizations[model_name][0]['average_births_dim_1'] = \
            bootstrapped_vectorizations['average_births_deaths_dim_1'][0]
        ppdd_scalars_and_generalizations[model_name][0]['average_deaths_dim_1'] = \
            bootstrapped_vectorizations['average_births_deaths_dim_1'][1]
        ppdd_scalars_and_generalizations[model_name][0]['persistent_entropy_dim_0'] = \
            bootstrapped_vectorizations['persistent_entropies_dim_0'][0]
        ppdd_scalars_and_generalizations[model_name][0]['persistent_entropy_dim_1'] = \
            bootstrapped_vectorizations['persistent_entropies_dim_1'][0]
        ppdd_scalars_and_generalizations[model_name][0]['standard_deviations_births_dim_0'] = \
            bootstrapped_vectorizations['standard_deviations_and_average_births_and_deaths_dim_0'][0]
        ppdd_scalars_and_generalizations[model_name][0]['standard_deviations_deaths_dim_0'] = \
            bootstrapped_vectorizations['standard_deviations_and_average_births_and_deaths_dim_0'][1]
        ppdd_scalars_and_generalizations[model_name][0]['standard_deviations_births_dim_1'] = \
            bootstrapped_vectorizations['standard_deviations_and_average_births_and_deaths_dim_1'][0]
        ppdd_scalars_and_generalizations[model_name][0]['standard_deviations_deaths_dim_1'] = \
            bootstrapped_vectorizations['standard_deviations_and_average_births_and_deaths_dim_1'][1]
    return ppdd_scalars_and_generalizations


def compute_mutual_information(pgdl_folderpath: str,
                               bootstrapped_vectorizations_and_generalizations: Dict[str, Tuple[Dict[str, npt.NDArray],
                               int]], sota_models_and_measures: Dict[str, Dict[str, npt.NDArray]], task: int):
    # Get config file for the task
    task_config_file = load_task_metadata(pgdl_folderpath, task)
    ppdd_scalars_and_generalizations = compute_scalars_for_exploratory_analysis_coming_from_persisence_diagrams(
        bootstrapped_vectorizations_and_generalizations)
    # Join the scalars from the persistence diagrams with the scalars from the SOTA models
    scalars_and_generalizations = copy.deepcopy(ppdd_scalars_and_generalizations)
    for model_name in scalars_and_generalizations.keys():
        # Convert from 1d arrays to scalars the values of the sota models and put them in the dict
        scalars_and_generalizations[model_name][0].update({scalar_name: scalar_value[0]
                                                           for (scalar_name, scalar_value)
                                                           in sota_models_and_measures[model_name].items()})
    # Compute the mutual information for each scalar
    scalar_names = scalars_and_generalizations[list(scalars_and_generalizations.keys())[0]][0].keys()
    mutual_information = dict()
    for scalar_name in scalar_names:
        predictions = {model_name: scalars_and_generalizations[model_name][0][scalar_name] for
                       model_name in scalars_and_generalizations.keys()}
        mutual_information[scalar_name] = conditional_mutual_information(predictions, task_config_file)
    mutual_information_df = pd.DataFrame(mutual_information, index=['Mutual information'])
    mutual_information_ordered = mutual_information_df.T.sort_values(by='Mutual information', ascending=False)
    return mutual_information_ordered


def compute_significance_test(r2_method_a: List[float], r2_method_b: List[float]):
    p_1, p_2, s_squared = [], [], []
    for i in range(0, 10, 2):
        p_1.append(r2_method_a[i] - r2_method_b[i])
        p_2.append(r2_method_a[i + 1] - r2_method_b[i + 1])
    for i in range(5):
        p_bar = (p_1[i] + p_2[i]) / 2.0
        s_squared.append((p_1[i] - p_bar) ** 2 + (p_2[i] - p_bar) ** 2)
    t_stat = p_1[0] / np.sqrt(np.sum(s_squared) / 5.0)
    p_value = stats.t.sf(np.abs(t_stat), 5) * 2  # Two-sided hypothesis test.
    return t_stat, p_value


def compute_all_significance_tests(results_prediction_gen_gap: pd.DataFrame):
    # Get all the rows of the dataframe
    comparison_methods_df = pd.DataFrame(columns=['method_1', 'method_2', 't_stat', 'p_value'])
    vectorizations_pairs_to_compare = itertools.combinations(features_to_compare, 2)
    # Perform the significance test
    for method_1, method_2 in vectorizations_pairs_to_compare:
        r_2_method_1 = results_prediction_gen_gap.loc[method_1].to_list()
        r_2_method_2 = results_prediction_gen_gap.loc[method_2].to_list()
        t_stat, p_value = compute_significance_test(r_2_method_1, r_2_method_2)
        comparison_methods_df = pd.concat(
            [comparison_methods_df, pd.DataFrame.from_records({'method_1': method_1, 'method_2': method_2,
                                                               't_stat': t_stat, 'p_value': p_value}, index=[0])],
            ignore_index=True)
    return comparison_methods_df


def compute_statistics_from_persistence_diagrams_task(pgdl_folderpath: str, results_folderpath: str, task: int,
                                                      output_folderpath: str):
    # First get the original vectorizations used to predict the generalization gap
    bootstrapped_vectorizations_and_generalizations = compute_bootstrapped_statistics_from_persistence_diagrams(
        pgdl_folderpath, results_folderpath, task)
    # Now get the results (vectorizations) for the competitors
    sota_models_and_measures = get_sota_models_and_measures(results_folderpath, task)
    # Perform prediction of generalization gap by fitting a linear model in a training dataset.
    results_prediction_gen_gap = predict_generalization_gap(bootstrapped_vectorizations_and_generalizations,
                                                            sota_models_and_measures)
    if not os.path.isdir(output_folderpath):
        os.makedirs(output_folderpath)
    results_predictions_significance_test = compute_all_significance_tests(results_prediction_gen_gap)
    results_predictions_significance_test.to_csv(
        f'{output_folderpath}/results_prediction_significance_test_task_{task}.csv')
    results_prediction_full_stats = compute_statistics_from_prediction_scores(results_prediction_gen_gap)
    results_prediction_full_stats.to_csv(f'{output_folderpath}/results_prediction_gen_gap_task_{task}.csv')
    # Now compute mutual informations instead of predicitions.
    results_mutual_information = compute_mutual_information(pgdl_folderpath,
                                                            bootstrapped_vectorizations_and_generalizations,
                                                            sota_models_and_measures, task)
    results_mutual_information.to_csv(f'{output_folderpath}/results_mutual_information_task_{task}.csv')


def compute_statistics_from_prediction_scores(results_prediction: pd.DataFrame):
    results_prediction = results_prediction.copy()
    results_prediction['Mean'] = results_prediction.mean(axis=1)
    results_prediction['Standard deviation'] = results_prediction.std(axis=1)
    results_prediction['Median'] = results_prediction.median(axis=1)
    results_prediction['Minimum'] = results_prediction.min(axis=1)
    results_prediction['Maximum'] = results_prediction.max(axis=1)
    # Order by mean
    results_prediction = results_prediction.sort_values(by='Mean', ascending=False)
    return results_prediction


def get_all_combinations_dimensions(max_dim: int):
    dims = list(range(max_dim + 1))
    combinations_fixed_lengths = [combinations(dims, length) for length in range(1, max_dim + 2)]
    return [comb for combinations_fixed_length in combinations_fixed_lengths for comb in combinations_fixed_length]


def compute_bootstrapped_statistics_from_persistence_diagrams(pgdl_folderpath: str, results_folderpath: str, task: int):
    # If the bootstrapped vectorizations and generalization gaps have already been computed, we load them
    if os.path.isfile(f'{results_folderpath}/bootstrapped_vectorizations_and_generalizations_task_{task}.pkl'):
        print(f'Loading bootstrapped vectorizations and generalization gaps for task {task}')
        with open(f'{results_folderpath}/bootstrapped_vectorizations_and_generalizations_task_{task}.pkl', 'rb') as f:
            bootstrapped_vectorizations_and_generalizations = pickle.load(f)
            print(f'Loaded bootstrapped vectorizations and generalization gaps for task {task}')
        return bootstrapped_vectorizations_and_generalizations
    # If not, we compute them
    print(f'Bootstraped vectorizations not found. '
          f'Computing bootstrapped vectorizations and generalization gaps for task {task}')
    start_time = time.time()
    # For each model, we have its list of persistence diagrams of different dimensions and its generalization gap
    diagrams_and_generalizations = get_persistence_diagrams_and_generalization_gaps(pgdl_folderpath, results_folderpath,
                                                                                    task)
    # For each model, we have its list of vectorized persistence diagrams for each dimension combinations
    # and its generalization gap
    vectorizations_and_generalizations = {
        model_name: [[get_studied_vectorizations_persistence_diagrams(persistence_diagrams)
                      for persistence_diagrams in diagrams_and_gen[0]], diagrams_and_gen[1]] for
        (model_name, diagrams_and_gen) in diagrams_and_generalizations.items()}
    bootstrapped_vectorizations_and_generalizations = {
        model_name: [get_bootstrapped_vectorizations(vectorizations_and_gens[0]), vectorizations_and_gens[1]]
        for (model_name, vectorizations_and_gens) in vectorizations_and_generalizations.items()}
    print(f'Computed bootstrapped vectorizations and generalization gaps for task {task} in '
          f'{time.time() - start_time} seconds')
    # Save the bootstrapped vectorizations and generalization gaps using pickle
    with open(f'{results_folderpath}/bootstrapped_vectorizations_and_generalizations_task_{task}.pkl', 'wb') as f:
        pickle.dump(bootstrapped_vectorizations_and_generalizations, f)
    return bootstrapped_vectorizations_and_generalizations


def predict_generalization_gap(bootstrapped_vectorizations_and_generalizations: Dict[str,
Tuple[Dict[str, npt.NDArray], int]], sota_models_and_measures: Dict[str, Dict[str, npt.NDArray]]):
    experiments_results = []
    # First we combine SOTA results with bootstrapped vectorizations and generalization gaps
    vectorizations_and_generalizations = copy.deepcopy(bootstrapped_vectorizations_and_generalizations)
    for model_name in bootstrapped_vectorizations_and_generalizations.keys():
        vectorizations_and_generalizations[model_name][0].update(sota_models_and_measures[model_name])
    map_indices_model_names = dict(enumerate(vectorizations_and_generalizations.keys()))
    model_indices = np.array(list(map_indices_model_names.keys()))
    vectorization_names = vectorizations_and_generalizations[map_indices_model_names[0]][0].keys()
    # Now we perform kfold with the indices, in order to have the same partition for all the vectorizations.
    # 5x2 fold cross validation
    kfold = RepeatedKFold(n_splits=2, n_repeats=5, random_state=random_seed_cross_validation)
    for i, (train_indices, test_indices) in enumerate(kfold.split(model_indices)):
        experiments_results_for_fold = []
        train_models = [map_indices_model_names[model_indices[index]] for index in train_indices]
        test_models = [map_indices_model_names[model_indices[index]] for index in test_indices]
        for vectorization_name in vectorization_names:
            train_vectorizations = [vectorizations_and_generalizations[model_name][0][vectorization_name]
                                    for model_name in train_models]
            test_vectorizations = [vectorizations_and_generalizations[model_name][0][vectorization_name]
                                   for model_name in test_models]
            train_generalization_gaps = [vectorizations_and_generalizations[model_name][1]
                                         for model_name in train_models]
            test_generalization_gaps = [vectorizations_and_generalizations[model_name][1]
                                        for model_name in test_models]
            train_vectorizations = np.vstack(train_vectorizations)
            test_vectorizations = np.vstack(test_vectorizations)
            train_generalization_gaps = np.vstack(train_generalization_gaps)
            test_generalization_gaps = np.vstack(test_generalization_gaps)
            # Linear regression to predict the generalization gap
            linear_regression = LinearRegression()
            linear_regression.fit(train_vectorizations, train_generalization_gaps)
            # Coefficient of determination R^2 of the prediction
            r2_score = linear_regression.score(test_vectorizations, test_generalization_gaps)
            experiments_results_for_fold.append((vectorization_name, r2_score))
        experiments_results.append(experiments_results_for_fold)
    # Create now a table with the results
    experiments_results_table = np.zeros((len(vectorization_names), 10))
    for i, experiments_results_for_fold in enumerate(experiments_results):
        for j, (_, r2_score) in enumerate(experiments_results_for_fold):
            experiments_results_table[j, i] = r2_score
    # Create a pandas dataframe with the results
    experiments_results_df = pd.DataFrame(experiments_results_table,
                                          columns=pd.Index([f'Fold {i}' for i in range(10)]),
                                          index=pd.Index(vectorization_names))
    return experiments_results_df


def perform_bootstraping_sample(vectors: List[npt.NDArray], number_of_samples=1000):
    bootstrap_sample_replications = []
    sample_size = len(vectors)
    for _ in range(number_of_samples):
        sample_indices = np.random.choice(len(vectors), sample_size, replace=True)
        sample = [vectors[i] for i in sample_indices]
        average_sample_vectorization = np.mean(sample, axis=0)
        bootstrap_sample_replications.append(average_sample_vectorization)
    average_of_bootstrap_sample_replications = np.mean(bootstrap_sample_replications, axis=0)
    return average_of_bootstrap_sample_replications


def get_bootstrapped_vectorizations(list_of_vectorizations: List[Dict[str, npt.NDArray]]):
    """
    We assume all the models have the same vectorizations and also are computed in the same combinations of dimensions.
    :param list_of_vectorizations: A list containing a dict with keys the names of the vectorizations and values
    the actual value of the vectorizations.
    :return:
    """
    bootstrapped_vectorizations = {}
    number_of_vectorizations_of_the_same_type = len(list_of_vectorizations)
    assert number_of_vectorizations_of_the_same_type > 0  # We have to have at least one persistence diagram
    vectorization_names = list_of_vectorizations[0].keys()
    # Check all the persistence diagrams have the same vectorizations
    for i in range(1, number_of_vectorizations_of_the_same_type):
        assert list_of_vectorizations[i].keys() == vectorization_names

    for vectorization_name in vectorization_names:
        vectorizations_for_all_ppdd = [list_of_vectorizations[i][vectorization_name]
                                       for i in range(number_of_vectorizations_of_the_same_type)]
        bootstrapped_vectorizations[vectorization_name] = perform_bootstraping_sample(vectorizations_for_all_ppdd)
    return bootstrapped_vectorizations


def combine_vectorizations(vectorizations_to_combine: List[List[npt.NDArray]]):
    max_dimension_studied = len(vectorizations_to_combine[0])
    total_vectorizations = len(vectorizations_to_combine)
    combined_vectorizations = [np.concatenate([np.atleast_1d(vectorizations_to_combine[i][dim])
                                               for i in range(total_vectorizations)], axis=0)
                               for dim in range(max_dimension_studied)]
    return combined_vectorizations


def get_original_and_squared_vectorizations(original_vectorizations: List[npt.NDArray]):
    max_dimension_studied = len(original_vectorizations)
    original_and_squared_vectorizations = [np.concatenate((np.atleast_1d(original_vectorizations[dim]),
                                                           np.atleast_1d(original_vectorizations[dim] ** 2)), axis=0)
                                           for dim in range(max_dimension_studied)]
    return original_and_squared_vectorizations


def get_original_and_logarithm_vectorizations(original_vectorizations: List[npt.NDArray]):
    max_dimension_studied = len(original_vectorizations)
    original_and_squared_vectorizations = [np.concatenate((np.atleast_1d(original_vectorizations[dim]),
                                                           np.atleast_1d(np.log(original_vectorizations[dim] + 1))),
                                                          axis=0)  # We sum 1 to avoid log(0)
                                           for dim in range(max_dimension_studied)]
    return original_and_squared_vectorizations


def get_studied_vectorizations_persistence_diagrams(persistence_diagrams: List[npt.NDArray]):
    studied_vectorizations = {}
    max_dimension_studied = len(persistence_diagrams) - 1
    dimension_combinations = get_all_combinations_dimensions(max_dimension_studied)

    def _add_vectorizations_for_all_combinations_of_dimensions(vectorizations: List[npt.NDArray],
                                                               vectorization_name: str):
        for dim_comb in dimension_combinations:
            dim_comb_string = '_'.join([str(i) for i in dim_comb])
            vectorization = np.atleast_1d(vectorizations[dim_comb[0]]) \
                if len(dim_comb) == 1 \
                else np.concatenate([np.atleast_1d(vectorizations[dim]) for dim in dim_comb], axis=0)
            studied_vectorizations[f'{vectorization_name}_dim_{dim_comb_string}'] = vectorization

    # Raw vectorizations
    persistence_pooling_vector = get_pooling_vectors(persistence_diagrams)
    average_lives = get_average_lives(persistence_diagrams)
    average_midlives = get_average_midlives(persistence_diagrams)
    average_births = get_average_births(persistence_diagrams)
    average_deaths = get_average_deaths(persistence_diagrams)
    standard_deviations_births = get_standard_deviations_births(persistence_diagrams)
    standard_deviations_deaths = get_standard_deviations_deaths(persistence_diagrams)
    complex_polynomials = get_complex_polynomials(persistence_diagrams)
    persistent_entropies = get_persistent_entropies(persistence_diagrams)
    # Combined vectorizations
    average_lives_midlives = combine_vectorizations([average_lives, average_midlives])
    original_and_squared_average_lives_and_midlives = get_original_and_squared_vectorizations(average_lives_midlives)
    average_births_and_deaths = combine_vectorizations([average_births, average_deaths])
    original_and_squared_average_births_and_deaths = get_original_and_squared_vectorizations(average_births_and_deaths)
    original_and_logarithm_average_births_and_deaths = get_original_and_logarithm_vectorizations(
        average_births_and_deaths)
    original_and_squared_average_lives_midlives_births_and_deaths = combine_vectorizations([
        original_and_squared_average_lives_and_midlives, original_and_squared_average_births_and_deaths])
    standard_deviations_births_and_deaths = combine_vectorizations([standard_deviations_births,
                                                                    standard_deviations_deaths])
    standard_deviations_and_average_births_and_deaths = combine_vectorizations([standard_deviations_births_and_deaths,
                                                                                average_births_and_deaths])
    original_and_squared_standard_deviations_and_average_births_and_deaths = get_original_and_squared_vectorizations(
        standard_deviations_and_average_births_and_deaths
    )
    # Adding vectorizations to the study
    _add_vectorizations_for_all_combinations_of_dimensions(persistence_pooling_vector, 'pers_pooling')
    _add_vectorizations_for_all_combinations_of_dimensions(average_lives_midlives, 'average_lives_midlives')
    _add_vectorizations_for_all_combinations_of_dimensions(original_and_squared_average_lives_and_midlives,
                                                           'original_and_squared_average_lives_midlives')
    _add_vectorizations_for_all_combinations_of_dimensions(average_births_and_deaths, 'average_births_deaths')
    _add_vectorizations_for_all_combinations_of_dimensions(original_and_squared_average_births_and_deaths,
                                                           'original_and_squared_average_births_deaths')
    _add_vectorizations_for_all_combinations_of_dimensions(original_and_logarithm_average_births_and_deaths,
                                                           'original_and_log_average_births_deaths')
    _add_vectorizations_for_all_combinations_of_dimensions(
        original_and_squared_average_lives_midlives_births_and_deaths,
        'original_and_squared_average_lives_midlives_births_and_deaths')
    _add_vectorizations_for_all_combinations_of_dimensions(persistent_entropies, 'persistent_entropies')
    _add_vectorizations_for_all_combinations_of_dimensions(standard_deviations_and_average_births_and_deaths,
                                                           'standard_deviations_and_average_births_and_deaths')
    _add_vectorizations_for_all_combinations_of_dimensions(
        original_and_squared_standard_deviations_and_average_births_and_deaths,
        'original_and_squared_standard_deviations_and_average_births_deaths')
    _add_vectorizations_for_all_combinations_of_dimensions(complex_polynomials, 'complex_polynomials')
    return studied_vectorizations


def get_persistence_diagrams_and_generalization_gaps(pgdl_folderpath: str, results_folderpath: str, task: int):
    generalization_gaps = load_task_generalization_gaps(pgdl_folderpath, task)
    persistence_diagrams = load_persistence_diagrams_task(results_folderpath, task)
    assert set(generalization_gaps.keys()) == set(persistence_diagrams.keys())
    diagrams_and_generalizations = {model_name: (ppdd, generalization_gaps[model_name])
                                    for model_name, ppdd in persistence_diagrams.items()}
    return diagrams_and_generalizations
