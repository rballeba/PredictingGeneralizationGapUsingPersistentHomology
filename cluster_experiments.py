import copy

import dash
import plotly.express as px
import os
import pickle
import time
from typing import Dict, Tuple, Any, Union, List
import pandas as pd
from scipy.stats import kendalltau, spearmanr, pearsonr

from extract_statistics_from_persistence_diagrams import compute_bootstrapped_statistics_from_persistence_diagrams, \
    compute_scalars_for_exploratory_analysis_coming_from_persisence_diagrams, get_sota_models_and_measures
from sklearn.cluster import KMeans
import numpy.typing as npt

from model.conf import random_seed_clusterization, clusters_by_depth_feature
from model.pgdl_data.pgdl_datasets import load_task_metadata

available_vectorizations = ["average_lives_dim_0", "average_midlives_dim_0", "average_lives_dim_1",
                            "average_midlives_dim_1", "average_births_dim_0", "average_deaths_dim_0",
                            "average_births_dim_1", "average_deaths_dim_1", "persistent_entropy_dim_0",
                            "persistent_entropy_dim_1", "standard_deviations_births_dim_0",
                            "standard_deviations_deaths_dim_0", "standard_deviations_births_dim_1",
                            "standard_deviations_deaths_dim_1"]

possible_metrics_for_generalization_in_clusters = ['kendalltau', 'spearmanr', 'pearsonr']


def explore_clusters(pgdl_folderpath: str, results_folderpath: str, task: int, output_folderpath: str):
    clusterized_vectorizations = dict()
    for vectorization_name in available_vectorizations:
        clusterized_vectorizations[vectorization_name] = compute_clusters(pgdl_folderpath, results_folderpath, task,
                                                                          vectorization_name)
    task_model_configs = load_task_metadata(pgdl_folderpath, task)
    sota_models_and_measures = get_sota_models_and_measures(results_folderpath, task)
    for metric in possible_metrics_for_generalization_in_clusters:
        compute_metric_in_depth_clusters(clusterized_vectorizations, sota_models_and_measures, task,
                                         task_model_configs, metric, output_folderpath)
    interactive_dashboard_clusters(clusterized_vectorizations, 'all', task, task_model_configs)


def get_config_file_without_parameter(parameter_to_remove: str, task_model_configs: Dict[str, Any],
                                      selected_models: Dict[str, List[float]]):
    # Copy the task_model_config but without the parameter
    task_model_configs_without_parameter = dict()
    included_model_names = set(selected_models.keys())
    for model_name in task_model_configs.keys():
        complete_model_name = model_name if model_name.startswith("model_") else f"model_{model_name}"
        if complete_model_name in included_model_names:
            task_model_configs_without_parameter[model_name] = dict()
            task_model_configs_without_parameter[model_name]["hparams"] = dict()
            for parameter in task_model_configs[model_name]["hparams"].keys():
                if parameter != parameter_to_remove:
                    task_model_configs_without_parameter[model_name]["hparams"][parameter] = \
                        task_model_configs[model_name]["hparams"][parameter]
            task_model_configs_without_parameter[model_name]["metrics"] = task_model_configs[model_name]["metrics"]
            task_model_configs_without_parameter[model_name]["dev_name"] = task_model_configs[model_name]["dev_name"]
    return task_model_configs_without_parameter


def compute_metric_in_depth_clusters(clusterized_vectorizations:
Union[Dict[str, Dict[str, Tuple[float, float, float]]], Dict[str, Tuple[float, float, float]]],
                                     sota_models_and_measures: Dict[str, Dict[str, npt.NDArray]],
                                     task: int, task_model_configs: Dict[str, Any],
                                     metric: str,
                                     output_folderpath: str):
    model_names_given_by_models_specs = set(task_model_configs.keys())

    def get_model_name_according_to_json_config(model_name: str):
        if model_name.startswith("model_"):
            model_name = model_name[6:]
        model_name = model_name if model_name in model_names_given_by_models_specs else f"model_{model_name}"
        return model_name

    cluster_property = clusters_by_depth_feature[task]
    model_names = list(clusterized_vectorizations[available_vectorizations[0]].keys())
    clusterized_vectorizations_flatten = {model_name: [clusterized_vectorizations[vectorization_name_it][model_name][0]
                                                       for vectorization_name_it in available_vectorizations]
                                          for model_name in model_names}
    # Add SOTA methods
    available_vectorizations_and_sota_names = copy.deepcopy(available_vectorizations)
    sota_names = list(sota_models_and_measures[list(sota_models_and_measures.keys())[0]])
    available_vectorizations_and_sota_names.extend(sota_names)
    for model_name in sota_models_and_measures.keys():
        clusterized_vectorizations_flatten[model_name].extend([sota_models_and_measures[model_name][sota_name][0]
                                                               for sota_name in sota_names])
    values_for_cluster_property = task_model_configs[get_model_name_according_to_json_config(
        model_names[0])]["hparams"][cluster_property]["possible_values"]
    restricted_metric = dict()
    for value in values_for_cluster_property:
        restricted_metric[value] = []
        # Generate a subdictionary with the models that have the same value for the cluster property
        selected_models = {model_name: clusterized_vectorizations_flatten[model_name] for model_name in
                           model_names
                           if task_model_configs[get_model_name_according_to_json_config(model_name)][
                               "hparams"][cluster_property]["current_value"] == value}
        for vectorization_name_it in range(len(available_vectorizations_and_sota_names)):
            # Compute the metric for the selected vectorization
            selected_vectorizations = {model_name: selected_models[model_name][vectorization_name_it]
                                       for model_name in selected_models.keys()}
            config_file_without_parameter = get_config_file_without_parameter(cluster_property, task_model_configs,
                                                                              selected_models)
            if metric in ['kendalltau', 'spearmanr', 'pearsonr']:
                selected_models_names = list(selected_models.keys())
                real_generalization_gaps = [task_model_configs[get_model_name_according_to_json_config(model_name)][
                                                "metrics"]["train_acc"] -
                                            task_model_configs[get_model_name_according_to_json_config(model_name)][
                                                "metrics"]["test_acc"]
                                            for model_name in selected_models_names]
                predicted_generalization_gaps = [selected_vectorizations[model_name]
                                                 for model_name in selected_models_names]
                if metric == 'kendalltau':
                    restricted_metric[value].append(
                        kendalltau(real_generalization_gaps, predicted_generalization_gaps)[0])
                elif metric == 'spearmanr':
                    restricted_metric[value].append(
                        spearmanr(real_generalization_gaps, predicted_generalization_gaps)[0])
                elif metric == 'pearsonr':
                    restricted_metric[value].append(
                        pearsonr(real_generalization_gaps, predicted_generalization_gaps)[0])
            else:
                raise ValueError(f"Metric {metric} not recognized.")
    df = pd.DataFrame.from_dict(restricted_metric, orient='index',
                                columns=available_vectorizations_and_sota_names)
    # Add a new column with the best metric that is not sota for each row
    df['best_for_pds'] = df.apply(lambda row: max([abs(row[column]) for column in available_vectorizations]),
                                  axis=1)
    df.index.name = cluster_property
    df.to_csv(f'{output_folderpath}/{metric}_{cluster_property}_task_{task}.csv')


def flatten_clusterized_vectorizations_and_available_parameters(clusterized_vectorizations, vectorization_name):
    # Generate a dataframe with the clusterized vectorizations
    if vectorization_name == 'all':
        clusterized_vectorizations_flatten = dict()
        for model_name in clusterized_vectorizations[available_vectorizations[0]].keys():
            clusterized_vectorizations_flatten[model_name] = []
            for vectorization_name_it in available_vectorizations:
                clusterized_vectorizations_flatten[model_name].append(
                    clusterized_vectorizations[vectorization_name_it][model_name][0])  # Vect value
                clusterized_vectorizations_flatten[model_name].append(
                    str(clusterized_vectorizations[vectorization_name_it][model_name][
                            2]))  # Cluster label according to vect.
            clusterized_vectorizations_flatten[model_name].append(
                clusterized_vectorizations[vectorization_name_it][model_name][1])  # Generalization gap
        df = pd.DataFrame.from_dict(clusterized_vectorizations_flatten, orient='index')
        columns = []
        for vectorization_name_it in available_vectorizations:
            columns.append(f'{vectorization_name_it}_vect')
            columns.append(f'{vectorization_name_it}_cluster')
        columns.append('generalization')
        df.columns = columns
    else:
        df = pd.DataFrame.from_dict(clusterized_vectorizations, columns=['vectorization', 'generalization', 'cluster'],
                                    orient='index')
        # Cluster as a string to have discrete values for the color
        df['cluster'] = df['cluster'].map(lambda x: f'cluster_{x}')
    return df


def get_parameter_from_model_name(model_name: str, parameter: str, task_model_configs: Dict[str, Any]):
    """
    It returns the value of the parameter of the model.
    :param model_name: Name of the model.
    :param parameter: Name of the parameter.
    :param task_model_configs: A JSON containing the model specifications  included in the PGDL task.
    :return: Value of the parameter.
    """
    model_names_given_by_models_specs = set(task_model_configs.keys()) # This is O(#models): however, as the number
    # of models is finite and low (<100), we can perform this operation each time we execute the method
    # get_parameter_from_model_name
    if model_name.startswith("model_"):
        model_name = model_name[6:]
    model_name = model_name if model_name in model_names_given_by_models_specs else f"model_{model_name}"
    return task_model_configs[model_name]["hparams"][parameter]["current_value"]


def interactive_dashboard_clusters(clusterized_vectorizations: Union[Dict[str, Dict[str, Tuple[float, float, float]]],
Dict[str, Tuple[float, float, float]]], vectorization_name: str, task: int, task_model_configs: Dict[str, Any]):
    """
    It creates an interactive dashboard using plotly and dash.
    :param clusterized_vectorizations: Clusterized vectorizations computed using compute_clusters.
    :param vectorization_name: String representing the vectorization name selected.
    :param task: Integer representing the task used to compute the clusters.
    :param task_model_configs: A JSON containing the model specifications  included in the PGDL task.
    :return: Nothing.
    """
    df = flatten_clusterized_vectorizations_and_available_parameters(clusterized_vectorizations, vectorization_name)
    # Add to the previous dataframe the configs
    available_parameters = list(task_model_configs[list(task_model_configs.keys())[0]]["hparams"].keys())
    hover_data = list(available_parameters)
    # Add the parameters to the dataframe
    for parameter in available_parameters:
        df[parameter] = df.index.map(lambda x: str(get_parameter_from_model_name(x, parameter, task_model_configs)))
    app = dash.Dash(__name__)
    app.layout = dash.html.Div([
        dash.html.H1(children='Cluster visualization', style={'textAlign': 'center'}),
        dash.html.H2(children=f'{vectorization_name} - Task: {task}', style={'textAlign': 'center'}),
        # Button to select if we paint differently the points according to their cluster or not
        # Allow to color by available parameters, generalization gap or cluster
        dash.dcc.Dropdown(
            id='color-radio',
            options=[
                        {'label': 'Color by cluster', 'value': 'cluster'},
                        {'label': 'Color by generalization gap', 'value': 'generalization'},
                    ] + [{'label': f'Color by {parameter}', 'value': parameter} for parameter in available_parameters
                         ],
            value=available_vectorizations[0]
        ),
        dash.dcc.Dropdown(
            id='vectorization-dropdown',
            options=[{'label': vectorization_name_it, 'value': vectorization_name_it} for vectorization_name_it in
                     available_vectorizations] if vectorization_name == 'all' else [vectorization_name],
            value=available_vectorizations[0]
        ),
        dash.dcc.Graph(id='graph-content'),
        # Write all the possible parameters
        dash.html.H3(children='Parameters and possible values:'),
        dash.dcc.Markdown(
            children='\n'.join([f'**{parameter}**: '
                                f'{task_model_configs[list(task_model_configs.keys())[0]]["hparams"][parameter]["possible_values"]}\n'
                                for parameter in available_parameters])),
    ])

    @app.callback(
        dash.dependencies.Output('graph-content', 'figure'),
        [dash.dependencies.Input('color-radio', 'value'),
         dash.dependencies.Input('vectorization-dropdown', 'value')])
    def update_graph_content(color_radio_value, vectorization_selected):
        """
        It updates the graph content according to the radio button value.
        :param vectorization_selected:  Value of the dropdown.
        :param color_radio_value: Value of the radio button.
        :return: A plotly figure.
        """
        if vectorization_name == 'all':
            if color_radio_value == 'cluster':
                return px.scatter(df, x=f'{vectorization_selected}_vect', y='generalization',
                                  color=f'{vectorization_selected}_cluster',
                                  hover_data=hover_data, hover_name=df.index)
            elif color_radio_value in available_parameters:
                return px.scatter(df, x=f'{vectorization_selected}_vect', y='generalization',
                                  color=color_radio_value,
                                  hover_data=hover_data, hover_name=df.index)
            else:
                return px.scatter(df, x=f'{vectorization_selected}_vect', y='generalization',
                                  color='generalization',
                                  hover_data=hover_data, hover_name=df.index)
        else:
            if color_radio_value == 'cluster':
                return px.scatter(df, x='vectorization', y='generalization', color='cluster',
                                  hover_data=hover_data, hover_name=df.index)
            elif color_radio_value in available_parameters:
                return px.scatter(df, x='vectorization', y='generalization', color=color_radio_value,
                                  hover_data=hover_data, hover_name=df.index)
            else:
                return px.scatter(df, x='vectorization', y='generalization', color='generalization',
                                  hover_data=hover_data, hover_name=df.index)

    starting_port = 8050
    app.run_server(debug=True, port=starting_port + task)


def compute_clusters(pgdl_folderpath: str, results_folderpath: str, task: int, vectorization_name: str):
    """
    It returns a dict with keys equal to the model names and values a 3D point containing in each component:
    1.- Vectorization value 2.- Generalization gap, 3.- Label assigned according to a K-means classification.
    :param pgdl_folderpath: Path (relative or absolute) to the source of the PGDL dataset.
    :param results_folderpath: Path (relative or absolute) to the path where the results are saved. Persistence diagrams
    and statistics must have been computed first.
    :param task: Task number.
    :param vectorization_name: Vectorization to cluster w.r.t. generalization gap. Available vectorizations contained in
    the variable available_vectorizations
    :return: Dict[str: npt.NDArray]
    """
    clusters_filepath = f'{results_folderpath}/clusters_{vectorization_name}_task_{task}.pkl'
    if os.path.isfile(clusters_filepath):
        with open(clusters_filepath, 'rb') as f:
            print(f"Loading clusterized vectorization and generalization gaps for vectorization {vectorization_name}"
                  f"and task {task}")
            vectorizations_and_generalizations_clustered = pickle.load(f)
        print("Loaded clusterized vectorizations.")
        return vectorizations_and_generalizations_clustered
    print(f'Clusterized vectorizations not found. '
          f'Computing clusterized vectorizations for vectorization {vectorization_name} and task  {task}')
    start_time = time.time()
    bootstrapped_vectorizations_and_generalizations = compute_bootstrapped_statistics_from_persistence_diagrams(
        pgdl_folderpath, results_folderpath, task)
    scalars_and_generalizations = compute_scalars_for_exploratory_analysis_coming_from_persisence_diagrams(
        bootstrapped_vectorizations_and_generalizations)
    assert vectorization_name in available_vectorizations
    number_of_models = len(list(bootstrapped_vectorizations_and_generalizations.keys()))
    vectorizations_and_generalizations = {model_name:
                                              (scalars_and_generalizations[model_name][0][
                                                   vectorization_name],
                                               scalars_and_generalizations[model_name][1])
                                          for model_name in scalars_and_generalizations.keys()}
    model_indices_map = dict(enumerate(vectorizations_and_generalizations.keys()))
    vect_and_gens_arr = [vectorizations_and_generalizations[model_indices_map[i]] for i in range(number_of_models)]
    kmeans = KMeans(n_clusters=2, random_state=random_seed_clusterization).fit(vect_and_gens_arr)
    kmeans_labels = kmeans.labels_
    # Assign the labels of the clusterization to the vector with generalizations
    vectorizations_and_generalizations_clustered = {
        model_indices_map[i]: (vectorizations_and_generalizations[model_indices_map[i]][0],
                               vectorizations_and_generalizations[model_indices_map[i]][1],
                               kmeans_labels[i])
        for i in range(number_of_models)}
    print(f"Computed clusterized vectorizations in {time.time() - start_time} seconds.")
    # Save the clusterizations
    with open(clusters_filepath, 'wb') as f:
        pickle.dump(vectorizations_and_generalizations_clustered, f)
    return vectorizations_and_generalizations_clustered
