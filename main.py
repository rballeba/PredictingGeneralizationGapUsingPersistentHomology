import os
import sys

import tensorflow as tf

from cluster_experiments import explore_clusters
from compute_figures_papers import plot_all_figures
from compute_persistence_diagrams_pgdl import compute_persistence_diagrams
from extract_statistics_from_persistence_diagrams import compute_statistics_from_persistence_diagrams
from model.sota_methods.get_sota_methods_measures import compute_sota_measures

possible_tasks = ["help", "compute_persistence_diagrams", "compute_sota", "compute_statistics", "compute_clusters",
                  "plot_figures"]
possible_pgdl_tasks = [1, 2, 4, 5, 6, 7, 8, 9]

# Set memory growth to true to avoid locating all the GPU memory at once.
# See https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


def execute_task():
    if len(sys.argv) < 2:
        print("Usage: python main.py <task> <task_args>. To see possible tasks and their arguments,"
              " run python main.py help")
        exit(1)
    task = sys.argv[1]
    if task not in possible_tasks:
        print("Usage: python main.py <task> <task_args>. To see possible tasks and their arguments,"
              " run python main.py help")
        exit(1)
    if task == "help":
        print("Possible tasks:")
        print("=====================================")
        print("Compute persistence diagrams for a specific PGDL task. Usage:")
        print("python main.py compute_persistence_diagrams <pgdl_folderpath> <number_of_the_task>"
              " <neurons_sampled> <inputs_sampled> <number_of_persistence_diagrams_per_model> <max_ph_dim> "
              "<output_folderpath>")
        print("<pgdl_folderpath> is the path to the folder where the PGDL data is stored.")
        print(f"<number_of_the_task> is an integer from the list {possible_pgdl_tasks} or 'all'. If 'all' is used, "
              f"the persistence diagrams for all tasks will be computed.")
        print("<neurons_sampled> is an integer determining the number of neurons sampled from the model.")
        print("<inputs_sampled> is an integer determining the number of inputs sampled from the training dataset.")
        print("<number_of_persistence_diagrams_per_model> is an integer determining the number of persistence diagrams"
              " computed for each model.")
        print("<max_ph_dim> is an integer determining the maximum homology dimension of the persistence diagrams.")
        print("<output_folderpath> is a path to the folder where the persistence diagrams will be saved.")
        print("=====================================")
        print("Compute SOTA measures for a specific PGDL task. Usage:")
        print("python main.py compute_sota <pgdl_folderpath> <number_of_the_task> <output_folderpath>")
        print("<pgdl_folderpath> is the path to the folder where the PGDL data is stored.")
        print(f"<number_of_the_task> is an integer from the list {possible_pgdl_tasks} or 'all'. If 'all' is used, "
              f"the SOTA measures for all tasks will be computed.")
        print("<output_folderpath> is a path to the folder where the SOTA measures will be saved.")
        print("=====================================")
        print("Compute statistics for a specific PGDL task. Usage:")
        print("python main.py compute_statistics <pgdl_folderpath> <results_folderpath> <number_of_the_task> "
              "<output_folderpath>")
        print("<pgdl_folderpath> is the path to the folder where the PGDL data is stored.")
        print("<results_folderpath> is a path to the folder where the results of the computations will be saved.")
        print(f"<number_of_the_task> is an integer from the list {possible_pgdl_tasks} or 'all'. If 'all' is used, "
              f"the statistics for all tasks will be computed.")
        print("<output_folderpath> is a path to the folder where the statistics will be saved.")
        print("=====================================")
        print("Generate clusters using K-means and show an interactive plot with data associated to these clusters. "
              "Also, computes mutual information on the clusters defined by the features selected in the config file. "
              "Usage:")
        print("python main.py compute_clusters <pgdl_folderpath> <number_of_the_task> "
              "<output_folderpath>")
        print("<pgdl_folderpath> is the path to the folder where the PGDL data is stored.")
        print(f"<number_of_the_task> is an integer from the list {possible_pgdl_tasks}.")
        print("<output_folderpath> is a path to the folder where the SOTA measures will be saved.")
        print("=====================================")
        print("Plot figures for the paper. Usage:")
        print("python main.py plot_figures <pgdl_folderpath> <results_folderpath>")
        print("<pgdl_folderpath> is the path to the folder where the PGDL data is stored.")
        print("<results_folderpath> is the path to the folder where the results of the computations are saved.")

    elif task == "compute_persistence_diagrams":
        compute_persistence_diagrams_for_specific_pgdl_task()

    elif task == "compute_statistics":
        compute_statistics_for_specific_pgdl_task()

    elif task == "compute_sota":
        compute_sota_measures_for_specific_pgdl_task()

    elif task == "compute_clusters":
        compute_clusters_for_specific_pgdl_task()

    elif task == "plot_figures":
        plot_figures()


def plot_figures():
    if len(sys.argv) != 4:
        print("Wrong number of arguments. Usage: python main.py plot_figures <pgdl_folderpath> <results_folderpath>")
        exit(1)
    possible_errors = []
    pgdl_folderpath = sys.argv[2]
    # Check if the folder exists
    if not os.path.isdir(pgdl_folderpath):
        possible_errors.append(f"Folder {pgdl_folderpath} does not exist.")
    results_folderpath = sys.argv[3]
    # Check if the folder exists
    if not os.path.isdir(results_folderpath):
        possible_errors.append(f"Folder {results_folderpath} does not exist.")
    if len(possible_errors) > 0:
        print("Errors:")
        for error in possible_errors:
            print(error)
        exit(1)
    plot_all_figures(pgdl_folderpath, results_folderpath)


def compute_sota_measures_for_specific_pgdl_task():
    if len(sys.argv) != 5:
        print("Wrong number of arguments. Usage: python main.py compute_sota <pgdl_folderpath>"
              " <number_of_the_task> <output_folderpath>")
        exit(1)
    possible_errors = []
    pgdl_folderpath = sys.argv[2]
    # Check if the folder exists
    if not os.path.isdir(pgdl_folderpath):
        possible_errors.append(f"Folder {pgdl_folderpath} does not exist.")
    try:
        task = int(sys.argv[3]) if sys.argv[3] != "all" else "all"
        if task not in possible_pgdl_tasks and task != "all":
            raise ValueError
    except ValueError:
        possible_errors.append(f"Parameter <number_of_the_task>: task number must be one of {possible_pgdl_tasks} "
                               f"or 'all'.")
    output_folderpath = sys.argv[4]
    if len(possible_errors) > 0:
        print("Errors:")
        for error in possible_errors:
            print(error)
        exit(1)
    compute_sota_measures(pgdl_folderpath, task, output_folderpath)


def compute_statistics_for_specific_pgdl_task():
    if len(sys.argv) != 6:
        print("Wrong number of arguments. Usage: python main.py compute_statistics <pgdl_folderpath>"
              " <results_folderpath> <number_of_the_task> <vectorization_name> <output_folderpath>")
        exit(1)
    possible_errors = []
    pgdl_folderpath = sys.argv[2]
    results_folderpath = sys.argv[3]
    output_folderpath = sys.argv[5]
    # Check if the folders exists
    if not os.path.isdir(pgdl_folderpath):
        possible_errors.append(f"Folder {pgdl_folderpath} does not exist.")
    if not os.path.isdir(results_folderpath):
        possible_errors.append(f"Folder {results_folderpath} does not exist.")
    try:
        task = int(sys.argv[4]) if sys.argv[4] != "all" else "all"
        if task not in possible_pgdl_tasks and task != "all":
            raise ValueError
    except ValueError:
        possible_errors.append(f"Parameter <number_of_the_task>: task number must be one of {possible_pgdl_tasks} "
                               f"or 'all'.")
    if len(possible_errors) > 0:
        print("Errors:")
        for error in possible_errors:
            print(error)
        exit(1)
    # Compute the statistics
    compute_statistics_from_persistence_diagrams(pgdl_folderpath, results_folderpath, task, output_folderpath)


def compute_clusters_for_specific_pgdl_task():
    if len(sys.argv) != 6:
        print("Wrong number of arguments. Usage: python main.py compute_clusters <pgdl_folderpath>"
              " <results_folderpath> <number_of_the_task> <output_folderpath>")
        exit(1)
    possible_errors = []
    pgdl_folderpath = sys.argv[2]
    results_folderpath = sys.argv[3]
    output_folderpath = sys.argv[5]
    # Check if the folders exists
    if not os.path.isdir(pgdl_folderpath):
        possible_errors.append(f"Folder {pgdl_folderpath} does not exist.")
    if not os.path.isdir(results_folderpath):
        possible_errors.append(f"Folder {results_folderpath} does not exist.")
    try:
        task = int(sys.argv[4])
        if task not in possible_pgdl_tasks:
            raise ValueError
    except ValueError:
        possible_errors.append(f"Parameter <number_of_the_task>: task number must be one of {possible_pgdl_tasks}.")
    if len(possible_errors) > 0:
        print("Errors:")
        for error in possible_errors:
            print(error)
        exit(1)
    # Compute the statistics
    explore_clusters(pgdl_folderpath, results_folderpath, task, output_folderpath)


def compute_persistence_diagrams_for_specific_pgdl_task():
    if len(sys.argv) != 9:
        print("Wrong number of arguments. Usage: python main.py compute_persistence_diagrams <pgdl_folderpath>"
              " <number_of_the_task> <neurons_sampled> <inputs_sampled> <number_of_persistence_diagrams_per_model> "
              "<max_ph_dim> <output_folderpath>")
        exit(1)
    possible_errors = []
    pgdl_folderpath = sys.argv[2]
    # Check if the folder exists
    if not os.path.isdir(pgdl_folderpath):
        possible_errors.append(f"Folder {pgdl_folderpath} does not exist.")
    try:
        task = int(sys.argv[3]) if sys.argv[3] != "all" else "all"
        if task not in possible_pgdl_tasks and task != "all":
            raise ValueError
    except ValueError:
        possible_errors.append(f"Parameter <number_of_the_task>: task number must be one of {possible_pgdl_tasks} "
                               f"or 'all'.")
    try:
        neurons_sampled = int(sys.argv[4])
        if neurons_sampled <= 0:
            raise ValueError
    except ValueError:
        possible_errors.append("Parameter <neurons_sampled>: Neurons sampled must be a positive integer.")
    try:
        inputs_sampled = int(sys.argv[5])
        if inputs_sampled <= 0:
            raise ValueError
    except ValueError:
        possible_errors.append("Parameter <inputs_sampled>: Inputs sampled must be a positive integer.")
    try:
        number_of_persistence_diagrams_per_model = int(sys.argv[6])
        if number_of_persistence_diagrams_per_model <= 0:
            raise ValueError
    except ValueError:
        possible_errors.append("Parameter <number_of_persistence_diagrams_per_model>: Number of persistence diagrams "
                               "per model must be an integer.")
    try:
        max_ph_dim = int(sys.argv[7])
        if max_ph_dim <= 0:
            raise ValueError
    except ValueError:
        possible_errors.append("Parameter <max_ph_dim>: Max PH dimension must be a positive integer.")
    output_folderpath = sys.argv[8]
    if len(possible_errors) > 0:
        print("Errors:")
        for error in possible_errors:
            print(error)
        exit(1)
    compute_persistence_diagrams(pgdl_folderpath, task, neurons_sampled, inputs_sampled,
                                 number_of_persistence_diagrams_per_model, max_ph_dim, output_folderpath)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    execute_task()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
