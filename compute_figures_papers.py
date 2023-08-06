import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from cluster_experiments import available_vectorizations, compute_clusters, \
    flatten_clusterized_vectorizations_and_available_parameters, get_parameter_from_model_name
from model.pgdl_data.pgdl_datasets import load_task_metadata

# Configurations for the plot
plt.style.use('seaborn')
paper_textwidth = 384

dimension_combinations = ((0,), (1,), (0, 1))
# vect_name : visualization_name
possible_vectorizations = {'pers_pooling': 'Persistence pooling',
                           'average_lives_midlives': 'Average of lives and midlives',
                           'original_and_squared_average_lives_midlives': 'Original and squared average of lives and midlives',
                           'average_births_deaths': 'Average of births and deaths',
                           'original_and_squared_average_births_deaths': 'Original and squared average of births and deaths',
                           'original_and_log_average_births_deaths': 'Original and log average of births and deaths',
                           'original_and_squared_average_lives_midlives_births_and_deaths': 'Original and squared average of lives, midlives, births, and deaths',
                           'persistent_entropies': 'Persistent entropies',
                           'standard_deviations_and_average_births_and_deaths': 'Standard deviations and average of births and deaths',
                           'original_and_squared_standard_deviations_and_average_births_deaths': 'Original and squared standard deviations and average of births and deaths',
                           'complex_polynomials': 'Complex polynomials'
                           }


def plot_all_figures(pgdl_folderpath: str, results_folderpath: str):
    # compute_heatmap_r2_scores(results_folderpath)
    plot_standard_deviations_and_averages_of_deaths(pgdl_folderpath, results_folderpath)


def set_size(width, ratio_width_height, fraction=1):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    :param ratio_width_height: float Ratio to multiply width by to get height
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches

    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5 ** .5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * ratio_width_height

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


def get_mean_and_standard_deviation_matrices(task_results):
    mean_matrix = np.zeros((len(dimension_combinations), len(possible_vectorizations)))
    std_matrix = np.zeros((len(dimension_combinations), len(possible_vectorizations)))
    mean_std_matrix = np.zeros((len(dimension_combinations), len(possible_vectorizations))).tolist()
    vect_names_ordered = list(possible_vectorizations.keys())
    for i, dim_combination in enumerate(dimension_combinations):
        for j, vectorization_name in enumerate(possible_vectorizations.keys()):
            dim_comb_string = '_'.join([str(i) for i in dim_combination])
            combined_name = f'{vectorization_name}_dim_{dim_comb_string}'
            mean_matrix[i, j] = task_results.loc[combined_name, 'Mean']
            std_matrix[i, j] = task_results.loc[combined_name, 'Standard deviation']
            mean_std_matrix[i][j] = '${:.2f}$'.format(mean_matrix[i, j]) if mean_matrix[i, j] > 0 else '$<0$'
    return mean_matrix, std_matrix, mean_std_matrix, vect_names_ordered


def compute_heatmap_r2_scores(results_folderpath: str):
    # Set fonts properly
    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 8,
        "font.size": 8,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        # Titles should have large size
        "axes.titlesize": 10
    }
    plt.rcParams.update(tex_fonts)

    def vectorization_figure_names(vectorization_names):
        return [r'($' + str(vect_num + 1) + r'$)' for vect_num in range(len(vectorization_names))]

    # Load task 1 and task 2 results for predicting the generalization gap and put them in a dataframe
    task_1_results = pd.read_csv(f'{results_folderpath}/predicting_gen_gap/results_prediction_gen_gap_task_1.csv',
                                 index_col=0, header='infer')
    task_2_results = pd.read_csv(f'{results_folderpath}/predicting_gen_gap/results_prediction_gen_gap_task_2.csv',
                                 index_col=0, header='infer')
    # Remove rows that are non-topological, i.e., indices always_generalize, brain, and interpex
    task_1_results = task_1_results.drop(['always_generalize', 'brain', 'interpex'])
    task_2_results = task_2_results.drop(['always_generalize', 'brain', 'interpex'])
    # Round the r2 mean and the r2 standard deviation columns to 2 decimals
    task_1_results['Mean'] = task_1_results['Mean'].round(2)
    task_1_results['Standard deviation'] = task_1_results['Standard deviation'].round(2)
    task_2_results['Mean'] = task_2_results['Mean'].round(2)
    task_2_results['Standard deviation'] = task_2_results['Standard deviation'].round(2)
    # Plot two heatmaps, one for task 1 and one for task 2.
    # The first heatmap is for task 1
    mean_matrix_task_1, std_matrix_task_1, mean_std_matrix_task_1, vect_names_ordered_task_1 = \
        get_mean_and_standard_deviation_matrices(task_1_results)
    # Create a seaborn heatmap for task 1
    f, ax = plt.subplots(figsize=set_size(paper_textwidth, 0.2, fraction=1))
    sns.heatmap(mean_matrix_task_1, annot=mean_std_matrix_task_1, fmt='', linewidths=.5,
                xticklabels=vectorization_figure_names(vect_names_ordered_task_1),
                cmap=sns.color_palette("Spectral", as_cmap=True).reversed(),
                yticklabels=[r'Dim. $0$', r'Dim. $1$', r'Dims. $0$, $1$'], ax=ax, vmax=1, vmin=0,
                cbar_kws={'pad': 0.01, 'ticks': [0, 0.5, 1]})
    ax.set_title(r'Heatmap of $R^2$ scores for Task $1$')
    # Save the heatmap
    plt.savefig(f'{results_folderpath}/predicting_gen_gap/heatmap_r2_scores_task_1.pdf', bbox_inches='tight')
    # Show the heatmap
    # Same for task 2
    mean_matrix_task_2, std_matrix_task_2, mean_std_matrix_task_2, vect_names_ordered_task_2 = \
        get_mean_and_standard_deviation_matrices(task_2_results)
    # Create a seaborn heatmap for task 2
    f, ax = plt.subplots(figsize=set_size(paper_textwidth, 0.2, fraction=1))
    sns.heatmap(mean_matrix_task_2, annot=mean_std_matrix_task_2, fmt='', linewidths=.5,
                xticklabels=vectorization_figure_names(vect_names_ordered_task_2),
                cmap=sns.color_palette("Spectral", as_cmap=True).reversed(),
                yticklabels=[r'Dim. $0$', r'Dim. $1$', r'Dims. $0$, $1$'], ax=ax, vmax=1, vmin=0,
                cbar_kws={'pad': 0.01, 'ticks': [0, 0.5, 1]})
    ax.set_title(r'Heatmap of $R^2$ scores for Task $2$')
    # Save the heatmap
    plt.savefig(f'{results_folderpath}/predicting_gen_gap/heatmap_r2_scores_task_2.pdf', bbox_inches='tight')
    print("Figure size: ", set_size(paper_textwidth, 0.2, fraction=1))
    print("Task 1 vectorizations names in order: ", vect_names_ordered_task_1)
    print("Task 2 vectorizations names in order: ", vect_names_ordered_task_2)


# This piece of code is repeated from cluster_experiments.py. It is repetead here as this script is only
# used for plotting the results of the experiments.

def create_clusterized_vectorizations(pgdl_folderpath, results_folderpath, task, task_config_file):
    clusterized_vectorizations = dict()
    for vectorization_name in available_vectorizations:
        clusterized_vectorizations[vectorization_name] = compute_clusters(pgdl_folderpath, results_folderpath, task,
                                                                          vectorization_name)
    return flatten_clusterized_vectorizations_and_available_parameters(clusterized_vectorizations, 'all')


def plot_standard_deviations_and_averages_of_deaths(pgdl_folderpath: str, results_folderpath: str):
    custom_params = {"axes.grid": False}
    sns.set_theme(style="whitegrid", rc=custom_params)
    # Set fonts properly
    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 8,
        "font.size": 7,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        # Titles should have large size
        "axes.titlesize": 10
    }
    plt.rcParams.update(tex_fonts)
    # Perform the plot
    vectorizations_studied = ['average_deaths_dim_0', 'standard_deviations_deaths_dim_0',
                              'average_deaths_dim_1', 'standard_deviations_deaths_dim_1']
    vectorizations_names = ['Mean deaths $H_0$',
                            'SD deaths $H_0$',
                            'Mean deaths $H_1$',
                            'SD deaths $H_1$'
                            ]

    task_1_config_file = load_task_metadata(pgdl_folderpath, 1)
    task_2_config_file = load_task_metadata(pgdl_folderpath, 2)
    clusterized_vectorizations_task_1 = create_clusterized_vectorizations(pgdl_folderpath, results_folderpath,
                                                                          1, task_1_config_file)
    clusterized_vectorizations_task_2 = create_clusterized_vectorizations(pgdl_folderpath, results_folderpath,
                                                                          2, task_2_config_file)
    # Add 'num_conv_blocks' for task 1
    max_x, min_x = [], []

    clusterized_vectorizations_task_1['num_conv_blocks'] = clusterized_vectorizations_task_1.index \
        .map(lambda x: str(get_parameter_from_model_name(x, 'num_conv_blocks', task_1_config_file)))
    # Add 'depth' value for task 2
    clusterized_vectorizations_task_2['depth'] = clusterized_vectorizations_task_2.index \
        .map(lambda x: str(get_parameter_from_model_name(x, 'depth', task_2_config_file)))
    # Now we can plot with vectorization value in x-axis, generalization gap in y-axis, and coloured by
    # 'num_conv_blocks' and 'depth'.
    f, axes = plt.subplots(nrows=2, ncols=4, figsize=set_size(paper_textwidth, 0.4, fraction=1), sharey='all')
    # Make the spines thinner for all the axes
    for ax in axes.flatten():
        ax.tick_params(axis='both', which='both', length=1, pad=2.5, color='black')
        # ax.grid(True, axis='both', linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_linewidth(0.1)
    # Set the y-labels
    axes[0, 0].set_ylabel('Gen. gap')
    axes[1, 0].set_ylabel('Gen. gap')
    for i in range(1, 3):  # Task number
        for j in range(len(vectorizations_studied)):  # Vectorization
            sns.scatterplot(data=eval(f'clusterized_vectorizations_task_{i}'),
                            x=f'{vectorizations_studied[j]}_vect', y='generalization',
                            hue='num_conv_blocks' if i == 1 else 'depth',
                            hue_order=['1', '3'] if i == 1 else ['6', '9', '12'],
                            ax=axes[i - 1, j], s=5)
            # Update the maximum and minimum values for the columns
            if i == 1:
                max_x.append(eval(f'clusterized_vectorizations_task_{i}')[f'{vectorizations_studied[j]}_vect'].max())
                min_x.append(eval(f'clusterized_vectorizations_task_{i}')[f'{vectorizations_studied[j]}_vect'].min())
                # Do not x-axis labels on the first row
                axes[i - 1, j].set_xticklabels([])
                # Also, do not show the x-axis labels of the dataframe
                axes[i - 1, j].set_xlabel('')
            else:
                current_max = eval(f'clusterized_vectorizations_task_{i}')[f'{vectorizations_studied[j]}_vect'].max()
                current_min = eval(f'clusterized_vectorizations_task_{i}')[f'{vectorizations_studied[j]}_vect'].min()
                if max_x[j] < current_max:
                    max_x[j] = current_max
                if min_x[j] > current_min:
                    min_x[j] = current_min
                # Show the x-axis labels on the second row
                axes[i - 1, j].set_xlabel(vectorizations_names[j])
            # Show legend only in the last column
            if j == len(vectorizations_studied) - 1:
                # Show legend outside the plot with a markerscale = 0.5
                axes[i - 1, j].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., markerscale=0.5,
                                      markerfirst=False, handletextpad=0.1, columnspacing=0.1)
                # Put a legend title
                axes[i - 1, j].get_legend().set_title('Blocks' if i == 1 else 'Depth')
                # align the legend title to the left
                axes[i - 1, j].get_legend().get_title().set_ha('right')
                # Set the legend title font size
                axes[i - 1, j].get_legend().get_title().set_fontsize(7)
            else:
                axes[i - 1, j].legend([], [], frameon=False)
    # Update the x-axis limits
    for i in range(2):
        for j in range(len(vectorizations_studied)):
            min_current, max_current = min_x[j] - 0.01, max_x[j] + 0.01
            axes[i, j].set_xlim(min_current, max_current)
            # Set 3 ticks on the x-axis, on the minimum, maximum and middle value. Put ticks rounded to 2 decimals
            x_margin = 0.02 if j % 2 == 1 else 0.07
            axes[i, j].set_xticks([round(min_current, 2) + x_margin, round((min_current + max_current) / 2, 2),
                                   round(max_x[j], 2) - x_margin])

            axes[i, j].set_xticks([round(min_current, 2) + x_margin, round((min_current + max_current) / 2, 2),
                                   round(max_current, 2) - x_margin])
    plt.savefig(f'{results_folderpath}/predicting_gen_gap/std_avg_plots.pdf', bbox_inches='tight')
