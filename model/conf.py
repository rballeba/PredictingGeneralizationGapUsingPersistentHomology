import os

# The following variable is used as a seed for the shuffling of a dataset using the method take_random_sample_dataset
# from model/dataset.py. This is done to ensure reproducibility of the results.
random_seed_shuffle_dataset = 1
# The following variable is used as buffer size for the shuffling of a dataset using the method
# take_random_sample_dataset from model/dataset.py. This is done to ensure reproducibility of the results.
buffer_size_shuffle_dataset = 10000
# The following variable defines the neuron sampling method used to compute_persistence_diagrams. Use the name of the
# class as a string.
activation_sampling_method = 'SampleActivationsByImportance'
# The following variable defines the distance function used to compute a distance matrix from a set of activations
# of a model given as a numpy array with shape (number_of_neurons, number_of_examples). Use the name of the class as a
# string.
distance_function_method = 'AbsoluteCorrelationDissimilarity'
# The following variable defines the temp folder where big matrices during the computation of persistence diagrams
# are stored.
temp_mapped_matrix_folder = os.path.abspath('./.temp_matrices')
# The following variable defines the number of layers that are skipped when computing the activations of a model.
num_skipped_layers = 0
# The following variable defines the random seed used for the computation of the k-folds when making
# cross-validation of the vectorizations to predict the generalization gap.
random_seed_cross_validation = 1
# The following variable defines the random seed used for the computation of the clusters of scalar
# vectorizations.
random_seed_clusterization = 0
# The following variable defines the feature of the neural network used to compute restricted mutual information in
# cluster_experiments.py for a specific task. The key is the task number and the value is the feature name.
clusters_by_depth_feature = {1: 'num_conv_blocks', 2: 'depth'}
# The following variable defines the list of vectorizations compared using significance tests in the prediction of the
# generalization gap.
features_to_compare = ['original_and_squared_standard_deviations_and_average_births_deaths_dim_0_1', 'interpex',
                       'always_generalize', 'brain', 'standard_deviations_and_average_births_and_deaths_dim_1']
