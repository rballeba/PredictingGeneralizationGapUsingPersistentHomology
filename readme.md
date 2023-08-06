# Predicting the generalization gap in neural networks using topological data analysis

This repository contains the code for the paper "Predicting the generalization gap in neural networks using topological data analysis" by Ballester et al. (2023).  In this article, we aim to predict the generalization gap, defined as the difference between the training and the test accuracies, of neural networks using linear models whose independent variable are persistence summaries computed from the persistence diagrams of neural network activations using the training dataset. 

The project includes a Dockerfile to build a Docker image with all the dependencies needed to run the code. The Docker image is available at [Docker Hub](https://hub.docker.com/repository/docker/rballeba/activations_neurocom). From now on, we assume that the Docker image is already built and available in the local machine tagged with the name `rballeba/activations_neurocom`. For other tags, just replace `rballeba/activations_neurocom` with the desired tag.

The experiments have been performed on the tasks 1 and 2 of the [PGDL dataset](https://github.com/google-research/google-research/tree/master/pgdl). To reproduce the experiments, you must download the [public dataset](http://storage.googleapis.com/gresearch/pgdl/public_data.zip) and place it in a `public_data` folder inside any other arbitrary folder that we denote `<google_data_path>`. We assume that you want to save the results of the experiments in a path denoted by `<results_path>`. To fully reproduce the experiments, you must perform the steps described in the following sections in the same order as they appear. This is important because some of the steps depend on the results of the previous steps. To get help more information about all the possible options of the scripts, you can execute `docker run rballeba/activations_neurocom run help`.

## Computing persistence diagrams

The first step is computing the 20 persistence diagrams from the neural networks of both tasks using a sample of 3000 neuron activations and a sample of 2000 samples from the training dataset. This is done by executing the following commands:

 - Task 1
```bash
docker run --gpus '"device=<nvidia-smi devices sepparated by a comma>"' -v <google_data_path>:/home/google_data -v <results_path>:/app/results rballeba/activations_neurocom compute_persistence_diagrams /home/google_data 1 3000 2000 20 1 /app/results
```

 - Task 2
```bash 
docker run --gpus '"device=<nvidia-smi devices sepparated by a comma>"' -v <google_data_path>:/home/google_data -v <results_path>:/app/results rballeba/activations_neurocom compute_persistence_diagrams /home/google_data 1 3000 2000 20 2 /app/results
```

## Computing SOTA measures

The second step is computing the SOTA measures (Always Generalize, Interpex, BrAIn) for both tasks. This is done by executing the following commands:

 - Task 1
```bash
docker run --gpus '"device=<nvidia-smi devices sepparated by a comma>"' -v <google_data_path>:/home/google_data -v <results_path>:/app/results rballeba/activations_neurocom compute_sota /home/google_data 1 /app/results
```
- Task 2
```bash
docker run --gpus '"device=<nvidia-smi devices sepparated by a comma>"' -v <google_data_path>:/home/google_data -v <results_path>:/app/results rballeba/activations_neurocom compute_sota /home/google_data 2 /app/results
```

## Computing experiments and extracting statistics

The following step is computing the linear models using the SOTA generalization measures and the persistence summaries as independent variables and extracting the statistics of the experiments such as average and standard deviation $R^2$ scores, $p$-values for the 5x2-fold cross validation significance test. This is done by executing the following commands:

 - Task 1
```bash
docker run --gpus '"device=<nvidia-smi devices sepparated by a comma>"' -v <google_data_path>:/home/google_data -v <results_path>:/app/results rballeba/activations_neurocom compute_statistics/home/google_data /app/results 1 /app/results/predicting_gen_gap
```
- Task 2
```bash
docker run --gpus '"device=<nvidia-smi devices sepparated by a comma>"' -v <google_data_path>:/home/google_data -v <results_path>:/app/results rballeba/activations_neurocom compute_statistics /home/google_data /app/results 2 /app/results/predicting_gen_gap
```

## Compute clusters

The previous steps were used to replicate the experiments performed in the paper. The following steps allows to run an interactive visualization of the bootstrapped persistence summaries and the generalization gaps for tasks 1 and 2, being able to select the property of the neural networks of each tasks to perform clusterization. This includes a clusterization using a K-means clusterization algorithm on the bootstrapped persistence sumaries concatenated with the generalization gap. See the function `compute_clusters` in the file `cluster_experiments.py` for more details. This is done by executing the following commands:

 - Task 1
```bash
docker run --gpus '"device=<nvidia-smi devices sepparated by a comma>"' -v <google_data_path>:/home/google_data -v <results_path>:/app/results -p 8051:8051 rballeba/activations_neurocom compute_clusters /home/google_data /app/results 1 /app/results/predicting_gen_gap
```
- Task 2
```bash
docker run --gpus '"device=<nvidia-smi devices sepparated by a comma>"' -v <google_data_path>:/home/google_data -v <results_path>:/app/results -p 8052:8052 rballeba/activations_neurocom compute_clusters /home/google_data /app/results 2 /app/results/predicting_gen_gap
```
After executing any of these two instructions, you only need to access http://localhost:8051/ for task 1 or http://localhost:8052/ for task 2 to visualize the plots.

## Compute plots

After executing all the previous steps, the following command can be used to generate the plots of the paper. The plots will be saved in the results folder.

```bash
docker run --gpus '"device=<nvidia-smi devices sepparated by a comma>"' -v <google_data_path>:/home/google_data -v <results_path>:/app/results rballeba/activations_neurocom plot_figures /home/google_data /app/results
```
