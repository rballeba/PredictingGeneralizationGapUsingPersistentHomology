import tensorflow as tf

from model.conf import random_seed_shuffle_dataset, buffer_size_shuffle_dataset


def take_random_sample_dataset(dataset: tf.data.Dataset, sample_size: int):
    sample_dataset = dataset.shuffle(buffer_size=buffer_size_shuffle_dataset, seed=random_seed_shuffle_dataset,
                                     reshuffle_each_iteration=False).take(sample_size)
    x_samples, y_samples = tuple(zip(*sample_dataset.as_numpy_iterator()))
    return tf.stack(x_samples, axis=0), tf.stack(y_samples, axis=0)
