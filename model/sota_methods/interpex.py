import tensorflow as tf

from model.sota_methods.interpex_complexitymeasures import *


def complexity(model, dataset, program_dir, measure='DBI, Mixup', augment=None):
    '''
    Wrapper Complexity Function to combine various complexity measures

    Parameters
    ----------
    model : tf.keras.Model()
        The Keras model for which the complexity measure is to be computed
    dataset : tf.data.Dataset
        Dataset object from PGDL data loader
    program_dir : str, optional
        The program directory to store and retrieve additional data
    measure : str, optional
        The complexity measure to compute, defaults to our winning solution of PGDL
    augment : str, optional
        Augmentation method to use, only relevant for some measures

    Returns
    -------
    float
        complexity measure
    '''

    if measure == 'DBI':
        complexityScore = complexityDB(model, dataset, program_dir=program_dir, pool=True, layer='initial',
                                       computeOver=400, batchSize=40)
    elif measure == 'Mixup':
        complexityScore = complexityMixup(model, dataset, program_dir=program_dir)
    elif measure == 'Margin':
        complexityScore = complexityMargin(model, dataset, augment=augment, program_dir=program_dir)
    elif measure == 'DBI, Mixup':
        complexityScore = complexityDB(model, dataset, program_dir=program_dir, pool=True, computeOver=400,
                                       batchSize=40) * (1 - complexityMixup(model, dataset, program_dir=program_dir))
    elif measure == 'ManifoldMixup':
        complexityScore = complexityManifoldMixup(model, dataset, program_dir=program_dir)
    else:
        complexityScore = complexityDB(model, dataset, program_dir=program_dir, pool=True, computeOver=400,
                                       batchSize=40) * (1 - complexityMixup(model, dataset, program_dir=program_dir))

    return complexityScore