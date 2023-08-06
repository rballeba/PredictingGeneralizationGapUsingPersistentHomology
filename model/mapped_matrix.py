import copy
import os
import tempfile
from enum import Enum

import numpy as np

from model.conf import temp_mapped_matrix_folder

FOLDER = temp_mapped_matrix_folder


class ConcatenationMode(Enum):
    COLUMN = 1
    ROW = 2


class MappedMatrix:
    def __init__(self, array=None, shape=None, concatenation_mode=ConcatenationMode.COLUMN):
        if array is not None and shape is not None:
            raise Exception("You can't specify both array and shape.")
        self._concatenation_mode = concatenation_mode
        _create_folder_if_not_existing()
        self._activations_file = tempfile.NamedTemporaryFile('w+b', dir=FOLDER)
        order = self._decide_order()
        if shape is not None:
            self.array = np.memmap(self._activations_file.name,
                                   dtype=float,
                                   mode='w+',
                                   shape=shape,
                                   order=order)
        else:
            self.array = np.memmap(self._activations_file.name,
                                   dtype=float,
                                   mode='w+',
                                   shape=array.shape,
                                   order=order)
            self.array[:] = array[:]

    def concatenate(self, array):
        previous_shape = self.array.shape
        new_shape = self._decide_new_shape(previous_shape, array.shape)
        self._expand_array(new_shape)
        self._copy_new_values(previous_shape, array)

    '''
    It returns the transposed matrix in a new file. Remember that you would need to delete two files instead of one, as 
    the transposed matrix does not share file with the original matrix
    '''

    def transpose(self):
        cp = copy.copy(self)
        cp._concatenation_mode = ConcatenationMode(3 - cp._concatenation_mode.value)
        cp.array = np.memmap(cp._activations_file.name,
                             dtype=float,
                             mode='r+',
                             shape=tuple(reversed(cp.array.shape)),
                             order=cp._decide_order())
        return cp

    def delete_matrix(self):
        self._activations_file.close()  # Automatically deletion when the file is closed
        del self.array  # It deletes the object (not file)

    def _expand_array(self, new_shape):
        self.array = np.memmap(self._activations_file.name,
                               dtype=float,
                               mode='r+',
                               shape=new_shape,
                               order=self._decide_order())

    def _copy_new_values(self, previous_shape, array):
        if self._concatenation_mode == ConcatenationMode.COLUMN:
            self.array[:, previous_shape[1]:] = array[:]
        elif self._concatenation_mode == ConcatenationMode.ROW:
            self.array[previous_shape[0]:, :] = array[:]

    def _decide_order(self):
        return 'F' if self._concatenation_mode == ConcatenationMode.COLUMN else 'C'

    def _decide_new_shape(self, previous_shape, concatenation_matrix_shape):
        rows = previous_shape[0]
        if self._concatenation_mode == ConcatenationMode.ROW:
            rows += concatenation_matrix_shape[0]
        columns = previous_shape[1]
        if self._concatenation_mode == ConcatenationMode.COLUMN:
            columns += concatenation_matrix_shape[1]
        return rows, columns


def _create_folder_if_not_existing():
    if not os.path.exists(FOLDER):
        os.makedirs(FOLDER)
