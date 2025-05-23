"""
here some functions to don't change functions or methods, only calling test functions
"""

import numpy as np


def expanding_dims_of_piece_vector(x_array):
    argmax_matrix = np.argmax(x_array[:, :, :-1], axis=-1) + 7 * x_array[:, :, -1]
    batch_size = x_array.shape[0]
    expand_table_rep = np.zeros(shape=(batch_size, 64, 14), dtype='float32')

    for b in range(batch_size):
        for i, element in enumerate(np.nditer(argmax_matrix)):
            element = int(element)
            expand_table_rep[b, i, element] = 1

    return np.reshape(expand_table_rep, (batch_size, 8, 8, 14))
