import numpy as np
import math


def normalize_matrix(matrix):
    return matrix / 8


def idx_to_x_y(idx, board_size):
    x = math.floor(idx/board_size)
    y = idx - x * board_size

    return x, y


def x_y_to_idx(x, y, board_size):
    return x * board_size + y
