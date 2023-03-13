import numpy as np
import random

MATRIX = np.array([[2., 4., 5.], [4., 12., 14.], [5., 14., 19.5]])


def is_zero(number, precision):
    return abs(number) < precision


def decompose_matrix(matrix_a, d, precision):
    for i in range(matrix_a[0].size):
        d = calculate_d(matrix_a, i, d)
        if is_zero(d[i], precision):
            return False
        for k in range(i + 1, matrix_a[0].size):
            matrix_a = calculate_l(matrix_a, d, k, i)

    return matrix_a, d


def calculate_d(matrix_a, col, d):
    diagonal_sum = 0
    for k in range(col):
        diagonal_sum += d[k] * matrix_a[col][k] * matrix_a[col][k]
    d[col] = matrix_a[col][col] - diagonal_sum
    return d


def calculate_l(matrix_a, d, row, col):
    l_sum = 0
    for k in range(col):
        l_sum += d[k] * matrix_a[row][k] * matrix_a[col][k]
    matrix_a[row][col] = (matrix_a[row][col] - l_sum) / d[col]

    return matrix_a


def calculate_z():



def calculate_y():
    pass


def calculate_x():
    pass


def generate_symmetrical_matrix(size):
    matrix_a = np.array([[0.0] * size] * size)
    for i in range(size):
        for j in range(size):
            if i <= j:
                matrix_a[i][j] = round(random.random()*100, 2)
            else:
                matrix_a[i][j] = matrix_a[j][i]
    return matrix_a


def run(size, precision, vect):
    matrix_a = generate_symmetrical_matrix(size)
    result = decompose_matrix(MATRIX, [0.0] * size, precision)
    if not result:
        print("There was a 0 in d")
    else:
        print(result)


run(3, 0.01, [0.0, 0.1, 1.5])
