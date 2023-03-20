# imports
import numpy as np
import random

# Given variables
MATRIX_A = []
N = 3
PRECISION = 2
S = []


def generate_symmetrical_matrix(size):
    matrix_a = np.array([[0.0] * size] * size)
    for i in range(size):
        for j in range(size):
            if i <= j:
                matrix_a[i][j] = round(random.random()*100, 2)
            else:
                matrix_a[i][j] = matrix_a[j][i]
    return matrix_a


def randomize_n():
    return random.randint(1, 200)


def randomize_precision():
    power = random.randint(1, 15)
    return 10 ** (-power)


def randomize_vector_s(n):
    return np.random.uniform(size=n)


def randomize_input():
    n = randomize_n()
    matrix_a = generate_symmetrical_matrix(n)
    precision = randomize_precision()
    s = randomize_vector_s(n)
    return matrix_a, n, precision, s


def get_input():
    answer = input('Do you want to use specific input? y/n\n')
    if answer == 'y':
        variables = MATRIX_A, N, PRECISION, S
    else:
        variables = randomize_input()
    print('The variables are: \n')
    print('MATRIX A: \n', variables[0])
    print('N: ', variables[1])
    print('PRECISION: ', variables[2])
    print('VECTOR S: ', variables[3])
    return variables


def compute_vector_b(matrix_a, s, n):
    pass


def compute_qr_decomposition(matrix_a):
    pass


def compute_x_qr_with_library(matrix_a, b):
    pass


def compute_x_householder(qr_dec):
    pass


def compute_difference(x_qr, x_householder):
    pass


def compute_errors(matrix_a, x_householder, x_qr, b, s):
    pass


def invert_using_qr_desc(qr_dec):
    pass


def invert_using_lib(matrix_a):
    pass


def compare_inverses(inverted_a, lib_inverted_a):
    pass


def run():
    # Task 6
    variables = get_input()
    matrix_a, n, precision, s = variables

    # Task 1
    b = compute_vector_b(matrix_a, s, n)

    # Task 2
    qr_dec = compute_qr_decomposition(matrix_a)

    # Task 3
    x_qr = compute_x_qr_with_library(matrix_a, b)
    x_householder = compute_x_householder(qr_dec)
    compute_difference(x_qr, x_householder)

    # Task 4
    compute_errors(matrix_a, x_householder, x_qr, b, s)

    # Task 5
    inverted_a = invert_using_qr_desc(qr_dec)
    lib_inverted_a = invert_using_lib(matrix_a)
    compare_inverses(inverted_a, lib_inverted_a)


run()
