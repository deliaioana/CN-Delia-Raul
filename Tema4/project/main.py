import random
import numpy as np

# Given variables
MATRIX_A = [[(2.5, 2), (102.5, 0)],
            [(3.5, 0), (0.33, 4), (1.05, 2), (104.88, 1)],
            [(100.0, 2)],
            [(1.3, 1), (101.3, 3)],
            [(1.5, 3), (0.73, 0), (102.23, 4)]]
NA = 5
VECTOR_B = [6., 7., 8., 9., 1.]
NB = 5
PRECISION = 10 ** (-6)


def get_input_from_files(file_a, file_b):
    n = None

    matrix_a = None

    vector_b = None
    precision = None

    return matrix_a, n, vector_b, n, precision,


def get_input():
    answer = input('Do you want to use specific input? y/n\n')
    if answer == 'y':
        variables = MATRIX_A, NA, VECTOR_B, NB, PRECISION
    else:
        variables = get_input_from_files('a.txt', 'b.txt')

    print('N (for matrix A): ', variables[0])
    print('MATRIX A: \n', variables[1])
    print('N (for vector B): ', variables[2])
    print('VECTOR B: \n', variables[3])
    print('PRECISION: \n', variables[4])
    return variables


def has_zero_on_diagonal():
    pass


def solve_with_gauss_seidel(matrix_a, na, vector_b, nb):
    pass


def compute_error(matrix_a, x, vector_b):
    pass


def run():
    # Task 1
    variables = get_input()
    matrix_a, na, vector_b, nb, precision = variables

    answer = has_zero_on_diagonal()
    print('ARE ALL ELEMENTS ON THE DIAGONAL NON-ZERO?')
    print(answer)

    # Task 2
    x, number_of_iterations = solve_with_gauss_seidel(matrix_a, na, vector_b, nb)
    print('SOLUTION FOR X: ', x)
    print('NUMBER OF ITERATIONS: ', number_of_iterations)

    # Task 3
    norm = compute_error(matrix_a, x, vector_b)
    print('NORM: ', norm)
    print('IS IT LOWER THAN PRECISION? ', (norm < precision))


run()
