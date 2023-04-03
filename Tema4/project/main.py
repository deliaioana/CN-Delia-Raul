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


def randomize_n():
    return random.randint(1, 200)


def randomize_precision():
    power = random.randint(1, 15)
    return 10 ** (-power)


def randomize_vector_s(size):
    return np.random.uniform(size=size)


def generate_random_sparse_matrix(size):
    number_of_elements = random.randint(max(0, size - int(size / 4)), size + int(size / 4))
    matrix_a = []

    for i in range(size):
        row = []


def generate_specific_sparse_matrix(size):
    pass


def randomize_input(allow_zero):
    n = randomize_n()

    if allow_zero:
        matrix_a = generate_random_sparse_matrix(n)
    else:
        matrix_a = generate_specific_sparse_matrix(n)

    vector_b = randomize_vector_s(n)
    precision = randomize_precision()
    return matrix_a, n, vector_b, n, precision,


def get_input():
    answer = input('Do you want to use specific input? y/n\n')
    if answer == 'y':
        variables = MATRIX_A, NA, VECTOR_B, NB, PRECISION
    else:
        answer = input('Do you allow zero values on the main diagonal? y/n\n')
        if answer == 'y':
            variables = randomize_input(True)
        else:
            variables = randomize_input(False)

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
