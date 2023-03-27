# imports
import copy

import numpy as np
import random

# Given variables
MATRIX_A = [[0., 0., 4.],
            [1., 2., 3.],
            [0., 1., 2.]]
N = 3
PRECISION = 10 ** (-6)
S = [3., 2., 1.]


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
    b = [0.] * n
    for i in range(n):
        sum_i = 0
        for j in range(n):
            sum_i += s[j] * matrix_a[i][j]
        b[i] = sum_i
    return b


def compute_qr_decomposition(matrix_a, n, b, precision):
    matrix_q_tilda = np.identity(n)
    u = [0.] * n

    for r in range(n-1):
        sigma = sum([matrix_a[i][r] * matrix_a[i][r] for i in range(r, n)])
        if sigma < precision:
            break
        k = np.sqrt(sigma)
        if matrix_a[r][r] > 0:
            k = -k
        beta = sigma - k * matrix_a[r][r]
        u[r] = matrix_a[r][r] - k

        for i in range(r+1, n):
            u[i] = matrix_a[i][r]

        for j in range(r+1, n):
            gamma = sum([u[i] * matrix_a[i][j] for i in range(r, n)]) / beta
            for i in range(r, n):
                matrix_a[i][j] = matrix_a[i][j] - gamma * u[i]

        matrix_a[r][r] = k
        for i in range(r+1, n):
            matrix_a[i][r] = 0

        gamma = sum([u[i] * b[i] for i in range(r, n)]) / beta

        for i in range(r, n):
            b[i] = b[i] - gamma * u[i]

        for j in range(n):
            gamma = sum([u[i] * matrix_q_tilda[i][j] for i in range(r, n)]) / beta
            for i in range(r, n):
                matrix_q_tilda[i][j] = matrix_q_tilda[i][j] - gamma * u[i]

    return np.transpose(matrix_q_tilda), matrix_a


def compute_x_qr_with_library(matrix_a, b):
    matrix_q, matrix_r = np.linalg.qr(matrix_a)
    matrix_qb = np.matmul(np.transpose(matrix_q), b)
    x_qr = list(np.linalg.solve(matrix_r, matrix_qb))
    return x_qr


def compute_x_householder(matrix_q, matrix_r, b):
    x_householder = list(np.linalg.solve(matrix_r, b))
    return x_householder


def compute_difference(x_qr, x_householder):
    # return np.linalg.norm([x_qr[i] - x_householder[i] for i in range(len(x_qr))])
    return np.linalg.norm(np.array(x_qr) - np.array(x_householder))


def compute_errors(matrix_a, x_householder, x_qr, b, s, precision):
    err_1 = np.linalg.norm(np.dot(matrix_a, x_householder) - b)
    err_2 = np.linalg.norm(np.dot(matrix_a, x_qr) - b)
    err_3 = np.linalg.norm(np.array(x_householder) - np.array(s)) / np.linalg.norm(s)
    err_4 = np.linalg.norm(np.array(x_qr) - np.array(s)) / np.linalg.norm(s)

    print(err_1 < precision, err_2 < precision, err_3 < precision, err_4 < precision)


def invert_using_qr_desc(matrix_a, matrix_q, matrix_r, n):
    if np.linalg.det(matrix_a) == 0:
        return 'Impossible'

    inverted_matrix = []

    for j in range(n):
        ej = [0.] * n
        for i in range(n):
            ej[i] = int(i == j)

        b = np.matmul(np.transpose(matrix_q), np.transpose(ej))
        x_star = compute_x(matrix_r, b, n)
        inverted_matrix.append(x_star)

    return np.transpose(inverted_matrix)


def compute_x(matrix_a, y, size):
    x = [0.0] * size
    for i in range(size-1, -1, -1):
        sum_1 = 0.
        for j in range(i+1, size):
            sum_1 += matrix_a[i][j] * x[j]
        x[i] = (y[i] - sum_1) / matrix_a[i][i]
    return x


def compare_inverses(inverted_a, lib_inverted_a, precision):
    print('MATRIX A INVERTED: \n', inverted_a)
    print('MATRIX A INVERTED WITH LIB: \n', lib_inverted_a)

    norm = np.linalg.norm(np.array(inverted_a) - np.array(lib_inverted_a))
    return norm < precision


def run():
    # Task 6
    variables = get_input()
    matrix_a, n, precision, s = variables
    matrix_a_copy = copy.deepcopy(matrix_a)

    # Task 1
    b = compute_vector_b(matrix_a, s, n)
    b_copy = copy.deepcopy(b)
    print('VECTOR B: \n', b)

    # Task 2
    matrix_q, matrix_r = compute_qr_decomposition(matrix_a, n, b, precision)
    print('Q: \n', matrix_q)
    print('R: \n', matrix_r)

    # Task 3
    x_qr = compute_x_qr_with_library(matrix_a, b)
    print('X QR WITH LIBRARY: ', x_qr)
    x_householder = compute_x_householder(matrix_q, matrix_r, b)
    print('X HOUSEHOLDER WITH LIBRARY: ', x_householder)
    print('DIFFERENCE: ', compute_difference(x_qr, x_householder))

    # Task 4
    compute_errors(matrix_a_copy, x_householder, x_qr, b_copy, s, precision)

    # Task 5
    inverted_a = invert_using_qr_desc(matrix_a_copy, matrix_q, matrix_r, n)
    if isinstance(inverted_a, str):
        print('A is not invertible')
    else:
        lib_inverted_a = np.linalg.inv(matrix_a_copy)
        print(compare_inverses(inverted_a, lib_inverted_a, precision))


run()
# please work
