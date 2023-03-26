# imports
import numpy as np
import random

# Given variables
MATRIX_A = [[0., 0., 4.],
            [1., 2., 3.],
            [0., 1., 2.]]
N = 3
PRECISION = 10 ** (-15)
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
    compute_errors(matrix_a, x_householder, x_qr, b, s)

    # Task 5
    inverted_a = invert_using_qr_desc(matrix_q)
    lib_inverted_a = invert_using_lib(matrix_a)
    compare_inverses(inverted_a, lib_inverted_a)


run()
# please work
