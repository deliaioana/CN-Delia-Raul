import random

import numpy as np

# Resources:
# https://www.youtube.com/watch?v=OSelhO6Qnlc
# https://www.topcoder.com/thrive/articles/strassenss-algorithm-for-matrix-multiplication

# TODO: check determinant
# TODO: random generate matrices and 'n'
# TODO: run strassen with all possible minimums (ex.: 4 -> 4, 2, 1) and compare times

A = np.array([
    [10, 10.4, 10, 10],
    [1, 1, 1, 1],
    [2, 3, 2, 3],
    [5, 5, 5, 5]])

B = np.array([
    [2, 1.2, 2, 3],
    [2, 1, 2, 2],
    [2, 2, 1, 2],
    [3, 2, 2, 2]])

C = np.array([
    [10.7, 10.6],
    [1.6, 1.9]])

D = np.array([
    [2.9, 2.0],
    [2.7, 1.8]])


def ex1():
    print('-------------------------------------')
    print('Ex 1\n')

    u = 1
    count = 0
    while 1 + u != 1:
        u /= 10
        count += 1

    print('u = ', u)
    print('m = ', count)
    return u


def ex2(machine_precision):
    print('-------------------------------------')
    print('Ex 2\n')

    one = 1.0
    left_sum = (one + machine_precision) + machine_precision
    right_sum = one + (machine_precision + machine_precision)

    if left_sum == right_sum:
        print('Equal sums')
    else:
        print('Not equal')
        print('(', one, '+', machine_precision, ') +', machine_precision, '==', left_sum)
        print(one, '+ (', machine_precision, '+', machine_precision, ') ==', right_sum)

    find_multiplication_example()


def find_multiplication_example():
    found = False
    counter = 0

    while not found:
        counter += 1
        a = random.random()
        b = random.random()
        c = random.random()

        left_product = (a * b) * c
        right_product = a * (b * c)

        if not left_product == right_product:
            print('Not equal')
            print('(', a, '+', b, ') +', c, '==', left_product)
            print(a, '+ (', b, '+', c, ') ==', right_product)

            print('Found random in ', counter, ' tries')
            break


def ex3(matrix_a, matrix_b, n, n_min):
    print('-------------------------------------')
    print('Ex 3\n')

    print('Matrix A:\n', matrix_a, '\n')
    print('Matrix B:\n', matrix_b, '\n')
    print('A * B:\n', multiply_strassen(matrix_a, matrix_b, n, n_min), '\n')


def dumb_multiply(matrix_a, matrix_b, n):
    # each cell is sum of products on A row and B column

    c = np.array([[0.0] * n for _ in range(n)])

    for row in range(n):
        for column in range(n):
            # getter is used to get the A column and B row
            for getter in range(n):
                c[row][column] += matrix_a[row][getter] * matrix_b[getter][column]
    return c


def multiply_strassen(matrix_a, matrix_b, n, n_min):
    if n == n_min:
        # if matrices are small enough
        return dumb_multiply(matrix_a, matrix_b, n)
        # return np.dot(matrix_a, matrix_b)

    else:
        # split matrix A
        a11 = matrix_a[:n // 2, :n // 2]
        a12 = matrix_a[:n // 2, n // 2:]
        a21 = matrix_a[n // 2:, :n // 2]
        a22 = matrix_a[n // 2:, n // 2:]

        # split matrix B
        b11 = matrix_b[:n // 2, :n // 2]
        b12 = matrix_b[:n // 2, n // 2:]
        b21 = matrix_b[n // 2:, :n // 2]
        b22 = matrix_b[n // 2:, n // 2:]

        # recursively calculate result parts
        p1 = multiply_strassen(a11 + a22, b11 + b22, n // 2, n_min)
        p2 = multiply_strassen(a21 + a22, b11, n // 2, n_min)
        p3 = multiply_strassen(a11, b12 - b22, n // 2, n_min)
        p4 = multiply_strassen(a22, b21 - b11, n // 2, n_min)
        p5 = multiply_strassen(a11 + a12, b22, n // 2, n_min)
        p6 = multiply_strassen(a21 - a11, b11 + b12, n // 2, n_min)
        p7 = multiply_strassen(a12 - a22, b21 + b22, n // 2, n_min)

        # combine result parts
        c11 = p1 + p4 - p5 + p7
        c12 = p3 + p5
        c21 = p2 + p4
        c22 = p1 + p3 - p2 + p6

        # construct result
        c = np.vstack((np.hstack((c11, c12)), np.hstack((c21, c22))))
        return c


precision = ex1()
ex2(precision)
ex3(A, B, 4, 1)
print('Simple multiplication: ', np.dot(A, B))

ex3(C, D, 2, 1)
