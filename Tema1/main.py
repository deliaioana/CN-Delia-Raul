import numpy as np

# Resources:
# https://www.youtube.com/watch?v=OSelhO6Qnlc
# https://www.topcoder.com/thrive/articles/strassenss-algorithm-for-matrix-multiplication

# TODO: check determinant
# TODO: random generate matrices and 'n'
# TODO: run strassen with all posible minimums (ex.: 4 -> 4, 2, 1) and compare times

A = np.array([
    [3, 5, 1, 3],
    [1, 2, 3, 4],
    [4, 5, 6, 8],
    [7, 8, 9, 3],
])

B = np.array([
    [4, 1, 2, 3],
    [1, 2, 1, 6],
    [2, 4, 6, 2],
    [6, 2, 5, 4],
])


def ex1():
    u = 1
    count = 0
    while 1 + u != 1:
        u /= 10
        count += 1
    print(u, count)
    return u


def ex2(machine_precision):
    one = 1.0
    if (one + machine_precision) + machine_precision == one + (machine_precision + machine_precision):
        print('Equal')
    else:
        print('Not equal')


def dumb_multiply(A, B, n):
    # each cell is sum of products on A row and B column

    C = np.array([[0] * n for x in range(n)])
    for row in range(n):
        for column in range(n):
            # getter is used to get the A column and B row
            for getter in range(n):
                C[row][column] += A[row][getter] * B[getter][column]
    return C


def multiply_Strassen(A, B, n, n_min):
    if n == n_min:
        # if matrixes are small enough
        return dumb_multiply(A, B, n)

    else:
        # split matrix A
        A11 = A[:n // 2, :n // 2]
        A12 = A[:n // 2, n // 2:]
        A21 = A[n // 2:, :n // 2]
        A22 = A[n // 2:, n // 2:]

        # split matrix B
        B11 = B[:n // 2, :n // 2]
        B12 = B[:n // 2, n // 2:]
        B21 = B[n // 2:, :n // 2]
        B22 = B[n // 2:, n // 2:]

        # recursively calculate result parts
        P1 = multiply_Strassen(A11 + A22, B11 + B22, n // 2, n_min)
        P2 = multiply_Strassen(A21 + A22, B11, n // 2, n_min)
        P3 = multiply_Strassen(A11, B12 - B22, n // 2, n_min)
        P4 = multiply_Strassen(A22, B21 - B11, n // 2, n_min)
        P5 = multiply_Strassen(A11 + A12, B22, n // 2, n_min)
        P6 = multiply_Strassen(A21 - A11, B11 + B12, n // 2, n_min)
        P7 = multiply_Strassen(A12 - A22, B21 + B22, n // 2, n_min)

        # combine result parts
        C11 = P1 + P4 - P5 + P7
        C12 = P3 + P5
        C21 = P2 + P4
        C22 = P1 + P3 - P2 + P6

        # construct result
        C = np.vstack((np.hstack((C11, C12)), np.hstack((C21, C22))))
        return C


# ex2(ex1())
# dumb_multiply(A, B, 4)
print(multiply_Strassen(A, B, 4, 1))
