import numpy as np

# Given variables
MATRIX_A = [[(2.5, 2), (102.5, 0), (3.0, 2)],
            [(3.5, 0), (0.33, 4), (1.05, 2), (104.88, 1)],
            [(100.0, 2)],
            [(1.3, 1), (101.3, 3)],
            [(1.5, 3), (0.73, 0), (102.23, 4)]]
NA = 5
VECTOR_B = [6., 7., 8., 9., 1.]
NB = 5
PRECISION = 10 ** (-6)


def get_input_from_files(file_a, file_b):
    with open(file_a) as f:
        size_a = [int(x) for x in next(f).split()][0]
        matrix_a = [[] for _ in range(size_a)]

        for line in f:
            values = line.replace(', ', ' ').split()
            value = float(values[0])
            row = int(values[1])
            col = int(values[2])

            new_value = value

            current_pair = [pair[0] for pair in matrix_a[row] if pair[1] == col]
            if current_pair:
                current_value = current_pair[0]
                matrix_a[row].remove((current_value, col))

                new_value = current_value + value

            matrix_a[row].append((new_value, col))

        diagonal = []

        for count, row in enumerate(matrix_a):
            non_zero_elem = False
            for pair in row:
                if pair[1] == count:
                    diagonal.append(pair[0])
                    non_zero_elem = True
            if not non_zero_elem:
                diagonal.append(0.)

        print('DIAGONAL: ', diagonal)

    with open(file_b) as f:
        size_b = [int(x) for x in next(f).split()][0]
        vector_b = []

        for line in f:
            value = float(line.split()[0])
            vector_b.append(value)

    n = size_b
    precision = PRECISION

    return matrix_a, n, vector_b, n, precision, diagonal


def get_input():
    answer = input('Do you want to use the file input? y/n\n')
    if answer == 'y':
        variables = get_input_from_files('input_files/a.txt', 'input_files/b.txt')
    else:
        variables = MATRIX_A, NA, VECTOR_B, NB, PRECISION

    print('N (for matrix A): ', variables[1])
    print('MATRIX A: \n', variables[0])
    print('N (for vector B): ', variables[3])
    print('VECTOR B: \n', variables[2])
    print('PRECISION: \n', variables[4])
    return variables


def has_zero_on_diagonal(matrix_a, size):
    zeros_on_diagonal = [x for x in range(size)]

    for i, row in enumerate(matrix_a):
        for element, j in row:
            if i == j and i in zeros_on_diagonal:
                zeros_on_diagonal.remove(i)

    return bool(zeros_on_diagonal)


def solve_with_gauss_seidel(matrix_a, na, vector_b, nb):
    pass


def compute_error(matrix_a, x, vector_b, diagonal):
    matrix_a_x = []

    for i in range(len(matrix_a)):
        row_sum = 0.
        for j in range(len(matrix_a[i])):
            row_sum += matrix_a[i][j][0] * x[matrix_a[i][j][1]]
        matrix_a_x.append(row_sum + diagonal[i] * x[i])

    print(matrix_a_x)

    return np.linalg.norm(np.array(matrix_a_x) - np.array(vector_b), ord=np.inf)


def run():
    # Task 1
    variables = get_input()
    matrix_a, na, vector_b, nb, precision, diagonal = variables

    answer = has_zero_on_diagonal(matrix_a, na)
    print('ARE ALL ELEMENTS ON THE DIAGONAL NON-ZERO?')
    print(not answer)

    # Task 2
    # x, number_of_iterations = solve_with_gauss_seidel(matrix_a, na, vector_b, nb)
    # print('SOLUTION FOR X: ', x)
    # print('NUMBER OF ITERATIONS: ', number_of_iterations)

    # Task 3
    norm = compute_error(matrix_a, [1. for _ in range(na)], vector_b, diagonal)
    print('NORM: ', norm)
    print('IS IT LOWER THAN PRECISION? ', (norm < precision))


run()
