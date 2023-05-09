import random

import numpy as np
import gradio as gr

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

E = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]])

F = np.array([
    [10.0, 11.0],
    [20.0, 21.0],
    [30.0, 31.0]])

G = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [0.0, 0.0, 0.0]])

H = np.array([
    [10.0, 11.0, 0.0],
    [20.0, 21.0, 0.0],
    [30.0, 31.0, 0.0]])

demo = gr.Blocks()
precision = 0


def ex1():
    global precision
    print('-------------------------------------')
    print('Ex 1\n')

    u = 1
    count = 0
    while 1 + u != 1:
        u /= 10
        count += 1

    print('u = ', u)
    print('m = ', count)
    precision = u

    return precision


def ex2(machine_precision):
    print('-------------------------------------')
    print('Ex 2\n')

    one = 1.0
    left_sum = (one + machine_precision) + machine_precision
    right_sum = one + (machine_precision + machine_precision)

    text = ''
    if left_sum == right_sum:
        text += "Equal sums\n"
        print('Equal sums')
    else:
        text += f"Equal sums\n"
        print('Not equal')
        print('(', one, '+', machine_precision, ') +', machine_precision, '==', left_sum)
        print(one, '+ (', machine_precision, '+', machine_precision, ') ==', right_sum)

    second_text = find_multiplication_example()
    return text + "\n" + second_text


def find_multiplication_example():
    found = False
    counter = 0
    text = ''

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

    text += f"Found random in {counter} tries"
    return text


def print_ex2():
    global precision
    text = ex2(precision)
    return text


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
    n = int(n)
    n_min = int(n_min)

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


def print_ex3(version: int):
    if version == 1:
        return multiply_strassen(A, B, 4, 1)
    elif version == 2:
        return multiply_strassen(C, D, 2, 1)
    else:
        return "Something went wrong"


def get_max_dimension(matrix_a, matrix_b):
    return max(len(matrix_a), len(matrix_b), len(matrix_a[0]), len(matrix_b[0]))


def apply_zeros(matrix, n):
    while matrix.shape[0] < n:
        row_of_zeros = [0.0] * matrix.shape[1]
        matrix = np.vstack([matrix, row_of_zeros])

    while matrix.shape[1] < n:
        column_of_zeros = np.array([0.0] * n)
        matrix = np.column_stack((matrix, column_of_zeros))

    return matrix


def get_next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def multiply_strassen_for_non_square_matrices(matrix_a, matrix_b):
    real_n = matrix_a.shape[0]
    real_m = matrix_b.shape[1]

    n = get_max_dimension(matrix_a, matrix_b)
    n = get_next_power_of_2(n)
    first_matrix = apply_zeros(matrix_a, n)
    second_matrix = apply_zeros(matrix_b, n)

    print(n)
    print(first_matrix)
    print(second_matrix)
    print(real_n)

    return multiply_strassen(first_matrix, second_matrix, n, 1)[:real_n, :real_m]


def generate_random_matrix(size, position):
    n = np.random.randint(1, 100)

    if position == 1:
        return np.random.rand(n, size)
    else:
        return np.random.rand(size, n)


def check_matrices_equality(result, lib_mul):
    current_precision = 10 ** (-6)
    for i in range(len(result)):
        for j in range(len(result[i])):
            if abs(result[i][j] - lib_mul[i][j]) > current_precision:
                return False
    return True


def print_bonus(version: int):
    if version == 1:
        result = multiply_strassen_for_non_square_matrices(E, F)
        lib_mul = np.matmul(E, F)
        text = f"First matrix:\n {E}\n\nSecond matrix:\n {F}\n\nMultiplication:\n {result}\n\nLibrary check:\n{lib_mul}"

    else:
        common_size = np.random.randint(1, 100)

        matrix_a = generate_random_matrix(common_size, 1)
        matrix_b = generate_random_matrix(common_size, 2)

        print('SHAPES', matrix_a.shape, matrix_b.shape)

        result = multiply_strassen_for_non_square_matrices(matrix_a, matrix_b)
        lib_mul = np.matmul(matrix_a, matrix_b)
        text = f"First matrix:\n {matrix_a}\n\nSecond matrix:\n {matrix_b}\n\nMultiplication:\n {result}\n\nLibrary " \
               f"check:\n{lib_mul}\n\nAre they equal? "

        correct = check_matrices_equality(result, lib_mul)
        text += str(correct)

    return text


def run():
    # global precision
    # precision = ex1()
    # ex2(precision)
    # ex3(A, B, 4, 1)
    # print('Simple multiplication: ', np.dot(A, B))
    # ex3(C, D, 2, 1)

    with demo:
        gr.Markdown(f"Exercise 1")
        ex1_button = gr.Button("Find machine precision")
        ex1_solution = gr.Textbox()
        ex1_button.click(ex1, outputs=ex1_solution)

        gr.Markdown(f"Exercise 2")
        ex2_button = gr.Button("Check property")
        ex2_solution = gr.Textbox()
        ex2_button.click(print_ex2, outputs=ex2_solution)

        gr.Markdown(f"Exercise 3")
        ex3_solution = gr.Textbox()
        with gr.Row():
            ex3_button_strassen_v1 = gr.Button(f"Strassen set 1")
            ex3_button_strassen_v2 = gr.Button(f"Strassen set 2")

            ex3_button_strassen_v1.click(print_ex3, inputs=[gr.Number(1, visible=False)], outputs=ex3_solution)
            ex3_button_strassen_v2.click(print_ex3, inputs=[gr.Number(2, visible=False)], outputs=ex3_solution)

        gr.Markdown(f"Bonus")
        bonus_solution = gr.Textbox()
        with gr.Row():
            bonus_button_v1 = gr.Button(f"Preset")
            bonus_button_v2 = gr.Button(f"Random input")

            bonus_button_v1.click(print_bonus, inputs=[gr.Number(1, visible=False)], outputs=bonus_solution)
            bonus_button_v2.click(print_bonus, inputs=[gr.Number(2, visible=False)], outputs=bonus_solution)

    demo.launch()


run()
