import numpy as np
import random
import gradio as gr
import scipy as scipy

MATRIX_V1 = np.array([[2., 4., 5.], [4., 12., 14.], [5., 14., 19.5]])
SIZE_V1 = 3
PRECISION_V1 = 10 ** (-8)
VECT_V1 = [12., 38., 68.]

MATRIX_V2 = np.array([[1., 2.5, 3.], [2.5, 8.25, 15.5], [3., 15.5, 43.]])
SIZE_V2 = 5
PRECISION_V2 = 10 ** (-8)
VECT_V2 = np.random.rand(5)

demo = gr.Blocks()


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
    diagonal_sum = 0.
    for k in range(col):
        diagonal_sum += d[k] * matrix_a[col][k] * matrix_a[col][k]
    d[col] = matrix_a[col][col] - diagonal_sum
    return d


def calculate_l(matrix_a, d, row, col):
    l_sum = 0.
    for k in range(col):
        l_sum += d[k] * matrix_a[row][k] * matrix_a[col][k]
    matrix_a[row][col] = (matrix_a[row][col] - l_sum) / d[col]

    return matrix_a


def calculate_z(matrix_a, size, vect):
    z = [0.0] * size
    for i in range(size):
        sum_1 = 0.
        for j in range(i):
            sum_1 += matrix_a[i][j] * z[j]
        z[i] = vect[i] - sum_1

    return z


def calculate_y(d, z, size):
    y = [0.0] * size
    for i in range(size):
        y[i] = z[i] / d[i]
    return y


def calculate_x(matrix_a, y, size):
    x = [0.0] * size
    for i in range(size-1, 0, -1):
        sum_1 = 0.
        for j in range(i+1, size):
            sum_1 += matrix_a[j][i] * x[j]
        x[i] = y[i] - sum_1
    return x


def generate_symmetrical_matrix(size):
    matrix_a = np.array([[0.0] * size] * size)
    for i in range(size):
        for j in range(size):
            if i <= j:
                matrix_a[i][j] = round(random.random()*100, 2)
            else:
                matrix_a[i][j] = matrix_a[j][i]
    return matrix_a


def solve(version: int):
    text = ''

    if version == 1:
        size = SIZE_V1
        precision = PRECISION_V1
        vect = VECT_V1
        matrix_a = MATRIX_V1.copy()

    elif version == 2:
        size = SIZE_V2
        precision = PRECISION_V2
        vect = VECT_V2
        matrix_a = generate_symmetrical_matrix(size)

    else:
        return "Something went wrong"

    copy = matrix_a.copy()

    result = decompose_matrix(matrix_a, [0.0] * size, precision)
    if not result:
        text += "There was a 0 in d\n"
        print("There was a 0 in d")
    else:
        matrix_a, d = result
        print('A: ', matrix_a)
        print('d: ', d)
        z = calculate_z(matrix_a, size, vect)
        print('Z: ', z)
        y = calculate_y(d, z, size)
        print('Y:', y)
        x = calculate_x(matrix_a, y, size)
        print('X: ', x)

        text += f"A: {matrix_a}\nd: {d}\nZ: {z}\nY: {y}\nX: {x}\n"

    # print(f'Det A: {np.linalg.det(copy)}')
    L = np.zeros((size, size), dtype=float)
    D = np.zeros((size, size), dtype=float)
    for i in range(size):
        for j in range(size):
            if i == j:
                L[i][j] = 1.
                D[i][j] = d[i]
            elif i < j:
                L[i][j] = 0.
                D[i][j] = 0.
            else:
                L[i][j] = matrix_a[i][j]
                D[i][j] = 0.
    Lt = np.transpose(L)

    text += f"Det LDLt: {np.linalg.det(L) * np.linalg.det(D) * np.linalg.det(Lt)}\n"
    print(f'Det LDLt: {np.linalg.det(L) * np.linalg.det(D) * np.linalg.det(Lt)}')

    auto_x = np.linalg.solve(copy, vect)
    text += f"{auto_x}\n"
    print(auto_x)

    auto_sol = np.linalg.norm(np.dot(copy, auto_x) - vect)
    text += f"Solution correct: {is_zero(auto_sol, precision)}\n"
    print(f'Solution correct: {is_zero(auto_sol, precision)}')

    return text


def run():
    with demo:
        solve_area = gr.Textbox(label="Solution:")
        with gr.Row():
            solve_button_v1 = gr.Button("Solve set 1")
            solve_button_v2 = gr.Button("Solve set 2")

            solve_button_v1.click(solve, inputs=[gr.Number(1, visible=False)], outputs=solve_area)
            solve_button_v2.click(solve, inputs=[gr.Number(2, visible=False)], outputs=solve_area)

    demo.launch()


run()
