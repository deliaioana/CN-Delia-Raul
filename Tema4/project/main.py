import numpy
import numpy as np
import gradio as gr

# Given variables
PRECISION = 10 ** (-6)
demo = gr.Blocks()
text = ''


def get_input_from_files(file_a, file_b):
    global text
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

        # text += f"DIAGONAL: {diagonal}\n"
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


def get_input(version: int):
    global text
    variables = get_input_from_files(f"input_files/a_v{int(version)}.txt", f"input_files/b_v{int(version)}.txt")

    # text += f"N (for matrix A): {variables[1]}\n" \
    #         f"MATRIX A: \n{variables[0]}\n" \
    #         f"N (for vector B): {variables[3]}\n" \
    #         f"VECTOR B: {variables[2]}\n" \
    #         f"PRECISION: {variables[4]}\n"
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


def solve_with_gauss_seidel(matrix_a, vector_b, n):
    x = [0.] * n

    k = 0

    while True:
        dif_sum = 0.
        for i in range(n):
            val = vector_b[i]
            diagonal_element = 0.

            for pair in matrix_a[i]:
                element, j = pair
                if i != j:
                    val = val - element * x[j]
                else:
                    diagonal_element = element

            val = val / diagonal_element
            dif = abs(x[i] - val)
            dif_sum += dif * dif
            x[i] = val
        norm = numpy.sqrt(dif_sum)

        k += 1
        if norm < PRECISION or k > 10000 or norm > 10 ** 8:
            break

    if norm < PRECISION:
        return x, k

    return 'Divergence'


def compute_error(matrix_a, x, vector_b, diagonal):
    matrix_a_x = []

    for i in range(len(matrix_a)):
        row_sum = 0.
        for j in range(len(matrix_a[i])):
            row_sum += matrix_a[i][j][0] * x[matrix_a[i][j][1]]
        matrix_a_x.append(row_sum)

    return np.linalg.norm(np.array(matrix_a_x) - np.array(vector_b), ord=np.inf)


def solve(version: int):
    global text
    text = ''

    # Task 1
    variables = get_input(version)
    matrix_a, na, vector_b, nb, precision, diagonal = variables

    answer = has_zero_on_diagonal(matrix_a, na)
    text += f"ARE ALL ELEMENTS ON THE DIAGONAL NON-ZERO? {not answer}\n"
    print('ARE ALL ELEMENTS ON THE DIAGONAL NON-ZERO?')
    print(not answer)

    # Task 2
    if not answer:
        solution = solve_with_gauss_seidel(matrix_a, vector_b, na)

        if solution == 'Divergence':
            text += "Divergence\n"
            print(solution)
        else:
            x, number_of_iterations = solution
            # text += f"SOLUTION FOR X: {x}\n" \
            text += f"NUMBER OF ITERATIONS: {number_of_iterations}\n"
            print('SOLUTION FOR X: ', x)
            print('NUMBER OF ITERATIONS: ', number_of_iterations)

            # Task 3
            norm = compute_error(matrix_a, x, vector_b, diagonal)
            text += f"NORM: {norm}\n" \
                    f"IS IT LOWER THAN PRECISION? {(norm < precision)}\n"
            print('NORM: ', norm)
            print('IS IT LOWER THAN PRECISION? ', (norm < precision))

    return text


def run():
    with demo:
        solve_area = gr.Textbox(label="Solution:")
        with gr.Row():
            solve_button_v1 = gr.Button("Solve set 1")
            solve_button_v2 = gr.Button("Solve set 2")
            solve_button_v3 = gr.Button("Solve set 3")

            solve_button_v1.click(solve, inputs=[gr.Number(0, visible=False)], outputs=solve_area)
            solve_button_v2.click(solve, inputs=[gr.Number(1, visible=False)], outputs=solve_area)
            solve_button_v3.click(solve, inputs=[gr.Number(2, visible=False)], outputs=solve_area)

    demo.launch()


run()
