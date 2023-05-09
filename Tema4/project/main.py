import numpy
import numpy as np
import gradio as gr

# Given variables
PRECISION = 10 ** (-6)
demo = gr.Blocks()
text = ''


def get_matrix_from_file(file):
    with open(file) as f:
        size_a = [int(x) for x in next(f).split()][0]
        matrix = [[] for _ in range(size_a)]

        for line in f:
            values = line.replace(', ', ' ').split()
            value = float(values[0])
            row = int(values[1])
            col = int(values[2])

            new_value = value

            current_pair = [pair[0] for pair in matrix[row] if pair[1] == col]
            if current_pair:
                current_value = current_pair[0]
                matrix[row].remove((current_value, col))

                new_value = current_value + value

            matrix[row].append((new_value, col))
        return matrix


def get_input_from_files(file_a, file_b):
    global text
    matrix_a = get_matrix_from_file(file_a)

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


def get_element_from_position(i, j, matrix):
    row = matrix[i]
    print(i, j, row)
    elements = [x for x in row if x[1] == j]
    return elements[0][0]


def compute_sum_of_matrices_and_check_with_file(matrix_a, matrix_b, sum_matrix, precision):
    if len(matrix_a) != len(matrix_b):
        return 1, False

    size = len(matrix_a)
    matrix_c = [[] for _ in range(size)]

    for i in range(size):
        for j in range(size):
            elem_a = next((pair[0] for pair in matrix_a[i] if pair[1] == j), 0)
            elem_b = next((pair[0] for pair in matrix_b[i] if pair[1] == j), 0)

            elem_sum = elem_a + elem_b
            if abs(elem_sum) > precision:
                matrix_c[i].append((elem_sum, j))

    if size != len(sum_matrix):
        return 2, False

    for i in range(size):
        if len(matrix_c[i]) != len(sum_matrix[i]):
            return 3, False
        for element_number in range(len(matrix_c[i])):
            j = matrix_c[i][element_number][1]

            element_from_file = matrix_c[i][element_number][0]
            element_from_sum = get_element_from_position(i, j, sum_matrix)

            if abs(element_from_file - element_from_sum) > precision:
                return (element_from_file, element_from_sum, i, j, sum_matrix), False

    return sum_matrix, True


def bonus():
    matrix_a = get_matrix_from_file('input_files/bonus_a.txt')
    matrix_b = get_matrix_from_file('input_files/bonus_b.txt')
    file_sum = get_matrix_from_file('input_files/bonus_sum.txt')
    computed_sum, correct = compute_sum_of_matrices_and_check_with_file(matrix_a, matrix_b, file_sum, PRECISION)

    return matrix_a, matrix_b, file_sum, computed_sum, correct


def get_object(obj):
    return obj


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

        matrix_a, matrix_b, sum_from_file, computed_sum, correct = bonus()

        bonus_area_1 = gr.Textbox(label="Bonus\nMatrix A:")
        solve_button_v1 = gr.Button("Get matrix A")

        bonus_area_2 = gr.Textbox(label="Matrix B:")
        solve_button_v2 = gr.Button("Get matrix B")

        bonus_area_3 = gr.Textbox(label="Sum from file:")
        solve_button_v3 = gr.Button("Get sum from file")

        bonus_area_4 = gr.Textbox(label="Compute sum:")
        solve_button_v4 = gr.Button("Compute sum")

        bonus_area_5 = gr.Textbox(label="Correct?")
        solve_button_v5 = gr.Button("Correct?")

        solve_button_v1.click(get_object, inputs=[gr.Variable(matrix_a)], outputs=bonus_area_1)
        solve_button_v2.click(get_object, inputs=[gr.Variable(matrix_b)], outputs=bonus_area_2)
        solve_button_v3.click(get_object, inputs=[gr.Variable(sum_from_file)], outputs=bonus_area_3)
        solve_button_v4.click(get_object, inputs=[gr.Variable(computed_sum)], outputs=bonus_area_4)
        solve_button_v5.click(get_object, inputs=[gr.Variable(correct)], outputs=bonus_area_5)

    demo.launch()


run()
