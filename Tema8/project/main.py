# HK 8
import math

PRECISION = 10 ** (-6)
function_selector = 0
H = 0


def function(x: float):
    if function_selector == 1:
        return function_v1(x)
    elif function_selector == 2:
        return function_v2(x)
    elif function_selector == 3:
        return function_v3(x)


def function_v1(x: float):
    return ((x ** 3) * (1 / 3)) - (2 * (x ** 2)) + (2 * x) + 3


def function_v2(x: float):
    return (x ** 2) * math.sin(x)


def function_v3(x: float):
    return (x ** 4) - (6 * (x ** 3)) + (13 * (x ** 2)) - (12 * x) + 4


def get_input():
    global function_selector, H
    function_selector = int(input("Select example: 1 / 2 / 3 ..."))
    power = int(input("Select power of h: "))
    H = 10 ** (-power)


def secant_method():
    if function_selector == 1:
        x = 3.41
        x0 = 3.40
    elif function_selector == 2:
        x = -0.44
        x0 = -0.43
    elif function_selector == 3:
        x = 0.99
        x0 = 0.98
    k = 0


def first_derivative_v1(x: float):
    return ((3 * function(x)) -
            (4 * function(x - H)) +
            (function(x - (2 * H)))) / (2 * H)


def first_derivative_v2(x: float):
    return (-(function(x + (2 * H))) +
            (8 * function(x + H)) -
            (8 * function(x - H)) +
            (function(x - (2 * H)))) / (12 * H)


def second_derivative(x: float):
    return (-(function(x + (2 * H))) +
            (16 * function(x + H)) -
            (30 * function(x)) +
            (16 * function(x - H)) -
            (function(x - (2 * H)))) / (12 * (H ** 2))


def run():
    get_input()
