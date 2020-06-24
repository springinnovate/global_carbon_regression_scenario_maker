import numpy as np


def get_array_from_two_dim_first_order_kernel_function(
        radius, starting_value, halflife):
    diameter = radius * 2
    x = np.linspace(-radius, radius + 1, diameter)
    y = np.linspace(-radius, radius + 1, diameter)

    X, Y = np.meshgrid(x, y)
    output_array = np.zeros((int(diameter), int(diameter)))

    for i in range(int(diameter)):
        for j in range(int(diameter)):
            x = i - radius
            y = j - radius
            output_array[i, j] = two_dim_first_order_kernel_function(
                x, y, starting_value, halflife)

    return output_array


def two_dim_first_order_kernel_function(x, y, starting_value, halflife):
    # Chosen to have a decent level of curvature difference across interval
    steepness = 4 / halflife
    return regular_sigmoidal_first_order(
        (x ** 2 + y ** 2) ** 0.5, left_value=starting_value,
        inflection_point=halflife, steepness=steepness)


def regular_sigmoidal_first_order(
        x, left_value=1.0, inflection_point=5.0, steepness=1.0,
        magnitude=1.0, scalar=1.0):
    return scalar * sigmoidal_curve(
        x, left_value, inflection_point, steepness, magnitude)


def sigmoidal_curve(
        x, left_value, inflection_point, steepness, magnitude, e=np.e):
    return left_value / ((1. / magnitude) + e ** (
        steepness * (x - inflection_point)))
