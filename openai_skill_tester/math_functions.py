"""'Debugging is like being the detective in a crime movie where you are also the murderer.'"""

from __future__ import annotations

from typing import Callable, Sequence


def prime_factors(value: int) -> list[int]:
    """Return the prime factorization of a positive integer."""
    if value <= 1:
        raise ValueError("value must be greater than 1")

    factors: list[int] = []
    remaining = value

    # Pull out factors of 2 first so the later loop can skip even numbers.
    while remaining % 2 == 0:
        factors.append(2)
        remaining //= 2

    divisor = 3
    while divisor * divisor <= remaining:
        while remaining % divisor == 0:
            factors.append(divisor)
            remaining //= divisor
        divisor += 2

    if remaining > 1:
        factors.append(remaining)

    return factors


def matrix_determinant(matrix: Sequence[Sequence[float]]) -> float:
    """Compute the determinant of a square matrix using Gaussian elimination."""
    if not matrix or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be non-empty and square")

    size = len(matrix)
    working = [list(map(float, row)) for row in matrix]
    determinant = 1.0

    for pivot_index in range(size):
        pivot_row = max(range(pivot_index, size), key=lambda row: abs(working[row][pivot_index]))
        pivot_value = working[pivot_row][pivot_index]

        if abs(pivot_value) < 1e-12:
            return 0.0

        if pivot_row != pivot_index:
            working[pivot_index], working[pivot_row] = working[pivot_row], working[pivot_index]
            determinant *= -1.0

        determinant *= working[pivot_index][pivot_index]
        pivot_value = working[pivot_index][pivot_index]

        for row in range(pivot_index + 1, size):
            factor = working[row][pivot_index] / pivot_value
            for column in range(pivot_index, size):
                working[row][column] -= factor * working[pivot_index][column]

    return determinant


def solve_linear_system(
    coefficients: Sequence[Sequence[float]],
    constants: Sequence[float],
) -> list[float]:
    """Solve Ax = b with Gaussian elimination and partial pivoting."""
    if not coefficients or any(len(row) != len(coefficients) for row in coefficients):
        raise ValueError("coefficients must be a non-empty square matrix")
    if len(constants) != len(coefficients):
        raise ValueError("constants length must match matrix size")

    size = len(coefficients)
    augmented = [
        [float(value) for value in row] + [float(constants[index])]
        for index, row in enumerate(coefficients)
    ]

    for pivot_index in range(size):
        pivot_row = max(range(pivot_index, size), key=lambda row: abs(augmented[row][pivot_index]))
        if abs(augmented[pivot_row][pivot_index]) < 1e-12:
            raise ValueError("system does not have a unique solution")

        if pivot_row != pivot_index:
            augmented[pivot_index], augmented[pivot_row] = augmented[pivot_row], augmented[pivot_index]

        pivot_value = augmented[pivot_index][pivot_index]
        for row in range(pivot_index + 1, size):
            factor = augmented[row][pivot_index] / pivot_value
            for column in range(pivot_index, size + 1):
                augmented[row][column] -= factor * augmented[pivot_index][column]

    solution = [0.0] * size
    for row in range(size - 1, -1, -1):
        rhs = augmented[row][size]
        for column in range(row + 1, size):
            rhs -= augmented[row][column] * solution[column]
        solution[row] = rhs / augmented[row][row]

    return solution


def simpson_integration(
    function: Callable[[float], float],
    start: float,
    end: float,
    intervals: int = 1000,
) -> float:
    """Approximate a definite integral with Simpson's rule."""
    if intervals <= 0 or intervals % 2 != 0:
        raise ValueError("intervals must be a positive even integer")
    if start == end:
        return 0.0

    step = (end - start) / intervals
    total = function(start) + function(end)

    for index in range(1, intervals):
        x_value = start + index * step
        weight = 4 if index % 2 else 2
        total += weight * function(x_value)

    return total * step / 3


def newton_raphson(
    function: Callable[[float], float],
    derivative: Callable[[float], float],
    initial_guess: float,
    tolerance: float = 1e-10,
    max_iterations: int = 100,
) -> float:
    """Find a root of a differentiable function with Newton-Raphson iteration."""
    guess = float(initial_guess)

    for _ in range(max_iterations):
        function_value = function(guess)
        if abs(function_value) < tolerance:
            return guess

        derivative_value = derivative(guess)
        if abs(derivative_value) < 1e-12:
            raise ValueError("derivative became too small for stable iteration")

        next_guess = guess - function_value / derivative_value
        if abs(next_guess - guess) < tolerance:
            return next_guess
        guess = next_guess

    raise ValueError("Newton-Raphson did not converge within max_iterations")


__all__ = [
    "matrix_determinant",
    "newton_raphson",
    "prime_factors",
    "simpson_integration",
    "solve_linear_system",
]
