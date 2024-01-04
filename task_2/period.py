import numpy as np
from numpy.linalg import matrix_power
from math import gcd
from functools import reduce


def get_example_matrix_str(text: str) -> np.ndarray:
    """
    Text to matrix
    """
    data_list = [float(number) for number in text.split()]

    mat = np.array(data_list).reshape(10, 10)

    return mat


# Define the function as per the user's description
def find_cycles_with_limit(matrix, start_row, loop_n=1):
    """
    Find cycles starting from the given row by following non-zero indices, limited by loop_n.
    """

    def get_nonzero_indices(row):
        """Get all indices with non-zero values in the given row."""
        return [i + 1 for i, value in enumerate(row) if value > 0]  # Convert to 1-indexed

    visited_rows = set()  # Keep track of visited rows to detect a cycle
    cycles = []  # Store the sequences of rows in cycles
    current_row = start_row
    loops_count = 0  # Count the number of complete loops
    period = 0
    while loops_count < loop_n:
        if current_row in visited_rows:
            # If the current row has already been visited, we've completed a loop
            loops_count += 1
            # If we've reached the desired number of loops, break the loop
            if loops_count == loop_n:
                period = len(visited_rows)
                break
            visited_rows.clear()  # Clear visited rows for the next loop

        visited_rows.add(current_row)
        row_indices = get_nonzero_indices(matrix[current_row - 1])  # Get non-zero indices of the current row
        cycles.append(row_indices)  # Add the non-zero indices as a cycle
        if not row_indices:  # If no non-zero indices, break the loop
            break
        current_row = row_indices[0]  # Continue with the first non-zero index

    return cycles, period


P = get_example_matrix_str(
    "0.0000 0.2953 0.4076 0.0000 0.0000 0.0000 0.0000 0.0000 0.2971 0.0000 0.0000 0.0000 0.0000 0.0000 0.5211 0.0000 0.0000 0.4789 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.5498 0.0000 0.0000 0.4502 0.0000 0.0000 0.0000 0.4618 0.2558 0.0000 0.0000 0.0000 0.0000 0.0000 0.2825 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.3296 0.6478 0.0000 0.0000 0.0227 0.2880 0.0000 0.0000 0.7120 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0639 0.0000 0.0000 0.9361 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.4837 0.1798 0.0000 0.0000 0.3364 0.0000 0.0000 0.0000 0.0000 0.5858 0.0000 0.0000 0.4142 0.0000 0.0000 0.8494 0.0000 0.0000 0.1506 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000")
print(P)

cycles_result, period = find_cycles_with_limit(P, 2, loop_n=3)
cyclic_classes = cycles_result[:period]
for i, cycle in enumerate(cyclic_classes):
    print(f"C{i} = {cycle}")
