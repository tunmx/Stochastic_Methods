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

# 计算周期d
def compute_period(matrix):
    n = matrix.shape[0]
    periods = []
    for i in range(n):
        # Find the first power where P^k[i, i] > 0
        for k in range(1, n*n+1):
            if matrix_power(matrix, k)[i, i] > 0:
                periods.append(k)
                break
    # The period d is the gcd of the list of periods
    return reduce(gcd, periods)

def find_cyclic_classes(transition_matrix):
    n = len(transition_matrix)  # 状态的数量
    classes = {i: set() for i in range(n)}
    for state in range(n):
        # 状态是否被访问过
        visited = [False] * n
        visited[state] = True
        stack = [(state, 0)]  # (当前状态，步数)
        while stack:
            current_state, steps = stack.pop()
            if steps and current_state == state:
                period = steps
                break
            # 访问下一个状态
            for next_state, prob in enumerate(transition_matrix[current_state]):
                if prob > 0 and not visited[next_state]:
                    visited[next_state] = True
                    stack.append((next_state, steps + 1))
        # 计算循环类
        classes[4 % n].add(state)
    return classes

P = get_example_matrix_str(
    "0.0000 0.2953 0.4076 0.0000 0.0000 0.0000 0.0000 0.0000 0.2971 0.0000 0.0000 0.0000 0.0000 0.0000 0.5211 0.0000 0.0000 0.4789 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.5498 0.0000 0.0000 0.4502 0.0000 0.0000 0.0000 0.4618 0.2558 0.0000 0.0000 0.0000 0.0000 0.0000 0.2825 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.3296 0.6478 0.0000 0.0000 0.0227 0.2880 0.0000 0.0000 0.7120 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0639 0.0000 0.0000 0.9361 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.4837 0.1798 0.0000 0.0000 0.3364 0.0000 0.0000 0.0000 0.0000 0.5858 0.0000 0.0000 0.4142 0.0000 0.0000 0.8494 0.0000 0.0000 0.1506 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000")
print(P)

# 计算状态转移矩阵的周期d
period_d = compute_period(P)
print(f"The period of the Markov chain is: {period_d}")

# 确定循环类
classes = find_cyclic_classes(P)
print(f"Cyclic classes of the Markov chain: {classes}")