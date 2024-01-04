import numpy as np
from numpy.linalg import matrix_power
from math import gcd
from functools import reduce
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=200, threshold=1000, suppress=True, formatter={'float': '{:0.5f}'.format})


def get_my_transition_matrix(xlsx_file, my_name, num_rows=12, use_cols='C:N', ) -> np.ndarray:
    """
    Read my personal transition matrix from an Excel (xlsx) file.
    """
    df = pd.read_excel(xlsx_file)
    row_index = df[df[df.columns[0]] == my_name].index
    if row_index.empty:
        raise FileNotFoundError(
            "transition matrix does not exist, please check the name is correct.")

    region = pd.read_excel(xlsx_file, usecols=use_cols, skiprows=row_index[0] + 1, nrows=num_rows)

    return region.to_numpy()


def compute_chain_period(transition_matrix):
    n = transition_matrix.shape[0]  # Number of states
    periods = []

    for i in range(n):
        state_periods = []
        # Start with the square of the matrix to avoid trivial self-loop
        for j in range(2, n * n):
            # Raise the transition matrix to the j-th power
            matrix_power_j = matrix_power(transition_matrix, j)
            if matrix_power_j[i, i] > 0:
                state_periods.append(j)
                break  # Only need the first occurrence

        # Compute all GCDs for the state i
        if state_periods:
            state_period = reduce(gcd, state_periods)
            periods.append(state_period)

    # Compute the GCD across all states to find the period d
    chain_period = reduce(gcd, periods)
    return chain_period


def get_example_matrix_str(text: str) -> np.ndarray:
    """
    Text to matrix
    """
    data_list = [float(number) for number in text.split()]

    mat = np.array(data_list).reshape(10, 10)

    return mat


class FindSteadyStateVector(object):

    def __init__(self, transaction_matrix: np.ndarray, max_power: int):
        self.P = transaction_matrix.copy()
        self.max_power = max_power
        self.loss_func = self.max_euclidean_distance
        self.loss_from_total_steps = None

    def run(self):
        ts = range(1, self.max_power + 1)
        self.loss_from_total_steps = list()
        print("Running Cesàro summation and loss calculation...")
        with tqdm(total=self.max_power, desc='Calculating', unit='step') as pbar:
            for t in ts:
                summation_matrix = self.cesaro_summation(self.P, t)
                loss = self.loss_func(summation_matrix)
                self.loss_from_total_steps.append(loss)
                pbar.update(1)  # 更新进度条
                pbar.set_postfix(Loss=f"{loss:.6f}")  # 显示额外信息

        print("Calculation completed.")

    def plot_loss(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_from_total_steps, 'b-', label='Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Loss over Cesàro Summation Steps')
        plt.grid(True)
        plt.legend()
        plt.show()

    @staticmethod
    def cesaro_summation(transition_matrix, t):
        powers = [matrix_power(transition_matrix, n) for n in range(t)]
        powers_stack = np.stack(powers)
        cesaro_mean = np.mean(powers_stack, axis=0)
        return cesaro_mean

    @staticmethod
    def max_euclidean_distance(matrix):
        # 初始化最大距离为0
        max_dist = 0
        # 获取矩阵的行数
        num_rows = matrix.shape[0]

        # 计算任意两行之间的欧几里得距离，并找到最大值
        for i in range(num_rows):
            for j in range(i + 1, num_rows):  # 仅比较不同的行，避免重复计算
                # 计算第i行和第j行之间的欧几里得距离
                l2_norm = np.linalg.norm(matrix[i] - matrix[j])
                # 更新最大距离
                max_dist = max(max_dist, l2_norm)

        return max_dist


mat = get_example_matrix_str(
    "0.0000 0.2953 0.4076 0.0000 0.0000 0.0000 0.0000 0.0000 0.2971 0.0000 0.0000 0.0000 0.0000 0.0000 0.5211 0.0000 0.0000 0.4789 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.5498 0.0000 0.0000 0.4502 0.0000 0.0000 0.0000 0.4618 0.2558 0.0000 0.0000 0.0000 0.0000 0.0000 0.2825 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.3296 0.6478 0.0000 0.0000 0.0227 0.2880 0.0000 0.0000 0.7120 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0639 0.0000 0.0000 0.9361 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.4837 0.1798 0.0000 0.0000 0.3364 0.0000 0.0000 0.0000 0.0000 0.5858 0.0000 0.0000 0.4142 0.0000 0.0000 0.8494 0.0000 0.0000 0.1506 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000")

# print(mat)

# mat = get_my_transition_matrix("TaskWorksheets2.xlsx", "Yan Jingyu")
# print(mat)
#
# chain_period = compute_chain_period(mat)
# print(chain_period)
#
t = 700
# P_t = FindSteadyStateVector.cesaro_summation(mat, t)
# print(P_t)
# print(max_euclidean_distance(P_t))

finder = FindSteadyStateVector(mat, 700)
finder.run()
finder.plot_loss()