import numpy as np


def markov():
    init_array = np.array([0.4, 0.12, 0.48])
    transfer_matrix = np.array([
        [0.9, 0.075, 0.025],
        [0.15, 0.8, 0.05],
        [0.25, 0.25, 0.5]
    ])
    res_tmp = init_array.copy()
    for i in range(25):
        res = np.dot(res_tmp, transfer_matrix)
        print(f"{i + 1}\t{res}")
        res_tmp = res


def matrix_power(p):
    transfer_matrix = np.array([
        [0.9, 0.075, 0.025],
        [0.15, 0.8, 0.05],
        [0.25, 0.25, 0.5]
    ])

    res_tmp = transfer_matrix.copy()
    for i in range(p):
        res = np.dot(res_tmp, transfer_matrix)
        print(f"{i + 1}\t{res}")
        res_tmp = res


if __name__ == '__main__':
    markov()

    matrix_power(25)
