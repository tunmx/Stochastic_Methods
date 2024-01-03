import pandas as pd
import numpy as np

np.set_printoptions(linewidth=200, threshold=1000, suppress=True)

matrix_power = np.linalg.matrix_power


def get_my_transition_matrix(xlsx_file, my_name="Yan Jingyu", num_rows=11, use_cols='C:M', ) -> np.ndarray:
    df = pd.read_excel(xlsx_file)
    row_index = df[df[df.columns[0]] == my_name].index
    if row_index.empty:
        raise FileNotFoundError(
            "transition matrix does not exist, please check the name is correct.")

    df = pd.read_excel(xlsx_file, usecols=use_cols, skiprows=row_index[0] + 1, nrows=num_rows)

    return df.to_numpy()


def get_example_matrix() -> np.ndarray:
    data_string = """
    0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.6471 0.0000 0.3529 0.0000 0.0000 0.0000 0.3786 0.0000 0.0000 0.0000 0.0000 0.0000 0.1072 0.0000 0.0000 0.0000 0.0000 0.0000 0.5142 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0832 0.1882 0.5169 0.0000 0.0000 0.0000 0.0000 0.0000 0.2116 0.0000 0.3532 0.0000 0.0000 0.0000 0.0518 0.0000 0.0000 0.1502 0.0000 0.0000 0.3496 0.0000 0.0952 0.0000 0.3051 0.0000 0.0000 0.0000 0.0000 0.1197 0.0000 0.0000 0.0000 0.0000 0.0000 0.3114 0.0000 0.0000 0.2638 0.0000 0.0000 0.0000 0.0000 0.1352 0.0000 0.0000 0.0000 0.5226 0.0000 0.0000 0.0000 0.3422 0.0000 0.0000 0.0000 0.0091 0.6572 0.0000 0.3337 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.4596 0.0000 0.0000 0.0000 0.0000 0.1501 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.3903 0.0000 0.0000 0.0000 0.1317 0.0000 0.7591 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.1093 0.0000 0.0000 0.0000 0.0000 0.0000 0.2183 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.2366 0.0000 0.5452 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.6985 0.0000 0.0000 0.0000 0.3015 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.2327 0.0000 0.0000 0.2122 0.2804 0.0000 0.0000 0.0000 0.2747 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.3387 0.0000 0.0000 0.0000 0.0000 0.0000 0.0021 0.0000 0.0000 0.0000 0.3923 0.0000 0.2669 0.0000 0.5361 0.0000 0.0000 0.3795 0.0000 0.0000 0.0612 0.0000 0.0232 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.3084 0.0000 0.0000 0.0000 0.4457 0.0000 0.2145 0.0000 0.0000 0.0314
    """
    data_list = [float(number) for number in data_string.split()]

    # 将列表转换为 NumPy 数组
    mat = np.array(data_list).reshape(15, 15)

    return mat


def show_power_steps(P: np.ndarray, max_power: int = 20):
    for k in range(1, max_power + 1):
        P_k = matrix_power(P, k)
        print(f"== Step: k={k} ==")
        print(P_k)


def check_significance_digits_mae(matrix):
    # Initialize an empty array to store the MAE of each row compared with other rows
    row_maes = np.zeros((len(matrix), len(matrix)))

    # Calculate MAE for each row compared with every other row
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if i != j:
                row_maes[i, j] = np.mean(np.abs(matrix[i] - matrix[j]))

    # Calculate the average MAE, ignoring the diagonal (comparison of the row with itself)
    np.fill_diagonal(row_maes, np.nan)
    average_mae = np.nanmean(row_maes)

    # Determine if the average MAE is within the desired precision
    accuracy_3_digits = average_mae < 0.001
    accuracy_4_digits = average_mae < 0.0001

    return accuracy_3_digits and accuracy_4_digits


def check_transition_matrix_is_regular(matrix: np.ndarray):
    return (matrix > 0).all()


def calculate_mean_first_passage_matrix():
    pass


mat = get_my_transition_matrix("TaskWorksheets1.xlsx")
print(mat)


P = get_example_matrix()

print(P)

show_power_steps(P, max_power=40)

P_40 = matrix_power(P, 40)
assert check_significance_digits_mae(P_40)
assert check_transition_matrix_is_regular(P_40)

w = P_40[0]
print(w.shape)
I = np.eye(w.shape[0])
print(I)
W = np.tile(w, (w.shape[0], 1))
print(W)
D = np.diag(1 / w)
print(D)
Z = np.linalg.inv(I - (P - W))
print(Z)

J = np.ones_like(Z)
J_dg = np.copy(J)
np.fill_diagonal(J_dg, np.diag(Z))

print(J_dg)

E = (I - Z + J_dg) @ D
print(E)
