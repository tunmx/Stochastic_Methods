import numpy as np


class Region(object):

    def __init__(self, xi, yi, pi, ci):
        self.xi = xi
        self.yi = yi
        self.pi = pi
        self.ci = ci




# 定义域的边界方程
def is_in_D1(x, y):
    return np.abs(x - 1.6) ** 4.3 + np.abs(y - 0.5) ** 4.3 <= 1


def is_in_D2(x, y):
    return np.abs(x) ** 4.3 + np.abs(y) ** 4.3 <= 1.5


def is_in_D3(x, y):
    return np.abs(x + 1.4) ** 3.7 + np.abs(y + 0.5) ** 3.7 <= 2


# 确定矩形R的边界
x_min = min(-1.6 - 1 ** (1 / 4.3), -1 ** (1 / 4.3), -1.4 - 2 ** (1 / 3.7))
x_max = max(1.6 + 1 ** (1 / 4.3), 1 ** (1 / 4.3), 1.4 + 2 ** (1 / 3.7))
y_min = min(-0.5 - 1 ** (1 / 4.3), -1 ** (1 / 4.3), -0.5 - 2 ** (1 / 3.7))
y_max = max(0.5 + 1 ** (1 / 4.3), 1 ** (1 / 4.3), 0.5 + 2 ** (1 / 3.7))

print([x_min, x_max])
print([y_min, y_max])

# 矩形R的面积
S_R = (x_max - x_min) * (y_max - y_min)


# 蒙特卡洛模拟
def monte_carlo_area(N):
    # 在矩形R内生成随机点
    x_random = np.random.uniform(x_min, x_max, N)
    y_random = np.random.uniform(y_min, y_max, N)

    # 计数落在域D内的点
    count_in_D = np.sum(is_in_D1(x_random, y_random) | is_in_D2(x_random, y_random) | is_in_D3(x_random, y_random))

    # 估算域D的面积
    estimated_area = S_R * (count_in_D / N)

    return estimated_area


# 使用大量样本点来提高估算的准确性
N = 1000000  # 样本点数量
estimated_area_D = monte_carlo_area(N)
print(estimated_area_D)
