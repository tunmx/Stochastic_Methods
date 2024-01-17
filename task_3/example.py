import numpy as np
import matplotlib.pyplot as plt


class Region(object):

    def __init__(self, xi, yi, pi, ci):
        self.xi = xi
        self.yi = yi
        self.pi = pi
        self.ci = ci


# 外部函数来计算区域R
def calculate_R(regions):
    x_min = float('inf')
    x_max = float('-inf')
    y_min = float('inf')
    y_max = float('-inf')

    # 遍历所有区域来更新R的边界值
    for region in regions:
        x_min = min(x_min, region.xi - region.ci ** (1 / region.pi))
        x_max = max(x_max, region.xi + region.ci ** (1 / region.pi))
        y_min = min(y_min, region.yi - region.ci ** (1 / region.pi))
        y_max = max(y_max, region.yi + region.ci ** (1 / region.pi))

    # 返回矩形R的边界
    return x_min, x_max, y_min, y_max


def visual_regions(regions, R_bounds, points=None, padding=0.1):
    x_min, x_max, y_min, y_max = R_bounds
    # 计算padding
    x_padding = (x_max - x_min) * padding
    y_padding = (y_max - y_min) * padding

    # 更新R边界，加上padding
    x_min_padded, x_max_padded = x_min - x_padding, x_max + x_padding
    y_min_padded, y_max_padded = y_min - y_padding, y_max + y_padding

    # 设置图像大小和分辨率
    plt.figure(figsize=(8, 8), dpi=80)

    # 创建一个网格，在R的范围内，包括padding
    x = np.linspace(x_min_padded, x_max_padded, 500)
    y = np.linspace(y_min_padded, y_max_padded, 500)
    X, Y = np.meshgrid(x, y)

    # 定义颜色和透明度
    colors = ['yellow', 'green', 'cyan', 'blue', 'red', 'magenta', 'black']
    alpha = 0.5

    # 绘制填充区域和文本标注
    for i, region in enumerate(regions):
        # 计算每个区域的值
        Z = (np.abs(X - region.xi) ** region.pi + np.abs(Y - region.yi) ** region.pi) - region.ci
        # 绘制每个区域的填充
        color = colors[i % len(colors)]
        plt.contourf(X, Y, Z, levels=[-region.ci, 0], colors=[color], alpha=alpha)
        # 在区域内添加文本标注
        plt.text(region.xi, region.yi, f'D{i + 1}', horizontalalignment='center', verticalalignment='center', )

    if points is not None:
        # Filter points that are inside the regions (indicated by the third column being 1)
        inside_points = points[points[:, 2] == 1]
        outside_points = points[points[:, 2] == 0]

        # Plot inside points with red 'o'
        plt.scatter(inside_points[:, 0], inside_points[:, 1], color='red', marker='o', s=2)

        # Plot outside points with black 'x'
        plt.scatter(outside_points[:, 0], outside_points[:, 1], color='black', marker='x', s=2)

    # 绘制外边框轮廓，加上padding
    plt.plot([x_min, x_max, x_max, x_min, x_min], [y_min, y_min, y_max, y_max, y_min], 'g--')

    plt.xlim(x_min_padded, x_max_padded)
    plt.ylim(y_min_padded, y_max_padded)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Filled Plot of Regions D1, D2, and D3 with Correct Aspect Ratio')
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')  # 设置x轴和y轴的比例相同
    plt.show()


def generating_random_points(region, n=1000):
    x_min, x_max, y_min, y_max = region
    random_points = np.random.rand(n, 2)
    random_points[:, 0] = random_points[:, 0] * (x_max - x_min) + x_min
    random_points[:, 1] = random_points[:, 1] * (y_max - y_min) + y_min

    return random_points


def is_point_in_regions(regions, points):
    # Initialize an array to store the result with an extra third column for the indicator
    points_with_indicator = np.hstack((points, np.zeros((points.shape[0], 1))))
    # Check each point against all regions
    for i, point in enumerate(points):
        for region in regions:
            # Calculate the region function value for the point
            if (np.abs(point[0] - region.xi) ** region.pi + np.abs(point[1] - region.yi) ** region.pi) <= region.ci:
                # If the point satisfies the region condition, mark it as inside (1)
                points_with_indicator[i, 2] = 1
                break  # No need to check other regions if it's already inside one

    # Count how many points fall inside the union of regions
    points_inside = np.sum(points_with_indicator[:, 2])

    return points_with_indicator, points_inside

def calculate_R_area(R_bounds):
    x_min, x_max, y_min, y_max = R_bounds
    # Calculate the width and height of the rectangle
    width = x_max - x_min
    height = y_max - y_min
    # Calculate the area
    area = width * height
    return area

# 创建Region实例
# regions = [
#     Region(0, -0.5, 2.1, 2),
#     Region(1.6, 0.5, 2.1, 1),
#     Region(0.5, 0, 2.6, 1)
# ]
# regions = [
#     Region(1.6, 0.5, 4.3, 1),
#     Region(0, 0, 4.3, 1.5),
#     Region(-1.4, -0.5, 3.7, 2)
# ]
regions = [
    Region(-0.5, -1.6, 1.5, 1.5),
    Region(-1.6, 0, 2.1, 2),
    Region(0.5, 0, 4.3, 1.5)
]

samples = 50

# 计算并输出矩形R的边界
R_bounds = calculate_R(regions)
print("矩形R的边界是:", R_bounds)

gen_points = generating_random_points(R_bounds, n=samples)
print(gen_points.shape)

results_points, in_num = is_point_in_regions(regions, gen_points)
N = len(results_points)
m_n = in_num / N
S = calculate_R_area(R_bounds)
S0 = m_n * S

D = (1 - m_n) * m_n
print(f"D = {D}")

epsilon = D / np.sqrt(N)

print(f"epsilon = {epsilon}")

print(f"m / n: {in_num}/{len(results_points)} = {m_n}")
print(f"S = {S}")
print(f"S0 = {S0}")

visual_pad = 0.1
visual_regions(regions, R_bounds, points=results_points, padding=visual_pad)
