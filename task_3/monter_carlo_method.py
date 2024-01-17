import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Type
from dataclasses import dataclass

from numpy import ndarray


class Region(object):

    def __init__(self, xi: float, yi: float, pi: float, ci: float):
        self.xi = xi
        self.yi = yi
        self.pi = pi
        self.ci = ci


@dataclass
class MCResult:
    m: int
    N: int
    m_N: float
    S0: float
    D_eta: float
    accuracy_of_p_epsilon: float
    abs_accuracy_of_S0: float
    rel_accuracy_of_S0: float
    CI: Tuple
    fall_points: np.ndarray


class MonterCarloSolution(object):

    def __init__(self, regions: List[Region]):
        self.regions = regions
        self.R = self.calculate_R()
        self.S = self.calculate_R_area()

    def fitting(self, max_N: int = 100, confidence=None) -> List[MCResult]:
        results = list()
        for N in range(1, max_N + 1):
            results.append(self.calculate(N, confidence))

        return results

    def calculate(self, N: int = 10, confidence: str = "95%") -> MCResult:
        gen_points = self.generating_random_points(self.R, N)
        fall_points, m = self.find_fall_is_points_in_regions(self.regions, gen_points)
        m_N = m / N
        S0 = m_N * self.S
        D_eta = self.calculate_D_eta(m_N)
        epsilon = self.calculate_accuracy_of_p_epsilon(D_eta, N, confidence)
        abs_accuracy = self.calculate_abs_accuracy_of_S(epsilon, self.S)
        rel_accuracy = self.calculate_rel_accuracy_of_S0(abs_accuracy, S0)
        CI = (S0 - abs_accuracy, S0 + abs_accuracy)

        return MCResult(m=int(m), N=N, m_N=m_N, S0=S0, D_eta=D_eta, accuracy_of_p_epsilon=epsilon,
                        abs_accuracy_of_S0=abs_accuracy, rel_accuracy_of_S0=rel_accuracy, CI=CI, fall_points=fall_points)

    def visual_regions(self, fall_points: np.ndarray = None, padding: float = 0.1, fill: bool = True) -> None:
        x_min, x_max, y_min, y_max = self.R
        # padding
        x_padding = (x_max - x_min) * padding
        y_padding = (y_max - y_min) * padding

        x_min_padded, x_max_padded = x_min - x_padding, x_max + x_padding
        y_min_padded, y_max_padded = y_min - y_padding, y_max + y_padding

        # setting pixel
        plt.figure(figsize=(8, 8), dpi=100)

        x = np.linspace(x_min_padded, x_max_padded, 500)
        y = np.linspace(y_min_padded, y_max_padded, 500)
        X, Y = np.meshgrid(x, y)

        colors = ["blue"] if fill else ['yellow', 'green', 'cyan', 'blue', 'red', 'magenta', 'black']

        alpha = 1.0 if fill else 0.5

        for i, region in enumerate(self.regions):
            Z = (np.abs(X - region.xi) ** region.pi + np.abs(Y - region.yi) ** region.pi) - region.ci
            # fill
            color = colors[i % len(colors)]
            plt.contourf(X, Y, Z, levels=[-region.ci, 0], colors=[color], alpha=alpha)
            # add text
            if not fill:
                plt.text(region.xi, region.yi, f'D{i + 1}', horizontalalignment='center', verticalalignment='center', )

        if fall_points is not None:
            # Filter points that are inside the regions (indicated by the third column being 1)
            inside_points = fall_points[fall_points[:, 2] == 1]
            outside_points = fall_points[fall_points[:, 2] == 0]

            # Plot inside points with red 'o'
            plt.scatter(inside_points[:, 0], inside_points[:, 1], color='red', marker='o', s=10)

            # Plot outside points with black 'x'
            plt.scatter(outside_points[:, 0], outside_points[:, 1], color='black', marker='x', s=10)

        # Added padding
        plt.plot([x_min, x_max, x_max, x_min, x_min], [y_min, y_min, y_max, y_max, y_min], 'g--')

        plt.xlim(x_min_padded, x_max_padded)
        plt.ylim(y_min_padded, y_max_padded)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Filled Plot of Regions D1, D2, and D3 with Correct Aspect Ratio')
        plt.grid(True)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

    def calculate_R_area(self) -> float:
        x_min, x_max, y_min, y_max = self.R
        # Calculate the width and height of the rectangle
        width = x_max - x_min
        height = y_max - y_min
        # Calculate the area
        area = width * height

        return area

    def calculate_R(self) -> Tuple:
        x_min = float('inf')
        x_max = float('-inf')
        y_min = float('inf')
        y_max = float('-inf')

        # 遍历所有区域来更新R的边界值
        for region in self.regions:
            x_min = min(x_min, region.xi - region.ci ** (1 / region.pi))
            x_max = max(x_max, region.xi + region.ci ** (1 / region.pi))
            y_min = min(y_min, region.yi - region.ci ** (1 / region.pi))
            y_max = max(y_max, region.yi + region.ci ** (1 / region.pi))

        # 返回矩形R的边界
        return x_min, x_max, y_min, y_max

    @staticmethod
    def calculate_D_eta(m_n: float) -> float:
        return (1 - m_n) * m_n

    @staticmethod
    def calculate_accuracy_of_p_epsilon(D_eta: float, N: int, confidence: str = None) -> float:
        confidence_level_table = {
            None: 1.0,
            "90%": 1.65,
            "95%": 1.96,
            "99%": 3.0,
        }
        z = confidence_level_table.get(confidence)
        if z is None:
            raise ValueError(f"Unsupported confidence level: {confidence}")
        return z * (np.sqrt(D_eta) / np.sqrt(N))

    @staticmethod
    def calculate_abs_accuracy_of_S(epsilon: float, S: float) -> float:
        return S * epsilon

    @staticmethod
    def calculate_rel_accuracy_of_S0(abs_accuracy: float, S0: float) -> float:
        return (abs_accuracy / S0) * 100

    @staticmethod
    def find_fall_is_points_in_regions(regions: List[Region], points: np.ndarray) -> Tuple[ndarray, ndarray]:
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

    @staticmethod
    def generating_random_points(rect: Tuple, n: int = 1000) -> np.ndarray:
        x_min, x_max, y_min, y_max = rect
        random_points = np.random.rand(n, 2)
        random_points[:, 0] = random_points[:, 0] * (x_max - x_min) + x_min
        random_points[:, 1] = random_points[:, 1] * (y_max - y_min) + y_min

        return random_points

    @staticmethod
    def plot_relative_accuracy(results: List[MCResult], interval: int = 200) -> None:
        # Extract sample sizes and relative accuracies
        sample_sizes = [result.N for result in results]
        relative_accuracies = [result.rel_accuracy_of_S0 for result in results]

        # Identify the indices where we want to highlight the points (every 20% interval and the last point)
        highlight_indices = [i for i in range(0, len(sample_sizes), interval)] + [len(sample_sizes) - 1]

        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(sample_sizes, relative_accuracies, linestyle='-', color='black')  # Line

        # Highlight the points at the specified interval and the last point
        for index in highlight_indices:
            plt.scatter(sample_sizes[index], relative_accuracies[index], color='blue')

        # Set the title and labels
        plt.title('The relative accuracy of the square S0 as a function of sample size N')
        plt.xlabel('Sample Size N')
        plt.ylabel('Relative Accuracy of S0 (%)')

        # Show grid
        plt.grid(True)

        # Show the plot
        plt.show()

regions = [
    Region(-0.5, -1.6, 1.5, 1.5),
    Region(-1.6, 0, 2.1, 2),
    Region(0.5, 0, 4.3, 1.5)
]
solution = MonterCarloSolution(regions)

num_of_samples = 1000
confidence = "90%"
# result = solution.calculate(N=num_of_samples, confidence=confidence)
#
# print(f"N = {result.N}")
# print(f"m/N = {result.m_N}")
# print(f"D_eta = {result.D_eta}")
# print(f"S0 = {result.S0}")
# print(f"abs.accuracy of S0 = {result.abs_accuracy_of_S0}")
# print(f"rel.accuracy of S0 = {result.rel_accuracy_of_S0}")
# print(f"{confidence} CI = {result.CI}")
# solution.visual_regions(result.fall_points, fill=True)

results = solution.fitting(num_of_samples, confidence)
solution.plot_relative_accuracy(results)