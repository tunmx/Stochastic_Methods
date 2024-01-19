# %load monter_carlo_method.py
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from dataclasses import dataclass

import tqdm
from matplotlib.ticker import FuncFormatter
from numpy import ndarray
from scipy.interpolate import PchipInterpolator

class Region(object):
    """
    Represents a geometric region defined by a boundary equation of the form:
    |x - xi|^pi + |y - yi|^pi = ci
    """

    def __init__(self, xi: float, yi: float, pi: float, ci: float):
        # Initialize the Region with the equation parameters
        self.xi = xi  # x-coordinate of the reference point
        self.yi = yi  # y-coordinate of the reference point
        self.pi = pi  # Power in the boundary equation
        self.ci = ci  # Constant in the boundary equation


@dataclass
class MCResult:
    """
    Stores the results from a Monte Carlo simulation for estimating area.
    """
    m: int  # Count of points within the region
    N: int  # Total number of sampled points
    m_N: float  # Ratio of points within the region
    S0: float  # Estimated area of the region
    D_eta: float  # Variance of the Bernoulli random variable
    accuracy_of_p_epsilon: float  # Precision of the probability estimate
    abs_accuracy_of_S0: float  # Absolute accuracy of the estimated area
    rel_accuracy_of_S0: float  # Relative accuracy of the estimated area
    CI: Tuple  # Confidence interval for the area estimate
    fall_points: np.ndarray  # Array of sampled points with inside/outside status


class MonterCarloSolution(object):

    def __init__(self, regions: List[Region]):
        self.regions = regions
        self.R = self.calculate_R()
        self.S = self.calculate_R_area()

    def fitting(self, max_N: int = 100, confidence=None) -> List[MCResult]:
        results = list()
        gen_points = self.generating_random_points(self.R, max_N)
        for N in tqdm.tqdm(range(1, max_N + 1)):
            result = self.calculate(N, confidence, gen_points[:N])
            results.append(
                result
            )
        return results

    def calculate(self, N: int = 10, confidence: str = "95%", gen_points=None) -> MCResult:
        if gen_points is None:
            gen_points = self.generating_random_points(self.R, N)
        fall_points, m = self.find_fall_is_points_in_regions(self.regions, gen_points)
        m_N = m / N
        S0 = m_N * self.S
        D_eta = self.calculate_D_eta(m_N)
        epsilon = self.calculate_accuracy_of_p_epsilon(D_eta, N, confidence)
        abs_accuracy = self.calculate_abs_accuracy_of_S(epsilon, self.S)
        rel_accuracy = self.calculate_rel_accuracy_of_S0(abs_accuracy, S0)
        CI = (round(S0 - abs_accuracy, 5), round(S0 + abs_accuracy, 5))

        return MCResult(m=int(m), N=N, m_N=m_N, S0=S0, D_eta=D_eta, accuracy_of_p_epsilon=epsilon,
                        abs_accuracy_of_S0=abs_accuracy, rel_accuracy_of_S0=rel_accuracy, CI=CI,
                        fall_points=fall_points)

    def visual_regions(self, result: MCResult = None, padding: float = 0.1, fill: bool = True, dpi=80) -> None:
        x_min, x_max, y_min, y_max = self.R
        # padding
        x_padding = (x_max - x_min) * padding
        y_padding = (y_max - y_min) * padding

        x_min_padded, x_max_padded = x_min - x_padding, x_max + x_padding
        y_min_padded, y_max_padded = y_min - y_padding, y_max + y_padding

        # setting pixel
        plt.figure(figsize=(8, 8), dpi=dpi)

        x = np.linspace(x_min_padded, x_max_padded, 500)
        y = np.linspace(y_min_padded, y_max_padded, 500)
        X, Y = np.meshgrid(x, y)

        colors = ["blue"] if fill else ['yellow', 'green', 'cyan', 'blue', 'red', 'magenta', 'black']

        alpha = 0.3 if fill else 0.5
        mask = np.zeros_like(X, dtype=bool)
        for i, region in enumerate(self.regions):
            Z = (np.abs(X - region.xi) ** region.pi + np.abs(Y - region.yi) ** region.pi) - region.ci
            # fill
            color = colors[i % len(colors)]
            if fill:
                inside = Z <= 0
                to_fill = inside & ~mask
                Z_masked = np.ma.array(Z, mask=~to_fill)
                plt.contourf(X, Y, Z_masked, levels=[-region.ci, 0], colors=[colors[i % len(colors)]], alpha=alpha)
                mask = mask | inside
            else:
                plt.contourf(X, Y, Z, levels=[-region.ci, 0], colors=[color], alpha=alpha)
            # add text
            if not fill:
                plt.text(region.xi, region.yi, f'D{i + 1}', horizontalalignment='center', verticalalignment='center', )

        if result is not None:
            # Filter points that are inside the regions (indicated by the third column being 1)
            inside_points = result.fall_points[result.fall_points[:, 2] == 1]
            outside_points = result.fall_points[result.fall_points[:, 2] == 0]

            # Plot inside points with red 'o'
            plt.scatter(inside_points[:, 0], inside_points[:, 1], color='red', marker='o', s=6)

            # Plot outside points with black 'x'
            plt.scatter(outside_points[:, 0], outside_points[:, 1], color='black', marker='x', s=6)

        if result is not None:
            # Add MCResult metrics to the plot as text
            result_text = (
                f'N = {result.N}\n'
                f'm/N = {result.m_N}\n'
                f'D_eta = {round(result.D_eta, 6)}\n'
                f'S = {round(self.S, 6)}, S0 = {round(result.S0, 6)}\n'
                f'abs.accuracy of S0 = {round(result.abs_accuracy_of_S0, 6)}\n'
                f'rel.accuracy of S0 = {round(result.rel_accuracy_of_S0, 6)}%\n'
                f'90% CI = {result.CI}'
            )
            plt.text(x_max_padded, y_max_padded, result_text, horizontalalignment='right', verticalalignment='top',
                     bbox=dict(facecolor='white', alpha=0.5))

        # Added padding
        plt.plot([x_min, x_max, x_max, x_min, x_min], [y_min, y_min, y_max, y_max, y_min], 'g--')

        plt.xlim(x_min_padded, x_max_padded)
        plt.ylim(y_min_padded, y_max_padded)
        plt.xlabel('x')
        plt.ylabel('y')
        if fill:
            plt.title('Monte Carlo Simulation of Geometric Area Estimation with Confidence Interval Analysis')
        else:
            plt.title('Filled Plot of Regions D1, D2, and D3')
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

        for region in self.regions:
            x_min = min(x_min, region.xi - region.ci ** (1 / region.pi))
            x_max = max(x_max, region.xi + region.ci ** (1 / region.pi))
            y_min = min(y_min, region.yi - region.ci ** (1 / region.pi))
            y_max = max(y_max, region.yi + region.ci ** (1 / region.pi))

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
        return z * (D_eta / np.sqrt(N))

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
    def plot_relative_accuracy(results: List[MCResult], checkpoints: List[int]) -> None:
        # Extract sample sizes and relative accuracies for all results
        sample_sizes = np.array([result.N for result in results])
        relative_accuracies = np.array([result.rel_accuracy_of_S0 for result in results])

        # Extract sample sizes and relative accuracies for checkpoints
        checkpoint_sizes = np.array([results[i].N for i in checkpoints])
        checkpoint_accuracies = np.array([results[i].rel_accuracy_of_S0 for i in checkpoints])

        # Create the plot
        plt.figure(figsize=(10, 6))

        # Use PchipInterpolator for a smooth monotonic curve
        pchip = PchipInterpolator(checkpoint_sizes, checkpoint_accuracies)
        smooth_sample_sizes = np.linspace(checkpoint_sizes.min(), checkpoint_sizes.max(), 400)
        smooth_relative_accuracies = pchip(smooth_sample_sizes)

        # Plot the smooth curve
        plt.plot(smooth_sample_sizes, smooth_relative_accuracies, linestyle='-', color='black')

        # Plot the checkpoints with a different style
        plt.scatter(checkpoint_sizes, checkpoint_accuracies, color='red', zorder=5)

        # Set the title and labels
        plt.title('The relative accuracy of the square S0 as a function of sample size N')
        plt.xlabel('Sample Size N')
        plt.ylabel('Relative Accuracy of S0 (%)')

        # Formatter to add a percentage sign
        def to_percent(y, position):
            s = str(int(y))
            return s + '%'

        # Create the formatter using the function to_percent
        formatter = FuncFormatter(to_percent)

        # Set the formatter
        plt.gca().yaxis.set_major_formatter(formatter)

        # Set the range of x-axis to show the full range of sample sizes
        plt.xlim(1, sample_sizes.max() + 50)

        # Show grid
        plt.grid(True)

        # Show the plot
        plt.show()


    @staticmethod
    def check_local_value(m_n, S, N, conf):
        eta = MonterCarloSolution.calculate_D_eta(m_n)
        eps = MonterCarloSolution.calculate_accuracy_of_p_epsilon(eta, N, conf)
        abs_acc = MonterCarloSolution.calculate_abs_accuracy_of_S(eps, S)
        print(f"D_eta = {eta}")
        print(f"eps = {eps}")
        print(f"abs_acc = {abs_acc}")


if __name__ == '__main__':
    regions = [
        # Region(-0.5, -1.6, 1.5, 1.5),
        # Region(-1.6, 0, 2.1, 2),
        # Region(0.5, 0, 4.3, 1.5),

        Region(0, -0.5, 2.1, 2),
        Region(1.6, 0.5, 2.1, 1),
        Region(0.5, 0, 2.6, 1),

        # Region(1.6, 0.5, 4.3, 1),
        # Region(0.0, 0.0, 4.3, 1.5),
        # Region(-1.4, -0.5, 3.7, 2)
    ]
    solution = MonterCarloSolution(regions)

    num_of_samples = 1000
    confidence = "90%"

    # result = solution.calculate(N=num_of_samples, confidence=confidence)
    # print(f"S = {solution.S}")
    # print(f"N = {result.N}")
    # print(f"m/N = {result.m_N}")
    # print(f"D_eta = {result.D_eta}")
    # print(f"epsilon = {result.accuracy_of_p_epsilon}")
    # print(f"S0 = {result.S0}")
    # print(f"abs.accuracy of S0 = {result.abs_accuracy_of_S0}")
    # print(f"rel.accuracy of S0 = {round(result.rel_accuracy_of_S0, 2)}%")
    # print(f"{confidence} CI = {result.CI}")
    # solution.visual_regions(result=None, fill=True, padding=0.4)

    checkpoints = [10, 25, 60, 150, 400, 1000]
    checkpoints = list(map(lambda x: x - 1, checkpoints))
    results = solution.fitting(num_of_samples, confidence)
    solution.plot_relative_accuracy(results, checkpoints)

    # solution.check_local_value(0.66, 19.7434, 50, "90%")

