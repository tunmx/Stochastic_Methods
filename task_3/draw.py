import numpy as np
import matplotlib.pyplot as plt


# Function to plot contours based on parameters
def plot_contours(params):
    # Create a grid of points
    x = np.linspace(-3, 3, 400)
    y = np.linspace(-3, 3, 400)
    x, y = np.meshgrid(x, y)

    plt.figure(figsize=(8, 8))

    # Loop through each set of parameters and plot the contour
    for i, (xi, yi, pi, ci) in enumerate(params):
        # Calculate the values for the equation based on parameters
        values = np.abs(x - xi) ** pi + np.abs(y - yi) ** pi

        # Plot the contour
        plt.contour(x, y, values, levels=[ci], label=f'Curve {i + 1}')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Contours of Given Equations')
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


# Parameters from the image
params = [
    (0, -1.6, 3.1, 1.3),
    (0, 0, 1.2, 2),
    (-1, 0.5, 4.3, 1.5)
]

# Call the function with the parameters
plot_contours(params)
