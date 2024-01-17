import numpy as np
import matplotlib.pyplot as plt


class Region(object):

    def __init__(self, xi, yi, pi, ci):
        self.xi = xi
        self.yi = yi
        self.pi = pi
        self.ci = ci


def calculate_R(regions):
    pass



def visual_regions(regions, R_bounds):
    pass

regions = [
    Region(0.5, 1.6, 1.5, 1.5),
    Region(1.6, 0, 2.1, 2),
    Region(-0.5, 0, 4.3, 1.5)
]

