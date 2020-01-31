import numpy as np


def Heaviside(x):
    return 0.5 * np.sign(x) + 0.5
