

from scipy.optimize import fsolve

import numpy as np
import pylab as p


w = 600.0
h = 200.0

# residuum of the compatibility relation


def a_fn(a):
    fn = np.cosh(w / 2.0 / a) - (h + a) / a
    return fn


# solve the curvature for the given w and h
a = fsolve(a_fn, 10.0)[0]
print('a', a)


def catenary(a, x, x0, y0):
    return a * np.cosh((x - x0) / a) + y0


# find the value of the catenary at x0 = 0
x0 = 0.0
y0 = -catenary(a, 0.0, x0, 0)

# evaluate the catenary
x = np.linspace(-w / 2, w / 2, 100)
y = catenary(a, x, x0, y0)

p.plot(x, y)
p.show()
