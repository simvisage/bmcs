#-------------------------------------------------------------------------------
#
# Copyright (c) 2009, IMB, RWTH Aachen.
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in simvisage/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.simvisage.com/licenses/BSD.txt
#
# Thanks for using Simvisage open source!
#
# Created on Feb 17, 2011 by: rch

from numpy import linspace
from time import time

'''Demonstrate the effect of gradual dimensional expansion during
the evaluation of the vectorized function.  
'''

if __name__ == '__main__':
    m = 1
    n = 100000

    a = linspace(0, 1000, m)
    b = linspace(0, 1000, n)

    n_e = 1000

    t = time()

    # 1. perform the first operation on a**2 (1D -> 1D) 
    #    in 1d first then expand to (a**2)*b (1D -> 2D)
    for e in range(n_e):
        R1 = (a[:, None] * a[:, None]) * b[None, :]

    print(('exec time 1', time() - t))

    t = time()

    # 2. perform the expansion a*b first 1D -> 2D matrix
    #    than multiply with a 2D -> 2D
    for e in range(n_e):
        R2 = a[:, None] * (a[:, None] * b[None, :])

    print(('exec time 2', time() - t))
