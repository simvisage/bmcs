'''
Created on Jan 23, 2018

@author: rch
'''

import numpy as np

ONE = np.ones((1,), dtype=np.float_)
DELTA = np.identity(3)
DELTA2D = np.identity(2)
DELTA1D = np.identity(1)

# Levi Civita symbol
EPS = np.zeros((3, 3, 3), dtype='f')
EPS[(0, 1, 2), (1, 2, 0), (2, 0, 1)] = 1
EPS[(2, 1, 0), (1, 0, 2), (0, 2, 1)] = -1

EPS2D = np.zeros((2, 2), dtype='f')
EPS2D[(0, 1)] = 1
EPS2D[(1, 0)] = -1

I_sym = (np.einsum('ac,bd->abcd', DELTA, DELTA) +
         np.einsum('ad,bc->abcd', DELTA, DELTA)) / 2.0

# expansion tensor
DELTA23_ab = np.array([[1, 0, 0],
                       [0, 1, 0]], dtype=np.float_)


if __name__ == '__main__':
    print I_sym

    print np.einsum('pq,pq->pq', EPS2D, EPS2D)
