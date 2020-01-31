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
    print(I_sym)
    print(np.einsum('pq,pq->pq', EPS2D, EPS2D))
    # Convert the tensor to an engineering tensor
    eps_tns = np.array([[[1, 2, 3],
                         [2, 1, 4],
                         [3, 4, 1]]], dtype=np.float)
    eps_eng1 = np.einsum(
        'ki,...ki->...k', DELTA, eps_tns
    )
    eps_eng2 = 0.5 * np.einsum(
        'kij,...ij->...k', np.fabs(EPS), eps_tns
    )
    eps_eng = np.concatenate((eps_eng1, eps_eng2), axis=-1)
    print('eps_eng', eps_eng)

    dim2 = int(eps_eng.shape[-1] / 2)
    eps = np.einsum(
        'ki,...k->...ki', DELTA, eps_eng[..., :dim2]
    ) + np.einsum(
        'kij,...k->...ij', np.fabs(EPS), eps_eng[..., dim2:]
    )

    print('eps', eps)
