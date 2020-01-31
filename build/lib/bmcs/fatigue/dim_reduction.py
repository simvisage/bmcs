'''
Created on Jan 27, 2018

@author: rch
'''

import numpy as np
import sympy as sp

delta = np.identity(3)

# -----------------------------------------------------------------------------------------------------
# Construct the fourth order elasticity tensor for the plane stress case (shape: (2,2,2,2))
# -----------------------------------------------------------------------------------------------------
E = 28000.0
nu = 0.2
# first Lame paramter
la = E * nu / ((1 + nu) * (1 - 2 * nu))
# second Lame parameter (shear modulus)
mu = E / (2 + 2 * nu)
K = la + (2. / 3.) * mu

C_abcd = (K * np.einsum('ab,cd->abcd', delta, delta) +
          mu * (np.einsum('ab,cd->acbd', delta, delta) +
                np.einsum('ab,cd->adbc', delta, delta) -
                2. / 3. * np.einsum('ab,cd->abcd', delta, delta))
          )

# elasticity matrix (shape: (3,3))
D_ab = np.zeros([3, 3])
D_ab[0, 0] = E / (1.0 - nu * nu)
D_ab[0, 1] = E / (1.0 - nu * nu) * nu
D_ab[1, 0] = E / (1.0 - nu * nu) * nu
D_ab[1, 1] = E / (1.0 - nu * nu)
D_ab[2, 2] = E / (1.0 - nu * nu) * (1.0 / 2.0 - nu / 2.0)


def map2d_ijkl2mn(i, j, k, l):
    '''
    Map the four-rank indexes to the two-rank matrix using the major
    and minor symmetry.
    '''
    # first two indices (ij)
    if i == 0 and j == 0:
        m = 0
    elif i == 1 and j == 1:
        m = 1
    elif (i == 0 and j == 1) or (i == 1 and j == 0):
        m = 2

    # second two indices (kl)
    if k == 0 and l == 0:
        n = 0
    elif k == 1 and l == 1:
        n = 1
    elif (k == 0 and l == 1) or (k == 1 and l == 0):
        n = 2

    return m, n


map2d_ijkl2a = np.array([[[[0, 0],
                           [0, 0]],
                          [[2, 2],
                           [2, 2]]],
                         [[[2, 2],
                           [2, 2]],
                          [[1, 1],
                           [1, 1]]
                          ]])
map2d_ijkl2b = np.array([[[[0, 2],
                           [2, 1]],
                          [[0, 2],
                           [2, 1]]],
                         [[[0, 2],
                           [2, 1]],
                          [[0, 2],
                           [2, 1]]
                          ]])

abcd2m = np.zeros((2, 2, 2, 2), np.int_)
abcd2n = np.zeros((2, 2, 2, 2), np.int_)
D_abef = np.zeros([2, 2, 2, 2])
for i in range(0, 2):
    for j in range(0, 2):
        for k in range(0, 2):
            for l in range(0, 2):
                a, b = map2d_ijkl2mn(i, j, k, l)
                abcd2m[i, j, k, l] = a
                abcd2n[i, j, k, l] = b
                D_abef[i, j, k, l] = D_ab[map2d_ijkl2mn(i, j, k, l)]

print(repr(abcd2m))
print(repr(abcd2n))

D_abcd = D_ab[abcd2m, abcd2n]

sig2d = np.zeros((6,), dtype=np.float_)
map2d_3d = np.array([[0, 0], [1, 1], [2, 2], [0, 1], [1, 2], [2, 0]])
sig3d = sig2d[map2d_3d]

print('sig3d', sig3d)
print('D_ab', D_ab)
print(D_abef - D_abcd)

E_, nu_ = sp.symbols('E,nu')
mu_ = E_ / (2 + 2 * nu_)
la_ = E_ * nu_ / ((1 + nu_) * (1 - 2 * nu_))
K_ = la_ + (2. / 3.) * mu_
sp.Matrix()

E_, nu_ = sp.symbols('E,nu')

D2D_ab = sp.Matrix(
    [[E_ / (1.0 - nu_ * nu_), E_ / (1.0 - nu_ * nu_) * nu_, 0],
     [E_ / (1.0 - nu_ * nu_) * nu_, E_ / (1.0 - nu_ * nu_), 0],
     [0, 0, E_ / (1.0 - nu_ * nu_) * (1.0 / 2.0 - nu_ / 2.0)]
     ])

D_factor = E_ * (1 - nu_) / ((1 + nu_) * (1 - 2 * nu_))
D3D_ab = D_factor * sp.Matrix(
    [[1, nu_ / (1 - nu_), nu_ / (1 - nu_), 0, 0, 0],
     [nu_ / (1 - nu_), 1, nu_ / (1 - nu_), 0, 0, 0],
     [nu_ / (1 - nu_), nu_ / (1 - nu_), 1, 0, 0, 0],
     [0, 0, 0, (1 - 2 * nu_) / (2 * (1 - nu_)), 0, 0],
     [0, 0, 0, 0, (1 - 2 * nu_) / (2 * (1 - nu_)), 0],
     [0, 0, 0, 0, 0, (1 - 2 * nu_) / (2 * (1 - nu_))]
     ])
sig3D_ab = sp.Matrix()
print(D3D_ab.subs({"E": E, "nu": nu}))

idx_1 = [0, 1, 3]
idx_2 = [2, 4, 5]
D3D_11 = D3D_ab[idx_1, idx_1]
D3D_12 = D3D_ab[idx_1, idx_2]
D3D_21 = D3D_ab[idx_2, idx_1]
D3D_22 = D3D_ab[idx_2, idx_2]
inv_D3D_22 = D3D_22.inv()
D2D_ab = D3D_11 - ((D3D_12 * inv_D3D_22) * D3D_21)
print(D2D_ab.subs({"E": E, "nu": nu}))
D2D_ab2 = E_ / (1 - nu_**2) * sp.Matrix([[1, nu_, 0],
                                         [nu_, 1, 0],
                                         [0, 0, (1 - nu_) / 2]])
print(D2D_ab2.subs({"E": E, "nu": nu}))

idx_1 = [0, 1, 3]
idx_2 = [2]
D3D_11 = D3D_ab[idx_1, idx_1]
D3D_12 = D3D_ab[idx_1, idx_2]
D3D_21 = D3D_ab[idx_2, idx_1]
D3D_22 = D3D_ab[idx_2, idx_2]
inv_D3D_22 = D3D_22.inv()
D2D_ab3 = D3D_11 - ((D3D_12 * inv_D3D_22) * D3D_21)

print(D2D_ab3.subs({"E": E, "nu": nu}))
