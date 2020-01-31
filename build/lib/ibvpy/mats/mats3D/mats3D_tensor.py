'''
Module providing subsidary functions for tensor mapping in 3D
'''

from numpy import \
    array, zeros, ix_, hstack, vstack

import numpy as np


#-------------------------------------------------------------------------
# Switch from engineering to tensorial notation
#-------------------------------------------------------------------------
def map3d_eps_eng_to_mtx(eps_eng):
    '''
    Switch from engineering notation to tensor notation for strains in 3D
    '''
    eps_mtx = array([[eps_eng[0], eps_eng[5] / 2., eps_eng[4] / 2.],
                     [eps_eng[5] / 2., eps_eng[1], eps_eng[3] / 2.],
                     [eps_eng[4] / 2., eps_eng[3] / 2, eps_eng[2]]])
    return eps_mtx


def map3d_sig_eng_to_mtx(sig_eng):
    '''
    Switch from engineering notation to tensor notation for stresses in 3D
    '''
    sig_mtx = array([[sig_eng[0], sig_eng[5], sig_eng[4]],
                     [sig_eng[5], sig_eng[1], sig_eng[3]],
                     [sig_eng[4], sig_eng[3], sig_eng[2]]])
    return sig_mtx


#-------------------------------------------------------------------------
# Switch from tensorial to engineering notation
#-------------------------------------------------------------------------

def map3d_eps_mtx_to_eng(eps_mtx):
    '''
    Switch from tensor notation to engineering notation for strains in 3D
    '''
    I = np.array([[0, 0], [1, 1], [2, 2]])
    eps_eng = array([eps_mtx[0, 0],
                     eps_mtx[1, 1],
                     eps_mtx[2, 2],
                     2.0 * eps_mtx[1, 2],
                     2.0 * eps_mtx[0, 2],
                     2.0 * eps_mtx[0, 1]])
    return eps_eng


def map3d_sig_mtx_to_eng(sig_mtx):
    '''
    Switch from tensor notation to engineering notation for stresses in 3D
    '''
    sig_eng = array([sig_mtx[0, 0],
                     sig_mtx[1, 1],
                     sig_mtx[2, 2],
                     sig_mtx[1, 2],
                     sig_mtx[0, 2],
                     sig_mtx[0, 1]])
    return sig_eng

#-------------------------------------------------------------------------
# Subsidiary index mapping functions for rank-four to rank-two tensors
#-------------------------------------------------------------------------


def map3d_ijkl2mn(i, j, k, l):
    '''
    Map the four-rank indexes to the two-rank matrix using the major
    and minor symmetry.
    '''
    # 3D-case:
    # first two indices (ij)
    if i == 0 and j == 0:
        m = 0
    elif i == 1 and j == 1:
        m = 1
    elif i == 2 and j == 2:
        m = 2
    elif (i == 1 and j == 2) or (i == 2 and j == 1):
        m = 3
    elif (i == 0 and j == 2) or (i == 2 and j == 0):
        m = 4
    elif (i == 0 and j == 1) or (i == 1 and j == 0):
        m = 5
    else:
        raise IndexError('error in the tensor index mapping')

    # second two indices (kl)
    if k == 0 and l == 0:
        n = 0
    elif k == 1 and l == 1:
        n = 1
    elif k == 2 and l == 2:
        n = 2
    elif (k == 1 and l == 2) or (k == 2 and l == 1):
        n = 3
    elif (k == 0 and l == 2) or (k == 2 and l == 0):
        n = 4
    elif (k == 0 and l == 1) or (k == 1 and l == 0):
        n = 5
    else:
        raise IndexError('error in the tensor index mapping')

    return m, n

#-------------------------------------------------------------------------
# Subsidiary mapping functions for rank-two to rank-four tensor
#-------------------------------------------------------------------------


def map3d_tns2_to_tns4(tns2):
    '''
    Map a matrix to a fourth order tensor assuming minor and major symmetry,
    e.g. D_mtx (6x6) in engineering notation to D_tns(3,3,3,3)).
    '''
    n_dim = 3
    tns4 = zeros([n_dim, n_dim, n_dim, n_dim])
    for i in range(0, n_dim):
        for j in range(0, n_dim):
            for k in range(0, n_dim):
                for l in range(0, n_dim):
                    tns4[i, j, k, l] = tns2[map3d_ijkl2mn(i, j, k, l)]
    return tns4


#-------------------------------------------------------------------------
# Subsidiary mapping functions for rank-four to rank-two tensor
#-------------------------------------------------------------------------

def map3d_tns4_to_tns2(tns4):
    '''
    Map a fourth order tensor to a matrix assuming minor and major symmetry,
    e.g. D_tns(3,3,3,3) to D_mtx (6x6) in engineering notation.
    (Note: Explicit assignment of components used for speed-up.)
    '''
    print('XXXXXXXXXXXXXXXXXXXXXXXXX')
    n_eng = 6
    tns2 = zeros([n_eng, n_eng])

    tns2[0, 0] = tns4[0, 0, 0, 0]
    tns2[0, 1] = tns2[1, 0] = tns4[0, 0, 1, 1]
    tns2[0, 2] = tns2[2, 0] = tns4[0, 0, 2, 2]
    tns2[0, 3] = tns2[3, 0] = tns4[0, 0, 1, 2]
    tns2[0, 4] = tns2[4, 0] = tns4[0, 0, 0, 2]
    tns2[0, 5] = tns2[5, 0] = tns4[0, 0, 0, 1]

    tns2[1, 1] = tns4[1, 1, 1, 1]
    tns2[1, 2] = tns2[2, 1] = tns4[1, 1, 2, 2]
    tns2[1, 3] = tns2[3, 1] = tns4[1, 1, 1, 2]
    tns2[1, 4] = tns2[4, 1] = tns4[1, 1, 0, 2]
    tns2[1, 5] = tns2[5, 1] = tns4[1, 1, 0, 1]

    tns2[2, 2] = tns4[2, 2, 2, 2]
    tns2[2, 3] = tns2[3, 2] = tns4[2, 2, 1, 2]
    tns2[2, 4] = tns2[4, 2] = tns4[2, 2, 0, 2]
    tns2[2, 5] = tns2[5, 2] = tns4[2, 2, 0, 1]

    tns2[3, 3] = tns4[1, 2, 1, 2]
    tns2[3, 4] = tns2[4, 3] = tns4[1, 2, 0, 2]
    tns2[3, 5] = tns2[5, 3] = tns4[1, 2, 0, 1]

    tns2[4, 4] = tns4[0, 2, 0, 2]
    tns2[4, 5] = tns2[5, 4] = tns4[0, 2, 0, 1]

    tns2[5, 5] = tns4[0, 1, 0, 1]

    return tns2


#-------------------------------------------------------------------------
# Compliance mapping 3D (used for inversion of the damage effect tensor in engineering notation
#-------------------------------------------------------------------------

def compliance_mapping3d(C_mtx_3d):
    '''
    The components of the compliance matrix are multiplied with factor 1,2 or 4 depending on their 
    position in the matrix (due to symmetry and switching of tensorial to engineering notation
    (Note: gamma_xy = 2*epsilon_xy, etc.). Necessary for evaluation of D=inv(C). 
    '''
    idx1 = [0, 1, 2]
    idx2 = [3, 4, 5]
    C11 = C_mtx_3d[ix_(idx1, idx1)]
    C12 = C_mtx_3d[ix_(idx1, idx2)]
    C21 = C_mtx_3d[ix_(idx2, idx1)]
    C22 = C_mtx_3d[ix_(idx2, idx2)]
    return vstack([hstack([C11,   2 * C12]),
                   hstack([2 * C21, 4 * C22])])
