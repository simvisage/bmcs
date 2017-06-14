'''
Created on 14.06.2017

@author: Yingxiong
'''
import numpy as np

min_sig = -20.0
max_sig = 5.0
n_sig = 100j
sig_1, sig_2, sig_3 = np.mgrid[min_sig: max_sig: n_sig,
                               min_sig: max_sig: n_sig,
                               min_sig: max_sig: n_sig]

sig_abcj = np.einsum('jabc->abcj', np.array([sig_1, sig_2, sig_3]))
DELTA = np.identity(3)
sig_abcij = np.einsum('abcj,jl->abcjl', sig_abcj, DELTA)

I1 = np.einsum('...ii,...ii', sig_abcij, DELTA)
s_ij = sig_abcij - np.einsum('...,ij->...ij', I1 / 3.0, DELTA)
J2 = np.einsum('...ii,...ii', s_ij, s_ij) / 2.0

k = 2.
f = J2 - k ** 2

import mayavi.mlab as mlab
f_pipe = mlab.contour3d(
    sig_1, sig_2, sig_3, f, contours=[0.0], color=(0, 1, 0))
# thr = mlab.pipeline.threshold(f_pipe, low=-0.1, up=0.1)
cut_plane = mlab.pipeline.scalar_cut_plane(f_pipe, plane_orientation='z_axes')
cut_plane.implicit_plane.normal = (1., 1., 1.)
mlab.outline()
mlab.show()
