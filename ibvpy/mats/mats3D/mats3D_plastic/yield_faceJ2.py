'''
Created on 14.06.2017

@author: Yingxiong
'''
import mayavi.mlab as m
import mayavi.mlab as mlab
import numpy as np
# define the boundaries of a cube in stress space
min_sig = -20.0  # maximum compression
max_sig = 5.0  # maximum tension
n_sig = 100j  # number of values in each stress direction
sig_1, sig_2, sig_3 = np.mgrid[min_sig: max_sig: n_sig,
                               min_sig: max_sig: n_sig,
                               min_sig: max_sig: n_sig]
# make a four dimensional array
sig_abcj = np.einsum('jabc->abcj', np.array([sig_1, sig_2, sig_3]))
# Kronecker delta
DELTA = np.identity(3)
sig_abcij = np.einsum('abcj,jl->abcjl', sig_abcj, DELTA)
# first invariant of the stress tensor
I1 = np.einsum('...ii,...ii', sig_abcij, DELTA)
# deviatoric stress tensor in each point
s_ij = sig_abcij - np.einsum('...,ij->...ij', I1 / 3.0, DELTA)
# second deviator of the stress tensor
J2 = np.einsum('...ii,...ii', s_ij, s_ij) / 2.0
# threshold defining the radial distance from hydrostatic axis
k = 2.
f = J2 - k ** 2
f_pipe = mlab.contour3d(
    sig_1, sig_2, sig_3, f, contours=[0.0], color=(0, 1, 0))
# thr = mlab.pipeline.threshold(f_pipe, low=-0.1, up=0.1)
cut_plane = mlab.pipeline.scalar_cut_plane(f_pipe, plane_orientation='z_axes')
cut_plane.implicit_plane.normal = (1., 1., 1.)
mlab.outline()
mlab.show()
