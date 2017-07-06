'''
Created on 14.06.2017
@author: Yingxiong
'''
import mayavi.mlab as mlab
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

f_pipe = mlab.contour3d(
    sig_1, sig_2, sig_3, f, contours=[0.0], color=(0, 1, 0))

# deviatoric_plane
pi_plane = mlab.pipeline.scalar_cut_plane(f_pipe, plane_orientation='z_axes')
pi_plane.implicit_plane.normal = (1., 1., 1.)
pi_plane.implicit_plane.widget.draw_plane = True
pi_plane.enable_contours = True
pi_plane.contour.minimum_contour = 0.0
pi_plane.contour.maximum_contour = 0.0
pi_plane.contour.number_of_contours = 1
pi_plane.actor.property.representation = 'points'

# plane stress
z_plane = mlab.pipeline.scalar_cut_plane(f_pipe, plane_orientation='z_axes')
z_plane.implicit_plane.origin = (0., 0., 0.)
z_plane.implicit_plane.widget.draw_plane = True
z_plane.enable_contours = True
z_plane.contour.minimum_contour = 0.0
z_plane.contour.maximum_contour = 0.0
z_plane.contour.number_of_contours = 1
z_plane.actor.property.representation = 'points'

# hydrostatic section
h_plane = mlab.pipeline.scalar_cut_plane(f_pipe, plane_orientation='z_axes')
h_plane.implicit_plane.normal = (1., -1., 0.)
h_plane.implicit_plane.widget.draw_plane = True
h_plane.enable_contours = True
h_plane.contour.minimum_contour = 0.0
h_plane.contour.maximum_contour = 0.0
h_plane.contour.number_of_contours = 1
h_plane.actor.property.representation = 'points'

mlab.axes(f_pipe)
# mlab.outline()
mlab.show()