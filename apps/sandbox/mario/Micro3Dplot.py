import os

import PIL
<<<<<<< HEAD
#import cv2
#import cv2
=======

>>>>>>> master
from matplotlib import cm
import matplotlib

from mpl_toolkits.mplot3d import Axes3D

from numpy import \
    array, zeros, trace, \
    einsum, zeros_like,\
    identity, sign, linspace, hstack, maximum,\
    sqrt, linalg
from scipy.interpolate import griddata


from traitsui.api import \
    View,  Include


import matplotlib as mpl

import matplotlib.pyplot as plt

import mayavi.mlab as m
<<<<<<< HEAD
import mayavi.mlab as m
#import moviepy.editor as mpy
import numpy as np
import numpy as np
=======


>>>>>>> master
import numpy as np


class Micro3Dplot():

    def get_3Dviz(self, w_N, w_T):

        normals = np.array([[.577350259, .577350259, .577350259],
                            [.577350259, .577350259, -.577350259],
                            [.577350259, -.577350259, .577350259],
                            [.577350259, -.577350259, -.577350259],
                            [.935113132, .250562787, .250562787],
                            [.935113132, .250562787, -.250562787],
                            [.935113132, -.250562787, .250562787],
                            [.935113132, -.250562787, -.250562787],
                            [.250562787, .935113132, .250562787],
                            [.250562787, .935113132, -.250562787],
                            [.250562787, -.935113132, .250562787],
                            [.250562787, -.935113132, -.250562787],
                            [.250562787, .250562787, .935113132],
                            [.250562787, .250562787, -.935113132],
                            [.250562787, -.250562787, .935113132],
                            [.250562787, -.250562787, -.935113132],
                            [.186156720, .694746614, .694746614],
                            [.186156720, .694746614, -.694746614],
                            [.186156720, -.694746614, .694746614],
                            [.186156720, -.694746614, -.694746614],
                            [.694746614, .186156720, .694746614],
                            [.694746614, .186156720, -.694746614],
                            [.694746614, -.186156720, .694746614],
                            [.694746614, -.186156720, -.694746614],
                            [.694746614, .694746614, .186156720],
                            [.694746614, .694746614, -.186156720],
                            [.694746614, -.694746614, .186156720],
                            [.694746614, -.694746614, -.186156720]])
        print(w_T)
        # controlling radius of the reference sphere (1e-2 for strain, 1 for
        # damage)
        factor = 1
        # controlling distance sphere-planes 3d (1.25 for strain, 1 for damage)
        factor_distance = 1
        # plot_sign = -1 for damage, = 1 for strain
        plot_sign = -1
        # tensor plotting option, = 1 true, = 0 false
        tensor_plotting = 0

        # Rotating normals and results
        Rotational_y = np.array([[-1, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, -1]])
        normals_aux = einsum(
            'ij,jk->ij', normals, Rotational_y)
        normals2 = np.concatenate((normals, normals_aux)) * factor
        origin_coord = np.zeros_like(normals2)

        # sets the origin of the microplane vectors that will be plotted. If origin_coord,
        # they will have their origin at the center of the sphere, if normals2 at
        # the shpere's surface

        origin = origin_coord

        min_x = -2 * factor  # sets the axis limits
        max_x = 2 * factor
        n_x = 200j  # number of values along each dimension
        x_1, x_2, x_3 = np.mgrid[min_x: max_x: n_x,
                                 min_x: max_x: n_x,
                                 min_x: max_x: n_x]
        # make a four dimensional array of coordinates covering the box, it starts with the back face coordinate x,
        # loops over all the y coordinates, on each y loops over the z
        # coordinates
        X_abcj = np.einsum('jabc->abcj', np.array([x_1, x_2, x_3]))
        # creates a matrix containing the normal of each vector from origin to
        # each grid coordinate
        norm_X_abc = np.sqrt(np.einsum('...j,...j->...', X_abcj, X_abcj))
        # creates a unit vector pointing to each grid coordinate from origin
        x_abc_j = X_abcj / norm_X_abc[..., np.newaxis]
        x_abc_j[(norm_X_abc <= 1e-6)] = 0.0

        # Output path for you animation images
        home_dir = os.path.expanduser('~')
        out_path = os.path.join(home_dir, 'anim')
    #     out_path = os.path.join(out_path, 'prueba')
        #out_path = os.path.join(out_path, 'eps_t_pi')

        filename6 = os.path.join(out_path, 'colorbar' + 'reds' + '.pdf')
        fig, ax = plt.subplots(figsize=(1.2, 8.49))
        fig.subplots_adjust(right=0.4, bottom=0.25)

        cmap = mpl.cm.Reds
        norm = mpl.colors.Normalize(vmin=0, vmax=1)

        cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                        norm=norm,
                                        orientation='vertical')

        plt.savefig(filename6)

        filename8 = os.path.join(out_path, 'colorbar' + 'Greens' + '.pdf')
        fig, ax = plt.subplots(figsize=(1.2, 8.49))
        fig.subplots_adjust(right=0.4, bottom=0.25)

        cmap = mpl.cm.Greens
        norm = mpl.colors.Normalize(vmin=0, vmax=1)

        cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                        norm=norm,
                                        orientation='vertical')

        plt.savefig(filename8)

        for i in range(len(w_T)):  # loop over the minima and maxima at cyclic loading

            # selects tensor to be displayed (eps_p_ij or phi_ij)
            #             eps_ij = phi_ij[i]
            # micro_vectors projects value onto normals. w_2_T[i] for damage, eps_P_N_2[i] for plastic normal strain
            # (np.linalg.norm(eps_Pi_T_2[i], axis=1)) dor plastic tangential strains
            micro_vectors = einsum('ij,i->ij', normals,
                                   w_T[i])

            micro_vectors_aux = einsum(
                'ij,jk->ij', micro_vectors, Rotational_y)
            micro_vectors = np.concatenate((micro_vectors, micro_vectors_aux))

            # first invariant of the tensor
            # tensor maps the vectors
#             eps_abc_i = np.einsum('...j,ji->...i', x_abc_j, eps_ij)
            # scalar product between "traction strain vector" and coordinate
            # vector
#             eps_abc = np.einsum('...j,...j->...', x_abc_j, eps_abc_i)

            # creates figure window, 900x900 pixels
            f = m.figure(bgcolor=(1, 1, 1), size=(900, 900))

            f_pipe1 = m.contour3d(x_1, x_2, x_3, (norm_X_abc - factor), opacity=0.3, contours=[0.0], color=(.9,
                                                                                                            .9, .9))

#             if tensor_plotting == 1:
#                 f_pipe2 = m.contour3d(x_1, x_2, x_3, plot_sign * eps_abc - (norm_X_abc - factor), opacity=0.15, contours=[0.0],
#                                       color=(1, 0, 0))

            idx1 = np.where(normals2[:, 2] > 0)
            value = np.concatenate((w_T[i], w_T[i]))
            print(value)

            f_pipe3 = m.points3d(normals2[idx1[:], 0], normals2[idx1[:], 1],
                                 origin_coord[idx1[:], 2], value[idx1[:]].reshape(1, 28), colormap='Reds', mode='sphere', resolution=100, scale_factor=0.15 * factor, scale_mode='none', vmax=1, vmin=0)

            xx = yy = zz = np.linspace(-1.05 * factor, 1.05 * factor, 10)
            yx = np.full_like(xx, -1.05 * factor)
            xy = np.full_like(xx, 1.05 * factor)
            xz = yz = np.full_like(xx, 0)

            m.plot3d(yx, yy, yz, color=(0, 0, 0),
                     line_width=0.001 * factor, tube_radius=0.01 * factor)
            m.plot3d(xx, xy, xz, color=(0, 0, 0),
                     line_width=0.001 * factor, tube_radius=0.01 * factor)

            m.text3d(-0.9 * factor, 1.25 * factor, 0,
                     '-1', color=(0, 0, 0), scale=0.1 * factor)
            m.text3d(-0.05 * factor, 1.2 * factor, 0, '0', color=(0, 0, 0),
                     scale=0.1 * factor)
            m.text3d(-0.04 * factor, 1.4 * factor, 0, 'x', color=(0, 0, 0),
                     scale=0.13 * factor)
            m.text3d(0.8 * factor, 1.2 * factor, 0,
                     '1', color=(0, 0, 0), scale=0.1 * factor)

            m.text3d(-1.2 * factor, 1 * factor, 0,
                     '-1', color=(0, 0, 0), scale=0.1 * factor)
            m.text3d(-1.2 * factor, 0.06 * factor, 0,
                     '0', color=(0, 0, 0), scale=0.1 * factor)
            m.text3d(-1.4 * factor, 0.07 * factor, 0,
                     'y', color=(0, 0, 0), scale=0.13 * factor)
            m.text3d(-1.2 * factor, -0.8 * factor, 0,
                     '1', color=(0, 0, 0), scale=0.1 * factor)

            m.view(azimuth=90, elevation=0)

            f.scene.render()

            filename1 = os.path.join(
                out_path, 'x,y' + 'animation' + 'tangential' + np.str(i) + '.png')

            m.savefig(filename=filename1)
            m.close()

            f = m.figure(bgcolor=(1, 1, 1), size=(900, 900))

            f_pipe1 = m.contour3d(x_1, x_2, x_3, (norm_X_abc - factor), opacity=0.3, contours=[0.0], color=(.9,
                                                                                                            .9, .9))
#             if tensor_plotting == 1:
#                 f_pipe2 = m.contour3d(x_1, x_2, x_3, plot_sign * eps_abc - (norm_X_abc - factor), opacity=0.15, contours=[0.0],
#                                       color=(1, 0, 0))

            idx2 = np.where(normals2[:, 1] > 0)

            f_pipe3 = m.points3d(normals2[idx2[:], 0], origin_coord[idx2[:], 1],
                                 normals2[idx2[:], 2], value[idx2[:]].reshape(1, 28), colormap='Reds', mode='sphere', resolution=100, scale_factor=0.15 * factor, scale_mode='none', vmax=1, vmin=0)

            xx = yy = zz = np.linspace(-1.05 * factor, 1.05 * factor, 10)
            xz = zx = np.full_like(xx, 1.05 * factor)
            xy = zy = np.full_like(xx, 0)

            m.plot3d(zx, zy, zz, color=(0, 0, 0),
                     line_width=0.001 * factor, tube_radius=0.01 * factor)
            m.plot3d(xx, xy, xz, color=(0, 0, 0),
                     line_width=0.001 * factor, tube_radius=0.01 * factor)

            m.text3d(0.9 * factor,  0, 1.25 * factor,
                     '-1', color=(0, 0, 0), scale=0.1 * factor)
            m.text3d(0.05 * factor, 0, 1.2 * factor, '0', color=(0, 0, 0),
                     scale=0.1 * factor)
            m.text3d(0.04 * factor, 0, 1.4 * factor, 'x', color=(0, 0, 0),
                     scale=0.13 * factor)
            m.text3d(-0.8 * factor,  0, 1.2 * factor,
                     '1', color=(0, 0, 0), scale=0.1 * factor)

            m.text3d(1.2 * factor, 0, 1 * factor,
                     '-1', color=(0, 0, 0), scale=0.1 * factor)
            m.text3d(1.2 * factor, 0, 0.06 * factor,
                     '0', color=(0, 0, 0), scale=0.1 * factor)
            m.text3d(1.4 * factor,  0, 0.07 * factor,
                     'z', color=(0, 0, 0), scale=0.13 * factor)
            m.text3d(1.2 * factor, 0, -0.8 * factor,
                     '1', color=(0, 0, 0), scale=0.1 * factor)

            m.view(azimuth=270, elevation=270)

            f.scene.render()

            filename2 = os.path.join(
                out_path, 'x,z' + 'animation' + 'tangential' + np.str(i) + '.png')

            m.savefig(filename=filename2)

            m.close()

            f = m.figure(bgcolor=(1, 1, 1), size=(900, 900))

            f_pipe1 = m.contour3d(x_1, x_2, x_3, (norm_X_abc - factor), opacity=0.3, contours=[0.0], color=(.9,
                                                                                                            .9, .9))
#             if tensor_plotting == 1:
#                 f_pipe2 = m.contour3d(x_1, x_2, x_3, plot_sign * eps_abc - (norm_X_abc - factor), opacity=0.15, contours=[0.0],
#                                       color=(1, 0, 0))

            idx3 = np.where(normals2[:, 0] > 0)

            f_pipe3 = m.points3d(origin_coord[idx3[:], 0], normals2[idx3[:], 1],
                                 normals2[idx3[:], 2], value[idx3[:]].reshape(1, 28), colormap='Reds', mode='sphere', resolution=100, scale_factor=0.15 * factor, scale_mode='none', vmax=1, vmin=0)

            xx = yy = zz = np.linspace(-1.05 * factor, 1.05 * factor, 10)
            yz = zy = np.full_like(xx, -1.05 * factor)
            yx = zx = np.full_like(xx, 0)

            m.plot3d(yx, yy, yz, color=(0, 0, 0),
                     line_width=0.001 * factor, tube_radius=0.01 * factor)
            m.plot3d(zx, zy, zz, color=(0, 0, 0),
                     line_width=0.001 * factor, tube_radius=0.01 * factor)

            m.text3d(0, -1 * factor, -1.2 * factor, '-1',
                     color=(0, 0, 0), scale=0.1 * factor)
            m.text3d(0, -0.06 * factor, -1.2 * factor, '0',
                     color=(0, 0, 0), scale=0.1 * factor)
            m.text3d(0, -0.07 * factor, -1.4 * factor, 'y', color=(0, 0, 0),
                     scale=0.13 * factor)
            m.text3d(0, 0.8 * factor, -1.2 * factor, '1',
                     color=(0, 0, 0), scale=0.1 * factor)
            m.text3d(0, -1.25 * factor, -0.9 * factor, '-1',
                     color=(0, 0, 0), scale=0.1 * factor)
            m.text3d(0, -1.2 * factor, -0.05 * factor, '0',
                     color=(0, 0, 0), scale=0.1 * factor)
            m.text3d(0, -1.4 * factor, -0.02 * factor, 'z',
                     color=(0, 0, 0), scale=0.13 * factor)
            m.text3d(0, -1.2 * factor, 0.8 * factor, '1',
                     color=(0, 0, 0), scale=0.1 * factor)

            m.view(azimuth=0, elevation=90)

            f.scene.render()

            filename3 = os.path.join(
                out_path, 'y,z' + 'animation' + 'tangential' + np.str(i) + '.png')

            m.savefig(filename=filename3)
            m.close()

            filename4 = os.path.join(
                out_path, 'comb' + 'animation' + 'tangential' + np.str(i) + '.pdf')
            filename5 = os.path.join(
                out_path, 'comb' + 'animation' + 'tangential' + np.str(i) + '.png')
    #         new_im.save(filename4)

            list_im = [filename1, filename2, filename3]
            imgs = [PIL.Image.open(i) for i in list_im]
            # pick the image which is the smallest, and resize the others to match
            # it (can be arbitrary image shape here)
            min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]
            imgs_comb = np.hstack(
                (np.asarray(i.resize(min_shape)) for i in imgs))

            # save that beautiful picture
            imgs_comb = PIL.Image.fromarray(imgs_comb)
            imgs_comb.save(filename4)
            imgs_comb.save(filename5)

            dx, pts = 1, 100j

            R1 = normals2[idx1, 0:2].reshape(28, 2)
            V1 = value[idx1].reshape(28,)
            X1, Y1 = np.mgrid[-dx:dx:pts, -dx:dx:pts]
            F1 = griddata(R1, V1, (X1, Y1), method='linear')

            R2 = normals2[idx2, 0:3:2].reshape(28, 2)
            V2 = value[idx2].reshape(28,)
            X2, Y2 = np.mgrid[-dx:dx:pts, -dx:dx:pts]
            F2 = griddata(R2, V2, (X2, Y2), method='linear')

            R3 = normals2[idx3, 1:3].reshape(28, 2)
            V3 = value[idx3].reshape(28,)
            X3, Y3 = np.mgrid[-dx:dx:pts, -dx:dx:pts]
            F3 = griddata(R3, V3, (X3, Y3), method='linear')

            plt.subplots(figsize=(27, 8.49))

            plt.subplot(131, xticks=[-1, 0, 1])
            plt.imshow(F1, extent=(-1, 1, -1, 1), cmap='Reds', vmax=1, vmin=0)
            plt.xlabel('y', fontsize=40)
            plt.ylabel('x', fontsize=40).set_rotation(0)
            plt.xticks([-1, 0, 1], fontsize=30)
            plt.yticks([-1, 0, 1], fontsize=30)

            plt.subplot(132)
            plt.imshow(F2, extent=(-1, 1, -1, 1), cmap='Reds', vmax=1, vmin=0)
            plt.xlabel('z', fontsize=40)
            plt.ylabel('x', fontsize=40).set_rotation(0)
            plt.xticks([-1, 0, 1], fontsize=30)
            plt.yticks([-1, 0, 1], fontsize=30)

            plt.subplot(133)
            plt.imshow(F3, extent=(-1, 1, -1, 1), cmap='Reds', vmax=1, vmin=0)
            plt.xlabel('y', fontsize=40)
            plt.ylabel('z', fontsize=40).set_rotation(0)
            plt.xticks([-1, 0, 1], fontsize=30)
            plt.yticks([-1, 0, 1], fontsize=30)

            filename7 = os.path.join(
                out_path, 'colormap' + 'tangential' + np.str(i) + '.pdf')
            filename9 = os.path.join(
                out_path, 'colormap' + 'tangential' + np.str(i) + '.png')
            plt.savefig(filename7)
            plt.savefig(filename9)

        for i in range(len(w_N)):  # loop over the minima and maxima at cyclic loading

            # selects tensor to be displayed (eps_p_ij or phi_ij)
            #             eps_ij = phi_ij[i]
            # micro_vectors projects value onto normals. w_2_T[i] for damage, eps_P_N_2[i] for plastic normal strain
            # (np.linalg.norm(eps_Pi_T_2[i], axis=1)) dor plastic tangential strains
            micro_vectors = einsum('ij,i->ij', normals,
                                   w_N[i])

            micro_vectors_aux = einsum(
                'ij,jk->ij', micro_vectors, Rotational_y)
            micro_vectors = np.concatenate((micro_vectors, micro_vectors_aux))

            # first invariant of the tensor
            # tensor maps the vectors
#             eps_abc_i = np.einsum('...j,ji->...i', x_abc_j, eps_ij)
            # scalar product between "traction strain vector" and coordinate
            # vector
#             eps_abc = np.einsum('...j,...j->...', x_abc_j, eps_abc_i)

            # creates figure window, 900x900 pixels
            f = m.figure(bgcolor=(1, 1, 1), size=(900, 900))
    #
            f_pipe1 = m.contour3d(x_1, x_2, x_3, (norm_X_abc - factor), opacity=0.3, contours=[0.0], color=(.9,
                                                                                                            .9, .9))

#             if tensor_plotting == 1:
#                 f_pipe2 = m.contour3d(x_1, x_2, x_3, plot_sign * eps_abc - (norm_X_abc - factor), opacity=0.15, contours=[0.0],
#                                       color=(1, 0, 0))

            idx1 = np.where(normals2[:, 2] > 0)
            value = np.concatenate((w_N[i], w_N[i]))

            f_pipe3 = m.points3d(normals2[idx1[:], 0], normals2[idx1[:], 1],
                                 origin_coord[idx1[:], 2], value[idx1[:]].reshape(1, 28), colormap='Greens', mode='sphere', resolution=100, scale_factor=0.15 * factor, scale_mode='none', vmax=1, vmin=0)

            xx = yy = zz = np.linspace(-1.05 * factor, 1.05 * factor, 10)
            yx = np.full_like(xx, -1.05 * factor)
            xy = np.full_like(xx, 1.05 * factor)
            xz = yz = np.full_like(xx, 0)

            m.plot3d(yx, yy, yz, color=(0, 0, 0),
                     line_width=0.001 * factor, tube_radius=0.01 * factor)
            m.plot3d(xx, xy, xz, color=(0, 0, 0),
                     line_width=0.001 * factor, tube_radius=0.01 * factor)

            m.text3d(-0.9 * factor, 1.25 * factor, 0,
                     '-1', color=(0, 0, 0), scale=0.1 * factor)
            m.text3d(-0.05 * factor, 1.2 * factor, 0, '0', color=(0, 0, 0),
                     scale=0.1 * factor)
            m.text3d(-0.04 * factor, 1.4 * factor, 0, 'x', color=(0, 0, 0),
                     scale=0.13 * factor)
            m.text3d(0.8 * factor, 1.2 * factor, 0,
                     '1', color=(0, 0, 0), scale=0.1 * factor)

            m.text3d(-1.2 * factor, 1 * factor, 0,
                     '-1', color=(0, 0, 0), scale=0.1 * factor)
            m.text3d(-1.2 * factor, 0.06 * factor, 0,
                     '0', color=(0, 0, 0), scale=0.1 * factor)
            m.text3d(-1.4 * factor, 0.07 * factor, 0,
                     'y', color=(0, 0, 0), scale=0.13 * factor)
            m.text3d(-1.2 * factor, -0.8 * factor, 0,
                     '1', color=(0, 0, 0), scale=0.1 * factor)

            m.view(azimuth=90, elevation=0)

            f.scene.render()

            filename1 = os.path.join(
                out_path, 'x,y' + 'animation' + 'normal' + np.str(i) + '.png')

            m.savefig(filename=filename1)
            m.close()

            f = m.figure(bgcolor=(1, 1, 1), size=(900, 900))

            f_pipe1 = m.contour3d(x_1, x_2, x_3, (norm_X_abc - factor), opacity=0.3, contours=[0.0], color=(.9,
                                                                                                            .9, .9))
#             if tensor_plotting == 1:
#                 f_pipe2 = m.contour3d(x_1, x_2, x_3, plot_sign * eps_abc - (norm_X_abc - factor), opacity=0.15, contours=[0.0],
#                                       color=(1, 0, 0))

            idx2 = np.where(normals2[:, 1] > 0)

            f_pipe3 = m.points3d(normals2[idx2[:], 0], origin_coord[idx2[:], 1],
                                 normals2[idx2[:], 2], value[idx2[:]].reshape(1, 28), colormap='Greens', mode='sphere', resolution=100, scale_factor=0.15 * factor, scale_mode='none', vmax=1, vmin=0)

            xx = yy = zz = np.linspace(-1.05 * factor, 1.05 * factor, 10)
            xz = zx = np.full_like(xx, 1.05 * factor)
            xy = zy = np.full_like(xx, 0)

            m.plot3d(zx, zy, zz, color=(0, 0, 0),
                     line_width=0.001 * factor, tube_radius=0.01 * factor)
            m.plot3d(xx, xy, xz, color=(0, 0, 0),
                     line_width=0.001 * factor, tube_radius=0.01 * factor)

            m.text3d(0.9 * factor,  0, 1.25 * factor,
                     '-1', color=(0, 0, 0), scale=0.1 * factor)
            m.text3d(0.05 * factor, 0, 1.2 * factor, '0', color=(0, 0, 0),
                     scale=0.1 * factor)
            m.text3d(0.04 * factor, 0, 1.4 * factor, 'x', color=(0, 0, 0),
                     scale=0.13 * factor)
            m.text3d(-0.8 * factor,  0, 1.2 * factor,
                     '1', color=(0, 0, 0), scale=0.1 * factor)

            m.text3d(1.2 * factor, 0, 1 * factor,
                     '-1', color=(0, 0, 0), scale=0.1 * factor)
            m.text3d(1.2 * factor, 0, 0.06 * factor,
                     '0', color=(0, 0, 0), scale=0.1 * factor)
            m.text3d(1.4 * factor,  0, 0.07 * factor,
                     'z', color=(0, 0, 0), scale=0.13 * factor)
            m.text3d(1.2 * factor, 0, -0.8 * factor,
                     '1', color=(0, 0, 0), scale=0.1 * factor)

            m.view(azimuth=270, elevation=270)

            f.scene.render()

            filename2 = os.path.join(
                out_path, 'x,z' + 'animation' + 'normal' + np.str(i) + '.png')

            m.savefig(filename=filename2)

            m.close()

            f = m.figure(bgcolor=(1, 1, 1), size=(900, 900))

            f_pipe1 = m.contour3d(x_1, x_2, x_3, (norm_X_abc - factor), opacity=0.3, contours=[0.0], color=(.9,
                                                                                                            .9, .9))
#             if tensor_plotting == 1:
#                 f_pipe2 = m.contour3d(x_1, x_2, x_3, plot_sign * eps_abc - (norm_X_abc - factor), opacity=0.15, contours=[0.0],
#                                       color=(1, 0, 0))

            idx3 = np.where(normals2[:, 0] > 0)

            f_pipe3 = m.points3d(origin_coord[idx3[:], 0], normals2[idx3[:], 1],
                                 normals2[idx3[:], 2], value[idx3[:]].reshape(1, 28), colormap='Greens', mode='sphere', resolution=100, scale_factor=0.15 * factor, scale_mode='none', vmax=1, vmin=0)

            xx = yy = zz = np.linspace(-1.05 * factor, 1.05 * factor, 10)
            yz = zy = np.full_like(xx, -1.05 * factor)
            yx = zx = np.full_like(xx, 0)

            m.plot3d(yx, yy, yz, color=(0, 0, 0),
                     line_width=0.001 * factor, tube_radius=0.01 * factor)
            m.plot3d(zx, zy, zz, color=(0, 0, 0),
                     line_width=0.001 * factor, tube_radius=0.01 * factor)

            m.text3d(0, -1 * factor, -1.2 * factor, '-1',
                     color=(0, 0, 0), scale=0.1 * factor)
            m.text3d(0, -0.06 * factor, -1.2 * factor, '0',
                     color=(0, 0, 0), scale=0.1 * factor)
            m.text3d(0, -0.07 * factor, -1.4 * factor, 'y', color=(0, 0, 0),
                     scale=0.13 * factor)
            m.text3d(0, 0.8 * factor, -1.2 * factor, '1',
                     color=(0, 0, 0), scale=0.1 * factor)
            m.text3d(0, -1.25 * factor, -0.9 * factor, '-1',
                     color=(0, 0, 0), scale=0.1 * factor)
            m.text3d(0, -1.2 * factor, -0.05 * factor, '0',
                     color=(0, 0, 0), scale=0.1 * factor)
            m.text3d(0, -1.4 * factor, -0.02 * factor, 'z',
                     color=(0, 0, 0), scale=0.13 * factor)
            m.text3d(0, -1.2 * factor, 0.8 * factor, '1',
                     color=(0, 0, 0), scale=0.1 * factor)

            m.view(azimuth=0, elevation=90)

            f.scene.render()

            filename3 = os.path.join(
                out_path, 'y,z' + 'animation' + 'normal' + np.str(i) + '.png')

            m.savefig(filename=filename3)
            m.close()

            filename4 = os.path.join(
                out_path, 'comb' + 'animation' + 'normal' + np.str(i) + '.pdf')
            filename5 = os.path.join(
                out_path, 'comb' + 'animation' + 'normal' + np.str(i) + '.png')
    #         new_im.save(filename4)

            list_im = [filename1, filename2, filename3]
            imgs = [PIL.Image.open(i) for i in list_im]
            # pick the image which is the smallest, and resize the others to match
            # it (can be arbitrary image shape here)
            min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]
            imgs_comb = np.hstack(
                (np.asarray(i.resize(min_shape)) for i in imgs))

            # save that beautiful picture
            imgs_comb = PIL.Image.fromarray(imgs_comb)
            imgs_comb.save(filename4)
            imgs_comb.save(filename5)

            dx, pts = 1, 100j

            R1 = normals2[idx1, 0:2].reshape(28, 2)
            V1 = value[idx1].reshape(28,)
            X1, Y1 = np.mgrid[-dx:dx:pts, -dx:dx:pts]
            F1 = griddata(R1, V1, (X1, Y1), method='linear')

            R2 = normals2[idx2, 0:3:2].reshape(28, 2)
            V2 = value[idx2].reshape(28,)
            X2, Y2 = np.mgrid[-dx:dx:pts, -dx:dx:pts]
            F2 = griddata(R2, V2, (X2, Y2), method='linear')

            R3 = normals2[idx3, 1:3].reshape(28, 2)
            V3 = value[idx3].reshape(28,)
            X3, Y3 = np.mgrid[-dx:dx:pts, -dx:dx:pts]
            F3 = griddata(R3, V3, (X3, Y3), method='linear')

            plt.subplots(figsize=(27, 8.49))

            plt.subplot(131, xticks=[-1, 0, 1])
            plt.imshow(F1, extent=(-1, 1, -1, 1),
                       cmap='Greens', vmax=1, vmin=0)
            plt.xlabel('y', fontsize=40)
            plt.ylabel('x', fontsize=40).set_rotation(0)
            plt.xticks([-1, 0, 1], fontsize=30)
            plt.yticks([-1, 0, 1], fontsize=30)

            plt.subplot(132)
            plt.imshow(F2, extent=(-1, 1, -1, 1),
                       cmap='Greens', vmax=1, vmin=0)
            plt.xlabel('z', fontsize=40)
            plt.ylabel('x', fontsize=40).set_rotation(0)
            plt.xticks([-1, 0, 1], fontsize=30)
            plt.yticks([-1, 0, 1], fontsize=30)

            plt.subplot(133)
            plt.imshow(F3, extent=(-1, 1, -1, 1),
                       cmap='Greens', vmax=1, vmin=0)
            plt.xlabel('y', fontsize=40)
            plt.ylabel('z', fontsize=40).set_rotation(0)
            plt.xticks([-1, 0, 1], fontsize=30)
            plt.yticks([-1, 0, 1], fontsize=30)

            filename7 = os.path.join(
                out_path, 'colormap' + 'normal' + np.str(i) + '.pdf')
            filename9 = os.path.join(
                out_path, 'colormap' + 'normal' + np.str(i) + '.png')
            plt.savefig(filename7)
            plt.savefig(filename9)
