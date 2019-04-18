#-------------------------------------------------------------------------
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
# Created on Aug 19, 2009 by: rch

from math import \
    cos, sin

from ibvpy.mats.mats2D.mats2D_eval import MATS2DEval
from ibvpy.mats.mats2D.mats2D_tensor import \
    map2d_tns2_to_tns4, \
    get_D_plane_stress, get_D_plane_strain, get_C_plane_stress, get_C_plane_strain
from ibvpy.mats.mats3D.mats3D_tensor import \
    map3d_tns4_to_tns2
from ibvpy.mats.matsXD.matsXD_cmdm.matsXD_cmdm import \
    MATSXDMicroplaneDamage
from ibvpy.mats.mats_eval import \
    IMATSEval
from mathkit.mfn.mfn_polar.mfn_polar import MFnPolar
from numpy import \
    array, ones, outer, \
    identity
from traits.api import \
    Enum, Property, cached_property, Constant, Type, \
    Int
from traitsui.api import \
    View, Include

import numpy as np


# @todo parameterize - should be specialized in the dimensional subclasses
class MATS2DMicroplaneDamage(MATSXDMicroplaneDamage, MATS2DEval):

    # implements(IMATSEval)

    # number of spatial dimensions
    #
    n_dim = Constant(2)

    # number of components of engineering tensor representation
    #
    n_eng = Constant(3)

    # planar constraint
    stress_state = Enum("plane_strain", "plane_stress")

    # Specify the class to use for directional dependence
    mfn_class = Type(MFnPolar)

    # get the normal vectors of the microplanes
    _MPN = Property(depends_on='n_mp')

    @cached_property
    def _get__MPN(self):
        return array([[cos(alpha), sin(alpha)] for alpha in self.alpha_list])

    # get the weights of the microplanes
    _MPW = Property(depends_on='n_mp')

    @cached_property
    def _get__MPW(self):
        return ones(self.n_mp) / self.n_mp * 2

    elasticity_tensors = Property(depends_on='E, nu, stress_state')

    @cached_property
    def _get_elasticity_tensors(self):
        '''
        Intialize the fourth order elasticity tensor
        for 3D or 2D plane strain or 2D plane stress
        '''
        # ----------------------------------------------------------------------------
        # Lame constants calculated from E and nu
        # ----------------------------------------------------------------------------
        E = self.E
        nu = self.nu

        # first Lame paramter
        la = E * nu / ((1 + nu) * (1 - 2 * nu))
        # second Lame parameter (shear modulus)
        mu = E / (2 + 2 * nu)

        # -----------------------------------------------------------------------------------------------------
        # Get the fourth order elasticity and compliance tensors for the 3D-case
        # -----------------------------------------------------------------------------------------------------

        # The following lines correspond to the tensorial expression:
        # (using numpy functionality in order to avoid the loop):
        #
        # D4_e_3D = zeros((3,3,3,3),dtype=float)
        # C4_e_3D = zeros((3,3,3,3),dtype=float)
        # delta = identity(3)
        # for i in range(0,3):
        #     for j in range(0,3):
        #         for k in range(0,3):
        #             for l in range(0,3):
        #                 # elasticity tensor (cf. Jir/Baz Inelastic analysis of structures Eq.D25):
        #                 D4_e_3D[i,j,k,l] = la * delta[i,j] * delta[k,l] + \
        #                                    mu * ( delta[i,k] * delta[j,l] + delta[i,l] * delta[j,k] )
        #                 # elastic compliance tensor (cf. Simo, Computational Inelasticity, Eq.(2.7.16) AND (2.1.16)):
        #                 C4_e_3D[i,j,k,l] = (1+nu)/(2*E) * \
        #                                    ( delta[i,k] * delta[j,l] + delta[i,l]* delta[j,k] ) - \
        #                                    nu / E * delta[i,j] * delta[k,l]
        #
        # NOTE: swapaxes returns a reference not a copy!
        # (the index notation always refers to the initial indexing (i=0,j=1,k=2,l=3))
        delta = identity(3)
        delta_ijkl = outer(delta, delta).reshape(3, 3, 3, 3)
        delta_ikjl = delta_ijkl.swapaxes(1, 2)
        delta_iljk = delta_ikjl.swapaxes(2, 3)
        D4_e_3D = la * delta_ijkl + mu * (delta_ikjl + delta_iljk)
        C4_e_3D = -nu / E * delta_ijkl + \
            (1 + nu) / (2 * E) * (delta_ikjl + delta_iljk)

        # -----------------------------------------------------------------------------------------------------
        # Get the fourth order elasticity and compliance tensors for the 2D-case
        # -----------------------------------------------------------------------------------------------------
        # 1. step: Get the (6x6)-elasticity and compliance matrices
        #          for the 3D-case:
        D2_e_3D = map3d_tns4_to_tns2(D4_e_3D)
        C2_e_3D = map3d_tns4_to_tns2(C4_e_3D)

        # 2. step: Get the (3x3)-elasticity and compliance matrices
        #          for the 2D-cases plane stress and plane strain:
        D2_e_2D_plane_stress = get_D_plane_stress(D2_e_3D)
        D2_e_2D_plane_strain = get_D_plane_strain(D2_e_3D)
        C2_e_2D_plane_stress = get_C_plane_stress(C2_e_3D)
        C2_e_2D_plane_strain = get_C_plane_strain(C2_e_3D)

        if self.stress_state == 'plane_stress':
            D2_e = D2_e_2D_plane_stress

        if self.stress_state == 'plane_strain':
            D2_e = D2_e_2D_plane_strain

        # 3. step: Get the fourth order elasticity and compliance tensors
        # for the 2D-cases plane stress and plane strain (D4.shape = (2,2,2,2))
        D4_e_2D_plane_stress = map2d_tns2_to_tns4(D2_e_2D_plane_stress)
        D4_e_2D_plane_strain = map2d_tns2_to_tns4(D2_e_2D_plane_strain)
        C4_e_2D_plane_stress = map2d_tns2_to_tns4(C2_e_2D_plane_stress)
        C4_e_2D_plane_strain = map2d_tns2_to_tns4(C2_e_2D_plane_strain)

        # -----------------------------------------------------------------------------------------------------
        # assign the fourth order elasticity and compliance tensors as return values
        # -----------------------------------------------------------------------------------------------------
        if self.stress_state == 'plane_stress':
            # print 'stress state:   plane-stress'
            D4_e = D4_e_2D_plane_stress
            C4_e = C4_e_2D_plane_stress

        if self.stress_state == 'plane_strain':
            # print 'stress state:   plane-strain'
            D4_e = D4_e_2D_plane_strain
            C4_e = C4_e_2D_plane_strain

        return D4_e, C4_e, D2_e

    def _get_explorer_config(self):
        '''Get the specific configuration of this material model in the explorer
        '''
        c = super(MATS2DMicroplaneDamage, self)._get_explorer_config()

        from ibvpy.mats.mats2D.mats2D_rtrace_cylinder import MATS2DRTraceCylinder

        # overload the default configuration
        c['rtrace_list'] += [
            MATS2DRTraceCylinder(name='Laterne',
                                 var_axis='time', idx_axis=0,
                                 var_surface='microplane_damage',
                                 record_on='update'),
        ]

        return c

    #-------------------------------------------------------------------------
    # Dock-based view with its own id
    #-------------------------------------------------------------------------
    traits_view = View(Include('polar_fn_group'),
                       dock='tab',
                       id='ibvpy.mats.mats3D.mats_2D_cmdm.MATS2D_cmdm',
                       kind='modal',
                       resizable=True,
                       scrollable=True,
                       width=0.6, height=0.8,
                       buttons=['OK', 'Cancel']
                       )


class MATS1DMicroplaneDamage(MATS2DMicroplaneDamage):

    n_mp = Int(2)
    _MPN = Property

    @cached_property
    def _get__MPN(self):
        # microplane normals:
        return np.array([[1, 0], [0, 1]], dtype='f')

    _MPW = Property

    @cached_property
    def _get__MPW(self):
        # microplane normals:
        return np.array([1.0, 1.0], dtype='f')


if __name__ == '__main__':
    m = MATS2DMicroplaneDamage()
    D4 = m._get_elasticity_tensors()
    print('D4', D4[2])

    # m.configure_traits(view='traits_view')
