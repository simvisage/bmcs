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

from numpy import \
    array, outer
from numpy import \
    identity
from traits.api import \
    Constant,  Property, cached_property
from traitsui.api import \
    View, Include
from ibvpy.mats.mats3D.mats3D_eval import MATS3DEval
from ibvpy.mats.matsXD.matsXD_cmdm.matsXD_cmdm import \
    MATSXDMicroplaneDamage


class MATS3DMicroplaneDamage(MATSXDMicroplaneDamage, MATS3DEval):

    # number of spatial dimensions
    #
    n_dim = Constant(3)

    # number of components of engineering tensor representation
    #
    n_eng = Constant(6)

    #-------------------------------------------------------------------------
    # PolarDiscr related data
    #-------------------------------------------------------------------------
    #
    # number of microplanes - currently fixed for 3D
    #
    n_mp = Constant(28)

    # get the normal vectors of the microplanes
    #
    _MPN = Property(depends_on='n_mp')

    @cached_property
    def _get__MPN(self):
        # microplane normals:
        return array([[.577350259, .577350259, .577350259],
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

    # get the weights of the microplanes
    #
    _MPW = Property(depends_on='n_mp')

    @cached_property
    def _get__MPW(self):
        # Note that the values in the array must be multiplied by 6 (cf. [Baz05])!
        # The sum of of the array equals 0.5. (cf. [BazLuz04]))
        # The values are given for an Gaussian integration over the unit
        # hemisphere.
        return array([.0160714276, .0160714276, .0160714276, .0160714276, .0204744730,
                      .0204744730, .0204744730, .0204744730, .0204744730, .0204744730,
                      .0204744730, .0204744730, .0204744730, .0204744730, .0204744730,
                      .0204744730, .0158350505, .0158350505, .0158350505, .0158350505,
                      .0158350505, .0158350505, .0158350505, .0158350505, .0158350505,
                      .0158350505, .0158350505, .0158350505]) * 6.0

    #-------------------------------------------------------------------------
    # Cached elasticity tensors
    #-------------------------------------------------------------------------

    elasticity_tensors = Property(
        depends_on='E, nu, dimensionality, stress_state')

    @cached_property
    def _get_elasticity_tensors(self):
        '''
        Intialize the fourth order elasticity tensor for 3D or 2D plane strain or 2D plane stress
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

        # The following line correspond to the tensorial expression:
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
        #                 C4_e_3D[i,j,k,l] = (1+nu)/(E) * \
        #                                    ( delta[i,k] * delta[j,l] + delta[i,l]* delta[j,k] ) - \
        #                                    nu / E * delta[i,j] * delta[k,l]
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
        # Get the fourth order elasticity and compliance tensors for the 3D-case
        # -----------------------------------------------------------------------------------------------------
        D2_e_3D = self.map_tns4_to_tns2(D4_e_3D)

        return D4_e_3D, C4_e_3D, D2_e_3D

    #-------------------------------------------------------------------------
    # Dock-based view with its own id
    #-------------------------------------------------------------------------
    traits_view = View(Include('polar_fn_group'),
                       dock='tab',
                       id='ibvpy.mats.mats3D.mats_3D_cmdm.MATS3D_cmdm',
                       kind='modal',
                       resizable=True,
                       scrollable=True,
                       width=0.6, height=0.8,
                       buttons=['OK', 'Cancel']
                       )


if __name__ == '__main__':
    m = MATS3DMicroplaneDamage()
    m.configure_traits()
