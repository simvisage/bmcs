#-------------------------------------------------------------------------------
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

from traits.api import \
    Property, cached_property, Instance, \
    DelegatesTo, Float

from traitsui.api import \
    View, Include

from ibvpy.mats.mats3D.mats3D_cmdm.mats3D_cmdm import \
    MATS3DMicroplaneDamage

from ibvpy.mats import \
    MATS2DMicroplaneDamage, PhiFnStrainSoftening, PhiFnGeneral, \
    PhiFnStrainHardening, PhiFnStrainHardeningLinear

from mathkit.mfn.mfn_polar.mfn_polar import MFnPolar

import numpy as np

class MATS2D5MicroplaneDamage(MATS3DMicroplaneDamage):
    '''
    Degenerated 3D microplane model
        assume linear elastic behavior in eps_z
        assume anistoropic damage in the in-plane components.
        
    needs the strain tensor in the surface coordinates 
    of a shell/plate. 
    '''

    # Specify the class to use for directional dependence
    mfn_class = MFnPolar

    mats2D_cmdm = Instance(MATS2DMicroplaneDamage)
    def _mats2D_cmdm_default(self):
        return MATS2DMicroplaneDamage()

    # @todo: handle the number of microplanes in a better way
    # currently, the get_n_mp does not return the same value
    # as set_n_mp
    #
    # introduce a hidden parameter in the MATSXD implementation
    # for 3D no user-definable n_mp is avialable - the hidden
    # value of MATSXD is just filled with constant value
    #
    # For MATS2D the public n_mp trait gets defined that 
    # seths the hidden _n_mp
    # For MATS2D5 the public n_mp increased by one is set as
    # the hidden _n_mp 
    #
    n_mp = Property(depends_on = 'mats2D_cmdm.n_mp')
    @cached_property
    def _get_n_mp(self):
        return self.mats2D_cmdm.n_mp + 1
    def _set_n_mp(self, value):
        self.mats2D_cmdm.n_mp + value

    #---------------------------------------------------------------------------------
    # Augmented polar discretization with a single z-normal
    #---------------------------------------------------------------------------------
    # get the normal vectors of the microplanes
    _MPN = Property(depends_on = 'n_mp')
    @cached_property
    def _get__MPN(self):
        mpn2D_arr = self.mats2D_cmdm._MPN
        mpnZ_arr = np.zeros((mpn2D_arr.shape[0], 1), dtype = 'float_')
        mpn3D_arr = np.append(mpn2D_arr, mpnZ_arr, axis = 1)
        return np.append(mpn3D_arr, [[0.0, 0.0, 1.0]], axis = 0)

    # get the weights of the microplanes
    _MPW = Property(depends_on = 'n_mp')
    @cached_property
    def _get__MPW(self):
        mpw2D_arr = self.mats2D_cmdm._MPW
        return np.append(mpw2D_arr, 1.0)

    #-----------------------------------------------------------------------------------------------
    # Get the damage state for all microplanes
    #-----------------------------------------------------------------------------------------------
    def get_phi_arr(self, sctx, e_max_arr):
        '''
        Return the damage coefficients
        '''
        # gather the coefficients for parameters depending on the orientation
        carr_list = [self.varpars[key].polar_fn_vectorized(self.alpha_list)
                     for key in self.phi_fn.identify_parameters() ]
        # vectorize the damage function evaluation
        n_arr = 1 + len(carr_list)
        phi_fn_vectorized = np.frompyfunc(self.phi_fn.get_value, n_arr, 1)
        # damage parameter for each microplane
        phi_arr = np.zeros_like(e_max_arr)
        carr_list = [ carr[:-1] for  carr in carr_list ]
        phi_arr[:-1] = phi_fn_vectorized(e_max_arr[:-1], *carr_list)
        phi_arr[-1] = 1.0
        return phi_arr
    #---------------------------------------------------------------------------------
    # Dock-based view with its own id
    #---------------------------------------------------------------------------------

if __name__ == '__main__':
    m = MATS2D5MicroplaneDamage()
    m.configure_traits(view = 'traits_view')

