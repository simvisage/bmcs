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
# Created on Sep 8, 2009 by: rch

from math import sin, cos, pi
from traits.api import Float, on_trait_change
from ibvpy.bcond.bc_dof import BCDof
#
# Supply the boundary conditions and construct the response graphs.
#


class BCDofProportional3D(BCDof):

    '''Convenience specialization of the BCDof with kinematic link between
    epsilon[0] and epsilon[1] corresponding to the angle - 
    proportion between the two strain components. 

    By changing the level alpha, the boundary conditions
    are  adjusted to the corresponding proportional loading
    of e1 and e2 in the strain space. This allows for elementary
    testing2 of the material point behavior for combined loadings.

    alpha = 0 and 90 means tensile loading in e1 and e2, respectively
    alpha = 45 represents simultaneous tension in e1 and e2
    alpha = 180 and 270 means pure pressure in e1 and e2, respectively

    Note that there is only a single kinematic constraint on strain,
    the remaining strain components are calculated using the 
    equilibrium loop    
    '''
#    alpha_degree = Range( 0., 360., 0.,
#                          label = 'Loading angle',
#                          auto_set = False)

    phi = Float(0.0,
                label='Loading angle 1',
                enter_set=True,
                auto_set=False)

    theta = Float(pi / 2.,
                  label='Loading angle 2',
                  enter_set=True,
                  auto_set=False)

    max_strain = Float

    # specialization of the BCDof traits
    var = 'u'
    dof = 0

    def __init__(self, **args):
        super(BCDofProportional3D, self).__init__(**args)
        self._reset()

    @on_trait_change('phi,theta,max_strain')
    def _reset(self):
        print('reseting bcond')
        value, coeff = self.get_value_and_coeff(
            self.max_strain, self.phi, self.theta)
        self.value = value
        self.link_dofs = [1, 2]
        self.link_coeffs = coeff

    @staticmethod
    def get_value_and_coeff(a, phi, theta):
        '''
        Helper function to construct the kinematic constraint for speficied
        proportion of principle strain components.

        The specification requires that the sum of the projections of e1, e2 and e3
        onto the specified inclined direction alpha is equal to a, i.e. 

        (cos(phi) * e[0] + sin(phi) * e[1]) * sin(theta) + e[2] * cos(theta) = a

        The kinematic constraint must be ascribed to a particular DOF and
        can be extended with dependency coefficients. Here we can rewrite 
        the above equation as

        e[0] = - sin(phi) / cos(phi) * e[1] \
               - cos(theta) / sin(theta) / cos(phi) * e[2] \
               + a / sin(theta) / cos(phi) 

        so that the corresponding BCDof specification has the form

        BCDof( var = 'u', # u is used by the time stepper for the kinematic variable
               value = a / sin(theta) / cos(phi),
               dof = 0,
               link_dofs = [1, 2],
               link_coeffs = [-sin(phi)/cos(phi), - cos(theta) / sin(theta) / cos(phi)] )
        '''
        print('phi', phi)
        print('theta', theta)
        coeff = [-sin(phi) / cos(phi), - cos(theta) / sin(theta) / cos(phi)]
        value = a / sin(theta) / cos(phi)
        return value, coeff
