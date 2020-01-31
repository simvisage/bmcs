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

from math import sin, cos
from numpy import pi as Pi
from traits.api import Range, Float, on_trait_change

from ibvpy.bcond.bc_dof import BCDof
#
# Supply the boundary conditions and construct the response graphs.
#


def get_value_and_coeff(a, alpha):
    '''
    Helper function to construct the kinematic constraint for speficied
    proportion of principle strain components.

    The specification requires that the sum of the projections of e1 and e2
    onto the specified inclined direction alpha is equal to a, i.e. 

    cos(alpha) * e[0] + sin(alpha) * e[1] = a

    The kinematic constraint must be ascribed to a particular DOF and
    can be extended with dependency coefficients. Here we can rewrite 
    the above equation as

    e[0] = - sin(alpha) / cos(alpha) * e[1] + a / cos(alpha) 

    so that the corresponding BCDof specification has the form

    BCDof( var = 'u', # u is used by the time stepper for the kinematic variable
           value = a / cos(alpha),
           dof = 0,
           link_dofs = [1],
           link_coeffs = [-sin(alpha)/cos(alpha) ] )
    '''
    ca = cos(alpha)
    sa = sin(alpha)
    coeff = -(sa / ca)
    value = a / ca
    return value, coeff


class BCDofProportional(BCDof):

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

    alpha_rad = Float(0.0,
                      label='Loading angle',
                      enter_set=True,
                      auto_set=False)

    max_strain = Float

    # specialization of the BCDof traits
    var = 'u'
    dof = 0

    def __init__(self, **args):
        super(BCDofProportional, self).__init__(**args)
        self._reset()

    @on_trait_change('alpha_rad,max_strain')
    def _reset(self):
        print('reseting bcond')
        alpha = self.alpha_rad  # Pi * self.alpha_degree / 180
        value, coeff = get_value_and_coeff(self.max_strain, alpha)
        self.value = value
        self.link_dofs = [1]
        self.link_coeffs = [coeff]
