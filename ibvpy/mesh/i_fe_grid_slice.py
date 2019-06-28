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
# Created on Jun 30, 2009 by: rchx


from traits.api import Interface, Int, Array, Property


class IFENodeSlice(Interface):
    '''
    Interface of the spatial domain.
    '''

    geo_X = Property

    dof_X = Property

    dofs = Property


class IFEGridSlice(Interface):
    '''
    Interface of the spatial domain.
    '''

    elems = Property(Array(Int))

    geo_X = Property

    dof_X = Property

    dofs = Property
