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
# Created on Jun 30, 2009 by: rchx

if __name__ == '__main__':
    def example_1d():
        from ibvpy.api import FEDomain, FERefinementGrid, FEGrid, TStepper as TS, \
            BCDofGroup, RTraceDomainListField
        from ibvpy.core.tloop import TLoop, TLine
        from ibvpy.mesh.xfe_subdomain import XFESubDomain
        from ibvpy.mats.mats1D.mats1D_elastic.mats1D_elastic import MATS1DElastic
        from ibvpy.fets.fets1D.fets1D2l import FETS1D2L
        from ibvpy.fets.fets1D.fets1D2l3u import FETS1D2L3U
        from ibvpy.fets.fets_ls.fets_crack import FETSCrack
        fets_eval = FETS1D2L( mats_eval = MATS1DElastic( E = 1. ) ) #, A=1.))
        #xfets_eval = fets_eval # use the same element for the enrichment
        xfets_eval = FETSCrack( parent_fets = fets_eval )
        # Discretization

        fe_domain = FEDomain()
        fe_level1 = FERefinementGrid( domain = fe_domain, fets_eval = fets_eval )
        fe_grid1 = FEGrid( coord_max = ( 4., 0., 0. ),
                           shape = ( 4, ),
                           fets_eval = fets_eval,
                           level = fe_level1 )


        enr = True
        if enr:
            fe_xdomain = XFESubDomain( domain = fe_domain,
                                       fets_eval = xfets_eval,
                                       fe_grid_slice = fe_grid1[  '(X - 2) **2 - 0.5 ' ] )
            fe_xdomain.deactivate_sliced_elems()

        print('elem_dof_map', fe_xdomain.elem_dof_map)

        fe_domain = FEDomain()
        fe_level1 = FERefinementGrid( domain = fe_domain, fets_eval = fets_eval )
        fe_grid1 = FEGrid( coord_max = ( 4 * 3.14, 0., 0. ),
                           shape = ( 8, ),
                           fets_eval = fets_eval,
                           level = fe_level1 )


        enr = True
        if enr:
            fe_xdomain = XFESubDomain( domain = fe_domain,
                                       fets_eval = xfets_eval,
                                       fe_grid_slice = fe_grid1[  'cos(X) - 0.5' ] )
            fe_xdomain.deactivate_sliced_elems()

        print('elem_dof_map2', fe_xdomain.elem_dof_map)


if __name__ == '__main__':
    example_1d()
