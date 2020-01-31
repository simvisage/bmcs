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
# Created on May 12, 2011 by: rch


#from sys_matrix import SysSparseMtx, SysDenseMtx
from ibvpy.api import \
    TStepper as TS, RTDofGraph, TLoop, \
    TLine, BCDof
from ibvpy.fets.fets1D.fets1D2l import FETS1D2L
from ibvpy.mats.mats1D.mats1D_elastic.mats1D_elastic import MATS1DElastic
from ibvpy.mesh.fe_grid import FEGrid


if __name__ == '__main__':

    fets_eval = FETS1D2L(mats_eval=MATS1DElastic(E=10.))

    # Discretization
    domain = FEGrid(coord_max=(10., 0., 0.),
                    shape=(1, ),
                    fets_eval=fets_eval)

    ts = TS(sdomain=domain,
            dof_resultants=True
            )
    tloop = TLoop(tstepper=ts, debug=False,
                  tline=TLine(min=0.0, step=1, max=1.0))

    '''Clamped bar loaded at the right end with unit displacement
    [00]-[01]-[02]-[03]-[04]-[05]-[06]-[07]-[08]-[09]-[10]
    'u[0] = 0, u[10] = 1'''

    domain.coord_max = (1, 0, 0)
    domain.shape = (3, )
    ts.bcond_list = [BCDof(var='u', dof=0, value=0.),
                     BCDof(var='u', dof=1, link_dofs=[2], link_coeffs=[0.5]),
                     BCDof(var='u', dof=3, value=1.)]
    ts.rtrace_list = [RTDofGraph(name='Fi,right over u_right (iteration)',
                                  var_y='F_int', idx_y=3,
                                  var_x='U_k', idx_x=3)]

    u = tloop.eval()
    # expected solution
    print('u', u)
    # compare the reaction at the left end
    F = ts.F_int[-1]

    print('F', F)

    ts.bcond_mngr.configure_traits()
