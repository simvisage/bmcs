

#from sys_matrix import SysSparseMtx, SysDenseMtx
import unittest

from ibvpy.api import \
    TStepper as TS, RTDofGraph, RTraceDomainField, TLoop, \
    TLine, BCDof
from ibvpy.fets.fets1D.fets1D2l import FETS1D2L
from ibvpy.mats.mats1D.mats1D_elastic.mats1D_elastic import MATS1DElastic
from ibvpy.mesh.fe_grid import FEGrid
from numpy import array, sqrt
from scipy.linalg import norm

if __name__ == '__main__':
    fets_eval = FETS1D2L(mats_eval=MATS1DElastic(E=10.))

    # Discretization
    domain = FEGrid(coord_max=(10., 0., 0.),
                    shape=(1,),
                    fets_eval=fets_eval
                    )

    ts = TS(sdomain=domain,
            dof_resultants=True
            )
    tloop = TLoop(tstepper=ts,
                  tline=TLine(min=0.0,  step=1, max=1.0))

    '''Clamped bar loaded at the right end with unit displacement
    [00]-[01]-[02]-[03]-[04]-[05]-[06]-[07]-[08]-[09]-[10]
    'u[0] = 0, u[10] = 1'''

    domain.coord_max = (10, 0, 0)
    domain.shape = (10,)
    ts.bcond_list = [BCDof(var='u', dof=0, value=0.),
                     BCDof(var='u', dof=10, value=1.)]
    ts.rtrace_list = [RTDofGraph(name='Fi,right over u_right (iteration)',
                                 var_y='F_int', idx_y=10,
                                 var_x='U_k', idx_x=10)]

    u = tloop.eval()
    # expected solution
    u_ex = array([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.],
                 dtype=float)
    difference = sqrt(norm(u - u_ex))
    print('difference')
    # compare the reaction at the left end
    F = ts.F_int[0]
    print(F)
