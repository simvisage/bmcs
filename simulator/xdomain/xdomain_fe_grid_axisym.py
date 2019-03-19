
from traits.api import \
    Property, cached_property, \
    provides, \
    Array

from mathkit.matrix_la.sys_mtx_assembly import SysMtxArray
import numpy as np
from simulator.api import IXDomain

from .xdomain_fe_grid import XDomainFEGrid


@provides(IXDomain)
class XDomainFEGridAxiSym(XDomainFEGrid):

    vtk_expand_operator = Array(np.float_)

    def _vtk_expand_operator_default(self):
        return np.identity(3)

    Diff0_abc = Array(np.float_)

    def _Diff0_abc_default(self):
        D3D_33 = np.array([[0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 1]], np.float_)
        D2D_11 = np.array([[0, 0],
                           [0, 1]], np.float_)
        return np.einsum('ab,cc->abc', D3D_33, D2D_11)

    Diff1_abcd = Array(np.float)

    def _Diff1_abcd_default(self):
        delta = np.vstack([np.identity(2), np.zeros((1, 2), dtype=np.float_)])
        return 0.5 * (
            np.einsum('ac,bd->abcd', delta, delta) +
            np.einsum('ad,bc->abcd', delta, delta)
        )

    B0_Eimabc = Property(depends_on='+input')

    @cached_property
    def _get_B0_Eimabc(self):
        x_Eia = self.x_Eia
        r_Em = np.einsum(
            'im,Eic->Emc',
            self.fets.N_im, x_Eia
        )[..., 1]
        B0_Eimabc = np.einsum(
            'abc,im, Em->Eimabc',
            self.Diff0_abc, self.fets.N_im, 1. / r_Em
        )
        return B0_Eimabc

    B_Eimabc = Property(depends_on='MESH,GEO,CS,FE')
    '''Kinematic mapping between displacements and strains in every
    integration point.
    '''
    @cached_property
    def _get_B_Eimabc(self):
        return self.B1_Eimabc + self.B0_Eimabc
