
from traits.api import \
    provides, \
    Array, Property, Instance, cached_property

from bmcs.mats.fets1d52ulrhfatigue import FETS1D52ULRHFatigue as FE1D
from ibvpy.mesh.i_fe_grid_slice import IFENodeSlice
from mathkit.matrix_la.sys_mtx_assembly import SysMtxArray
import numpy as np
from simulator.api import IXDomain

from .xdomain_fe_grid import XDomainFEGrid


@provides(IXDomain)
class XDomainFEInterface(XDomainFEGrid):

    vtk_expand_operator = Array(np.float_)

    def _vtk_expand_operator_default(self):
        return np.identity(3)

    DELTA_p = Array()

    def _DELTA_p_default(self):
        return np.array([-1, 1], np.float_)

    I = Instance(IFENodeSlice)
    J = Instance(IFENodeSlice)

    o_pEia = Property(depends_on='changed')

    @cached_property
    def _get_o_pEia(self):
        o_Ia = self.I.dofs
        o_Ja = self.J.dofs
        o_piEa = np.array([[o_Ia[:-1], o_Ia[1:]],
                           [o_Ja[:-1], o_Ja[1:]]])
        return np.einsum('piEa->pEia', o_piEa)

    X_pEia = Property(depends_on='changed')

    @cached_property
    def _get_X_pEia(self):
        X_Ia = self.I.geo_X
        X_Ja = self.J.geo_X
        X_piEa = np.array([[X_Ia[:-1], X_Ia[1:]],
                           [X_Ja[:-1], X_Ja[1:]]])
        return np.einsum('piEa->pEia', X_piEa)

    B_Eimabc = Property(depends_on='changed')

    @cached_property
    def _get_B_Eimabc(self):

        x_Eia = self.x_Eia
        B_Eimabc = np.einsum(
            'abc,im->Eimabc',
            self.D0_abc, self.fets.N_im
        )
        return B_Eimabc
