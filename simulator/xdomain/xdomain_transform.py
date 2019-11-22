
from traits.api import \
    Property, cached_property, \
    provides, Array

from mathkit import EPS
import numpy as np
from simulator.api import IXDomain

from .xdomain_fe_grid import XDomainFEGrid


@provides(IXDomain)
class XDomainFEGridTransform(XDomainFEGrid):

    vtk_expand_operator = Array(np.float_)

    def _vtk_expand_operator_default(self):
        return np.identity(3)

    T_Emra = Property(depends_on='MESH,GEO,CS,FE')

    @cached_property
    def _get_T_Emra(self):
        To3D = self.fets.vtk_expand_operator
        J_Emar = np.einsum(
            'ik,...ij,jl->...kl',
            To3D, self.J_Emar, To3D
        )
        m_0_Ema = J_Emar[..., 0]

        m_2_Ema = np.einsum(
            '...i,...j,ijk->...k',
            m_0_Ema, J_Emar[..., 1], EPS
        )
        m_1_Ema = np.einsum(
            '...i,...j,ijk->...k',
            m_2_Ema, m_0_Ema, EPS)
        M_rEma = np.array([m_0_Ema, m_1_Ema, m_2_Ema])
        M_Emra = np.einsum('rEma->Emra', M_rEma)
        norm_M_Emra = np.sqrt(
            np.einsum('...ij,...ij->...i', M_Emra, M_Emra)
        )[..., np.newaxis]

        T_Emra = M_Emra / norm_M_Emra
        return T_Emra

    B_Eimabc = Property(depends_on='MESH,GEO,CS,FE')
    '''Kinematic mapping between displacements and strains in every
    integration point.
    '''
    @cached_property
    def _get_B_Eimabc(self):
        # return self.B1_Eimabc
        return np.einsum(
            'Emra,Emsb,Eimabc->Eimrsc',
            self.T_Emra[..., :, :2], self.T_Emra[..., :, :2], self.B1_Eimabc
        )
