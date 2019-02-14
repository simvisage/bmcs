'''
Created on Feb 14, 2019

@author: rch
'''

from ibvpy.api import MATSEval
import numpy as np
from simulator.i_model import IModel
import traits.api as tr


@tr.provides(IModel)
class MATS1D5Elastic(MATSEval):

    node_name = "multilinear bond law"

    E_s = tr.Float(100.0, tooltip='Shear stiffness of the interface [MPa]',
                   MAT=True, unit='MPa', symbol='E_\mathrm{s}',
                   desc='Shear-modulus of the interface',
                   auto_set=True, enter_set=True)

    E_n = tr.Float(100.0, tooltip='Normal stiffness of the interface [MPa]',
                   MAT=True, unit='MPa', symbol='E_\mathrm{n}',
                   desc='Normal stiffness of the interface',
                   auto_set=False, enter_set=True)

    state_var_shapes = {}

    D_rs = tr.Property(depends_on='E_n,E_s')

    @tr.cached_property
    def _get_D_rs(self):
        return np.array([[self.E_s, 0],
                         [0, self.E_n]], dtype=np.float_)

    def get_corr_pred(self, u_r, tn1):
        tau = np.einsum(
            'rs,...s->...r',
            self.D_rs, u_r)
        grid_shape = tuple([1 for _ in range(len(u_r.shape[:-1]))])
        D = self.D_rs.reshape(grid_shape + (2, 2))
        return tau, D
