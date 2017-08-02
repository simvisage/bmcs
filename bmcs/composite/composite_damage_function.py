'''
Created on Jul 26, 2017

@author: rch
'''

from traits.api import HasStrictTraits, Float, Property
from view.ui import BMCSLeafNode
import numpy as np

'''How to predict the strain-hardening response 
from a test with modified reinforcement ratio.

The bond is assumed to be the same, the matrix 
strength scales with the reinforcement ratio.
'''


class CompositeCrossSection(BMCSLeafNode):

    E_f = Float(230000,
                MAT=True,
                auto_set=False, enter_set=True,
                symbol='$E_\mathrm{f}$', unit='mm',
                desc='E modulus of the fabrics',
                )

    E_m = Float(29000,
                MAT=True,
                auto_set=False, enter_set=True,
                symbol='$E_\mathrm{m}$', unit='mm',
                desc='E modulus of the matrix',
                )

    rho = Float(0.01,
                CS=True,
                auto_set=False, enter_set=True,
                symbol='$\rho$', unit='-',
                desc='reinforcement ratio',
                )

    omega_f = Property

    def _get_omega_f(self):
        omega_f = (1 - self.rho * self.E_f /
                   ((1 - self.rho) * self.E_m + self.rho * self.E_f))
        return omega_f


import pylab as p

if __name__ == '__main__':
    ccs = CompositeCrossSection()
    rho_arr = np.linspace(0, 1, 20)
    E_f_arr = [29000, 72000, 180000, 230000]
    omega_l = [[ccs.trait_set(rho=rho, E_f=E_f).omega_f
                for rho in rho_arr]
               for E_f in E_f_arr]
    omega_a = np.array(omega_l)
    p.plot(rho_arr, omega_a.T)
    p.show()
