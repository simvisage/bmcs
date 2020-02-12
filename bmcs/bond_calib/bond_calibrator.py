'''
Created on Feb 5, 2020

@author: rch
'''


from os.path import join

from bmcs.bond_calib.inverse.fem_inverse import \
    MATSEval, FETS1D52ULRH, TStepper, TLoop
from ibvpy.api import BCDof
from view.ui import BMCSRootNode
import numpy as np
import traits.api as tr


class BondCalib(BMCSRootNode):
    '''Method for calibration of a bond-slip law based on the measured 
    pull-out response.
    '''
    E_m = tr.Float(32701,
                   MAT=True,
                   auto_set=False,
                   enter_set=True,
                   symbol=r'E_\mathrm{m}',
                   unit=r'MPa',
                   desc='Elasticity modulus of matrix'
                   )

    A_m = tr.Float(100. * 15. - 8. * 1.85,
                   CS=True,
                   auto_set=False,
                   enter_set=True,
                   symbol=r'A_\mathrm{m}',
                   unit=r'mm^2',
                   desc='Matrix area'
                   )

    E_f = tr.Float(210000,
                   MAT=True,
                   auto_set=False,
                   enter_set=True,
                   symbol=r'E_\mathrm{m}',
                   unit=r'MPa',
                   desc='Elasticity modulus of reinforcement'
                   )

    A_f = tr.Float(8. * 1.85,
                   CS=True,
                   auto_set=False,
                   enter_set=True,
                   symbol=r'A_\mathrm{f}',
                   unit=r'mm^2',
                   desc='Reinforcement area'
                   )

    P_b = tr.Float(1,
                   CS=True,
                   auto_set=False,
                   enter_set=True,
                   symbol=r'P_\mathrm{b}',
                   unit=r'mm',
                   desc='Perimeter'
                   )

    L_x = tr.Float(100.0,
                   GEO=True,
                   auto_set=False,
                   enter_set=True,
                   symbol=r'L',
                   unit=r'mm',
                   desc='Length of bond zone'
                   )

    n_e_x = tr.Int(20,
                   MESH=True,
                   auto_set=False,
                   enter_set=True,
                   symbol=r'n_E',
                   unit=r'-',
                   desc='Number of discretization element along bond zone'
                   )

    n_reg = tr.Int(4,
                   ALG=True,
                   auto_set=False,
                   enter_set=True,
                   symbol=r'n_R',
                   unit=r'-',
                   desc='Range of regularization'
                   )

    k_max = tr.Int(2000,
                   ALG=True,
                   auto_set=False,
                   enter_set=True,
                   symbol=r'K_\mathrm{max}',
                   unit=r'-',
                   desc='maximum number of iteration'
                   )

    tolerance = tr.Float(1e-6,
                         ALG=True,
                         auto_set=False,
                         enter_set=True,
                         symbol=r'\eta',
                         unit=r'-',
                         desc='tolerance'
                         )

    w_arr = tr.Array(np.float_,
                     auto_set=False,
                     enter_set=True,
                     symbol=r'w',
                     unit=r'mm',
                     desc='Control displacement array'
                     )

    P_arr = tr.Array(np.float_,
                     auto_set=False,
                     enter_set=True,
                     symbol=r'P',
                     unit=r'N',
                     desc='Pullout force array'
                     )

    def get_bond_slip(self):
        mats = MATSEval(E_m=self.E_m, E_f=self.E_f)

        fets = FETS1D52ULRH(A_m=self.A_m,
                            A_f=self.A_f,
                            L_b=self.P_b)

        ts = TStepper(mats_eval=mats,
                      fets_eval=fets,
                      L_x=self.L_x,  # half of specimen length
                      n_e_x=self.n_e_x  # number of elements
                      )

        n_dofs = ts.domain.n_dofs
        ts.bc_list = [BCDof(var='u', dof=n_dofs - 2, value=0.0),  # the fixed DOF
                      BCDof(var='u', dof=n_dofs - 1, value=1.0)]  # the DOF on which the displacement is applied

        tl = TLoop(ts=ts, k_max=self.k_max,
                   tolerance=self.tolerance, w_arr=self.w_arr,
                   pf_arr=self.P_arr, n=self.n_reg)

        slip, bond = tl.eval()
        return bond, slip


if __name__ == '__main__':

    w_data = [0, 0.01, 0.05, 0.1, 0.2, 0.5,
              1, 2,     3,   4,   5,   6, 7, 8]
    f_data = [0, 1.0, 1.8, 2.4, 2.6, 2.75,
              2.8, 2.7, 2.5, 1.6, 0.7, 0.4, 0.3, 0.2]

    w = np.array(w_data, dtype=np.float_)
    P = np.array(f_data, dtype=np.float_)

    bcc = BondCalib(w_arr=w, P_arr=P, n_reg=2)
    bond, slip = bcc.get_bond_slip()
    np.set_printoptions(precision=4)
    print('slip')
    print([np.array(slip)])
    print('bond')
    print([np.array(bond)])
    import matplotlib.pyplot as plt
    plt.plot(np.array(slip), np.array(bond))
    plt.show()