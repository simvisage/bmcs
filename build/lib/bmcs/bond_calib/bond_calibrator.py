'''
Created on Feb 5, 2020

@author: rch
'''


from os.path import join

from bmcs.bond_calibration.inverse.fem_inverse import \
    MATSEval, FETS1D52ULRH, TStepper, TLoop
from ibvpy.api import BCDof
from matresdev.db.simdb.simdb import simdb

import numpy as np
import traits.api as tr


class BondCalib(tr.HasStrictTraits):

    E_m = tr.Float(32701)
    A_m = tr.Float(100. * 15. - 8. * 1.85)
    E_f = tr.Float(210000)
    A_f = tr.Float(8. * 1.85)
    L_x = tr.Float(100.0)
    n_e_x = tr.Int(20)

    w_arr = tr.Array(np.float_)
    P_arr = tr.Array(np.float_)

    def get_bond_slip(self):
        mats = MATSEval(E_m=self.E_m)

        fets = FETS1D52ULRH(A_m=self.A_m,
                            A_f=self.A_f)

        ts = TStepper(mats_eval=mats,
                      fets_eval=fets,
                      L_x=self.L_x,  # half of specimen length
                      n_e_x=self.n_e_x  # number of elements
                      )

        n_dofs = ts.domain.n_dofs
        ts.bc_list = [BCDof(var='u', dof=n_dofs - 2, value=0.0),  # the fixed DOF
                      BCDof(var='u', dof=n_dofs - 1, value=1.0)]  # the DOF on which the displacement is applied

        tl = TLoop(ts=ts, w_arr=self.w_arr, pf_arr=self.P_arr, n=3)

        slip, bond = tl.eval()
        return bond, slip


if __name__ == '__main__':

    w_arr = np.hstack(
        (np.linspace(0, 0.15, 13), np.linspace(0.4, 4.2, 31)))
    folder = join(simdb.exdata_dir,
                  'double_pullout_tests',
                  '2018-02-14_DPO_Leipzig',
                  )
    dm3, fm3 = np.loadtxt(join(folder,  'DPOUC23A.txt'))
    d3 = np.hstack((0, np.linspace(0.135, 8, 100)))
    f3 = np.interp(d3, dm3, fm3)

    P_arr = np.interp(w_arr, d3 / 2., f3) * 1000.

    bcc = BondCalib(w_arr=w_arr, P_arr=P_arr)

    bond, slip = bcc.get_bond_slip()
    np.set_printoptions(precision=4)
    print('slip')
    print([np.array(slip)])
    print('bond')
    print([np.array(bond)])
