'''
Created on Feb 8, 2018

@author: rch
'''

from ibvpy.mats.mats2D.vmats2D_eval import MATS2D
import numpy as np


class MATS2DElastic(MATS2D):
    '''Elastic material model.
    '''
    state_array_shapes = {}

    def get_corr_pred(self, eps_Emcd, deps_Emcd, t_n, t_n1,
                      update_state):
        D_Emabcd = self.D_abcd[None, None, :, :, :, :]
        sig_Emab = np.einsum('...abcd,...cd->...ab', D_Emabcd, eps_Emcd)
        return D_Emabcd, sig_Emab
