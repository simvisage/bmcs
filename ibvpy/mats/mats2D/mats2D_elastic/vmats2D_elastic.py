'''
Created on Feb 8, 2018

@author: rch
'''

from ibvpy.mats.mats2D.mats2D_eval import MATS2DEval
import numpy as np


class MATS2DElastic(MATS2DEval):
    '''Elastic material model.
    '''
    state_var_shapes = {}

    def get_corr_pred(self, eps_Emcd, t_n1):
        D_Emabcd = self.D_abcd[None, None, :, :, :, :]
        sig_Emab = np.einsum('abcd,...cd->...ab', self.D_abcd, eps_Emcd)
        return sig_Emab, D_Emabcd
