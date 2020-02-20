'''
Created on 7 Feb 2020

@author: fseemab
'''

from ibvpy.api import MATSEval
from scipy.optimize import  root
from simulator.i_model import IModel

import numpy as np
import traits.api as tr
import traitsui.api as ui


@tr.provides(IModel)
class MATS1D5DP2D(MATSEval):

    node_name = 'Pressure sensitive cumulative damage plasticity'

    E_N = tr.Float(30000, label='E_N',
                   desc='Normal stiffness of the interface',
                   MAT=True,
                   enter_set=True, auto_set=False)

    E_T = tr.Float(12900, label='E_T',
                   desc='Shear modulus of the interface',
                   MAT=True,
                   enter_set=True, auto_set=False)

    gamma = tr.Float(55.0, label='gamma',
                     desc='Kinematic Hardening Modulus',
                     MAT=True,
                     enter_set=True, auto_set=False)

    K = tr.Float(11, label='K',
                 desc='Isotropic hardening modulus',
                 MAT=True,
                 enter_set=True, auto_set=False)

    S_T = tr.Float(0.005, label='S_T',
                 desc='Damage accumulation parameter',
                 MAT=True,
                 enter_set=True, auto_set=False)
    
    S_N = tr.Float(0.005, label='S_N',
                 desc='Damage accumulation parameter',
                 MAT=True,
                 enter_set=True, auto_set=False)

    c_N = tr.Float(1, Label='c_N',
                 desc='Damage accumulation parameter',
                 MAT=True,
                 enter_set=True, auto_set=False)
    
    c_T = tr.Float(1, Label='c_T',
                 desc='Damage accumulation parameter',
                 MAT=True,
                 enter_set=True, auto_set=False)
    
    m = tr.Float(0.3, label='m',
                 desc='Lateral Pressure Coefficient',
                 MAT=True,
                 enter_set=True, auto_set=False)

    sigma_o = tr.Float(4.2, label='sigma_o',
                       desc='Reversibility limit',
                       MAT=True,
                       enter_set=True, auto_set=False)
    
    sig_t = tr.Float(5.0, label='sig_t',
                     MAT=True,
                     enter_set=True, auto_set=False)
    
    b = tr.Float(0.2, label='b',
                 MAT=True,
                 enter_set=True, auto_set=False)

    state_var_shapes = dict(ep_p_N=(),
                            ep_pi_T=(),
                            alpha=(),
                            z=(),
                            omega_T=(),
                            omega_N=())

    D_rs = tr.Property(depends_on='E_N,E_T')

    @tr.cached_property
    def _get_D_rs(self):
        print('recalculating D_rs')
        return np.array([[self.E_T, 0],
                         [0, self.E_N]], dtype=np.float_)

    def init(self, ep_p_N, ep_pi_T, alpha, z, omega_T, omega_N):
        r'''
        Initialize the state variables.
        '''
        ep_p_N[...] = 0
        ep_pi_T[...] = 0
        alpha[...] = 0
        z[...] = 0
        omega_T[...] = 0
        omega_N[...] = 0

    algorithmic = tr.Bool(True)

    def get_corr_pred(self, u_r, t_n, ep_p_N, ep_pi_T, alpha, z, omega_T, omega_N):

        ep_T = u_r[..., 0]
        ep_N = u_r[..., 1]
        
        sig_N_eff_trial = self.E_N * (ep_N - ep_p_N)
        sig_T_eff_trial = self.E_T * (ep_T - ep_pi_T) 
        
        Z = self.K * z
        X = self.gamma * alpha
        f_trial = np.fabs(sig_T_eff_trial - self.gamma * alpha) - \
         (self.sigma_o + self.K * z - self.m * sig_N_eff_trial) * (1.0 - np.heaviside((sig_N_eff_trial), 1) * ((sig_N_eff_trial) ** 2 / (self.sig_t) ** 2))

        I = f_trial > 1e-8

        delta_lamda=0
        sig_N_eff=0
        sig_T_eff=0
                
        N = (self.m + np.heaviside((sig_N_eff), 1) * ((-self.m * sig_N_eff**2) / (self.sig_t)**2 + \
                                        (self.sigma_o + self.K * (z[I] + delta_lamda) - self.m * sig_N_eff) * (2 * sig_N_eff / (self.sig_t)**2))) / (1. - omega_N[I])
                  
        f1 = self.sigma_o + self.K * (z[I] + delta_lamda) - self.m * sig_N_eff
        #print(f1)         
        ft = 1.0 - np.heaviside(sig_N_eff, 1) * (sig_N_eff ** 2 / (self.sig_t) ** 2)
                  
        f_lamda = np.fabs(sig_T_eff_trial[I] - self.gamma * alpha[I]) - delta_lamda * (self.E_T / (1. - omega_T[I]) + self.gamma) - f1 * ft
 
        f_N = sig_N_eff - sig_N_eff_trial[I] + (delta_lamda / (1 - omega_N[I])) * self.E_N * N
                 
        f_tau = sig_T_eff - sig_T_eff_trial + (delta_lamda / (1 - omega_T)) * self.E_T * np.sign(sig_T_eff_trial - X)
        
        sig_N = (1.0 - omega_N) * sig_N_eff
        
        sig_T = (1.0 - omega_T) * self.E_T * (ep_T - ep_pi_T)
             
        x0 = np.zeros_like(ep_T)
             
        def get_sig_N_eff():
                sol = root(lambda sig_N_eff : f(f_N(delta_lamda,sig_N_eff),f_lamda(delta_lamda,sig_N_eff)), x0 = x0 ,method='lm', tol=1e-6)
                return sol.x
        
        def get_sig_T_eff():
                sol = root(lambda sig_T_eff : f_tau(delta_lamda,sig_T_eff), x0 = x0 ,method='lm', tol=1e-6)
                return sol.x
        
        def get_delta_lamda():
                sol = root(lambda delta_lamda : f(f_lamda(sig_N_eff, delta_lamda), f_N(sig_N_eff, delta_lamda), f_tau(sig_T_eff, delta_lamda)), x0 = x0 ,method='lm', tol=1e-6) 
                return sol.x
        print(delta_lamda)
        
        ep_p_N[I] += delta_lamda * N / (1 - omega_N[I])
        #print(np.shape(omega_N[I]))
        ep_pi_T[I] += delta_lamda * np.sign(sig_T_eff_trial[I] - self.gamma * alpha[I]) / (1.0 - omega_T[I])
            
            
        z[I] += delta_lamda
        alpha[I] += delta_lamda * np.sign(sig_T_eff_trial[I] - X[I])
            
        Y_N = 0.5 * self.E_N * (ep_N - ep_p_N) ** 2.0
        Y_T = 0.5 * self.E_T * (ep_T - ep_pi_T) ** 2.0
            
        omega_N[I] += delta_lamda * (1 - omega_N[I]) ** self.c_N * (Y_N[I] / self.S_N + self.b * Y_T[I] / self.S_T) * np.heaviside(ep_N[I], 1)
            
        omega_T[I] += delta_lamda * (1 - omega_T[I]) ** self.c_T * (Y_T[I] / self.S_T + self.b * Y_N[I] / self.S_N)
            
        sig_N[I] = (1.0 - omega_N[I]) * sig_N_eff
        
        sig_T[I] = (1.0 - omega_T[I]) * self.E_T * (ep_T[I] - ep_pi_T[I]) 
            
        f = np.fabs(sig_T[I] / (1 - omega_T[I]) - self.gamma * alpha[I]) - \
            (self.sigma_o + self.K * z[I] - self.m * sig_N[I] / (1 - omega_N[I])) * (1.0 - np.heaviside((sig_N[I] / (1 - omega_N[I])), 1) * ((sig_N[I] / (1 - omega_N[I])) ** 2 / (self.sig_t) ** 2))   
             
                
        #=======================================================================
        # sig_N_i = (1 - omega_N) * sig_N_i_eff_trial 
        #     
        # sig_T_i = (1 - omega_T) * sig_T_i_eff_trial 
        #=======================================================================
        
        f = f_trial
        #print(sig_T[I])
        # Unloading stiffness
        E_alg_T = (1 - omega_T) * self.E_T
        E_alg_N = (1 - omega_N) * self.E_N
        
        ep = np.zeros_like(u_r)
        ep[..., 0] = ep_T
        ep[..., 1] = ep_N
        E_TN = np.einsum('abEm->Emab',
                         np.array(
                             [
                                 [E_alg_T, np.zeros_like(E_alg_T)],
                                 [np.zeros_like(E_alg_N), E_alg_N]
                             ])
                         )
        #print('omega-T', omega_T)
        return ep, E_TN

    def _get_var_dict(self):
        var_dict = super(MATS1D5DP2D, self)._get_var_dict()
        var_dict.update(
            slip=self.get_slip,
            s_el=self.get_s_el,
            shear=self.get_shear,
            omega_T=self.get_omega_T,
            omega_N=self.get_omega_N,
            ep_pi_T=self.get_ep_pi_T,
            ep_p_N =self.get_ep_p_N,
            alpha=self.get_alpha,
            z=self.get_z
        )
        return var_dict

    def get_slip(self, u_r, tn1, **state):
        return self.get_eps(u_r, tn1)[..., 0]

    def get_shear(self, u_r, tn1, **state):
        return self.get_sig(u_r, tn1, **state)[..., 0]

    def get_omega_T(self, u_r, tn1, ep_pi_T, alpha, z, omega_T):
        return omega_T

    def get_omega_N(self, u_r, tn1, ep_p_N, omega_N):
        return omega_N

    def get_ep_pi_T(self, u_r, tn1, ep_pi_T, alpha, z, omega_T):
        return ep_pi_T
    
    def get_ep_p_N(self, u_r, tn1, ep_p_N, omega_N):
        return ep_p_N

    def get_alpha(self, u_r, tn1, ep_pi_T, alpha, z, omega_T):
        return alpha

    def get_z(self, u_r, tn1, ep_pi_T, alpha, z, omega_T):
        return z

    def get_s_el(self, u_r, tn1, **state):
        ep_T = self.get_slip(u_r, tn1, **state)
        ep_pi_T = self.get_ep_pi_T(u_r, tn1, **state)
        ep_e_T = ep_T - ep_pi_T
        return ep_e_T


    tree_view = ui.View(
        ui.Item('E_N'),
        ui.Item('E_T'),
        ui.Item('gamma'),
        ui.Item('K'),
        ui.Item('S_T'),
        ui.Item('S_N'),
        ui.Item('c_T'),
        ui.Item('c_N'),
        ui.Item('m'),
        ui.Item('sigma_o'),
        ui.Item('sig_t'),
        ui.Item('b'),
        ui.Item('D_rs', style='readonly')
    )

    traits_view = tree_view


if __name__ == '__main__':
    m = MATS1D5DP2D()
    print(m.D_rs)
    m.E_T = 100
    print(m.D_rs)
    m.configure_traits()