'''
Created on Feb 14, 2019

@author: rch
'''

from ibvpy.api import MATSEval
import numpy as np
from simulator.i_model import IModel
import traits.api as tr
import traitsui.api as tru
from sqlalchemy.orm import state
from view.ui.bmcs_tree_node import BMCSTreeNode

'''
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
    
   ''' 
    
class MATSBONDSLIPPRESSURESENSITIVE(MATSEval):
    
    node_name = 'Pressure Sensitive bond model'
    
    E_N = tr.Float(100, label = 'E_N',
                 desc = 'Normal stiffness of the interface',
                 MAT = True,
                 enter_set = True,auto_set = False)
    
    E_T = tr.Float(100, label = 'E_T',
                desc = 'Shear modulus of the interface',
                MAT = True,
                enter_set = True, auto_set = False)
    
    gamma = tr.Float(40.0, label = 'gamma',
                  desc = 'Kinematic Hardening Modulus',
                  MAT = True,
                  enter_set = True, auto_set = False)
    
    K = tr.Float(1, label = 'K',
              desc = 'Isotropic hardening modulus',
              MAT = True,
              enter_set = True, auto_set = False)
    
    S = tr.Float(0.1, label = 'S',
              desc = 'Damage accumulation parameter',
              MAT = True,
              enter_set = True, auto_set = False)
    
    r = tr.Float(0.1, label = 'r',
              desc = 'Damage accumulation parameter',
              MAT = True,
              enter_set = True, auto_set = False)
    
    c = tr.Float(1, Label = 'c',
              desc = 'Damage accumulation parameter',
              MAT = True,
              enter_set = True, auto_set = False)
    
    tau_0 = tr.Float(1, label = 'tau_0',
                  desc = 'Reversibility limit',
                  MAT = True,
                  enter_set = True, auto_set = False)
    
    m = tr.Float(1, label = 'm',
              desc= 'Lateral Pressure Coefficient',
              MAT = True,
              enter_set = True, auto_set = False)
    
    tau_bar = tr.Float(1.1)
    
    
    def get_corr_pred(self, s,s_pi, alpha,z, omega_t, eps_n, eta):
        
        
        
        # For normal
        
        if eps_n >= 0:
            H = 1 
        elif eps_n < 0:
            H = 0
        
        pressure_N = (H(-eps_n)) * self.E_N * (eps_n)
        
        
        # For tangential
        
        Y = 0.5 * self.E_T * (s-s_pi)**2
        
        tau_pi_trial = self.E_T * (s - s_pi)
        
        Z = self.K * z
        
        X = self.gamma * alpha
        
        f = np.fabs(tau_pi_trial - X) - Z - self.tau_0 + self.m * pressure_N
        
        elas = f <= 1e-6
        plas = f > 1e-6
        
        # Return mapping
        
        delta_lambda = f[plas] / (self.E_T/(1-omega_t) + self.gamma + self.K)
        
        #update all state variables
        
        s_plas = s_pi[plas]
        
        s_pi[plas] += (delta_lambda[plas]*np.sign(tau_pi_trial[plas]-X[plas])/(1-omega_t[plas]))
        
        Y = 0.5 * self.E_T * (s[plas] - s_pi[plas])**2
        
        omega_t += (delta_lambda[plas]*(1-omega_t)**self.c *(Y[plas] / self.S)**self.r)
        
        tau_pi_trial[plas] = (1-omega_t[plas])*self.E_T*(s[plas]-s_pi[plas])
        
        alpha[plas] += (delta_lambda[plas]) * np.sign(tau_pi_trial[plas]-X[plas])
        
        z[plas] += delta_lambda[plas]
        
        
        #Algorithmic Stiffness
        
        E_alg_T = ((1-omega_t)*self.E_T - ((self.E_T**2 * (1 - omega_t)) / (self.E_T + (self.gamma + self.K)*(1 - omega_t)))
                  -((1 - omega_t)**self.c * ((Y / self.S)**self.r) * self.E_T**2 * ((s - s_pi) \
                   * (self.tau_bar/(self.tau_bar-(self.m * self.pressure)))) * np.sign(tau_pi_trial - X)) / (self.E_T/(1 - omega_t) + self.gamma + self.K))
        
        return E_alg_T, s_pi, alpha, z, omega_t
    
    
                                 
       
                
       
       
       
       
       
       










