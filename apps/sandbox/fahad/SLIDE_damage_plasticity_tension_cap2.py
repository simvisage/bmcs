'''
Created on 14.11.2016

@author: abaktheer

two dimensional SLIDE (with Desmorat short-cut return mapping)
'''


import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import  root




def get_response(eps_N_arr, eps_T_arr, sig_0, sig_t, E_T, E_N, m, S_N, S_T, c_N, c_T, K_T, gamma_T):

    # arrays to store the values
    sig_N_arr= np.zeros_like(eps_N_arr)
    sig_T_arr= np.zeros_like(eps_T_arr)
    sig_N_trial_arr = np.zeros_like(eps_N_arr)
    sig_T_trial_arr = np.zeros_like(eps_T_arr)
    eps_N_p_arr = np.zeros_like(eps_N_arr)
    eps_T_p_arr = np.zeros_like(eps_T_arr)
    omega_N_arr = np.zeros_like(eps_N_arr)
    omega_T_arr = np.zeros_like(eps_T_arr)
    z_T_arr = np.zeros_like(eps_T_arr)
    alpha_T_arr = np.zeros_like(eps_T_arr)
    f_arr = np.zeros_like(eps_T_arr)

    # state variables
    eps_N_p_i = 0.0
    omega_N_i = 0.0
    eps_T_p_i = 0.0
    omega_T_i = 0.0
    z_T_i = 0.0 
    alpha_T_i = 0.0
    
    delta_lamda = 0.0
    
    for i in range(1, len(eps_N_arr)):
        
        eps_N_i = eps_N_arr[i]
        eps_T_i = eps_T_arr[i]
        
        sig_N_i_eff_trial = E_N * (eps_N_i - eps_N_p_i)
        sig_T_i_eff_trial  = E_T* (eps_T_i - eps_T_p_i)
        
        
        f_trial = np.fabs(sig_T_i_eff_trial - gamma_T * alpha_T_i)  -\
         (sig_0  + K_T * z_T_i - m * sig_N_i_eff_trial) * (1.0 - np.heaviside((sig_N_i_eff_trial), 1) * ((sig_N_i_eff_trial)**2 / (sig_t)**2 ))
         
        if f_trial > 1e-8: 
            
            def f(vars):
                
                delta_lamda ,   sig_N_eff  = vars
                
                N =  (m  + np.heaviside((sig_N_eff), 1) * ((-m * sig_N_eff **2)/ (sig_t)**2    +\
                                                                       (sig_0  + K_T * (z_T_i + delta_lamda) - m * sig_N_eff) * (2 * sig_N_eff / (sig_t)**2 ) )) /(1. - omega_N_i)
            
                f1 = sig_0 + K_T * (z_T_i + delta_lamda) - m * sig_N_eff
                
                ft =  1.0 - np.heaviside(sig_N_eff, 1) * (sig_N_eff**2 / (sig_t)**2)
                
                f_lamda  = np.fabs(sig_T_i_eff_trial - gamma_T * alpha_T_i) - delta_lamda * ( E_T/(1. - omega_T_i) + gamma_T)  - f1 * ft
                f_N = sig_N_eff - sig_N_i_eff_trial +  delta_lamda * E_N * N

                return [f_lamda, f_N]
            
            x0 = np.array([0, 0])
            
            sol   =  root(f, x0 = x0,  method='lm')
            delta_lamda,  sig_N_eff= sol.x
            
            eps_N_p_i +=  delta_lamda* (m  - np.heaviside((sig_N_eff), 1) * ((m * sig_N_eff **2)/ (sig_t)**2 - (sig_0  - m * sig_N_eff) * (2 * sig_N_eff / (sig_t)**2 ) ))
            eps_T_p_i += delta_lamda * np.sign(sig_T_i_eff_trial - gamma_T * alpha_T_i) / (1.0 - omega_T_i)
            
            z_T_i += delta_lamda
            alpha_T_i += delta_lamda * np.sign(sig_T_i_eff_trial - gamma_T * alpha_T_i)
            
            Y_N = 0.5 * E_N * (eps_N_i - eps_N_p_i)**2.0
            Y_T = 0.5 * E_T * (eps_T_i - eps_T_p_i)**2.0
            
            omega_N_i +=  delta_lamda*(1 - omega_N_i)**c_N * (Y_N / S_N + b * Y_T /S_T) * np.heaviside(eps_N_i, 1)
            omega_T_i +=  delta_lamda*(1 - omega_T_i)**c_T * (Y_T / S_T + b * Y_N /S_N)
            
            sig_N_i = (1.0 - omega_N_i) *   sig_N_eff
            
            sig_T_i = (1.0 - omega_T_i) * E_T * (eps_T_i - eps_T_p_i) 
            
            f = np.fabs(sig_T_i /(1 - omega_T_i)  - gamma_T * alpha_T_i)  -\
            (sig_0  + K_T * z_T_i - m * sig_N_i /(1 - omega_N_i)) * (1.0 - np.heaviside(( sig_N_i /(1 - omega_N_i)), 1) * (( sig_N_i /(1 - omega_N_i))**2 / (sig_t)**2 ))   
                       
            
        else: 
                
            sig_N_i = (1 - omega_N_i) * sig_N_i_eff_trial 
            
            sig_T_i = (1 - omega_T_i) * sig_T_i_eff_trial 
        
            f  = f_trial

        sig_N_trial_arr[i] = sig_N_i_eff_trial 
        sig_T_trial_arr[i] = sig_T_i_eff_trial
        sig_N_arr[i] = sig_N_i 
        sig_T_arr[i] = sig_T_i 
        eps_T_p_arr[i] = eps_T_p_i
        eps_N_p_arr[i] = eps_N_p_i
        omega_N_arr[i] = omega_N_i
        omega_T_arr[i] = omega_T_i
        z_T_arr[i] = z_T_i
        alpha_T_arr[i] = alpha_T_i
        f_arr[i] = f
        
    return  sig_N_trial_arr, sig_T_trial_arr, sig_N_arr, sig_T_arr, eps_T_p_arr, eps_N_p_arr, omega_N_arr, omega_T_arr, z_T_arr, alpha_T_arr, f_arr 


if __name__ == '__main__':
    
    inc = 10

    s_history = np.array([0, 0.0002])
    eps_N_arr = np.hstack([np.linspace(s_history[i], s_history[i + 1], inc)
                         for i in range(len(s_history) - 1)])
    
    t_N_arr= t_arr = np.linspace(0, 1, len(eps_N_arr))
    
    

    s_history = np.array([0, 0.0004])
    eps_T_arr = np.hstack([np.linspace(s_history[i], s_history[i + 1], inc)
                         for i in range(len(s_history) - 1)])
    
    t_T_arr= t_arr = np.linspace(0, 1, len(eps_T_arr))


    
    sig_0=10.0
    E_N=50000.
    E_T=20000.
    sig_t = 5.0
    S_N = 0.00000001
    S_T = 0.0000005
    c_N = 1.2
    c_T = 2.2
    K_T = 0
    gamma_T = 0

    m=0.2
    b=0.2


    sig_N_trial_arr, sig_T_trial_arr, sig_N_arr, sig_T_arr, eps_T_p_arr, eps_N_p_arr, omega_N_arr, omega_T_arr, z_T_arr, alpha_T_arr, f_arr = get_response(
        eps_N_arr, eps_T_arr, sig_0, sig_t, E_T, E_N, m, S_N, S_T, c_N, c_T, K_T, gamma_T)
    
    
    
    
    
    
    #  loading history
    ax1 = plt.subplot(231)
    ax1.plot(t_N_arr, eps_N_arr, 'k', linewidth=2,
             label='normal')
    ax1.plot(t_T_arr, eps_T_arr, 'r', linewidth=2,
             label='tangential')
    plt.xlabel('time')
    plt.ylabel('strain')
    plt.title('Loading history')

    
    #  stress-strain
    ax1 = plt.subplot(232)
    ax1.plot(eps_N_arr, sig_N_arr, 'k', linewidth=2,
             label='normal')
    ax1.plot(eps_N_arr, sig_N_trial_arr, '--k', linewidth=2,
             label='normal')
    ax1.plot(eps_T_arr, sig_T_arr, 'r', linewidth=2,
             label='tangential')
    ax1.plot(eps_T_arr, sig_T_trial_arr, '--r', linewidth=2,
             label='tangential')

    plt.xlabel('strain')
    plt.ylabel('Stress(MPa)')

    
    # damage
    ax2 = plt.subplot(233)
    ax2.plot(t_N_arr, omega_N_arr, 'k', linewidth=2,
             label='normal')
    ax2.plot(t_T_arr, omega_T_arr, 'r', linewidth=2,
             label='tangential')

    plt.title('Damage evolution')
    plt.ylim(0, 1)
    plt.xlabel('time')
    plt.ylabel('Damage')
    plt.legend(loc=4)
    
    
    
    # plastic strains
    ax2 = plt.subplot(234)

    ax2.plot(eps_N_arr, eps_N_p_arr, 'k', linewidth=2,
             label='normal')
    ax2.plot(eps_T_arr, eps_T_p_arr, 'r', linewidth=2,
             label='tangential')

    ax2.axhline(y=0, color='k', linewidth=1, alpha=0.5)
    ax2.axvline(x=0, color='k', linewidth=1, alpha=0.5)
    plt.title('plastic strains')
    plt.xlabel('strain')
    plt.ylabel('plastic strains')


    # threshold
    ax2 = plt.subplot(235)
    ax2.plot(eps_N_arr, f_arr, 'k', linewidth=2,
             label='normal')
    ax2.plot(eps_T_arr, f_arr, 'k', linewidth=2,
             label='normal')

    ax2.axhline(y=0, color='k', linewidth=1, alpha=0.5)
    ax2.axvline(x=0, color='k', linewidth=1, alpha=0.5)
    plt.ylim(0, 0.1)
    plt.title('Threshol - f')
    plt.xlabel('strain')
    plt.ylabel('f')
    plt.legend(loc=4)
    
    
    
    
    
    
    '''
    # plot the return mapping
    '''

    sig_N = np.linspace(-30 , 0.1* sig_0 / m, 1000) 
     
    f_1 = (sig_0 - m * sig_N) 
     
    f_t = (1.0 - np.heaviside((sig_N), 1) * ((sig_N)**2 / (sig_t)**2  ))
     
    f = f_1 * f_t 
     
 
      
    sig_N_arr = sig_N_arr/(1 -omega_N_arr)
    sig_T_arr = sig_T_arr/(1 -omega_T_arr)  
     
    plt.subplot(236) 
    plt.plot(sig_N , f_1, 'k', linewidth=2.0, alpha=1.0)
    plt.plot(sig_N , f, 'r', linewidth=2.0, alpha=1.0)
     
     
    plt.plot(sig_N_trial_arr[:], np.abs(sig_T_trial_arr)[:], 'go', alpha=1.0)
    plt.plot(sig_N_arr[:], np.abs(sig_T_arr)[:], 'yo', linewidth=2.0, alpha=1.0)

     
    for i in np.arange(0, inc):
         
        plt.arrow(sig_N_trial_arr[i], np.abs(sig_T_trial_arr)[i] ,
                sig_N_arr[i] -  sig_N_trial_arr[i], 
                np.abs(sig_T_arr)[i]- np.abs(sig_T_trial_arr)[i], 
                head_width=0.0005,head_length=0.001,color = 'r')
     
    plt.ylim([0,20])

     
    plt.axhline(y=0, color='k', linewidth=0.5, alpha=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5, alpha=0.5)


    plt.show()

