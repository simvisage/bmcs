import os

import matplotlib

from apps.sandbox.mario.Framcos.Micro2Dplot import Micro2Dplot
from apps.sandbox.mario.Framcos.vmats2D_mpl_csd_eeq import MATS2DMplCSDEEQ

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def get_eps_ab(eps_O):

    eps_ab = np.zeros((2, 2))
    eps_ab[0, 0] = eps_O[0]
    eps_ab[0, 1] = eps_O[2]
    eps_ab[1, 0] = eps_O[2]
    eps_ab[1, 1] = eps_O[1]

    return eps_ab


def get_sig_O(sig_ab):

    sig_O = np.zeros((3, 1))
    sig_O[0] = sig_ab[0, 0]
    sig_O[1] = sig_ab[1, 1]
    sig_O[2] = sig_ab[1, 0]

    return sig_O


def get_K_OP(D_abcd):

    K_OP = np.zeros((3, 3))

    K_OP[0, 0] = D_abcd[0, 0, 0, 0]
    K_OP[0, 1] = D_abcd[0, 0, 1, 1]
    K_OP[0, 2] = 2 * D_abcd[0, 0, 0, 1]

    K_OP[1, 0] = D_abcd[1, 1, 0, 0]
    K_OP[1, 1] = D_abcd[1, 1, 1, 1]
    K_OP[1, 2] = 2 * D_abcd[1, 1, 0, 1]

    K_OP[2, 0] = D_abcd[0, 1, 0, 0]
    K_OP[2, 1] = D_abcd[0, 1, 1, 1]
    K_OP[2, 2] = 2 * D_abcd[0, 1, 0, 1]

    return K_OP


def get_UF_t(time_function, n_t):


    int_var = np.zeros((n_mp, 21))
    int_var_aux = np.zeros((n_mp, 21))
    dissip = np.zeros((n_mp, 8))
    save = np.concatenate((int_var, dissip), axis=1)
    df = pd.DataFrame(save)
    df.to_hdf(path, 'first', mode='w', format='table')

    # total number of DOFs
    n_O = 3
    # Global vectors
    F_ext = np.zeros((n_O,), np.float_)
    F_O = np.zeros((n_O,), np.float_)
    U_P = np.zeros((n_O,), np.float_)
    U_k_O = np.zeros((n_O,), dtype=np.float_)
    eps_aux = get_eps_ab(U_k_O)
    # Construct index maps distinguishing the controlled displacements
    # and free displacements. Here we simply say the the control displacement
    # is the first one. Then index maps are constructed using the np.where
    # function returning the indexes of True positions in a logical array.
    CONTROL = 0
    FREE = slice(1, None)  # This means all except the first index, i.e. [1:]
    # Setup the system matrix with displacement constraints
    # Time stepping parameters
    t_n1, t_max, t_step = 0, len(time_function), 1 / n_t
    t_n = 0
    # Iteration parameters
    k_max, R_acc = 1000, 1e-5
    # Record solutions
    U_t_list, F_t_list, U_P_list = [np.copy(U_k_O)], [np.copy(F_O)], [np.copy(U_P)]
    D = np.zeros((2, 2, 2, 2))
    D = D[np.newaxis, :, :, :, :]

    # Load increment loop
    while t_n1 <= t_max - 1:
        #print('t:', t_n1)
        # Get the displacement increment for this step
        delta_U = time_function[t_n1] - time_function[t_n]
        k = 0
        # Equilibrium iteration loop
        while k < k_max:
            # Transform the primary vector to field
            eps_ab = get_eps_ab(U_k_O)
            # Stress and material stiffness
            D_abcd, sig_ab, eps_p_Emab = m.get_corr_pred(
                eps_ab, 1, int_var, eps_aux, F_ext
            )
            # Internal force
            F_O = get_sig_O(sig_ab).reshape(3,)
            U_P = get_sig_O(eps_p_Emab).reshape(3, )
            # System matrix
            K_OP = get_K_OP(D_abcd)
            # Get the balancing forces - NOTE - for more displacements
            # this should be an assembly operator.
            # KU remains a 2-d array so we have to make it a vector
            KU = K_OP[:, CONTROL] * delta_U
            # Residuum
            R_O = F_ext - F_O - KU
            # Convergence criterion
            R_norm = np.linalg.norm(R_O[FREE])
            if R_norm < R_acc:
                # Convergence reached
                break
            # Next iteration -
            delta_U_O = np.linalg.solve(K_OP[FREE, FREE], R_O[FREE])
            # Update total displacement
            U_k_O[FREE] += delta_U_O
            # Update control displacement
            U_k_O[CONTROL] += delta_U
            # Note - control displacement nonzero only in the first iteration.
            delta_U = 0
            k += 1
        else:
            print('no convergence')
            break

        # Update states variables after convergence
        int_var = m._get_state_variables(eps_ab, int_var, eps_aux)

        # Definition internal variables / forces per column:  1) damage N, 2)iso N, 3)kin N, 4) consolidation N, 5) eps p N,
        # 6) sigma N, 7) iso F N, 8) kin F N, 9) energy release N, 10) damage T, 11) iso T, 12-13) kin T, 14-15) eps p T,
        # 16-17) sigma T, 18) iso F T, 19-20) kin F T, 21) energy release T

        # Definition dissipation components per column: 1) damage N, 2) damage T, 3) eps p N, 4) eps p T, 5) iso N
        # 6) iso T, 7) kin N, 8) kin T

        dissip[:, 0] += np.einsum('...n,...n->...n', int_var[:, 0] - int_var_aux[:, 0], int_var[:, 8])
        dissip[:, 1] += np.einsum('...n,...n->...n', int_var[:, 9] - int_var_aux[:, 9], int_var[:, 20])
        dissip[:, 2] += np.einsum('...n,...n->...n', int_var[:, 4] - int_var_aux[:, 4], int_var[:, 5])
        dissip[:, 3] += np.einsum('...n,...n->...', int_var[:, 13:15] - int_var_aux[:, 13:15], int_var[:, 15:17])
        dissip[:, 4] += np.einsum('...n,...n->...n', int_var[:, 1] - int_var_aux[:, 1], int_var[:, 6])
        dissip[:, 5] += np.einsum('...n,...n->...n', int_var[:, 10] - int_var_aux[:, 10], int_var[:, 17])
        dissip[:, 6] += np.einsum('...n,...n->...n', int_var[:, 2] - int_var_aux[:, 2], int_var[:, 7])
        dissip[:, 7] += np.einsum('...n,...n->...', int_var[:, 11:13] - int_var_aux[:, 11:13], int_var[:, 18:20])

        int_var_aux = int_var * 1

        save = np.concatenate((int_var, dissip), axis=1)

        df = pd.DataFrame(save)
        df.to_hdf(path, 'middle' + np.str(t_n1), append=True)

        U_t_list.append(np.copy(U_k_O))
        F_t_list.append(F_O)
        U_P_list.append(U_P)
        eps_aux = get_eps_ab(U_k_O)
        D_aux = D_abcd[np.newaxis, :, :, :, :]
        D = np.concatenate((D, D_aux))
        t_n = t_n1
        t_n1 += 1

    U_t, F_t, U_p = np.array(U_t_list), np.array(F_t_list),np.array(U_P_list)
    return U_t, F_t, D, U_p

def get_int_var(path, size, n_mp):  # unpacks saved data

    S = np.zeros((len(F), n_mp, 29))

    S[0] = np.array(pd.read_hdf(path, 'first'))

    for i in range(1, size):
        S[i] = np.array(pd.read_hdf(path, 'middle' + np.str(i - 1)))

    omega_N_Emn = S[:, :, 0]
    z_N_Emn = S[:, :, 1]
    alpha_N_Emn = S[:, :, 2]
    r_N_Emn = S[:, :, 3]
    eps_N_p_Emn = S[:, :, 4]
    sigma_N_Emn = S[:, :, 5]
    Z_N_Emn = S[:, :, 6]
    X_N_Emn = S[:, :, 7]
    Y_N_Emn = S[:, :, 8]

    omega_T_Emn = S[:, :, 9]
    z_T_Emn = S[:, :, 10]
    alpha_T_Emna = S[:, :, 11:13]
    eps_T_pi_Emna = S[:, :, 13:15]
    sigma_T_Emna = S[:, :, 15:17]
    Z_T_pi_Emn = S[:, :, 17]
    X_T_pi_Emna = S[:, :, 18:20]
    Y_T_pi_Emn = S[:, :, 20]

    Disip_omena_N_Emn = S[:, :, 21]
    Disip_omena_T_Emn = S[:, :, 22]
    Disip_eps_p_N_Emn = S[:, :, 23]
    Disip_eps_p_T_Emn = S[:, :, 24]
    Disip_iso_N_Emn = S[:, :, 25]
    Disip_iso_T_Emn = S[:, :, 26]
    Disip_kin_N_Emn = S[:, :, 27]
    Disip_kin_T_Emn = S[:, :, 28]

    return omega_N_Emn, z_N_Emn, alpha_N_Emn, r_N_Emn, eps_N_p_Emn, sigma_N_Emn, Z_N_Emn, X_N_Emn, Y_N_Emn, \
           omega_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna, sigma_T_Emna, Z_T_pi_Emn, X_T_pi_Emna, Y_T_pi_Emn, \
           Disip_omena_N_Emn, Disip_omena_T_Emn, Disip_eps_p_N_Emn, Disip_eps_p_T_Emn, Disip_iso_N_Emn, \
           Disip_iso_T_Emn, Disip_kin_N_Emn, Disip_kin_T_Emn



concrete_type= 6 # 0:C40MA, 1:C80MA, 2:120MA, 3:Tensile, 4:Compressive, 5:Biaxial, 6: Paper 2D redistribution

Concrete_Type_string = ['C40MA', 'C80MA','C120MA', 'Tensile', 'Compressive', 'Biaxial', 'Paper_2D']

loading_scenario = 'monotonic'   # monotonic, cyclic

M_plot = 1  # Plot microplanes polar graphs. 1: yes, 0: no

t_steps = 500
n_mp = 360

if loading_scenario == 'monotonic':

    eps = -0.01

    load = np.linspace(0, eps, t_steps)

if loading_scenario == 'cyclic':

    eps = [-0.002745, -0.0016, -0.0047, -0.0025, -0.0065, -0.003, -0.0085, -0.004, -0.01]

    load1 = np.linspace(0, eps[0], t_steps)
    load2 = np.linspace(load1[-1], eps[1], t_steps)
    load3 = np.linspace(load2[-1], eps[2], t_steps)
    load4 = np.linspace(load3[-1], eps[3], t_steps)
    load5 = np.linspace(load4[-1], eps[4], t_steps)
    load6 = np.linspace(load5[-1], eps[5], t_steps)
    load7 = np.linspace(load6[-1], eps[6], t_steps)
    load8 = np.linspace(load7[-1], eps[7], t_steps)
    load9 = np.linspace(load8[-1], eps[8], t_steps)

    load = np.concatenate((load1, load2[1::], load3[1::], load4[1::], load5[1::], load6[1::], load7[1::], load8[1::], load9[1::]))


t_steps = len(load)

# Path saving data

home_dir = os.path.expanduser('~')

if not os.path.exists('Data Processing'):
    os.makedirs('Data Processing')

path = os.path.join(
   home_dir, 'Data Processing/' + 'dc' + Concrete_Type_string[concrete_type] + loading_scenario + '.hdf5')

m = MATS2DMplCSDEEQ(concrete_type)
plot = Micro2Dplot()


U, F, D, U_p = get_UF_t(
    load,
    t_steps
)

[omega_N_Emn, z_N_Emn, alpha_N_Emn, r_N_Emn, eps_N_p_Emn, sigma_N_Emn, Z_N_Emn, X_N_Emn, Y_N_Emn, omega_T_Emn, z_T_Emn,
 alpha_T_Emna, eps_T_pi_Emna, sigma_T_Emna, Z_T_pi_Emn, X_T_pi_Emna, Y_T_pi_Emn, Disip_omena_N_Emn, Disip_omena_T_Emn,
 Disip_eps_p_N_Emn, Disip_eps_p_T_Emn, Disip_iso_N_Emn, Disip_iso_T_Emn, Disip_kin_N_Emn, Disip_kin_T_Emn] \
    = get_int_var(path, len(F), n_mp)

font = {'family': 'DejaVu Sans',
        'size': 18}

matplotlib.rc('font', **font)


f, (ax2) = plt.subplots(1, 1, figsize=(5, 4))

ax2.plot(np.abs(U[:, 0]), np.abs(F[:, 0]), 'k', linewidth=3.5)
ax2.set_xlabel(r'$|\varepsilon_{11}$| [-]', fontsize=25)
ax2.set_ylabel(r'$|\sigma{11}$| [-]', fontsize=25)

print(np.max(np.abs(F[:, 0])), 'fc')

plt.show()



if M_plot == 1:

    # For plotting polar plots, first convert vectorial variables into scalars (by taking norm)

    eps_N_Emn = np.zeros_like(eps_N_p_Emn)
    eps_T_Emna = np.zeros_like(eps_T_pi_Emna)
    eps_T_Emn = np.zeros_like(eps_N_p_Emn)
    eps_T_pi_Emn = np.zeros_like(eps_N_p_Emn)
    sigma_T_Emn = np.zeros_like(eps_N_p_Emn)
    X_T_pi_Emn = np.zeros_like(eps_N_p_Emn)
    alpha_T_Emn = np.zeros_like(eps_N_p_Emn)

    for i in range(len(F)):
        eps = get_eps_ab(U[i])
        eps_N_Emn[i] = m._get_e_N_Emn_2(eps)
        eps = get_eps_ab(U[i])
        eps_T_Emna[i] = m._get_e_T_Emnar_2(eps)


    eps_T_sign_Emn = np.zeros_like(eps_N_p_Emn)
    eps_pi_T = np.zeros_like(eps_N_p_Emn)
    eps_pi_T_sign = np.zeros_like(eps_N_p_Emn)
    sigma_T = np.zeros_like(eps_N_p_Emn)
    sigma_T_sign = np.zeros_like(eps_N_p_Emn)
    X_T = np.zeros_like(eps_N_p_Emn)
    X_T_sign = np.zeros_like(eps_N_p_Emn)


    eps_T_Emn = np.sqrt(np.einsum('...i,...i->... ', eps_T_Emna, eps_T_Emna))
    eps_T_pi_Emn = np.sqrt(np.einsum('...i,...i->... ', eps_T_pi_Emna, eps_T_pi_Emna))
    sigma_T_Emn = np.sqrt(np.einsum('...i,...i->... ', sigma_T_Emna, sigma_T_Emna))
    X_T_pi_Emn = np.sqrt(np.einsum('...i,...i->... ', X_T_pi_Emna, X_T_pi_Emna))
    alpha_T_Emn = np.sqrt(np.einsum('...i,...i->... ', alpha_T_Emna, alpha_T_Emna))


    plot.get_2Dviz(n_mp, eps_N_Emn, eps_T_Emn, omega_N_Emn, z_N_Emn, alpha_N_Emn, r_N_Emn, eps_N_p_Emn, sigma_N_Emn, Z_N_Emn, X_N_Emn, Y_N_Emn, \
           omega_T_Emn, z_T_Emn, alpha_T_Emn, eps_T_pi_Emn, sigma_T_Emn, Z_T_pi_Emn, X_T_pi_Emn, Y_T_pi_Emn, \
           Disip_omena_N_Emn, Disip_omena_T_Emn, Disip_eps_p_N_Emn, Disip_eps_p_T_Emn, Disip_iso_N_Emn, \
           Disip_iso_T_Emn, Disip_kin_N_Emn, Disip_kin_T_Emn)
