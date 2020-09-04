import os

import matplotlib

from apps.sandbox.mario.Framcos.vmats3D_mpl_csd_eeq import MATS3DMplCSDEEQ

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DELTA = np.identity(3)

EPS = np.zeros((3, 3, 3), dtype='f')
EPS[(0, 1, 2), (1, 2, 0), (2, 0, 1)] = 1
EPS[(2, 1, 0), (1, 0, 2), (0, 2, 1)] = -1


DD = np.hstack([DELTA, np.zeros_like(DELTA)])
EEPS = np.hstack([np.zeros_like(EPS), EPS])

GAMMA = np.einsum(
    'ik,jk->kij', DD, DD
) + np.einsum(
    'ikj->kij', np.fabs(EEPS)
)


def get_eps_ab(eps_O): return np.einsum(
    'Oab,...O->...ab', GAMMA, eps_O
)[np.newaxis, ...]


GAMMA_inv = np.einsum(
    'aO,bO->Oab', DD, DD
) + 0.5 * np.einsum(
    'aOb->Oab', np.fabs(EEPS)
)


def get_sig_O(sig_ab): return np.einsum(
    'Oab,...ab->...O', GAMMA_inv, sig_ab
)[0, ...]


GG = np.einsum(
    'Oab,Pcd->OPabcd', GAMMA_inv, GAMMA_inv
)


def get_K_OP(D_abcd):
    return np.einsum(
        'OPabcd,abcd->OP', GG, D_abcd
    )

# The above operators provide the three mappings
# map the primary variable to from vector to field
# map the residuum field to evctor (assembly operator)
# map the gradient of the residuum field to system matrix


def get_UF_t(time_function, n_t):


    int_var = np.zeros((n_mp, 25))
    int_var_aux = np.zeros((n_mp, 25))
    dissip = np.zeros((n_mp, 8))
    save = np.concatenate((int_var, dissip), axis=1)
    df = pd.DataFrame(save)
    df.to_hdf(path, 'first', mode='w', format='table')
    D = np.zeros((3, 3, 3, 3))
    D = D[np.newaxis, :, :, :, :]

    # total number of DOFs
    n_O = 6
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
    k_max, R_acc = 1000, 1e-3
    # Record solutions
    U_t_list, F_t_list, U_P_list = [np.copy(U_k_O)], [np.copy(F_O)], [np.copy(U_P)]

    # Load increment loop
    while t_n1 <= t_max - 1:
        #print('t:', t_n1)
        # Get the displacement increment for this step
        delta_U = time_function[t_n1] - time_function[t_n]
        k = 0
        # Equilibrium iteration loop
        while k < k_max:
            # Transform the primary vector to field
            eps_ab = get_eps_ab(U_k_O).reshape(3, 3)
            # Stress and material stiffness
            D_abcd, sig_ab, eps_p_Emab = m.get_corr_pred(
                eps_ab, 1, int_var, eps_aux, F_ext
            )
            # Internal force
            F_O = get_sig_O(sig_ab.reshape(1, 3, 3)).reshape(6, )

            # System matrix
            K_OP = get_K_OP(D_abcd)
            #Beta = get_K_OP(beta_Emabcd)
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

        # Target time of the next load increment
#         U_t_list.append(np.copy(U_k_O))
#         F_t_list.append(F_O)

            # Update states variables after convergence
        int_var = m._get_state_variables(eps_ab, int_var, eps_aux)

        # Definition internal variables / forces per column:  1) damage N, 2)iso N, 3)kin N, 4) consolidation N, 5) eps p N,
        # 6) sigma N, 7) iso F N, 8) kin F N, 9) energy release N, 10) damage T, 11) iso T, 12-14) kin T, 15-17) eps p T,
        # 18-20) sigma T, 21) iso F T, 22-24) kin F T, 25) energy release T

        # Definition dissipation components per column: 1) damage N, 2) damage T, 3) eps p N, 4) eps p T, 5) iso N
        # 6) iso T, 7) kin N, 8) kin T

        dissip[:, 0] += np.einsum('...n,...n->...n', int_var[:, 0] - int_var_aux[:, 0], int_var[:, 8])
        dissip[:, 1] += np.einsum('...n,...n->...n', int_var[:, 9] - int_var_aux[:, 9], int_var[:, 20])
        dissip[:, 2] += np.einsum('...n,...n->...n', int_var[:, 4] - int_var_aux[:, 4], int_var[:, 5])
        dissip[:, 3] += np.einsum('...n,...n->...', int_var[:, 14:17] - int_var_aux[:, 14:17], int_var[:, 17:20])
        dissip[:, 4] += np.einsum('...n,...n->...n', int_var[:, 1] - int_var_aux[:, 1], int_var[:, 6])
        dissip[:, 5] += np.einsum('...n,...n->...n', int_var[:, 10] - int_var_aux[:, 10], int_var[:, 17])
        dissip[:, 6] += np.einsum('...n,...n->...n', int_var[:, 2] - int_var_aux[:, 2], int_var[:, 7])
        dissip[:, 7] += np.einsum('...n,...n->...', int_var[:, 11:14] - int_var_aux[:, 11:14], int_var[:, 21:24])

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

    U_t, F_t, U_p = np.array(U_t_list), np.array(F_t_list), np.array(U_P_list)
    return U_t, F_t, D, U_p

def get_int_var(path, size, n_mp):  # unpacks saved data

    S = np.zeros((len(F), n_mp, 33))

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
    alpha_T_Emna = S[:, :, 11:14]
    eps_T_pi_Emna = S[:, :, 14:17]
    sigma_T_Emna = S[:, :, 17:20]
    Z_T_pi_Emn = S[:, :, 20]
    X_T_pi_Emna = S[:, :, 21:24]
    Y_T_pi_Emn = S[:, :, 24]

    Disip_omena_N_Emn = S[:, :, 25]
    Disip_omena_T_Emn = S[:, :, 26]
    Disip_eps_p_N_Emn = S[:, :, 27]
    Disip_eps_p_T_Emn = S[:, :, 28]
    Disip_iso_N_Emn = S[:, :, 29]
    Disip_iso_T_Emn = S[:, :, 30]
    Disip_kin_N_Emn = S[:, :, 31]
    Disip_kin_T_Emn = S[:, :, 32]

    return omega_N_Emn, z_N_Emn, alpha_N_Emn, r_N_Emn, eps_N_p_Emn, sigma_N_Emn, Z_N_Emn, X_N_Emn, Y_N_Emn, \
           omega_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna, sigma_T_Emna, Z_T_pi_Emn, X_T_pi_Emna, Y_T_pi_Emn, \
           Disip_omena_N_Emn, Disip_omena_T_Emn, Disip_eps_p_N_Emn, Disip_eps_p_T_Emn, Disip_iso_N_Emn, \
           Disip_iso_T_Emn, Disip_kin_N_Emn, Disip_kin_T_Emn


concrete_type= 0        # 0:C40MA, 1:C80MA, 2:120MA, 3:Tensile, 4:Compressive, 5:Biaxial

Concrete_Type_string = ['C40MA', 'C80MA','C120MA', 'Tensile', 'Compressive', 'Biaxial']

loading_scenario = 'monotonic'   # monotonic, cyclic

M_plot = 1  # Plot microplanes polar graphs. 1: yes, 0: no

t_steps = 100
n_mp = 28

if loading_scenario == 'monotonic':

    eps = -0.01

    load = np.linspace(0, eps, t_steps)

if loading_scenario == 'cyclic':

    eps = [-0.002785, -0.000225050506, -0.0055, -0.0025, -0.0065, -0.003, -0.0085, -0.004, -0.01]

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
   home_dir, 'Data Processing/'+ '3Ddc' + Concrete_Type_string[concrete_type] + loading_scenario + '.hdf5')

m = MATS3DMplCSDEEQ(concrete_type)



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

#plot.get_3Dviz(S[:, :, 0], S[:, :, 8])
