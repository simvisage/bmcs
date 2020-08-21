#!/usr/bin/env python
# coding: utf-8

# # Simulation of fatigue for uniaxial stress state

# Assume a uniaxial stress state with $\sigma_{11} = \bar{\sigma}(\theta)$ representing the loading function. All other components of the stress tensor are assumed zero
# \begin{align}
# \sigma_{ab} = 0; \forall a,b \in (0,1,2), a = b \neq 1
# \end{align}
#

# In[1]:

import copy
import matplotlib

from apps.sandbox.mario.Framcos.Micro2Dplot import Micro2Dplot
from apps.sandbox.mario.Framcos.vmats2D_mpl_csd_eeq import MATS2DMplCSDEEQ

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

#from .vmats3D_mpl_csd_eeq import MATS3DMplCSDEEQ





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


# The above operators provide the three mappings
# map the primary variable to from vector to field
# map the residuum field to evctor (assembly operator)
# map the gradient of the residuum field to system matrix

m = MATS2DMplCSDEEQ()
plot = Micro2Dplot()


def get_UF_t(eps_max, time_function, n_t):

    n_mp = 100
    omegaN = np.zeros((n_mp, ))
    omegaN_aux = np.zeros((n_mp,))
    z_N_Emn = np.zeros((n_mp, ))
    alpha_N_Emn = np.zeros((n_mp, ))
    r_N_Emn = np.zeros((n_mp, ))
    eps_N_p_Emn = np.zeros((n_mp, ))
    eps_N_p_Emn_aux = np.zeros((n_mp,))
    sigma_N_Emn = np.zeros((n_mp, ))
    Y_n = np.zeros((n_mp, ))
    R_n = np.zeros((n_mp, ))
    w_T_Emn = np.zeros((n_mp, ))
    w_T_Emn_aux = np.zeros((n_mp,))
    z_T_Emn = np.zeros((n_mp, ))
    alpha_T_Emna = np.zeros((n_mp, 2))
    eps_T_pi_Emna = np.zeros((n_mp, 2))
    eps_T_pi_Emna_aux = np.zeros((n_mp, 2))
    # eps_T_pi_Emna_aux = eps_T_pi_Emna_aux[np.newaxis, :, :]
    sigma_T_Emna = np.zeros((n_mp, 2))
    X_T = np.zeros((n_mp, 2))
    Y_T = np.zeros((n_mp, ))
    sctx = np.zeros((n_mp, 19))
    sctx = sctx[np.newaxis, :, :]
    D = np.zeros((2, 2, 2, 2))
    D = D[np.newaxis, :, :, :, :]
    D_E = np.zeros((1,))
    MPW = np.ones(n_mp) / n_mp * 2

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
                eps_ab, 1, omegaN, z_N_Emn,
                alpha_N_Emn, r_N_Emn, eps_N_p_Emn, sigma_N_Emn,
                w_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna, eps_aux, F_O
            )
            # Internal force
            F_O = get_sig_O(sig_ab).reshape(3,)
            U_P = get_sig_O(eps_p_Emab).reshape(3, )
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


        [omegaN, z_N_Emn, alpha_N_Emn, r_N_Emn, eps_N_p_Emn, sigma_N_Emn, Y_n, R_n, w_T_Emn, z_T_Emn,
            alpha_T_Emna, eps_T_pi_Emna, sigma_T_Emna, Y_T, X_T] = m._get_state_variables(
                eps_ab, 1, omegaN, z_N_Emn, alpha_N_Emn, r_N_Emn, eps_N_p_Emn, sigma_N_Emn,
                w_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna, eps_aux, F_O)

        #D_E_s = np.sum(np.einsum('...n,...n->...n',w_T_Emn-w_T_Emn_aux,Y_T) + np.einsum('...n,...n->...n',omegaN-omegaN_aux,Y_n) + np.einsum('...n,...n->...',eps_T_pi_Emna-eps_T_pi_Emna_aux,sigma_T_Emna) + np.einsum('...n,...n->...n',eps_N_p_Emn-eps_N_p_Emn_aux,sigma_N_Emn))
        D_E_s = np.sum(np.einsum('...n,...n->...n',MPW,np.einsum('...n,...n->...n',w_T_Emn-w_T_Emn_aux,Y_T)) + np.einsum('...n,...n->...n',MPW,np.einsum('...n,...n->...n',omegaN-omegaN_aux,Y_n)) + np.einsum('...n,...n->...n',MPW,np.einsum('...n,...n->...',eps_T_pi_Emna-eps_T_pi_Emna_aux,sigma_T_Emna)) + np.einsum('...n,...n->...n',MPW,np.einsum('...n,...n->...n',eps_N_p_Emn-eps_N_p_Emn_aux,sigma_N_Emn)))

        print(eps_T_pi_Emna-eps_T_pi_Emna_aux)
        omegaN = omegaN.reshape(n_mp, )
        omegaN_aux=omegaN*1
        z_N_Emn = z_N_Emn.reshape(n_mp, )
        alpha_N_Emn = alpha_N_Emn.reshape(n_mp, )
        r_N_Emn = r_N_Emn.reshape(n_mp, )
        eps_N_p_Emn = eps_N_p_Emn.reshape(n_mp, )
        eps_N_p_Emn_aux=eps_N_p_Emn*1
        sigma_N_Emn = sigma_N_Emn.reshape(n_mp,)
        Y_n = Y_n.reshape(n_mp,)
        R_n = R_n.reshape(n_mp,)
        w_T_Emn = w_T_Emn.reshape(n_mp, )
        w_T_Emn_aux=w_T_Emn*1
        z_T_Emn = z_T_Emn.reshape(n_mp, )
        alpha_T_Emna = alpha_T_Emna.reshape(n_mp, 2)
        eps_T_pi_Emna = eps_T_pi_Emna.reshape(n_mp, 2)
        eps_T_pi_Emna_aux=eps_T_pi_Emna*1
        # eps_T_pi_Emna_aux2 = eps_T_pi_Emna[np.newaxis, :, :]
        # eps_T_pi_Emna_aux= np.concatenate((eps_T_pi_Emna_aux, eps_T_pi_Emna_aux2))
        sigma_T_Emna = sigma_T_Emna.reshape(n_mp, 2)
        X_T = X_T.reshape(n_mp, 2)
        Y_T = Y_T.reshape(n_mp, )
        D_E_s = D_E_s.reshape(1, ) + D_E[-1]
        D_E = np.concatenate((D_E, D_E_s))


        sctx_aux = np.concatenate((omegaN.reshape(n_mp, 1), z_N_Emn.reshape(n_mp, 1), alpha_N_Emn.reshape(n_mp, 1),
                                   r_N_Emn.reshape(n_mp, 1), eps_N_p_Emn.reshape(
                                       n_mp, 1), sigma_N_Emn.reshape(n_mp, 1), Y_n.reshape(n_mp, 1), R_n.reshape(n_mp, 1),
                                   w_T_Emn.reshape(n_mp, 1), z_T_Emn.reshape(n_mp, 1), alpha_T_Emna, eps_T_pi_Emna, sigma_T_Emna, Y_T.reshape(n_mp, 1), X_T), axis=1)

        sctx_aux = sctx_aux[np.newaxis, :, :]
        sctx = np.concatenate((sctx, sctx_aux))
        D_aux = D_abcd[np.newaxis, :, :, :, :]
        D = np.concatenate((D, D_aux))
        U_t_list.append(np.copy(U_k_O))
        F_t_list.append(F_O)
        U_P_list.append(U_P)
        eps_aux = get_eps_ab(U_k_O)


        t_n = t_n1
        t_n1 += 1
        # print(t_n1)
        # print(Beta)

    U_t, F_t, U_p = np.array(U_t_list), np.array(F_t_list),np.array(U_P_list)
    return U_t, F_t, sctx, D, D_E,U_p


n_cycles = 1
T = 1 / n_cycles
eps_max = -0.01

t_steps = 100 * n_cycles

eps1 = -0.002785
eps2 = -0.000225050506
eps3 = -0.0055
eps4 = -0.0025
eps5 = -0.0065
eps6 = -0.003
eps7 = -0.0085
eps8 = -0.004
eps9 = -0.01

load1 = np.linspace(0, eps1, t_steps)


load2 = np.linspace(load1[-1], eps2, t_steps)
# load3 = np.linspace(load2[-1], eps3, t_steps)
# load4 = np.linspace(load3[-1], eps4, t_steps)
# load5 = np.linspace(load4[-1], eps5, t_steps)
# load6 = np.linspace(load5[-1], eps6, t_steps)
# load7 = np.linspace(load6[-1], eps7, t_steps)
# load8 = np.linspace(load7[-1], eps8, t_steps)
# load9 = np.linspace(load8[-1], eps9, t_steps)


# sin_load = np.concatenate(
#     (load1, load2[1::], load3[1::], load4[1::], load5[1::], load6[1::], load7[1::], load8[1::], load9[1::]))
# t = np.linspace(0, 1, len(sin_load))


t = np.linspace(0, 1, t_steps)
#load = eps_max * (np.sin(np.pi / T * (t - T / 2)) + 1) / 2
sin_load = np.linspace(0, eps_max, t_steps)
# sin_load = np.concatenate((load1, load2[1::]))
# # plt.plot(t, load)
# plt.show()

t_steps_total = len(sin_load)


def loading_history(t): return (np.sin(np.pi / T * (t - T / 2)) + 1) / 2


U, F, S, D, D_E, U_p = get_UF_t(
    eps_max,
    sin_load,
    t_steps_total
)

font = {'family': 'normal',
        'size': 18}

matplotlib.rc('font', **font)

# f, (ax1) = plt.subplots(1, 1, figsize=(5, 4))
#
# ax1.plot(t, np.abs(sin_load), 'k', linewidth=3.5, )
# ax1.set_xlabel('Pseudotime [-]', fontsize=25)
# ax1.set_ylabel(r'$|\varepsilon_{11}$| [-]', fontsize=25)
# plt.show()

f, (ax2) = plt.subplots(1, 1, figsize=(5, 4))

ax2.plot(np.abs(U[:, 0]), np.abs(F[:, 0]), 'k', linewidth=3.5)
ax2.set_xlabel(r'$|\varepsilon_{11}$| [-]', fontsize=25)
ax2.set_ylabel(r'$|\sigma{11}$| [-]', fontsize=25)

print(np.max(np.abs(F[:, 0])), 'fc')
# ax2.set_xlim(0.0000, 0.004)
# #
# ax2.set_ylim(0.0000, 130)


plt.show()
y_int = integrate.cumtrapz(np.abs(F[:, 0]),np.abs(U[:, 0]))
print(y_int[-1])
print(0.5*U_p[-1,0]*F[-1, 0])
print(0.5*(U[-1, 0]-U_p[-1,0])*F[-1, 0])
print(y_int[-1] - U_p[-1,0]*F[-1, 0])
print(D_E[-1])

# f, (ax2) = plt.subplots(1, 1, figsize=(5, 4))
#
# ax2.plot(np.abs(U[:, 0]), np.abs(U[:, 1]) /
#          np.abs(U[:, 0]), 'k', linewidth=3.5)
# ax2.set_xlabel(r'$|\varepsilon_{11}$| [-]', fontsize=25)
# ax2.set_ylabel(r'$|\sigma{11}$| [-]', fontsize=25)
# plt.show()

n_mp = 200
rads = np.arange(0, (2 * np.pi), (2 * np.pi) / n_mp)
eps_N = np.zeros((len(S), len(S[1])))

for i in range(len(F)):
    eps = get_eps_ab(U[i])
    eps_N[i] = m._get_e_N_Emn_2(eps)

omegaN = S[:, :, 0]
eps_p_N = S[:, :, 4]

eps_T = np.zeros((len(S), len(S[1])))
eps_T_sign = np.zeros((len(S), len(S[1])))
eps_pi_T = np.zeros((len(S), len(S[1])))
eps_pi_T_sign = np.zeros((len(S), len(S[1])))
sigma_T = np.zeros((len(S), len(S[1])))
sigma_T_sign = np.zeros((len(S), len(S[1])))
X_T = np.zeros((len(S), len(S[1])))
X_T_sign = np.zeros((len(S), len(S[1])))


for i in range(len(F)):
    eps = get_eps_ab(U[i])
    eps_T_aux = m._get_e_T_Emnar_2(eps)
    eps_T[i] = np.sqrt(np.einsum('...i,...i->... ', eps_T_aux, eps_T_aux))
    eps_pi_T[i] = np.sqrt(
        np.einsum('...i,...i->... ', S[i, :, 12:14], S[i, :, 12:14]))
    sigma_T[i] = np.sqrt(
        np.einsum('...i,...i->... ', S[i, :, 14:16], S[i, :, 14:16]))
    X_T[i] = np.sqrt(
        np.einsum('...i,...i->... ', S[i, :, 17:19], S[i, :, 17:19]))

    sign_T = np.sign(eps_T_aux)

    sign_pi_T = np.sign(S[i, :, 12:14])
    sign_sigma_T = np.sign(S[i, :, 14:16])
    sign_X_T = np.sign(S[i, :, 17:19])

    eps_T_sign[i] = np.einsum(
        '...n,...n->...n', np.sign(np.einsum('...ni,...ni->...n', sign_T, sign_T)), eps_T[i])
    eps_pi_T_sign[i] = np.einsum(
        '...n,...n->...n', np.sign(np.einsum('...ni,...ni->...n', sign_pi_T, sign_pi_T)), eps_pi_T[i])

    sigma_T_sign[i] = np.einsum(
        '...n,...n->...n', np.sign(np.einsum('...ni,...ni->...n', sign_sigma_T, sign_sigma_T)), sigma_T[i])
    X_T_sign[i] = np.einsum(
        '...n,...n->...n', np.sign(np.einsum('...ni,...ni->...n', sign_X_T, sign_X_T)), X_T[i])


omegaT = S[:, :, 8]


sigma_N = S[:, :, 5]
Y_N = S[:, :, 6]
X_N = S[:, :, 7]
Y_T = S[:, :, 16]


eps_global_norm = np.zeros((len(S), len(S[1])))
sigma_global_norm = np.zeros((len(S), len(S[1])))
eps = np.zeros((len(S), 2, 2))
sigma = np.zeros((len(S), 2, 2))
n_mp = 100

for i in range(len(F)):
    eps[i] = get_eps_ab(U[i])
    sigma[i] = get_eps_ab(F[i])
    eps_micro = np.einsum('...ij,...j->...i', eps[i], m._get__MPN())
    eps_global_norm[i] = np.einsum('...n,...n->...n', np.sign(np.einsum('...i,...i->...',
                                                                        eps_micro, m._get__MPN())), np.sqrt(np.einsum('...i,...i->... ', eps_micro, eps_micro)))
    sigma_micro = np.einsum('...ij,...j->...i', sigma[i], m._get__MPN())
    sigma_global_norm[i] = np.einsum('...n,...n->...n', np.sign(np.einsum('...i,...i->...',
                                                                          sigma_micro, m._get__MPN())), np.sqrt(np.einsum('...i,...i->... ', sigma_micro, sigma_micro)))
eps_T = m._get_e_T_Emnar_2(eps)
eps_Emna = np.einsum(
    '...i,...i->...i', eps_N.reshape(len(F), n_mp, 1), m._get__MPN()) + eps_T
eps_micro_norm = np.einsum('...n,...n->...n', np.sign(np.einsum('...i,...i->...',
                                                                eps_Emna,  m._get__MPN())), np.sqrt(np.einsum('...i,...i->... ', eps_Emna, eps_Emna)))

sigma_Emna = np.einsum(
    '...i,...i->...i', sigma_N.reshape(len(F), n_mp, 1), m._get__MPN()) + S[:, :, 14:16]
sigma_micro_norm = np.einsum('...n,...n->...n', np.sign(np.einsum('...i,...i->...',
                                                                  sigma_Emna,  m._get__MPN())), np.sqrt(np.einsum('...i,...i->... ', sigma_Emna, sigma_Emna)))

D_1 = D[:, 0, 0, :, :]
D_1_Emna = np.einsum('...ij,...nj->...ni', D_1, m._get__MPN())

D_1_norm = np.einsum('...n,...n->...n', np.sign(np.einsum('...ni,...ni->...n',
                                                          D_1_Emna, m._get__MPN())), np.sqrt(np.einsum('...i,...i->... ', D_1_Emna, D_1_Emna)))

D_2 = D[:, 1, 1, :, :]
D_2_Emna = np.einsum(
    '...ij,...nj->...ni', D_2, m._get__MPN())
D_2_norm = np.einsum('...n,...n->...n', np.sign(np.einsum('...i,...i->...',
                                                          D_2_Emna, m._get__MPN())), np.sqrt(np.einsum('...i,...i->... ', D_2_Emna, D_2_Emna)))

D_12 = D[:, 0, 1, :, :]
D_12_Emna = np.einsum(
    '...ij,...nj->...ni', D_12, m._get__MPN())
D_12_norm = np.einsum('...n,...n->...n', np.sign(np.einsum('...ni,...ni->...n',
                                                           D_12_Emna, D_12_Emna)), np.sqrt(np.einsum('...i,...i->... ', D_12_Emna, D_12_Emna)))

D_21 = D[:, 1, 0, :, :]
D_21_Emna = np.einsum(
    '...ij,...nj->...ni', D_21, m._get__MPN())
D_21_norm = np.einsum('...n,...n->...n', np.sign(np.einsum('...ni,...ni->...n',
                                                           D_21_Emna, D_21_Emna)), np.sqrt(np.einsum('...i,...i->... ', D_21_Emna, D_21_Emna)))


# plot.get_2Dviz(n_mp, eps_global_norm, sigma_global_norm, eps_micro_norm, sigma_micro_norm, D_1_norm, D_2_norm, D_12_norm,
#                D_21_norm, eps_N, eps_p_N, sigma_N, omegaN, eps_T_sign, eps_pi_T_sign,
#                sigma_T_sign, omegaT, Y_N, X_N, Y_T, X_T_sign, F, U, t_steps)
