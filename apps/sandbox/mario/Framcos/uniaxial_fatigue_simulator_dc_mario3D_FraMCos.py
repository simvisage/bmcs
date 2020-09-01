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

from apps.sandbox.mario.Framcos.vmats3D_mpl_csd_eeq import MATS3DMplCSDEEQ
from apps.sandbox.mario.Micro3Dplot import Micro3Dplot
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


#from .vmats3D_mpl_csd_eeq import MATS3DMplCSDEEQ
sp.init_printing()

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


m = MATS3DMplCSDEEQ()
plot = Micro3Dplot()


def get_UF_t(eps_max, time_function, n_t):

    n_mp = 28
    omegaN = np.zeros((n_mp, ))
    z_N_Emn = np.zeros((n_mp, ))
    alpha_N_Emn = np.zeros((n_mp, ))
    r_N_Emn = np.zeros((n_mp, ))
    eps_N_p_Emn = np.zeros((n_mp, ))
    sigma_N_Emn = np.zeros((n_mp, ))
    Y_n = np.zeros((n_mp, ))
    R_n = np.zeros((n_mp, ))
    w_T_Emn = np.zeros((n_mp, ))
    z_T_Emn = np.zeros((n_mp, ))
    alpha_T_Emna = np.zeros((n_mp, 3))
    eps_T_pi_Emna = np.zeros((n_mp, 3))
    sigma_T_Emna = np.zeros((n_mp, 3))
    X_T = np.zeros((n_mp, 3))
    Y_T = np.zeros((n_mp, ))
    sctx = np.zeros((n_mp, 23))
    sctx = sctx[np.newaxis, :, :]
    D = np.zeros((3, 3, 3, 3))
    D = D[np.newaxis, :, :, :, :]

    # total number of DOFs
    n_O = 6
    # Global vectors
    F_ext = np.zeros((n_O,), np.float_)
    F_O = np.zeros((n_O,), np.float_)
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
    U_t_list, F_t_list = [np.copy(U_k_O)], [np.copy(F_O)]

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
            sig_ab, D_abcd = m.get_corr_pred(
                eps_ab, 1, omegaN, z_N_Emn,
                alpha_N_Emn, r_N_Emn, eps_N_p_Emn, sigma_N_Emn,
                w_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna, eps_aux
            )
            D_abcd = D_abcd.reshape(3, 3, 3, 3)

            # Internal force
            F_O = get_sig_O(sig_ab).reshape(6,)

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
                w_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna, eps_aux)

        omegaN = omegaN.reshape(n_mp, )
        z_N_Emn = z_N_Emn.reshape(n_mp, )
        alpha_N_Emn = alpha_N_Emn.reshape(n_mp, )
        r_N_Emn = r_N_Emn.reshape(n_mp, )
        eps_N_p_Emn = eps_N_p_Emn.reshape(n_mp, )
        sigma_N_Emn = sigma_N_Emn.reshape(n_mp,)
        Y_n = Y_n.reshape(n_mp,)
        R_n = R_n.reshape(n_mp,)
        w_T_Emn = w_T_Emn.reshape(n_mp, )
        z_T_Emn = z_T_Emn.reshape(n_mp, )
        alpha_T_Emna = alpha_T_Emna.reshape(n_mp, 3)
        eps_T_pi_Emna = eps_T_pi_Emna.reshape(n_mp, 3)
        sigma_T_Emna = sigma_T_Emna.reshape(n_mp, 3)
        X_T = X_T.reshape(n_mp, 3)
        Y_T = Y_T.reshape(n_mp, )

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
        eps_aux = get_eps_ab(U_k_O)

        t_n = t_n1
        t_n1 += 1
        # print(t_n1)
        # print(Beta)

    U_t, F_t = np.array(U_t_list), np.array(F_t_list)
    return U_t, F_t, sctx, D


n_cycles = 1
T = 1 / n_cycles
eps_max = -0.01
t_steps = 50 * n_cycles


t = np.linspace(0, 1, t_steps)

sin_load = np.linspace(0, eps_max, t_steps)

t_steps_total = len(sin_load)


U, F, S, D = get_UF_t(
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
ax2.set_xlim(-0.00001, 0.01)

ax2.set_ylim(-0.00001, 50)
plt.show()

f, (ax2) = plt.subplots(1, 1, figsize=(5, 4))

ax2.plot(np.abs(U[:, 0]), np.abs(U[:, 1]) /
         np.abs(U[:, 0]), 'k', linewidth=3.5)
ax2.set_xlabel(r'$|\varepsilon_{11}$| [-]', fontsize=25)
ax2.set_ylabel(r'$|\sigma{11}$| [-]', fontsize=25)
plt.show()

# **Examples of postprocessing**:
# Plot the axial strain against the lateral strain

# print(np.array([sv['w_T_Emn'] for sv in S2]))
# z_N_Emn
norm_alphaT = np.zeros((len(S[0]), len(F)))
norm_eps_pT = np.zeros((len(S[0]), len(F)))
print(norm_alphaT.shape)
print(S[:, :, 10:13].shape)
print(S[:, :, 13:16].shape)

for j in range(len(F)):
    for i in range(len(S[0])):

        norm_alphaT[i, j] = np.linalg.norm(S[j, i, 10:13])
        norm_eps_pT[i, j] = np.linalg.norm(S[j, i, 13:16])


t = np.arange(len(F))
fig, axs = plt.subplots(2, 5)
for i in range(len(S[0])):
    axs[0, 0].plot(F[:, 0], S[:, i, 0])
axs[0, 0].set_title('omegaN')

for i in range(len(S[0])):
    axs[0, 1].plot(F[:, 0], S[:, i, 1])
axs[0, 1].set_title('rN')

for i in range(len(S[0])):
    axs[0, 2].plot(F[:, 0], S[:, i, 2])
axs[0, 2].set_title('alphaN')

for i in range(len(S[0])):
    axs[0, 3].plot(F[:, 0], S[:, i, 3])
axs[0, 3].set_title('zN')

for i in range(len(S[0])):
    axs[0, 4].plot(F[:, 0], S[:, i, 4])
axs[0, 4].set_title('eps_pN')

for i in range(len(S[0])):
    axs[1, 0].plot(F[:, 0], S[:, i, 6])
axs[1, 0].set_title('omegaT')

for i in range(len(S[0])):
    axs[1, 1].plot(F[:, 0], S[:, i, 7])
axs[1, 1].set_title('zT')

for i in range(len(S[0])):
    axs[1, 2].plot(F[:, 0], norm_alphaT[i, :])
axs[1, 2].set_title('alphaT')

for i in range(len(S[0])):
    axs[1, 2].plot(F[:, 0], norm_eps_pT[i, :])
axs[1, 3].set_title('eps_pT')

# axs[1, 4].plot(F[:, 0], S[:, 6, 4], 'g')
# #axs[1, 2].plot(F[:, 0], S[:, 26, 2], 'r')
# axs[1, 4].set_title('eps_pN')


plt.show()


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 4))
ax1.plot(U[:, 0], U[:, (1, 2)])
ax2.plot(U[:, 0], F[:, 0])
plt.show()

plot.get_3Dviz(S[:, :, 0], S[:, :, 8])
