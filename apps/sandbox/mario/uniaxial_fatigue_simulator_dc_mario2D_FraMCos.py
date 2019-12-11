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

from apps.sandbox.mario.Micro2Dplot import Micro2Dplot
from apps.sandbox.mario.vmats2D_mpl_csd_eeq import MATSXDMplCDSEEQ
from apps.sandbox.mario.vmats2D_mpl_d_eeq import MATSXDMicroplaneDamageEEQ
from ibvpy.mats.mats3D.mats3D_plastic.vmats3D_desmorat import MATS3DDesmorat
from ibvpy.mats.mats3D.mats3D_sdamage.vmats3D_sdamage import MATS3DScalarDamage

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

#from .vmats3D_mpl_csd_eeq import MATS3DMplCSDEEQ


sp.init_printing()


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


def get_UF_t(eps_max, time_function, n_t):
    m = MATSXDMplCDSEEQ()
    n_mp = 360
    omegaN = np.zeros((n_mp, ))
    z_N_Emn = np.zeros((n_mp, ))
    alpha_N_Emn = np.zeros((n_mp, ))
    r_N_Emn = np.zeros((n_mp, ))
    eps_N_p_Emn = np.zeros((n_mp, ))
    sigma_N_Emn = np.zeros((n_mp, ))
    w_T_Emn = np.zeros((n_mp, ))
    z_T_Emn = np.zeros((n_mp, ))
    alpha_T_Emna = np.zeros((n_mp, 2))
    eps_T_pi_Emna = np.zeros((n_mp, 2))
    sctx = np.zeros((n_mp, 12))
    sctx = sctx[np.newaxis, :, :]

    # total number of DOFs
    n_O = 3
    # Global vectors
    F_ext = np.zeros((n_O,), np.float_)
    F_O = np.zeros((n_O,), np.float_)
    U_k_O = np.zeros((n_O,), dtype=np.float_)
    # Construct index maps distinguishing the controlled displacements
    # and free displacements. Here we simply say the the control displacement
    # is the first one. Then index maps are constructed using the np.where
    # function returning the indexes of True positions in a logical array.
    CONTROL = 0
    FREE = slice(1, None)  # This means all except the first index, i.e. [1:]
    # Setup the system matrix with displacement constraints
    # Time stepping parameters
    t_n1, t_max, t_step = 0, 1, 1 / n_t
    t_n = 0
    # Iteration parameters
    k_max, R_acc = 100, 1e-3
    # Record solutions
    U_t_list, F_t_list = [np.copy(U_k_O)], [np.copy(F_O)]

    # Load increment loop
    while t_n1 <= t_max:
        #print('t:', t_n1)
        # Get the displacement increment for this step
        delta_U = eps_max * (time_function(t_n1) - time_function(t_n))
        k = 0
        # Equilibrium iteration loop
        while k < k_max:
            # Transform the primary vector to field
            eps_ab = get_eps_ab(U_k_O)
            # Stress and material stiffness
            D_abcd, sig_ab = m.get_corr_pred(
                eps_ab, 1, omegaN, z_N_Emn,
                alpha_N_Emn, r_N_Emn, eps_N_p_Emn, sigma_N_Emn,
                w_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna
            )
            # Internal force
            F_O = get_sig_O(sig_ab).reshape(3,)
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

        # Target time of the next load increment
        U_t_list.append(np.copy(U_k_O))
        F_t_list.append(F_O)

        [omegaN, z_N_Emn, alpha_N_Emn, r_N_Emn, eps_N_p_Emn, sigma_N_Emn, w_T_Emn, z_T_Emn,
            alpha_T_Emna, eps_T_pi_Emna] = m._get_state_variables(
                eps_ab, 1, omegaN, z_N_Emn, alpha_N_Emn, r_N_Emn, eps_N_p_Emn, sigma_N_Emn,
                w_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna)

        omegaN = omegaN.reshape(n_mp, )
        z_N_Emn = z_N_Emn.reshape(n_mp, )
        alpha_N_Emn = alpha_N_Emn.reshape(n_mp, )
        r_N_Emn = r_N_Emn.reshape(n_mp, )
        eps_N_p_Emn = eps_N_p_Emn.reshape(n_mp, )
        sigma_N_Emn = sigma_N_Emn.reshape(n_mp,)
        w_T_Emn = w_T_Emn.reshape(n_mp, )
        z_T_Emn = z_T_Emn.reshape(n_mp, )
        alpha_T_Emna = alpha_T_Emna.reshape(n_mp, 2)
        eps_T_pi_Emna = eps_T_pi_Emna.reshape(n_mp, 2)

        sctx_aux = np.concatenate((omegaN.reshape(n_mp, 1), z_N_Emn.reshape(n_mp, 1), alpha_N_Emn.reshape(n_mp, 1),
                                   r_N_Emn.reshape(n_mp, 1), eps_N_p_Emn.reshape(
                                       n_mp, 1), sigma_N_Emn.reshape(n_mp, 1),
                                   w_T_Emn.reshape(n_mp, 1), z_T_Emn.reshape(n_mp, 1), alpha_T_Emna, eps_T_pi_Emna), axis=1)

        sctx_aux = sctx_aux[np.newaxis, :, :]
        sctx = np.concatenate((sctx, sctx_aux))
        t_n = t_n1
        t_n1 += t_step

    U_t, F_t = np.array(U_t_list), np.array(F_t_list)
    return U_t, F_t, sctx


# n_cycles = 8
# T = 1 / n_cycles
#
#
# def loading_history(t): return (np.sin(np.pi / T * (t - T / 2)) + 1) / 2
#
#
# U, F, S = get_UF_t(
#     eps_max=-0.02,
#     time_function=lambda t: loading_history(t),
#     n_t=50
# )
#
#
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
# ax1.plot(U[:, (1, 2)], U[:, 0])
# ax1.set_ylabel(r'$\varepsilon_{\mathrm{axial}}$')
# ax1.set_xlabel(r'$\varepsilon_{\mathrm{lateral}}$')
# ax2.plot(U[:, 0], F[:, 0])
# ax2.set_ylabel(r'$\sigma_{\mathrm{axial}}$')
# ax2.set_xlabel(r'$\varepsilon_{\mathrm{axial}}$')
# plt.show()
