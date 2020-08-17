#!/usr/bin/env python
# coding: utf-8

# # Simulation of fatigue for uniaxial stress state

# Assume a uniaxial stress state with $\sigma_{11} = \bar{\sigma}(\theta)$ representing the loading function. All other components of the stress tensor are assumed zero
# \begin{align}
# \sigma_{ab} = 0; \forall a,b \in (0,1,2), a = b \neq 1
# \end{align}
#

# In[1]:


import os

import matplotlib

from apps.sandbox.mario.Framcos.Micro2Dplot import Micro2Dplot
from apps.sandbox.mario.Framcos.vmats2D_mpl_csd_eeq import MATS2DMplCSDEEQ
#import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


#folder = '0.8'
home_dir = os.path.expanduser('~')
path = os.path.join(
   home_dir, 'Data Processing/0.75_internals.hdf5')
# path2 = os.path.join(
#     home_dir, 'Desktop/Master/HiWi/Master Thesis/Framcos/Calibration/sixth approach/0.75/0.75_macro.hdf5')
# path3 = os.path.join(
# home_dir, 'Desktop/Master/HiWi/Master Thesis/Framcos/Calibration/sixth
# approach/0.75/0.75_D.hdf5')


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


m = MATS2DMplCSDEEQ()
plot = Micro2Dplot()


def get_UF_t(F, n_t, load, factor_H1, factor_H2, factor_L):

    n_mp = 360
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
    alpha_T_Emna = np.zeros((n_mp, 2))
    eps_T_pi_Emna = np.zeros((n_mp, 2))
    sigma_T_Emna = np.zeros((n_mp, 2))
    X_T = np.zeros((n_mp, 2))
    Y_T = np.zeros((n_mp, ))
    sctx = np.zeros((n_mp, 20))
    macro = np.zeros((3, 2))
    D_E=np.zeros((1, ))

    #sctx = sctx[np.newaxis, :, :]
    macro2 = macro[np.newaxis, :, :]

    df = pd.DataFrame(sctx)
    df.to_hdf(path, 'first', mode='w', format='table')

#     df2 = pd.DataFrame(macro)
#     df2.to_hdf(path2, 'first', mode='w', format='table')

#     df3 = pd.DataFrame(D)
#     df3.to_hdf(path3, 'first', mode='w', format='table')

    # total number of DOFs
    n_O = 3
    # Global vectors
    F_ext = np.zeros((n_O,), np.float_)
    F_O = np.zeros((n_O,), np.float_)
    U_k_O = np.zeros((n_O,), dtype=np.float_)
    eps_aux = get_eps_ab(U_k_O)
    # Setup the system matrix with displacement constraints
    # Time stepping parameters
    t_aux, t_n1, t_max, t_step = 0, 0, len(F), 1 / n_t
    # Iteration parameters
    k_max, R_acc = 1000, 1e-3
    # Record solutions
    U_t_list, F_t_list = [np.copy(U_k_O)], [np.copy(F_O)]
    D = np.zeros((2, 2, 2, 2))
    D = D[np.newaxis, :, :, :, :]

    # Load increment loop
    while t_n1 <= t_max - 1:
        #print('t:', t_n1)
        F_ext[0] = F[t_n1]
        F_ext[1] = 0. * F[t_n1]

        k = 0
        # Equilibrium iteration loop
        while k < k_max:
            # Transform the primary vector to field
            eps_ab = get_eps_ab(U_k_O)
            # Stress and material stiffness

            D_abcd, sig_ab = m.get_corr_pred(
                eps_ab, 1, omegaN, z_N_Emn,
                alpha_N_Emn, r_N_Emn, eps_N_p_Emn, sigma_N_Emn,
                w_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna, eps_aux, F_ext
            )

            # Internal force
            F_O = get_sig_O(sig_ab).reshape(3,)
            # Residuum
            R_O = F_ext - F_O
            # System matrix
            K_OP = get_K_OP(D_abcd)
            # Convergence criterion
            R_norm = np.linalg.norm(R_O)
            delta_U_O = np.linalg.solve(K_OP, R_O)
            U_k_O += delta_U_O
            if R_norm < R_acc:
                # Convergence reached
                break
            # Next iteration
            k += 1

        else:
            print('no convergence')

            break

        # Update states variables after convergence
        [omegaN, z_N_Emn, alpha_N_Emn, r_N_Emn, eps_N_p_Emn, sigma_N_Emn, Y_n, R_n, w_T_Emn, z_T_Emn,
            alpha_T_Emna, eps_T_pi_Emna, sigma_T_Emna, Y_T, X_T, D_E_s] = m._get_state_variables(
                eps_ab, 1, omegaN, z_N_Emn, alpha_N_Emn, r_N_Emn, eps_N_p_Emn, sigma_N_Emn,
                w_T_Emn, z_T_Emn, alpha_T_Emna, eps_T_pi_Emna, eps_aux, F_ext)

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
        alpha_T_Emna = alpha_T_Emna.reshape(n_mp, 2)
        eps_T_pi_Emna = eps_T_pi_Emna.reshape(n_mp, 2)
        sigma_T_Emna = sigma_T_Emna.reshape(n_mp, 2)
        X_T = X_T.reshape(n_mp, 2)
        Y_T = Y_T.reshape(n_mp, )
        D_E_s= D_E_s.reshape(1, )
#
        # if F[t_n1] == 0 or F[t_n1] == factor_H1 * load or F[t_n1] ==
        # factor_H2 * load or F[t_n1] == factor_L * load:
        if F[t_n1] == 0 or F[t_n1] == factor_H1 * load or F[t_n1] == factor_L * load:

            sctx_aux = np.concatenate((omegaN.reshape(n_mp, 1), z_N_Emn.reshape(n_mp, 1), alpha_N_Emn.reshape(n_mp, 1), r_N_Emn.reshape(n_mp, 1), eps_N_p_Emn.reshape(n_mp, 1), sigma_N_Emn.reshape(
                n_mp, 1), Y_n.reshape(n_mp, 1), R_n.reshape(n_mp, 1),
                w_T_Emn.reshape(n_mp, 1), z_T_Emn.reshape(
                n_mp, 1), alpha_T_Emna, eps_T_pi_Emna, sigma_T_Emna,
                Y_T.reshape(n_mp, 1), X_T), axis=1)

            #             U_k = U_k_O.reshape((3, 1))
            #             F_k = F_O.reshape((3, 1))

            #             macro = np.concatenate((U_k, F_k), axis=1)
            #         D = D_abcd.reshape((2, 8))
            #         sctx_aux = sctx_aux[np.newaxis, :, :]
            #             macro_aux=np.concatenate((U_k, F_k), axis=1)
            #             macro_aux=macro_aux[np.newaxis, :, :]
            #                     macro2 = np.concatenate((macro2, macro_aux))
            #
            df = pd.DataFrame(sctx_aux)
            df.to_hdf(path, 'middle' + np.str(t_aux), append=True)
            #             df2 = pd.DataFrame(macro)
            #             df2.to_hdf(path2, 'middle' + np.str(t_aux), append=True)
            #df3 = pd.DataFrame(D)
            #df3.to_hdf(path3, 'middle' + np.str(t_aux), append=True)

    #         sctx = np.concatenate((sctx, sctx_aux))
            U_t_list.append(np.copy(U_k_O))
            F_t_list.append(F_O)
            eps_aux = get_eps_ab(U_k_O)
            D_aux = D_abcd[np.newaxis, :, :, :, :]
            D = np.concatenate((D, D_aux))
            #D_E_aux=D_E_s[np.newaxis, :]
            D_E = np.concatenate((D_E, D_E_s))
            t_aux += 1
            # print(t_aux)

        t_n1 += 1

    U_t, F_t = np.array(U_t_list), np.array(F_t_list)
    return U_t, F_t, t_n1 / t_max, t_aux, D,D_E

# load = -60.54301292467442
#
load = -97.38726895936968
# load = -73.43966477329167
# load = -123.82847695050454


# load = 41.81
# FINAL LOADINGS
# load = -91.69221808121128

# load = -119.33166543189739


# load = -60.54301292467442



l_H1 = 0.85
cycles1 = 5808 / 0.65

l_H2 = 0.15
cycles2 = 32

factor_H1 = 0.85
factor_H2 = 0.95
factor_L = 0.2


max_load1 = load * factor_H1
max_load2 = load * factor_H2
max_load3 = load * 0.6
max_load4 = load * 0.65
max_load5 = load * 0.7
max_load6 = load * 0.75
max_load7 = load * 0.8
max_load8 = load * 0.85
max_load9 = load * 0.9
max_load10 = load * 0.95
max_load11 = load * 1.0
min_load = load * factor_L


# n_cycles1 = 88
# n_cycles2 = 99912

n_cycles1 = 100000
n_cycles2 = 4192
n_cycles3 = 10
n_cycles4 = 10
n_cycles5 = 10
n_cycles6 = 10
n_cycles7 = 10
n_cycles8 = 10
n_cycles9 = 10
n_cycles10 = 10
n_cycles11 = 10


t_steps_cycle = 20

monotonic = np.linspace(0, max_load1, t_steps_cycle)

first_load = np.concatenate((np.linspace(0, max_load1, t_steps_cycle), np.linspace(
    max_load1, min_load, t_steps_cycle)[1:]))
cycle1 = np.concatenate((np.linspace(min_load, max_load1, t_steps_cycle)[1:], np.linspace(max_load1, min_load, t_steps_cycle)[
                        1:]))
cycle1 = np.tile(cycle1, n_cycles1 - 1)

change_order = np.concatenate((np.linspace(min_load, max_load2, 632)[1:], np.linspace(max_load2, min_load, 632)[
    1:]))

cycle2 = np.concatenate((np.linspace(min_load, max_load2, t_steps_cycle)[1:], np.linspace(max_load2, min_load, t_steps_cycle)[
                        1:]))
cycle2 = np.tile(cycle2, n_cycles2)

cycle3 = np.concatenate((np.linspace(min_load, max_load3, t_steps_cycle)[1:], np.linspace(max_load3, min_load, t_steps_cycle)[
                        1:]))
cycle3 = np.tile(cycle3, n_cycles3)

cycle4 = np.concatenate((np.linspace(min_load, max_load4, t_steps_cycle)[1:], np.linspace(max_load4, min_load, t_steps_cycle)[
                        1:]))
cycle4 = np.tile(cycle4, n_cycles4)

cycle5 = np.concatenate((np.linspace(min_load, max_load5, t_steps_cycle)[1:], np.linspace(max_load5, min_load, t_steps_cycle)[
                        1:]))
cycle5 = np.tile(cycle5, n_cycles5)

cycle6 = np.concatenate((np.linspace(min_load, max_load6, t_steps_cycle)[1:], np.linspace(max_load6, min_load, t_steps_cycle)[
                        1:]))
cycle6 = np.tile(cycle6, n_cycles6)

cycle7 = np.concatenate((np.linspace(min_load, max_load7, t_steps_cycle)[1:], np.linspace(max_load7, min_load, t_steps_cycle)[
                        1:]))
cycle7 = np.tile(cycle7, n_cycles7)

cycle8 = np.concatenate((np.linspace(min_load, max_load8, t_steps_cycle)[1:], np.linspace(max_load8, min_load, t_steps_cycle)[
                        1:]))
cycle8 = np.tile(cycle8, n_cycles8)

cycle9 = np.concatenate((np.linspace(min_load, max_load9, t_steps_cycle)[1:], np.linspace(max_load9, min_load, t_steps_cycle)[
                        1:]))
cycle9 = np.tile(cycle9, n_cycles9)

cycle10 = np.concatenate((np.linspace(min_load, max_load10, t_steps_cycle)[1:], np.linspace(max_load10, min_load, t_steps_cycle)[
    1:]))
cycle10 = np.tile(cycle10, n_cycles10)

cycle11 = np.concatenate((np.linspace(min_load, max_load11, t_steps_cycle)[1:], np.linspace(max_load11, min_load, t_steps_cycle)[
    1:]))
cycle11 = np.tile(cycle11, n_cycles11)

# sin_load = np.concatenate((first_load, cycle2, cycle3,
#                            cycle4, cycle5, cycle6, cycle7, cycle8, cycle9, cycle10, cycle11))
#
#sin_load = np.concatenate((first_load, cycle1, change_order, cycle2))


sin_load = np.concatenate((first_load, cycle1))

# sin_load = monotonic


t_steps = len(sin_load)
T = 1 / n_cycles1
t = np.linspace(0, 1, len(sin_load))
# plt.plot(t, (np.sin(np.pi / T * (t - T / 2)) + 1) / 2)
# plt.show()

# T1 = (1 - n_cycles1 / (n_cycles1 + n_cycles2)) / n_cycles1
# T2 = (1 - n_cycles2 / (n_cycles1 + n_cycles2)) / n_cycles2
# A1 = (max_load1 - min_load) / 2
# A2 = (max_load2 - min_load) / 2
# t_steps_cycle = 10
#
# t_steps1 = t_steps_cycle * n_cycles1
# t_steps2 = t_steps_cycle * n_cycles2
#
# sin_load = np.linspace(0, A1 + min_load, t_steps_cycle)
#
# t1 = np.linspace(0, t_steps1 / (t_steps1 + t_steps2), t_steps1)
# t2 = np.linspace(t_steps2 / (t_steps1 + t_steps2), 1, t_steps2)
#
# sin_load = np.concatenate(
#     (sin_load, A1 * (np.sin(np.pi / T1 * (t1[:-1]))) + A1 + min_load, A2 * (np.sin(np.pi / T2 * (t2[1:]))) + A2 + min_load))
# t_steps = t_steps_cycle * (n_cycles1 + n_cycles2 + 1)
# t = np.linspace(0, 1, len(sin_load))

# print(sin_load)


# def loading_history(t): return (np.sin(np.pi / T * (t)))
#
#
# def first_loading(t): return ((A1 + min_load) / t_steps_cycle * t)
#
#
# def F(t):
#     if t < t_steps_cycle / len(sin_load):
#         return first_loading(t)
#     if t <= 0.5:
#         return A1 * loading_history(t) + A1 + min_load
#     if t > 0.5:
#         return A2 * loading_history(t)


# font = {'family': 'normal',
#         'size': 18}
#
# matplotlib.rc('font', **font)
#
# load = np.zeros_like(t)
# for i in range(len(t)):
#     load[i] = F(t[i])
#
# f, (ax) = plt.subplots(1, 1, figsize=(5, 4))
# ax.plot(t, load, linewidth=2.5)
# ax.set_xlabel('pseudotime [-]', fontsize=25)
# ax.set_ylabel('Loading [Mpa]', fontsize=25)
# plt.show()
#
#

U, F, cyc, number_cyc, D,D_E = get_UF_t(
    sin_load, t_steps, load, factor_H1, factor_H2, factor_L)


# macro = np.zeros((np.int(number_cyc - 1), 3, 2))
# #D = np.zeros((np.int(number_cyc - 1), 2, 8))
#
# macro[0] = np.array(pd.read_hdf(path2, 'first'))
# #D[0] = np.array(pd.read_hdf(path3, 'first'))
#
# for i in range(1, np.int(number_cyc - 1)):
#     macro[i] = np.array(pd.read_hdf(path2, 'middle' + np.str(i - 1)))
#     #D[i] = np.array(pd.read_hdf(path3, 'middle' + np.str(i - 1)))
#
# U = macro[:, :, 0]
# F = macro[:, :, 1]
#D = D.reshape((np.int(number_cyc - 1), 2, 2, 2, 2))

font = {'family': 'normal',
        'size': 18}

matplotlib.rc('font', **font)

f, (ax1) = plt.subplots(1, 1, figsize=(5, 4))

ax1.plot(t[0:], np.abs(sin_load / load)[0:], linewidth=2.5)
ax1.set_xlabel('pseudotime [-]', fontsize=25)
ax1.set_ylabel(r'$|S_{max}$| [-]', fontsize=25)
ax1.set_title('L-H')

print(np.max(np.abs(F[:, 0])), 'sigma1')
print(np.max(np.abs(F[:, 1])), 'sigma2')

f, (ax2) = plt.subplots(1, 1, figsize=(5, 4))

ax2.plot(np.abs(U[:, 0]), np.abs(F[:, 0]), linewidth=2.5)
ax2.set_xlabel(r'$|\varepsilon_{11}$|', fontsize=25)
ax2.set_ylabel(r'$|\sigma_{11}$| [-]', fontsize=25)
ax2.set_title(str((n_cycles1)) + ',' + str(cyc))
# ax2.set_ylim(0.00, 60)
# ax2.set_xlim(0.000, 0.00333)
plt.show()

f, (ax) = plt.subplots(1, 1, figsize=(5, 4))
ax.plot((np.arange(len(U[2::2, 0])) + 1) / len(U[1::2, 0]),
        np.abs(U[2::2, 0]), linewidth=2.5)
plt.show()


# f, (ax2) = plt.subplots(1, 1, figsize=(5, 4))
#
# ax2.plot(np.abs(U[0:t_steps_cycle + 2 * (t_steps_cycle - 1), 0]),
#          np.abs(F[0:t_steps_cycle + 2 * (t_steps_cycle - 1), 0]), linewidth=2.5)
# ax2.plot(np.abs(U[t_steps_cycle + 2 * (19 - 1) * (t_steps_cycle - 1):, 0]),
#          np.abs(F[t_steps_cycle + 2 * (19 - 1) * (t_steps_cycle - 1):, 0]), linewidth=2.5)
# ax2.set_xlabel(r'$|\varepsilon_{11}$|', fontsize=25)
# ax2.set_ylabel(r'$|\sigma_{11}$| [-]', fontsize=25)
# ax2.set_title(str((n_cycles1)) + ',' + str(cyc))
# plt.show()


# f, (ax2) = plt.subplots(1, 1, figsize=(5, 4))
#
# ax2.plot(np.abs(U[t_steps_cycle + 2 * (5 - 1) * (t_steps_cycle - 1):t_steps_cycle + 2 * 5 * (t_steps_cycle - 1), 0]),
#          np.abs(F[t_steps_cycle + 2 * (5 - 1) * (t_steps_cycle - 1):t_steps_cycle + 2 * 5 * (t_steps_cycle - 1), 0]), linewidth=2.5)
# ax2.set_xlabel(r'$|\varepsilon_{11}$|', fontsize=25)
# ax2.set_ylabel(r'$|\sigma_{11}$| [-]', fontsize=25)
# ax2.set_title(str((n_cycles1)) + ',' + str(cyc))
# plt.show()
#
#
# f, (ax2) = plt.subplots(1, 1, figsize=(5, 4))
#
# ax2.plot(np.abs(U[t_steps_cycle + 2 * (8 - 1) * (t_steps_cycle - 1):, 0]),
#          np.abs(F[t_steps_cycle + 2 * (8 - 1) * (t_steps_cycle - 1):, 0]), linewidth=2.5)
# ax2.set_xlabel(r'$|\varepsilon_{11}$|', fontsize=25)
# ax2.set_ylabel(r'$|\sigma_{11}$| [-]', fontsize=25)
# ax2.set_title(str((n_cycles1)) + ',' + str(cyc))
# plt.show()


f, (ax) = plt.subplots(1, 1, figsize=(5, 4))

# ax.plot(np.arange(len(U[(t_steps_cycle)::2 * (t_steps_cycle - 1), 0])) + 1,
# np.abs(U[(t_steps_cycle)::2 * (t_steps_cycle - 1), 0]), linewidth=2.5)

ax.plot(np.arange(len(U[0:-2:2, 0])) + 1,
        np.abs(U[2::2, 0]), linewidth=2.5)
ax.set_ylim(0.002, 0.0045)
ax.set_xlim(0, 1400)
#ax.set_xlim(0, ((n_cycles1)) + 1)
ax.set_xlabel('number of cycles [N]', fontsize=25)
ax.set_ylabel(r'$|\varepsilon_{11}^{max}$|', fontsize=25)
plt.title('creep fatigue Smax = 0.85')
plt.show()

# print(U[:, 0])
# print(U[(t_steps_cycle)::2 * (t_steps_cycle - 1), 0])


f, (ax) = plt.subplots(1, 1, figsize=(5, 4))


# ax.plot((np.arange(len(U[(t_steps_cycle)::2 * (t_steps_cycle - 1), 0])) + 1),
#         np.abs(U[(t_steps_cycle)::2 * (t_steps_cycle - 1), 0]), linewidth=2.5)
# ax.plot((np.arange(len(U[0::2 * (t_steps_cycle - 1), 0])) + 1),
#         np.abs(U[1::2 * (t_steps_cycle - 1), 0]), linewidth=2.5)

# ax.plot((np.arange(len(U[(t_steps_cycle)::2 * (t_steps_cycle - 1), 0])) + 1) / len(U[(t_steps_cycle)::2 * (t_steps_cycle - 1), 0]),
#         np.abs(U[(t_steps_cycle)::2 * (t_steps_cycle - 1), 0]), linewidth=2.5)
# ax.plot((np.arange(len(U[0::2 * (t_steps_cycle - 1), 0])) + 1) / len(U[0::2 * (t_steps_cycle - 1), 0]),
#         np.abs(U[1::2 * (t_steps_cycle - 1), 0]), linewidth=2.5)

ax.plot((np.arange(len(U[2::2, 0])) + 1) / len(U[1::2, 0]),
        np.abs(U[2::2, 0]), linewidth=2.5)
ax.plot((np.arange(len(U[1::2, 0])) + 1) / len(U[0::2, 0]),
        np.abs(U[1::2, 0]), linewidth=2.5)


# X_axis1 = np.array(np.arange(l_H1 * cycles1) + 1)[1:] / cycles1
# X_axis1 = np.concatenate((np.array([0]), X_axis1))
# Y_axis1 = np.abs(U[2:np.int(2 * l_H1 * cycles1) + 2:2, 0])
# # Y_axis1 = np.concatenate((np.array([Y_axis1[0]]), Y_axis1))
#
#
# print(U[2:np.int(2 * l_H1 * cycles1) + 2, 0])
# print(X_axis1.shape)
# print(Y_axis1.shape)
#
# print(len(U[2::2, 0]))
# X_axis2 = np.array((np.arange(len(U[2::2, 0]) -
#                               (l_H1 * cycles1)) + 1) / (cycles2) + l_H1)
# X_axis2 = np.concatenate((np.array([X_axis1[-1]]), X_axis2))
#
# Y_axis2 = np.abs(U[np.int(2 * l_H1 * cycles1) + 2::2, 0])
# Y_axis2 = np.concatenate(
#     (np.array([Y_axis2[0]]), Y_axis2))
# X_axis = np.concatenate((X_axis1, X_axis2))
#
# print(U[np.int(2 * l_H1 * cycles1):np.int(2 * l_H1 * cycles1) + 10, 0])
# print(X_axis2.shape)
# print(Y_axis2.shape)
#
#
# x = U[2, 0]
# Y_axis = np.concatenate((np.array([x]), np.array(U[2::2, 0])))
# ax.plot(X_axis1, Y_axis1, 'k', linewidth=2.5)
# ax.plot(X_axis2, Y_axis2, 'k', linewidth=2.5)
# # ax.plot(X_axis, Y_axis, 'r', linewidth=2.5)
# ax.plot([X_axis1[-1], X_axis2[0]],
#         [Y_axis1[-1], Y_axis2[0]], 'k', linewidth=2.5)


# ax.plot(np.arange(cycles1) / l_H1 * cycles1,
#         np.abs(U[2::2, 0]), linewidth=2.5)


ax.set_ylim(0.002, 0.0045)
ax.set_xlim(-0.1, 1.1)
#ax.set_xlim(0, ((n_cycles1 + n_cycles2)) + 1)
ax.set_xlabel('N/Nf', fontsize=25)
ax.set_ylabel('strain', fontsize=25)
plt.title('creep fatigue Smax = 0.85')
plt.show()

plt.plot(np.arange(len(U[2::2, 0])), U[2::2, 0])
plt.show()


n_mp = 360

S = np.zeros((len(F), n_mp, 19))

S[0] = np.array(pd.read_hdf(path, 'first'))

for i in range(1, len(F)):
    S[i] = np.array(pd.read_hdf(path, 'middle' + np.str(i - 1)))


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
n_mp = 360

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
                                                                  sigma_Emna,  m._get__MPN())), np.sqrt(np.einsum('...i,...i->... ',
                                                                                                                  sigma_Emna, sigma_Emna)))

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


plot.get_2Dviz(n_mp, eps_global_norm, sigma_global_norm, eps_micro_norm, sigma_micro_norm, D_1_norm, D_2_norm, D_12_norm,
               D_21_norm, eps_N, eps_p_N, sigma_N, omegaN, eps_T_sign, eps_pi_T_sign,
               sigma_T_sign, omegaT, Y_N, X_N, Y_T, X_T_sign, F, U, t_steps_cycle)
#
#
# A = np.array(range(len(F)))
# A = A[1::]
#
#
# plt.subplot(131)
#
# plt.plot(np.arange(
#     len(A)) / 2, D_1_norm[A, 0] / D_1_norm[A[0], 0], linewidth=2.5)
#
# plt.title('eps_T_sign 20L-80H ')
#
# #===================
# # Normal damage
# #===================
# plt.subplot(132)
#
# plt.plot(np.arange(
#     len(A)) / 2, D_2_norm[A, np.int(np.floor(n_mp / 4))] / D_2_norm[A[0], np.int(np.floor(n_mp / 4))], linewidth=2.5)
#
# plt.title('eps_pi_T_sign')
#
# #===================
# # Normal plastic strain
# #===================
# plt.subplot(133)
#
# plt.plot(np.arange(
#     len(A)) / 2, D_12_norm[A, np.int(np.floor(n_mp / 8))] / D_12_norm[A[0], np.int(np.floor(n_mp / 8))], linewidth=2.5)
#
# plt.title('sigma_T_sign')
#
# plt.show()

#
#
# f, (ax) = plt.subplots(1, 1, figsize=(5, 4))
# ax.plot(U[:, 0], F[:, 0], linewidth=2.5)
# ax.set_xlabel('strain loading direction [-]', fontsize=25)
# ax.set_ylabel('Loading [Mpa]', fontsize=25)
# plt.show()
#
#
# # f, (ax) = plt.subplots(1, 1, figsize=(5, 4))
# # ax.plot(np.arange(n_cycles / 2) + 1,
# #         np.abs(U[(t_steps_cycle + 1)::2 * t_steps_cycle, 0]), linewidth=2.5)
# # ax.set_xlim(0, (n_cycles / 2) + 1)
# # ax.set_xlabel('number of cycles', fontsize=25)
# # ax.set_ylabel('max eps', fontsize=25)
# # plt.show()
#
# n_mp = 100
# rads = np.arange(0, (2 * np.pi), (2 * np.pi) / n_mp)
# eps_N = np.zeros((len(S), len(S[1])))
# for i in range(len(F)):
#     eps = get_eps_ab(U[i])
#     eps_N[i] = m._get_e_N_Emn_2(eps)
#
# omegaN = S[:, :, 0]
# eps_p_N = S[:, :, 4]
#
#
# plt.subplot(131, projection='polar')
# for i in range(len(F)):
#     #print('idx', idx.shape)
#     plt.plot(rads, eps_N[i, :])
# plt.ylim(-1.2 * np.max(np.abs(eps_N)), 0.8 * np.max(np.abs(eps_N)))
# plt.title('eps_N')
#
# plt.subplot(132, projection='polar')
# for i in range(len(F)):
#     plt.plot(rads, omegaN[i, :])
# plt.title('omegaN')
#
# plt.subplot(133, projection='polar')
# for i in range(len(F)):
#     plt.plot(rads, eps_p_N[i, :])
# plt.ylim(-1.2 * np.max(np.abs(eps_p_N)), 0.8 * np.max(np.abs(eps_p_N)))
# plt.title('eps_p_N')
# plt.show()
#
# eps_T = np.zeros((len(S), len(S[1])))
# eps_T_sign = np.zeros((len(S), len(S[1])))
# eps_pi_T = np.zeros((len(S), len(S[1])))
# eps_pi_T_sign = np.zeros((len(S), len(S[1])))
# sigma_T = np.zeros((len(S), len(S[1])))
# sigma_T_sign = np.zeros((len(S), len(S[1])))
# X_T = np.zeros((len(S), len(S[1])))
# X_T_sign = np.zeros((len(S), len(S[1])))
#
#
# for i in range(len(F)):
#     eps = get_eps_ab(U[i])
#     eps_T_aux = m._get_e_T_Emnar_2(eps)
#     eps_T[i] = np.sqrt(np.einsum('...i,...i->... ', eps_T_aux, eps_T_aux))
#     eps_pi_T[i] = np.sqrt(
#         np.einsum('...i,...i->... ', S[i, :, 12:14], S[i, :, 12:14]))
#     sigma_T[i] = np.sqrt(
#         np.einsum('...i,...i->... ', S[i, :, 14:16], S[i, :, 14:16]))
#     X_T[i] = np.sqrt(
#         np.einsum('...i,...i->... ', S[i, :, 17:19], S[i, :, 17:19]))
#
#     sign_T = np.sign(eps_T_aux)
#
#     sign_pi_T = np.sign(S[i, :, 12:14])
#     sign_sigma_T = np.sign(S[i, :, 14:16])
#     sign_X_T = np.sign(S[i, :, 17:19])
#
#     eps_T_sign[i] = np.einsum(
#         '...n,...n->...n', np.sign(np.einsum('...ni,...ni->...n', sign_T, sign_T)), eps_T[i])
#     eps_pi_T_sign[i] = np.einsum(
#         '...n,...n->...n', np.sign(np.einsum('...ni,...ni->...n', sign_pi_T, sign_pi_T)), eps_pi_T[i])
#
#     sigma_T_sign[i] = np.einsum(
#         '...n,...n->...n', np.sign(np.einsum('...ni,...ni->...n', sign_sigma_T, sign_sigma_T)), sigma_T[i])
#     X_T_sign[i] = np.einsum(
#         '...n,...n->...n', np.sign(np.einsum('...ni,...ni->...n', sign_X_T, sign_X_T)), X_T[i])
#
#
# omegaT = S[:, :, 8]
#
# plt.subplot(131, projection='polar')
# for i in range(len(F)):
#     plt.plot(rads, eps_T[i, :])
# plt.title('eps_T')
#
# #===================
# # Tangential damage
# #===================
# plt.subplot(132, projection='polar')
# for i in range(len(F)):
#     plt.plot(rads, omegaT[i, :])
# plt.title('omegaT')
#
# #===================
# # Tangential plastic strain
# #===================
# plt.subplot(133, projection='polar')
# for i in range(len(F)):
#     plt.plot(rads, eps_pi_T[i, :])
# plt.title('eps_pi_T')
#
# plt.show()
#
# sigma_n = S[:, :, 5]
# Y_N = S[:, :, 6]
# X_N = S[:, :, 7]
# Y_T = S[:, :, 16]
#
# plot.get_2Dviz(n_mp, eps_N, eps_p_N, sigma_n, omegaN, Y_N, X_N,
#                eps_T_sign, eps_pi_T_sign, sigma_T_sign, omegaT, Y_T, X_T_sign)
#
# # n_cycles = 1
# # T = 1 / n_cycles
# #
# #
# # def loading_history(t): return (np.sin(np.pi / T * (t - T / 2)) + 1) / 2
# #
# #
# # U, F, S = get_UF_t(
# #     F=lambda t: -100 * loading_history(t),
# #     n_t=50 * n_cycles
# # )
#
#
# # **Examples of postprocessing**:
# # Plot the axial strain against the lateral strain
#
# # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 4))
# # ax1.plot(U[:, 0], U[:, 1])
# # ax2.plot(U[:, 0], F[:, 0])
# #
# # plt.show()
# #
# # eps_N = np.zeros((len(S), len(S[1])))
# # eps_T = np.zeros((len(S), len(S[1])))
# # norm_alphaT = np.zeros((len(S), len(S[1])))
# # norm_eps_pT = np.zeros((len(S), len(S[1])))
# #
# # for i in range(len(F)):
# #     eps = get_eps_ab(U[i])
# #     eps_N[i] = m._get_e_N_Emn_2(eps)
# #     eps_T_aux = m._get_e_T_Emna(eps)
# #     eps_T[i] = np.sqrt(np.einsum('...i,...i->... ', eps_T_aux, eps_T_aux))
# #     norm_alphaT[i] = np.sqrt(
# #         np.einsum('...i,...i->... ', S[i, :, 8:10], S[i, :, 8:10]))
# #     norm_eps_pT[i] = np.sqrt(
# #         np.einsum('...i,...i->... ', S[i, :, 10:12], S[i, :, 10:12]))
# #
# # n_mp = 100
# # plot.get_2Dviz(n_mp, eps_N, S[:, :, 0], S[:, :, 4],
# #                eps_T, S[:, :, 6], norm_eps_pT)
