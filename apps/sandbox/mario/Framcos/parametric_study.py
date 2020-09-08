'''
Created on 04.07.2019
@author: Abdul
plotting tool for microplane models
'''

import matplotlib


import matplotlib.pyplot as plt
import numpy as np



# font = {'family': 'normal',
#         'size': 18}
#
# matplotlib.rc('font', **font)


E = 34.e3

nu = 0.2

gamma_T = 10000.

K_T = 000.0

S_T = 0.05

# r_T_serie = np.array([1, 2, 4, 6, 8, 10, 12, 14])
r_T = 1.
# c_T_serie = np.array([1, 2, 4, 6, 8, 10, 12, 14])
c_T = 1.


e_T_serie = np.array([1,2,3,4,5])
#e_T = 1.


a = 0.0081

tau_pi_bar = 0.5


# E = 90.e3
#
# nu = 0.2
#
# gamma_T = 100000.
#
# K_T = 000.0
#
# # S_T_serie = np.array([0.00000001, 0.00000005, 0.0000001, 0.0000005, 0.000001])
# S_T = .0000001
# # r_T_serie = np.array([1.2, 1.15, 1.1, 1.05, 1.])
# r_T = 1.
# # c_T_serie = np.array([1, 1.5, 2, 2.5, 3])
# c_T = 1.
#
#
# e_T_serie = np.array([1,2,3,4,5])
# # e_T = 1.
#
#
# a = 0.0
#
# tau_pi_bar = 1.000


# S_max_serie = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65]
S_max_serie = [1.]
cycles_S = np.zeros_like(e_T_serie)


for k in range(len(S_max_serie)):

    S_max = S_max_serie[k]
    max_sigma_N_Emn = -61.5 * S_max
    t_steps_cycle = 500
    min_load = -1.
    n_cycles1 = 5

    max_eps_T = np.linspace(0.0004 * S_max, 1 * 0.0004 * S_max, n_cycles1)

    eps_T = np.linspace(0, 0.001 * S_max, t_steps_cycle)
    sigma_N_Emn = np.linspace(0, max_sigma_N_Emn, t_steps_cycle)

#     first_load = np.concatenate(
#         (np.linspace(0, 1, t_steps_cycle), np.linspace(1, min_load, t_steps_cycle)[1:]))
#     cycle1 = np.concatenate((np.linspace(min_load, 1., t_steps_cycle)[1:], np.linspace(1., min_load, t_steps_cycle)[
#         1:]))
#     cycle1 = np.tile(cycle1, n_cycles1 - 1)
#     cycle1 = np.concatenate((first_load, cycle1[:-1:]))
#
#     eps_T_aux = np.repeat(max_eps_T, (t_steps_cycle - 1) * 2)
#
#     eps_T = np.einsum('...n,...n->...n', eps_T_aux, cycle1)
#
#     first_load_sigma = np.concatenate((np.linspace(0, max_sigma_N_Emn, t_steps_cycle), np.linspace(
#         max_sigma_N_Emn, min_load, t_steps_cycle)[1:]))
#     cycle1_sigma = np.concatenate((np.linspace(min_load, max_sigma_N_Emn, t_steps_cycle)[1:], np.linspace(max_sigma_N_Emn, min_load, t_steps_cycle)[
#         1:]))
#     cycle1_sigma = np.tile(cycle1_sigma, n_cycles1 - 1)
#     sigma_N_Emn = np.concatenate((first_load_sigma, cycle1_sigma))

    for j in range(len(e_T_serie)):
        e_T = e_T_serie[j]

        eps_T_Emna = np.zeros((len(eps_T), 1, 2))
        eps_T_Emna[:, 0, 0] = eps_T
        D_int = np.zeros((len(eps_T),))
        psi_t = np.zeros((len(eps_T),))
        w_T_Emn = np.zeros((len(eps_T),))
        Y_aux = np.zeros((len(eps_T),))
        eps_pi_plot = np.zeros((len(eps_T),))
        sigma_pi_plot = np.zeros((len(eps_T),))
        w_T_Emn = np.zeros((len(eps_T),))
        z_T_Emn = np.zeros_like(w_T_Emn)
        alpha_T_Emna = np.zeros_like(eps_T_Emna)
        eps_T_pi_Emna = np.zeros_like(eps_T_Emna)
        sigma_T_Emna = np.zeros_like(eps_T_Emna)

        Z = np.zeros_like(w_T_Emn)
        X = np.zeros_like(eps_T_Emna)

#         w_T_Emn = 0.
#         z_T_Emn = 0.
#         alpha_T_Emna = np.zeros((1, 2))
#         eps_T_pi_Emna = np.zeros((1, 2))
#         sigma_T_Emna = np.zeros((1, 2))
#
#         Z = np.zeros_like(w_T_Emn)
#         X = np.zeros_like(eps_T_Emna)

        E_T = E * (1.0 - 4 * nu) / \
            ((1.0 + nu) * (1.0 - 2 * nu))

        cycles = 10

#         for i in range(1, len(eps_T)):
        for i in range(len(eps_T)):

            sig_pi_trial = E_T * (eps_T_Emna[i] - eps_T_pi_Emna[i - 1])
            Z[i] = K_T * z_T_Emn[i - 1]
            X[i] = gamma_T * alpha_T_Emna[i - 1]
            norm_1 = np.sqrt(
                np.einsum(
                    '...na,...na->...n',
                    (sig_pi_trial - X[i]), (sig_pi_trial - X[i]))
            )
            Y = 0.5 * E_T * \
                np.einsum(
                    '...na,...na->...n',
                    (eps_T_Emna[i] - eps_T_pi_Emna[i - 1]),
                    (eps_T_Emna[i] - eps_T_pi_Emna[i - 1]))

            f = norm_1 - tau_pi_bar - \
                Z[i] + a * sigma_N_Emn[i]

            plas_1 = f > 1e-6
            elas_1 = f < 1e-6

            delta_lamda = f / \
                (E_T / (1.0 - w_T_Emn[i - 1]) + gamma_T + K_T) * plas_1

            norm_2 = 1.0 * elas_1 + np.sqrt(
                np.einsum(
                    '...na,...na->...n',
                    (sig_pi_trial - X[i]), (sig_pi_trial - X[i]))) * plas_1

            eps_T_pi_Emna[i][..., 0] = eps_T_pi_Emna[i - 1][..., 0] + plas_1 * delta_lamda * \
                ((sig_pi_trial[..., 0] - X[i][..., 0]) /
                 (1.0 - w_T_Emn[i - 1])) / norm_2
            eps_T_pi_Emna[i][..., 1] = eps_T_pi_Emna[i - 1][..., 1] + plas_1 * delta_lamda * \
                ((sig_pi_trial[..., 1] - X[i][..., 1]) /
                 (1.0 - w_T_Emn[i - 1])) / norm_2

            w_T_Emn[i] = w_T_Emn[i - 1] + ((1 - w_T_Emn[i - 1]) ** c_T) * \
                (delta_lamda * (Y / S_T) ** r_T) * \
                 (tau_pi_bar / (tau_pi_bar + a * sigma_N_Emn[i])) ** e_T

            alpha_T_Emna[i][..., 0] = alpha_T_Emna[i - 1][..., 0] + plas_1 * delta_lamda * \
                (sig_pi_trial[..., 0] - X[i][..., 0]) / norm_2
            alpha_T_Emna[i][..., 1] = alpha_T_Emna[i - 1][..., 1] + plas_1 * delta_lamda * \
                (sig_pi_trial[..., 1] - X[i][..., 1]) / norm_2

            z_T_Emn[i] = z_T_Emn[i - 1] + delta_lamda

            sigma_T_Emna[i] = (1 - w_T_Emn[i]) * E_T * \
                (eps_T_Emna[i] - eps_T_pi_Emna[i])

            eps_pi_plot[i] = np.linalg.norm(
                eps_T_pi_Emna[i]) * np.sign(eps_T_pi_Emna[i])[0, 0]
            sigma_pi_plot[i] = np.linalg.norm(
                sigma_T_Emna[i]) * np.sign(sigma_T_Emna[i])[0, 0]

            Y = 0.5 * E_T * \
                np.einsum(
                    '...na,...na->...n',
                    (eps_T_Emna[i] - eps_T_pi_Emna[i]),
                    (eps_T_Emna[i] - eps_T_pi_Emna[i]))

            D_int[i] = D_int[i - 1] + Y * ((1 - w_T_Emn[i - 1]) ** c_T) * (delta_lamda * (Y / S_T) ** r_T) + np.einsum('i...,i...->...', sigma_T_Emna[i][0],
                                                                                                                       eps_T_pi_Emna[i][0]) - gamma_T * np.einsum('i...,i...->...', alpha_T_Emna[i][0], (alpha_T_Emna[i][0] - alpha_T_Emna[i - 1][0])) - K_T * z_T_Emn[i] * delta_lamda
            psi_t[i] = 0.5 * (1 - w_T_Emn[i]) * E_T * np.einsum('i...,i...->...', (eps_T_Emna[i][0] - eps_T_pi_Emna[i][0]), (eps_T_Emna[i][0] -
                                                                                                                             eps_T_pi_Emna[i][0])) + 0.5 * K_T * z_T_Emn[i]**2 + 0.5 * gamma_T * np.einsum('i...,i...->...', (alpha_T_Emna[i][0]), (alpha_T_Emna[i][0]))

            if sigma_N_Emn[i] == max_sigma_N_Emn:
                cycles += 1

            if w_T_Emn[i] > 0.99:
                print('broken')
                break


        plt.subplot(111)
        plt.plot(np.abs(sigma_N_Emn), w_T_Emn, linewidth=2.5 +1)
        plt.yscale('log')
        plt.xlim(0, 60)
        plt.ylim(1e-10, 1e0)
        plt.xlabel('$\sigma_N$', fontsize=25)
        plt.ylabel('$\omega$', fontsize=25)
        plt.title('microplane damage')

#         plt.subplot(111)
#         plt.xlim(0, 6)
#         plt.ylim(0.63, 0.97)
#         plt.scatter(np.log10(cycles), S_max, linewidth=3.5)
#
#         print(cycles)
#         plt.subplot(233)
#         plt.plot(
#                  sigma_pi_plot,w_T_Emn, linewidth=2.5)
#
#         plt.xlabel('$\sigma_N$', fontsize=25)
#         plt.ylabel('$\dot{\omega}$', fontsize=25)
#         plt.title('microplane damage')
#
#         plt.subplot(234)
#         # plt.plot(eps_T, Y_aux, linewidth=3.5)
#         plt.plot(eps_T,
#                  eps_pi_plot, linewidth=2.5)
#
#         plt.xlabel('$\sigma_N$', fontsize=25)
#         plt.ylabel('$\dot{\omega}$', fontsize=25)
#         plt.title('microplane damage')
#
#         plt.subplot(232)
#
#         plt.plot(eps_T, w_T_Emn, linewidth=3.5)
#
#         plt.xlabel('$\sigma_1$', fontsize=25)
#         plt.ylabel('$\dot{\omega}$', fontsize=25)
#
#         plt.subplot(231)
#
#         plt.plot(np.arange(len(eps_T)), eps_T, linewidth=3.5)
#
#         plt.xlabel('$\sigma_1$', fontsize=25)
#         plt.ylabel('$\dot{\omega}$', fontsize=25)
#
#         plt.subplot(235)
#
#         plt.plot(eps_T, D_int, linewidth=3.5)
#
#         plt.xlabel('$\sigma_1$', fontsize=25)
#         plt.ylabel('$\dot{\omega}$', fontsize=25)
#
#         plt.subplot(236)
#
#         plt.plot(eps_T, psi_t, linewidth=3.5)
#
#         plt.xlabel('$\sigma_1$', fontsize=25)
#         plt.ylabel('$\dot{\omega}$', fontsize=25)
    cycles_S[k] = cycles



# plt.subplot(111)
# plt.xlim(0, 6)
# plt.ylim(0.63, 0.97)
# plt.plot(np.log10(cycles_S), S_max_serie, linewidth=3.5)


# s_min = 0.0
# S = np.linspace(1., 0.)
#
# Y = (0.45 + 1.8 * s_min) / (1 + 1.8 * s_min - 0.3 * s_min**2)
# cyc = (8 / (Y - 1)) * (S - 1)
# plt.subplot(131)
#
# plt.plot(cyc, S, linewidth=3.5)
# plt.title('wohler curve')
# plt.xlim(0, 6)
# plt.ylim(0.63, 0.97)
# plt.xlabel('Log(N)', fontsize=25)
# plt.ylabel('Smax', fontsize=25)
#
#
# plt.subplot(132)
#
# plt.plot(eps_T, w_T_Emn, linewidth=3.5)
# plt.yscale('log')
#
# plt.ylim(1e-6, 1e0)
# plt.xlabel('$\sigma_1$', fontsize=25)
# plt.ylabel('$\dot{\omega}$', fontsize=25)
#
# plt.title('required damage evolution')

plt.show()


# E = 30.e3
#
# nu = 0.2
#
# gamma_N = 00000.
#
# K_N = 100000.0
#
# # A_d_serie = np.array([10000, 5000, 1000, 500, 100])
# A_d_serie = np.array([10000])
# # c_T_serie = np.array([1, 1.5, 2, 2.5, 3])
# eps_0 = 0.0001
# sigma_0 = 20.
#
#
# # e_T_serie = np.array([0])
#
#
#
# tau_pi_bar = 1.000
#
#
# # S_max_serie = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65]
# S_max_serie = [1.]
# cycles_S = np.zeros_like(S_max_serie)
#
#
# for k in range(len(S_max_serie)):
#
#     S_max = S_max_serie[k]
#
#     t_steps_cycle = 1000
#     min_load = 0.4
#     n_cycles1 = 5
#
#     max_eps_N = np.linspace(-0.01 * S_max, 1 * -0.01 * S_max, n_cycles1)
#
#     eps_N_Emn = np.linspace(0, -0.001 * S_max, t_steps_cycle)
#
#     first_load = np.concatenate(
#         (np.linspace(0, 1, t_steps_cycle), np.linspace(1, min_load, t_steps_cycle)[1:]))
#     cycle1 = np.concatenate((np.linspace(min_load, 1., t_steps_cycle)[1:], np.linspace(1., min_load, t_steps_cycle)[
#         1:]))
#     cycle1 = np.tile(cycle1, n_cycles1 - 1)
#     cycle1 = np.concatenate((first_load, cycle1[:-1:]))
#
#     eps_N_aux = np.repeat(max_eps_N, (t_steps_cycle - 1) * 2)
#
#     eps_N_Emn = np.einsum('...n,...n->...n', eps_N_aux, cycle1)
# #
#
#     for j in range(len(A_d_serie)):
#
#         Ad = A_d_serie[j]
#         D_int = np.zeros_like(eps_N_Emn)
#         psi_n = np.zeros_like(eps_N_Emn)
#         w_N_Emn = np.zeros_like(eps_N_Emn)
#         r_N_Emn = np.zeros_like(eps_N_Emn)
#         Y_N = np.zeros_like(eps_N_Emn)
#         eps_pi_plot = np.zeros_like(eps_N_Emn)
#         sigma_pi_plot = np.zeros_like(eps_N_Emn)
#         z_N_Emn = np.zeros_like(eps_N_Emn)
#         alpha_N_Emn = np.zeros_like(eps_N_Emn)
#         eps_N_p_Emn = np.zeros_like(eps_N_Emn)
#         sigma_N_Emn = np.zeros_like(eps_N_Emn)
#
#         Z = np.zeros_like(eps_N_Emn)
#         X = np.zeros_like(eps_N_Emn)
#
#         E_N = E / (1.0 - 2.0 * nu)
#
#         cycles = 10
#
#         for i in range(1, len(eps_N_Emn)):
#
#             sigma_trial = E_N * (eps_N_Emn[i] - eps_N_p_Emn[i - 1])
#             pos = sigma_trial > 1e-6
#             pos2 = sigma_trial < -1e-6
#             H = 1.0 * pos
#             H2 = 1.0 * pos2
#
#             sigma_n_trial = (
#                 1.0 - H * w_N_Emn[i - 1]) * E_N * (eps_N_Emn[i] - eps_N_p_Emn[i - 1])
#             sigma_N_Emn_tilde = E_N * (eps_N_Emn[i] - eps_N_p_Emn[i - 1])
#             Z[i] = K_N * r_N_Emn[i - 1] * H2
#             X[i] = gamma_N * alpha_N_Emn[i - 1] * H2
#             h = (sigma_0 + Z[i]) * H2
#
#             f_trial = (abs(sigma_N_Emn_tilde - X[i]) - h) * H2
#
#             thres_1 = f_trial > 1e-6
#
#             delta_lamda = f_trial / \
#                 (E_N / (1 - w_N_Emn[i - 1]) + abs(K_N) + gamma_N) * thres_1
#             eps_N_p_Emn[i] = eps_N_p_Emn[i - 1] + delta_lamda * \
#                 np.sign(sigma_N_Emn_tilde - X[i])
#             r_N_Emn[i] = r_N_Emn[i - 1] + delta_lamda
#             alpha_N_Emn[i] = alpha_N_Emn[i - 1] + delta_lamda * \
#                 np.sign(sigma_N_Emn_tilde - X[i])
#
#             def Z_N(z_N_Emn): return (1.0 / Ad) * (-z_N_Emn / (1.0 + z_N_Emn))
#             #print((eps_N_Emn[i] - eps_N_p_Emn[i]) ** 2.0)
#             Y_N[i] = 0.5 * H * E_N * (eps_N_Emn[i] - eps_N_p_Emn[i]) ** 2.0
#             Y_0 = 0.5 * E_N * eps_0 ** 2.0
#
#             f = (Y_N[i] - (Y_0 + Z_N(z_N_Emn[i - 1])))
#
#             thres_2 = f > 1e-6
#             thres_3 = f < 1e-6
#
#             def f_w(Y): return 1.0 - 1.0 / (1.0 + Ad * (Y - Y_0))
#
#             w_N_Emn[i] = w_N_Emn[i - 1] * thres_3 + f_w(Y_N[i]) * thres_2
#             z_N_Emn[i] = -w_N_Emn[i]
#
#             sigma_N_Emn[i] = (1.0 - H * w_N_Emn[i]) * E_N * \
#                 (eps_N_Emn[i] - eps_N_p_Emn[i])
#             D_int[i] = Y_N[i] * (w_N_Emn[i] - w_N_Emn[i - 1])
#
#         plt.subplot(233)
#         plt.plot(eps_N_Emn,
#                  sigma_N_Emn, linewidth=2.5)
#
#         plt.xlabel('$\sigma_N$', fontsize=25)
#         plt.ylabel('$\dot{\omega}$', fontsize=25)
#         plt.title('microplane damage')
#
#         plt.subplot(234)
#         # plt.plot(eps_T, Y_aux, linewidth=3.5)
#         plt.plot(eps_N_Emn,
#                  eps_N_p_Emn, linewidth=2.5)
#
#         plt.xlabel('$\sigma_N$', fontsize=25)
#         plt.ylabel('$\dot{\omega}$', fontsize=25)
#         plt.title('microplane damage')
#
#         plt.subplot(232)
#
#         plt.plot(eps_N_Emn, w_N_Emn, linewidth=3.5)
#
#         plt.xlabel('$\sigma_1$', fontsize=25)
#         plt.ylabel('$\dot{\omega}$', fontsize=25)
#
#         plt.subplot(231)
#
#         plt.plot(np.arange(len(eps_N_Emn)), eps_N_Emn, linewidth=3.5)
#
#         plt.xlabel('$\sigma_1$', fontsize=25)
#         plt.ylabel('$\dot{\omega}$', fontsize=25)
#
#         plt.subplot(235)
#
#         plt.plot(eps_N_Emn, D_int, linewidth=3.5)
#
#         plt.xlabel('$\sigma_1$', fontsize=25)
#         plt.ylabel('$\dot{\omega}$', fontsize=25)
#
# #         plt.subplot(236)
# #
# #         plt.plot(eps_T, psi_t, linewidth=3.5)
# #
# #         plt.xlabel('$\sigma_1$', fontsize=25)
# #         plt.ylabel('$\dot{\omega}$', fontsize=25)
# plt.show()
