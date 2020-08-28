'''
Created on 04.07.2019
@author: Abdul
plotting tool for microplane models
'''

import os

import matplotlib

import matplotlib.pyplot as plt
import numpy as np


class Micro2Dplot():

    def get_2Dviz(self, n_mp, eps_N_Emn, eps_T_Emn, omega_N_Emn, z_N_Emn, alpha_N_Emn, r_N_Emn, eps_N_p_Emn, sigma_N_Emn, Z_N_Emn, X_N_Emn, Y_N_Emn, \
           omega_T_Emn, z_T_Emn, alpha_T_Emn, eps_T_pi_Emn, sigma_T_Emn, Z_T_pi_Emn, X_T_pi_Emn, Y_T_pi_Emn, \
           Disip_omena_N_Emn, Disip_omena_T_Emn, Disip_eps_p_N_Emn, Disip_eps_p_T_Emn, Disip_iso_N_Emn, \
           Disip_iso_T_Emn, Disip_kin_N_Emn, Disip_kin_T_Emn):

        rads = np.arange(0, (2 * np.pi), (2 * np.pi) / n_mp)
        font = {'family': 'DejaVu Sans',
                'size': 18}

        matplotlib.rc('font', **font)
        A = np.array(range(len(eps_N_p_Emn)))

        plt.figure(figsize=(9, 3))
        plt.subplot(241, projection='polar')
        for i in A:
            #print('idx', idx.shape)
            plt.plot(rads, eps_N_Emn[i, :], 'k')
        plt.title(r'$\varepsilon_N$')

        plt.subplot(242, projection='polar')
        for i in A:
            plt.plot(rads, omega_N_Emn[i, :], 'g')
        plt.title(r'$\omega_N$')

        plt.subplot(243, projection='polar')
        for i in A:
            plt.plot(rads, eps_N_p_Emn[i, :], 'g')
        plt.ylim(-1.2 * np.max(np.abs(eps_N_p_Emn)),
                 0.8 * np.max(np.abs(eps_N_p_Emn)))
        plt.title(r'$\varepsilon^p_N$')

        plt.subplot(244, projection='polar')
        for i in A:
            plt.plot(rads, sigma_N_Emn[i, :], 'b')
        plt.ylim(-1.2 * np.max(np.abs(sigma_N_Emn)),
                 0.8 * np.max(np.abs(sigma_N_Emn)))
        plt.title(r'$\sigma_N$')

        plt.subplot(245, projection='polar')
        for i in A:
            plt.plot(rads, np.abs(eps_T_Emn[i, :]), 'k')
        plt.title(r'$\varepsilon_T$')

        plt.subplot(246, projection='polar')
        for i in A:
            plt.plot(rads, omega_T_Emn[i, :], 'r')
        plt.title(r'$\omega_T$')

        plt.subplot(247, projection='polar')
        for i in A:
            plt.plot(rads, np.abs(eps_T_pi_Emn[i, :]), 'r')
        plt.title(r'$\varepsilon^{\pi}_T$')

        plt.subplot(248, projection='polar')
        for i in A:
            plt.plot(rads, np.abs(sigma_T_Emn[i, :]), 'b')
        plt.title(r'$\sigma_T$')

        plt.show()


        plt.figure(figsize=(9, 3))

        plt.subplot(241, projection='polar')
        print(Disip_omena_N_Emn[42, :])
        for i in A:
            print(Disip_omena_N_Emn[i, :].shape, i)
            plt.plot(rads, np.array(Disip_omena_N_Emn[i, :]), 'k')
        plt.title(r'$D \omega_N_N$')

        plt.subplot(242, projection='polar')
        for i in A:
            plt.plot(rads, Disip_omena_T_Emn[i, :], 'g')
        plt.title(r'$D \omega_T$')

        plt.subplot(243, projection='polar')
        for i in A:
            plt.plot(rads, Disip_eps_p_N_Emn[i, :], 'g')
        plt.title(r'$\D varepsilon^p_N$')

        plt.subplot(244, projection='polar')
        for i in A:
            plt.plot(rads, Disip_eps_p_T_Emn[i, :], 'b')
        plt.title(r'$\D varepsilon^p_T$')

        plt.subplot(245, projection='polar')
        for i in A:
            plt.plot(rads, Disip_iso_N_Emn[i, :], 'k')
        plt.title(r'$D iso N$')

        plt.subplot(246, projection='polar')
        for i in A:
            plt.plot(rads, Disip_iso_T_Emn[i, :], 'k')
        plt.title(r'$D iso T$')

        plt.subplot(247, projection='polar')
        for i in A:
            plt.plot(rads, Disip_kin_N_Emn[i, :], 'r')
        plt.title(r'$D kin N$')

        plt.subplot(248, projection='polar')
        for i in A:
            plt.plot(rads, Disip_kin_T_Emn[i, :], 'r')
        plt.title(r'$D kin T$')

        plt.show()
#
#         plt.subplot(241, projection='polar')
#         for i in A:
#             #print('idx', idx.shape)
#             plt.plot(rads, eps_global_norm[i, :])
#         plt.ylim(-1.2 * np.max(np.abs(eps_global_norm)),
#                  0.8 * np.max(np.abs(eps_global_norm)))
#         plt.title('eps_macro 20L-80H')
#
#         plt.subplot(242, projection='polar')
#         for i in A:
#             plt.plot(rads, sigma_global_norm[i, :])
#         plt.ylim(-1.2 * np.max(np.abs(sigma_global_norm)),
#                  0.8 * np.max(np.abs(sigma_global_norm)))
#         plt.title('sigma_macro')
#
#         plt.subplot(243, projection='polar')
#         for i in A:
#             plt.plot(rads, D_1_norm[i, :])
#         #plt.ylim(-1000, 1000)
#         plt.title('D_11')
#
#         plt.subplot(244, projection='polar')
#         for i in A:
#             plt.plot(rads, D_2_norm[i, :])
#         #plt.ylim(-1000, 1000)
#         plt.title('D_22')
#
#         plt.subplot(245, projection='polar')
#         for i in A:
#             plt.plot(rads, eps_micro_norm[i, :])
# #         plt.ylim(-1.2 * np.max(np.abs(eps_micro_norm)),
# #                  0.8 * np.max(np.abs(eps_micro_norm)))
#         plt.title('eps_micro')
#
#         plt.subplot(246, projection='polar')
#         for i in A:
#             plt.plot(rads, sigma_micro_norm[i, :])
# #         plt.ylim(-1.2 * np.max(np.abs(sigma_micro_norm)),
# #                  0.8 * np.max(np.abs(sigma_micro_norm)))
#         plt.title('sigma_micro')
#
#         plt.subplot(247, projection='polar')
#         for i in A:
#             plt.plot(rads, D_12_norm[i, :])
#         #plt.ylim(-1000, 1000)
#         plt.title('D_12')
#
#         plt.subplot(248, projection='polar')
#         for i in A:
#             plt.plot(rads, D_21_norm[i, :])
#         #plt.ylim(-1000, 1000)
#         plt.title('D_21')
#
#         plt.show()
#
#         plt.subplot(221, projection='polar')
#         for i in A:
#             #print('idx', idx.shape)
#             plt.plot(rads, eps_N[i, :])
#         #plt.ylim(-1.2 * np.max(np.abs(eps_N)), 0.8 * np.max(np.abs(eps_N)))
#         plt.title('eps_N 20L-80H')
#
#         plt.subplot(222, projection='polar')
#         for i in A:
#             plt.plot(rads, eps_p_N[i, :])
#         plt.ylim(-1.2 * np.max(np.abs(eps_p_N)), 0.8 * np.max(np.abs(eps_p_N)))
#         plt.title('eps_p_N')
#
#         plt.subplot(223, projection='polar')
#         for i in A:
#             plt.plot(rads, sigma_N[i, :])
# #         plt.ylim(-1.2 * np.max(np.abs(sigma_N)), 0.8 * np.max(np.abs(sigma_N)))
#         plt.title('sigma_N')
#
#         plt.subplot(224, projection='polar')
#         for i in A:
#             plt.plot(rads, omegaN[i, :])
#         plt.title('omegaN')
#
#         plt.show()
#
#         plt.subplot(221, projection='polar')
#         for i in A:
#             plt.plot(rads, np.abs(eps_T_sign[i, :]))
#         plt.title('eps_T 20L-80H')
#
#         plt.subplot(222, projection='polar')
#         for i in A:
#             plt.plot(rads, np.abs(eps_pi_T_sign[i, :]))
#         plt.title('eps_pi_T')
#
#         plt.subplot(223, projection='polar')
#         for i in A:
#             plt.plot(rads, np.abs(sigma_T_sign[i, :]))
#         plt.title('sigma_T')
#
#         plt.subplot(224, projection='polar')
#         for i in A:
#             plt.plot(rads, omegaT[i, :])
#         plt.title('omegaT')
#
#         plt.show()
#
#         'Plotting in cycles'
#         #===================
#         # Normal strain
#         #===================
#         n_mp = 360
#         plt.subplot(231)
#
#         plt.plot(np.arange(len(A)), eps_N[A, 0], linewidth=2.5)
#         # plt.plot(np.arange(len(A)), eps_N[A, np.int(n_mp / 20)], linewidth=2.5)
#         # plt.plot(np.arange(len(A)), eps_N[A, np.int(n_mp / 15)], linewidth=2.5)
#         # plt.plot(np.arange(len(A)), eps_N[A, np.int(n_mp / 10)], linewidth=2.5)
#         # plt.plot(np.arange(len(A)), eps_N[A, np.int(n_mp / 8)], linewidth=2.5)
#         # plt.plot(np.arange(len(A)), eps_N[A, np.int(n_mp / 6)], linewidth=2.5)
#         # plt.plot(np.arange(len(A)), eps_N[A, np.int(n_mp / 5)], linewidth=2.5)
#         plt.plot(np.arange(len(A)), eps_N[A, np.int(n_mp / 4)], linewidth=2.5)
#
#         plt.title('eps_N 20L-80H')
#
#         #===================
#         # Normal damage
#         #===================
#         plt.subplot(232)
#
# #         plt.plot(np.arange(
# #             len(A)), eps_p_N[A, 0], linewidth=2.5)
# #         plt.plot(np.arange(
# #             len(A)), eps_p_N[A, np.int(n_mp / 4)], linewidth=2.5)
#
#         plt.plot(eps_N[:, 0], eps_p_N[:, 0], linewidth=2.5)
#         plt.plot(eps_N[:, np.int(n_mp / 4)],
#                  eps_p_N[:, np.int(n_mp / 4)], linewidth=2.5)
#
#         plt.title('eps_p_N')
#
#         #===================
#         # Normal plastic strain
#         #===================
#         plt.subplot(233)
#
# #         plt.plot(np.arange(
# #             len(A)), sigma_N[A, 0], linewidth=2.5)
# #         plt.plot(np.arange(
# #             len(A)), sigma_N[A, np.int(n_mp / 8)], linewidth=2.5)
# #         plt.plot(np.arange(
# #             len(A)), sigma_N[A, np.int(n_mp / 4)], linewidth=2.5)
#         plt.plot(np.abs(F[A, 0]), sigma_N[A, 0], linewidth=2.5)
#         # plt.plot(np.abs(F[A, 0]), sigma_N[A, 5], linewidth=2.5)
#         # plt.plot(np.abs(F[A, 0]), sigma_N[A, 10], linewidth=2.5)
#         # plt.plot(np.abs(F[A, 0]), sigma_N[A, 15], linewidth=2.5)
#         # plt.plot(np.abs(F[A, 0]), sigma_N[A, 20], linewidth=2.5)
#         # plt.plot(np.abs(F[A, 0]), sigma_N[A, 25], linewidth=2.5)
#         # plt.plot(np.abs(F[A, 0]), sigma_N[A, 30], linewidth=2.5)
#         # plt.plot(np.abs(F[A, 0]), sigma_N[A, 35], linewidth=2.5)
#         # plt.plot(np.abs(F[A, 0]), sigma_N[A, 40], linewidth=2.5)
#         # plt.plot(np.abs(F[A, 0]), sigma_N[A, 45], linewidth=2.5)
#         # plt.plot(np.abs(F[A, 0]), sigma_N[A, 50], linewidth=2.5)
#         # plt.plot(np.abs(F[A, 0]), sigma_N[A, 55], linewidth=2.5)
#         # plt.plot(np.abs(F[A, 0]), sigma_N[A, 60], linewidth=2.5)
#         # plt.plot(np.abs(F[A, 0]), sigma_N[A, 65], linewidth=2.5)
#         # plt.plot(np.abs(F[A, 0]), sigma_N[A, 70], linewidth=2.5)
#         # plt.plot(np.abs(F[A, 0]), sigma_N[A, 75], linewidth=2.5)
#         # plt.plot(np.abs(F[A, 0]), sigma_N[A, 80], linewidth=2.5)
#         # plt.plot(np.abs(F[A, 0]), sigma_N[A, 85], linewidth=2.5)
#         plt.plot(np.abs(F[A, 0]), sigma_N[A, np.int(n_mp / 4)], linewidth=2.5)
#
# #         plt.plot(np.arange(len(A)), sigma_N[A, 0], linewidth=2.5)
# #         plt.plot(np.arange(len(A)),
# #                  sigma_N[A, np.int(n_mp / 20)], linewidth=2.5)
# #         plt.plot(np.arange(len(A)),
# #                  sigma_N[A, np.int(n_mp / 15)], linewidth=2.5)
# #         plt.plot(np.arange(len(A)),
# #                  sigma_N[A, np.int(n_mp / 10)], linewidth=2.5)
# #         plt.plot(np.arange(len(A)),
# #                  sigma_N[A, np.int(n_mp / 8)], linewidth=2.5)
# #         plt.plot(np.arange(len(A)),
# #                  sigma_N[A, np.int(n_mp / 6)], linewidth=2.5)
# #         plt.plot(np.arange(len(A)),
# #                  sigma_N[A, np.int(n_mp / 5)], linewidth=2.5)
# #         plt.plot(np.arange(len(A)),
# #                  sigma_N[A, np.int(n_mp / 4)], linewidth=2.5)
#         plt.title('sigma_n')
#
#         #===================
#         # Tangential strain
#         #===================
#         plt.subplot(234)
#
# #         plt.plot(np.arange(
# #             len(A)), omegaN[A, 0], linewidth=2.5)
# #         plt.plot(np.arange(
# #             len(A)), omegaN[A, np.int(n_mp / 4)], linewidth=2.5)
#
# #         plt.plot(np.abs(F[A, 0]), omegaN[A, np.int(n_mp / 8)], linewidth=2.5)
# #         plt.plot(np.abs(F[A, 0]), omegaN[A, np.int(n_mp / 6)], linewidth=2.5)
# #         plt.plot(np.abs(F[A, 0]), omegaN[A, np.int(n_mp / 5)], linewidth=2.5)
# #         plt.plot(np.abs(F[A, 0]), omegaN[A, np.int(n_mp / 4)], linewidth=2.5)
#
#         plt.plot(np.arange(len(A)), omegaN[A, np.int(n_mp / 4)], linewidth=2.5)
#         plt.plot(np.arange(len(A)), omegaN[A, 85], linewidth=2.5)
#         plt.plot(np.arange(len(A)), omegaN[A, 80], linewidth=2.5)
#         plt.plot(np.arange(len(A)), omegaN[A, 75], linewidth=2.5)
#         plt.plot(np.arange(len(A)), omegaN[A, 70], linewidth=2.5)
#         plt.plot(np.arange(len(A)), omegaN[A, 65], linewidth=2.5)
#         plt.plot(np.arange(len(A)), omegaN[A, 60], linewidth=2.5)
#         plt.plot(np.arange(len(A)), omegaN[A, 55], linewidth=2.5)
#         plt.plot(np.arange(len(A)), omegaN[A, 50], linewidth=2.5)
#         plt.plot(np.arange(len(A)), omegaN[A, 45], linewidth=2.5)
#         plt.plot(np.arange(len(A)), omegaN[A, 40], linewidth=2.5)
#         plt.plot(np.arange(len(A)), omegaN[A, 35], linewidth=2.5)
#         plt.plot(np.arange(len(A)), omegaN[A, 30], linewidth=2.5)
#         plt.plot(np.arange(len(A)), omegaN[A, 25], linewidth=2.5)
#         plt.plot(np.arange(len(A)), omegaN[A, 20], linewidth=2.5)
#         plt.plot(np.arange(len(A)), omegaN[A, 15], linewidth=2.5)
#         plt.plot(np.arange(len(A)), omegaN[A, 10], linewidth=2.5)
#         plt.plot(np.arange(len(A)), omegaN[A, 5], linewidth=2.5)
#         plt.plot(np.arange(len(A)), omegaN[A, 0], linewidth=2.5)
#
#         # plt.plot(np.arange(len(A)),
#         #          omegaN[A, np.int(n_mp / 4.5)], linewidth=2.5)
#         # plt.plot(np.arange(len(A)), omegaN[A, np.int(n_mp / 5)], linewidth=2.5)
#         # plt.plot(np.arange(len(A)),
#         #          omegaN[A, np.int(n_mp / 5.5)], linewidth=2.5)
#
#         plt.title('omegaN')
#
#         #===================
#         # Tangential damage
#         #===================
#         plt.subplot(235)
#
#         plt.plot(np.arange(len(A)), Y_N[A, 0], linewidth=2.5)
#         plt.plot(np.arange(len(A)), Y_N[A, np.int(n_mp / 4)], linewidth=2.5)
#
#         plt.title('lambda')
#
#         #===================
#         # Tangential plastic strain
#         #===================
#         plt.subplot(236)
#
#         plt.plot(np.arange(len(A)), X_N[A, 0], linewidth=2.5)
#         plt.plot(np.arange(len(A)), X_N[A, np.int(n_mp / 4)], linewidth=2.5)
#
#         plt.title('X_N')
#
#         plt.show()
#
#         plt.subplot(231)
#
#         plt.plot(np.arange(
#             len(A)), eps_T_sign[A, np.int(np.floor(n_mp / 8))], linewidth=2.5)
#         plt.plot(np.arange(
#             len(A)), eps_T_sign[A, np.int(n_mp / 10)], linewidth=2.5)
#
#         plt.title('eps_T_sign 20L-80H ')
#
#         #===================
#         # Normal damage
#         #===================
#         plt.subplot(232)
#
#         plt.plot(np.arange(
#             len(A)), eps_pi_T_sign[A, np.int(np.floor(n_mp / 8))], linewidth=2.5)
#         plt.plot(np.arange(
#             len(A)), eps_pi_T_sign[A, np.int(n_mp / 10)], linewidth=2.5)
#
#         plt.title('eps_pi_T_sign')
#
#         #===================
#         # Normal plastic strain
#         #===================
#         plt.subplot(233)
#
#         plt.plot(np.arange(
#             len(A)), sigma_T_sign[A, np.int(np.floor(n_mp / 8))], linewidth=2.5)
#         plt.plot(np.arange(
#             len(A)), sigma_T_sign[A, np.int(n_mp / 10)], linewidth=2.5)
#
#         plt.title('sigma_T_sign')
#
#         #===================
#         # Tangential strain
#         #===================
#         plt.subplot(234)
#
# #         plt.plot(np.arange(len(A)), omegaT[A, np.int(
# #             np.floor(n_mp / 8))], linewidth=2.5)
# #         plt.plot(np.arange(len(A)), omegaT[A, np.int(
# #             np.floor(n_mp / 10))], linewidth=2.5)
# #
#         #plt.plot(np.abs(F[:, 0]), omegaT[:, np.int(n_mp / 8)], linewidth=2.5)
# #         plt.plot(np.abs(F[:, 0]), omegaT[:, np.int(n_mp / 10)], linewidth=2.5)
#
#         plt.plot(np.abs(F[A, 0]), omegaT[A, np.int(n_mp / 8)], linewidth=2.5)
#         plt.plot(np.abs(F[A, 0]), omegaT[A, np.int(n_mp / 10)], linewidth=2.5)
#         fc = np.max(np.abs(F[:, 0]))
# #         print(fc)
#         plt.xlim(0.75 * fc, 0.85 * fc)
#         plt.ylim(1e-7, 1e-4)
#         plt.yscale('log')
#
#         plt.title('omegaT')
# #
#         #===================
#         # Tangential damage
#         #===================
#         plt.subplot(235)
#
# #         plt.plot(np.arange(len(A)), omegaT[A, np.int(
# #             np.floor(n_mp / 8))], linewidth=2.5)
# #         plt.plot(np.arange(len(A)), omegaT[A, np.int(
# #             np.floor(n_mp / 10))], linewidth=2.5)
# #
#         #plt.plot(np.abs(F[:, 0]), omegaT[:, np.int(n_mp / 8)], linewidth=2.5)
# #         plt.plot(np.abs(F[:, 0]), omegaT[:, np.int(n_mp / 10)], linewidth=2.5)
#
#         plt.plot(np.arange(len(A)), omegaT[A, np.int(n_mp / 8)], linewidth=2.5)
#         plt.plot(np.arange(len(A)), omegaT[A, 40], linewidth=2.5)
#         plt.plot(np.arange(len(A)), omegaT[A, 35], linewidth=2.5)
#         plt.plot(np.arange(len(A)), omegaT[A, 30], linewidth=2.5)
#         plt.plot(np.arange(len(A)), omegaT[A, 25], linewidth=2.5)
#         plt.plot(np.arange(len(A)), omegaT[A, 20], linewidth=2.5)
#         plt.plot(np.arange(len(A)), omegaT[A, 15], linewidth=2.5)
#         plt.plot(np.arange(len(A)), omegaT[A, 10], linewidth=2.5)
#         plt.plot(np.arange(len(A)), omegaT[A, 5], linewidth=2.5)
#         plt.plot(np.arange(len(A)), omegaT[A, 0], linewidth=2.5)
#
#
#
# #         plt.plot(np.abs(F[A, 0]), omegaT[A, np.int(n_mp / 8)], linewidth=2.5)
# #         plt.plot(np.abs(F[A, 0]), omegaT[A, np.int(n_mp / 9)], linewidth=2.5)
# #         plt.plot(np.abs(F[A, 0]), omegaT[A, np.int(n_mp / 10)], linewidth=2.5)
# #         plt.plot(np.abs(F[A, 0]), omegaT[A, np.int(n_mp / 11)], linewidth=2.5)
#         fc = np.max(np.abs(F[:, 0]))
# #         print(fc)
# #         plt.xlim(0.7 * fc, 0.95 * fc)
# #         plt.ylim(1e-7, 1)
# #         plt.yscale('log')
#
#         plt.title('omegaT_lin')
#
#         plt.subplot(236)
#
#         plt.plot(np.arange(len(A)), omegaT[A, np.int(
#             np.floor(n_mp / 8))], linewidth=2.5)
#         plt.plot(np.arange(len(A)), omegaT[A, np.int(
#             np.floor(n_mp / 10))], linewidth=2.5)
# #
#         #plt.plot(np.abs(F[:, 0]), omegaT[:, np.int(n_mp / 8)], linewidth=2.5)
# #         plt.plot(np.abs(F[:, 0]), omegaT[:, np.int(n_mp / 10)], linewidth=2.5)
#
# #         plt.plot(np.abs(F[A, 0]), omegaT[A, np.int(n_mp / 8)], linewidth=2.5)
# #         plt.plot(np.abs(F[A, 0]), omegaT[A, np.int(n_mp / 10)], linewidth=2.5)
# #         fc = np.max(np.abs(F[:, 0]))
# # #         print(fc)
# #         plt.xlim(0.7 * fc, 0.8 * fc)
# #         plt.ylim(1e-7, 1e-4)
# #         plt.yscale('log')
#
#         plt.title('omegaT')
#
#         plt.show()
#
#         plt.subplot(131)
#
#         plt.plot(np.arange(
#             len(A)) / 2, D_1_norm[A, 0] / D_1_norm[A[0], 0], linewidth=2.5)
#
#         plt.title('eps_T_sign 20L-80H ')
#
#         #===================
#         # Normal damage
#         #===================
#         plt.subplot(132)
#
#         plt.plot(np.arange(
#             len(A)) / 2, D_2_norm[A, np.int(np.floor(n_mp / 4))] / D_2_norm[A[0], np.int(np.floor(n_mp / 4))], linewidth=2.5)
#
#         plt.title('eps_pi_T_sign')
#
#         #===================
#         # Normal plastic strain
#         #===================
#         plt.subplot(133)
#
#         plt.plot(np.arange(
#             len(A)) / 2, D_12_norm[A, np.int(np.floor(n_mp / 8))] / D_12_norm[A[0], np.int(np.floor(n_mp / 8))], linewidth=2.5)
#
#         plt.title('sigma_T_sign')
#
#         plt.show()
