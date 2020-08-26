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

    def get_2Dviz(self, n_mp, eps_global_norm, sigma_global_norm, eps_micro_norm, sigma_micro_norm, D_1_norm, D_2_norm, D_12_norm, D_21_norm, eps_N, eps_p_N, sigma_N, omegaN, eps_T_sign, eps_pi_T_sign, sigma_T_sign, omegaT, Y_N, X_N, Y_T, X_T, F, U, t_steps_cycle,D_E_damage_N_M,D_E_damage_T_M, D_E_plast_N_M,D_E_plast_T_M, D_E_hard_iso_N_M,D_E_hard_iso_T_M,D_E_hard_kin_N_M,D_E_hard_kin_T_M):

        rads = np.arange(0, (2 * np.pi), (2 * np.pi) / n_mp)
        font = {'family': 'DejaVu Sans',
                'size': 18}

        matplotlib.rc('font', **font)
        A = np.array(range(len(F)))

        # A = A[1::5]
        #
        # home_dir = os.path.expanduser('~')
        # out_path = os.path.join(home_dir, 'anim')
        # out_path = os.path.join(out_path, '2D')

#         for i in A:
#
#             f, (ax2) = plt.subplots(1, 1, figsize=(5, 4))
#
#             ax2.plot(np.abs(U[:i, 0]), np.abs(F[:i, 0]), 'k', linewidth=3.5)
#
#             ax2.set_xlim(0.0000, 0.01)
#             #
#             ax2.set_ylim(0.0000, 70)
#
#             filename1 = os.path.join(
#                 out_path, 'F-U' + 'animation' + np.str(i) + '.png')
#             f.savefig(fname=filename1)
#             plt.close()
#
#             f, (ax) = plt.subplots(2, 4, figsize=(25, 20))
#             ax = plt.subplot(241, projection='polar')
#             ax.plot(rads, eps_N[i, :], 'k')
#             ax.set_ylim(-0.012, 0.012)
#             # ax.set_title(r'$\varepsilon_N$')
#
#             ax = plt.subplot(242, projection='polar')
#             ax.plot(rads, omegaN[i, :], 'g')
#             ax.set_ylim(0., 1.05)
#             # ax.set_title(r'$\omega_N$')
#
#             ax = plt.subplot(243, projection='polar')
#             ax.plot(rads, eps_p_N[i, :], 'g')
#             ax.set_ylim(-0.01, 0.002)
# #             ax.set_ylim(-1.2 * np.max(np.abs(eps_p_N)),
# #                         0.8 * np.max(np.abs(eps_p_N)))
#             # ax.set_title(r'$\varepsilon^p_N$')
#
#             ax = plt.subplot(244, projection='polar')
#             ax.plot(rads, sigma_N[i, :], 'b')
#             ax.set_ylim(-600, 100)
# #             ax.set_ylim(-1.2 * np.max(np.abs(sigma_N)),
# #                         0.8 * np.max(np.abs(sigma_N)))
#             # ax.set_title(r'$\sigma_N$')
#
#             ax = plt.subplot(245, projection='polar')
#             ax.plot(rads, np.abs(eps_T_sign[i, :]), 'k')
#             ax.set_ylim(0, 0.0105)
# #             ax.set_title(r'$\varepsilon_T$')
#
#             ax = plt.subplot(246, projection='polar')
#             ax.plot(rads, omegaT[i, :], 'r')
#             ax.set_ylim(0.0, 1.05)
# #             ax.set_title(r'$\omega_T$')
#
#             ax = plt.subplot(247, projection='polar')
#             ax.plot(rads, np.abs(eps_pi_T_sign[i, :]), 'r')
#             ax.set_ylim(0, 0.007)
# #             ax.set_title(r'$\varepsilon^{\pi}_T$')
#
#             ax = plt.subplot(248, projection='polar')
#             ax.plot(rads, np.abs(sigma_T_sign[i, :]), 'b')
#             ax.set_ylim(0, 16)
# #             ax.set_title(r'$\sigma_T$')
#
#             filename1 = os.path.join(
#                 out_path, 'microplane' + 'animation' + np.str(i) + '.png')
#             f.savefig(fname=filename1)
#             plt.close()

        # A = A[1::10]
        #A = A[(t_steps_cycle - 1) + 1::2 * (t_steps_cycle - 1)]
        plt.figure(figsize=(9, 3))
        plt.subplot(241, projection='polar')
        for i in A:
            #print('idx', idx.shape)
            plt.plot(rads, eps_N[i, :], 'k')
        plt.title(r'$\varepsilon_N$')

        plt.subplot(242, projection='polar')
        for i in A:
            plt.plot(rads, omegaN[i, :], 'g')
        plt.title(r'$\omega_N$')

        plt.subplot(243, projection='polar')
        for i in A:
            plt.plot(rads, eps_p_N[i, :], 'g')
        plt.ylim(-1.2 * np.max(np.abs(eps_p_N)),
                 0.8 * np.max(np.abs(eps_p_N)))
        plt.title(r'$\varepsilon^p_N$')

        plt.subplot(244, projection='polar')
        for i in A:
            plt.plot(rads, sigma_N[i, :], 'b')
        plt.ylim(-1.2 * np.max(np.abs(sigma_N)),
                 0.8 * np.max(np.abs(sigma_N)))
        plt.title(r'$\sigma_N$')

        plt.subplot(245, projection='polar')
        for i in A:
            plt.plot(rads, np.abs(eps_T_sign[i, :]), 'k')
        plt.title(r'$\varepsilon_T$')

        plt.subplot(246, projection='polar')
        for i in A:
            plt.plot(rads, omegaT[i, :], 'r')
        plt.title(r'$\omega_T$')

        plt.subplot(247, projection='polar')
        for i in A:
            plt.plot(rads, np.abs(eps_pi_T_sign[i, :]), 'r')
        plt.title(r'$\varepsilon^{\pi}_T$')

        plt.subplot(248, projection='polar')
        for i in A:
            plt.plot(rads, np.abs(sigma_T_sign[i, :]), 'b')
        plt.title(r'$\sigma_T$')

        plt.show()

        print(D_E_damage_N_M.shape)

        plt.figure(figsize=(9, 3))

        plt.subplot(241, projection='polar')
        for i in A:
            # print('idx', idx.shape)
            plt.plot(rads, D_E_damage_N_M.transpose()[i, :], 'k')
        plt.title(r'$D \omega_N_N$')

        plt.subplot(242, projection='polar')
        for i in A:
            plt.plot(rads, D_E_damage_T_M.transpose()[i, :], 'g')
        plt.title(r'$D \omega_T$')

        plt.subplot(243, projection='polar')
        for i in A:
            plt.plot(rads, D_E_plast_N_M.transpose()[i, :], 'g')
        plt.title(r'$\D varepsilon^p_N$')

        plt.subplot(244, projection='polar')
        for i in A:
            plt.plot(rads, D_E_plast_T_M.transpose()[i, :], 'b')
        plt.title(r'$\D varepsilon^p_T$')

        plt.subplot(245, projection='polar')
        for i in A:
            plt.plot(rads, -D_E_hard_iso_N_M.transpose()[i, :], 'k')
        plt.title(r'$D iso N$')

        plt.subplot(246, projection='polar')
        for i in A:
            plt.plot(rads, -D_E_hard_iso_T_M.transpose()[i, :], 'k')
        plt.title(r'$D iso T$')

        plt.subplot(247, projection='polar')
        for i in A:
            plt.plot(rads, D_E_hard_kin_N_M.transpose()[i, :], 'r')
        plt.title(r'$D kin N$')

        plt.subplot(248, projection='polar')
        for i in A:
            plt.plot(rads, D_E_hard_kin_T_M.transpose()[i, :], 'r')
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
