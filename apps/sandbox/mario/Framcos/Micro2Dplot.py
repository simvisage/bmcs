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
        for i in A:
            plt.plot(rads, np.array(Disip_omena_N_Emn[i, :]), 'k')
        plt.title(r'$D \omega_N$')

        plt.subplot(242, projection='polar')
        for i in A:
            plt.plot(rads, Disip_omena_T_Emn[i, :], 'g')
        plt.title(r'$D \omega_T$')

        plt.subplot(243, projection='polar')
        for i in A:
            plt.plot(rads, Disip_eps_p_N_Emn[i, :], 'g')
        plt.title(r'$D \varepsilon^p_N$')

        plt.subplot(244, projection='polar')
        for i in A:
            plt.plot(rads, Disip_eps_p_T_Emn[i, :], 'b')
        plt.title(r'$D \varepsilon^p_T$')

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