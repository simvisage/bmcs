'''
Created on 04.07.2019
@author: Abdul
plotting tool for microplane models
'''

import matplotlib.pyplot as plt
import numpy as np


class Micro2Dplot():

    def get_2Dviz(self, n_mp, eps_N, omegaN, eps_p_N, eps_T, omegaT, eps_pi_T):

        rads = np.arange(0, (2 * np.pi), (2 * np.pi) / n_mp)

        'Plotting in cycles'
        #===================
        # Normal strain
        #===================

        peaks = np.arange(len(eps_N))

        plt.subplot(231, projection='polar')
        for i in peaks:
            #print('idx', idx.shape)
            plt.plot(rads, eps_N[i, :])
        plt.ylim(-1.2 * np.max(np.abs(eps_N)), 0.8 * np.max(np.abs(eps_N)))
        plt.title('eps_N')

        #===================
        # Normal damage
        #===================
        plt.subplot(232, projection='polar')
        for i in peaks:
            #plt.polar(rads, w_1_T[i, :])
            plt.plot(rads, omegaN[i, :])
        plt.title('omegaN')

        #===================
        # Normal plastic strain
        #===================
        plt.subplot(233, projection='polar')
        for i in peaks:
            plt.plot(rads, eps_p_N[i, :])
        plt.ylim(-1.2 * np.max(np.abs(eps_p_N)), 0.8 * np.max(np.abs(eps_p_N)))
        plt.title('eps_p_N')

        #===================
        # Tangential strain
        #===================
        plt.subplot(234, projection='polar')
        for i in peaks:
            #print('idx', idx.shape)
            plt.plot(rads, eps_T[i, :])
        plt.title('eps_T')

        #===================
        # Tangential damage
        #===================
        plt.subplot(235, projection='polar')
        for i in peaks:
            #plt.polar(rads, w_1_T[i, :])
            plt.plot(rads, omegaT[i, :])
        plt.title('omegaT')

        #===================
        # Tangential plastic strain
        #===================
        plt.subplot(236, projection='polar')
        for i in peaks:
            plt.plot(rads, eps_pi_T[i, :])
        plt.title('eps_pi_T')

        plt.show()
