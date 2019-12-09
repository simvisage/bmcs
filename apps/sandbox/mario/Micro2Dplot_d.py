'''
Created on 04.07.2019
@author: Abdul
plotting tool for microplane models
'''

import matplotlib.pyplot as plt
import numpy as np


class Micro2Dplot_d():

    def get_2Dviz(self, n_mp, eps_max, omega, norm_eps_T, eps_N):

        rads = np.arange(0, (2 * np.pi), (2 * np.pi) / n_mp)

        'Plotting in cycles'
        #===================
        # tangential strain
        #===================

        peaks = np.arange(len(eps_max))

        plt.subplot(221, projection='polar')
        for i in peaks:
            #plt.polar(rads, w_1_T[i, :])
            plt.plot(rads, norm_eps_T[i, :])
        plt.title('eps_T')

        #===================
        # normal strain
        #===================

        plt.subplot(222, projection='polar')
        for i in peaks:
            #print('idx', idx.shape)
            plt.plot(rads, eps_N[i, :])
        #plt.ylim(-1.2 * np.max(np.abs(eps_max)), 0.8 * np.max(np.abs(eps_max)))
        plt.title('eps_N')

        #===================
        # max eps
        #===================
        plt.subplot(223, projection='polar')
        for i in peaks:
            #plt.polar(rads, w_1_T[i, :])
            plt.plot(rads, eps_max[i, :])
        plt.title('eps max')

        #===================
        # damage
        #===================
        plt.subplot(224, projection='polar')
        for i in peaks:
            #plt.polar(rads, w_1_T[i, :])
            plt.plot(rads, omega[i, :])
        plt.title('omega')

        plt.show()
