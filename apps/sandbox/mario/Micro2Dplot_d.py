'''
Created on 04.07.2019
@author: Abdul
plotting tool for microplane models
'''

import matplotlib.pyplot as plt
import numpy as np


def get_2Dviz(S, n_mp):

    eps_max = S[:, :, 0],
    omega = S[:, :, 1]
    eps_T = S[:, :, 2:4]
    eps_N = S[:, :, 4]
    norm_eps_T = np.zeros((len(eps_T), len(eps_T[1])))
    for i in range(len(eps_T)):
        norm_eps_T[i] = np.sqrt(
            np.einsum('...i,...i->... ', S[i, :, 2:4], S[i, :, 2:4]))

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
