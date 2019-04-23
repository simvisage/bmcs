''''
Created on 20.10.2018

@author: Mario Aguilar
'''

import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    import os

    home_dir = os.path.expanduser('~')
    path = os.path.join(home_dir, 'Uniaxial Desmorat')
    path = os.path.join(path, 'Original parameters')
    #path = os.path.join(path, 'data')
    path = os.path.join(path, 'Wohler')

    N_0 = np.load(os.path.join(path, 'N_S0.npy'))
    S_0 = np.load(os.path.join(path, 'S_maxc0.npy'))

    N_01 = np.load(os.path.join(path, 'N_S0.1.npy'))
    S_01 = np.load(os.path.join(path, 'S_maxc0.1.npy'))

    N_02 = np.load(os.path.join(path, 'N_S0.2.npy'))
    S_02 = np.load(os.path.join(path, 'S_maxc0.2.npy'))

    N_04 = np.load(os.path.join(path, 'N_S0.4.npy'))
    S_04 = np.load(os.path.join(path, 'S_maxc0.4.npy'))

#     N_01_ = np.load(os.path.join(path, 'N_S_10.npy'))
#     S_01_ = np.load(os.path.join(path, 'S_max_10.npy'))


#     N_02_ = np.load(os.path.join(path, 'eps_m_0.npy'))
#     S_02_ = np.load(os.path.join(path, 'sigma_m_0.npy'))

#     N_04_ = np.load(os.path.join(path, 'N_Sc-0.4.npy'))
#     S_04_ = np.load(os.path.join(path, 'S_maxc-0.4.npy'))

    plt.subplot(111)
    axes = plt.gca()
    axes.set_ylim([0, 1.2])
    plt.semilogx(N_0, S_0, 'k')
    plt.semilogx(N_01, S_01, 'r')
    plt.semilogx(N_02, S_02, 'b')
    plt.semilogx(N_04, S_04, 'g')
#     plt.semilogx(N_01_, S_01_, 'y')
    #plt.plot(N_02_, S_02_, 'b')
    #plt.semilogx(N_04_, S_04_, 'y')
    plt.xlabel('Number of cycles')
    plt.ylabel('Smax')

    plt.show()
