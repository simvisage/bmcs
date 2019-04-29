''''
Created on 20.10.2018

@author: Mario Aguilar
'''

import matplotlib as mpl
import numpy as np


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os

    name = 'CT80-42_3610_Zykl'
    name = 'CT80-39_6322_Zykl'

    home_dir = os.path.expanduser('~')
    path_master = os.path.join(home_dir, 'Data Processing')
    path_master = os.path.join(path_master, 'CT')
    path_master = os.path.join(path_master, 'C80')
    path_master = os.path.join(path_master, name)
    path_master = os.path.join(path_master, 'NPY')
    #not filtered
    F_original = np.load(os.path.join(
        path_master, name + '_Force_nofilter.npy'))
    disp1_original = np.load(os.path.join(path_master, name +
                                          '_Displacement_machine_nofilter.npy'))
    disp2_original = np.load(os.path.join(path_master, name +
                                          '_Displacement_sliding1_nofilter.npy'))
    disp3_original = np.load(os.path.join(path_master, name +
                                          '_Displacement_sliding2_nofilter.npy'))
    disp4_original = np.load(os.path.join(path_master, name +
                                          '_Displacement_crack1_nofilter.npy'))
#     disp5_original = np.load(os.path.join(path_master, name +
#                                           '_Displacement_crack2_nofilter.npy'))

    # filered
    F_max1 = np.load(os.path.join(
        path_master, name + '_Force1_max.npy'))
    F_max2 = np.load(os.path.join(
        path_master, name + '_Force2_max.npy'))
    disp_max1 = np.load(os.path.join(
        path_master, name + '_Creep_displacement_machine_max.npy'))
    disp_max2 = np.load(os.path.join(
        path_master, name + '_Creep_displacement_sliding1_max.npy'))
    disp_max3 = np.load(os.path.join(
        path_master, name + '_Creep_displacement_sliding2_max.npy'))
    disp_max4 = np.load(os.path.join(
        path_master, name + '_Creep_displacement_crack1_max.npy'))
#     disp_max5 = np.load(os.path.join(
#         path_master, name + '_Creep_displacement_crack2_max.npy'))

    F_min1 = np.load(os.path.join(
        path_master, name + '_Force1_min.npy'))
    F_min2 = np.load(os.path.join(
        path_master, name + '_Force2_min.npy'))

    disp_min1 = np.load(os.path.join(
        path_master, name + '_Creep_displacement_machine_min.npy'))
    disp_min2 = np.load(os.path.join(
        path_master, name + '_Creep_displacement_sliding1_min.npy'))
    disp_min3 = np.load(os.path.join(
        path_master, name + '_Creep_displacement_sliding2_min.npy'))
    disp_min4 = np.load(os.path.join(
        path_master, name + '_Creep_displacement_crack1_min.npy'))
#     disp_min5 = np.load(os.path.join(
#         path_master, name + '_Creep_displacement_crack2_min.npy'))

    N_max1 = np.load(os.path.join(
        path_master, name + '_Creep_n_load_max1.npy'))
    N_max2 = np.load(os.path.join(
        path_master, name + '_Creep_n_load_max2.npy'))
    N_min1 = np.load(os.path.join(
        path_master, name + '_Creep_n_load_min1.npy'))
    N_min2 = np.load(os.path.join(
        path_master, name + '_Creep_n_load_min2.npy'))

    F1 = np.load(os.path.join(
        path_master, name + '_Force1.npy'))
    F2 = np.load(os.path.join(
        path_master, name + '_Force2.npy'))
    disp1 = np.load(os.path.join(
        path_master, name + '_Displacement_machine.npy'))
    disp2 = np.load(os.path.join(
        path_master, name + '_Displacement_sliding1.npy'))
    disp3 = np.load(os.path.join(
        path_master, name + '_Displacement_sliding2.npy'))
    disp4 = np.load(os.path.join(
        path_master, name + '_Displacement_crack1.npy'))
#     disp5 = np.load(os.path.join(
#         path_master, name + '_Displacement_crack2.npy'))
    mpl.rcParams['agg.path.chunksize'] = 10000

    plt.figure(num=name)

    plt.subplot(131)

    plt.plot(disp2_original, F_original, 'k')
    plt.xlabel('Displacement [mm]')
    plt.ylabel('kN')
    plt.title('original data', fontsize=20)
    plt.xlim(0, 1.4)
    plt.ylim(-800, 0)

    plt.subplot(132)

    plt.plot(disp2, F1, 'k')
    plt.xlabel('Displacement [mm]')
    plt.ylabel('kN')
    plt.title('filtered data', fontsize=20)
    plt.xlim(0, 1.4)
    plt.ylim(-800, 0)

    plt.subplot(133)

    plt.plot(N_min1, abs(disp_min2), 'k')
    plt.plot(N_max1, abs(disp_max2), 'k')

    plt.plot(N_min1, abs(disp_min3), 'r')
    plt.plot(N_max1, abs(disp_max3), 'r')

    plt.plot(N_min1, abs(disp_min4), 'g')
    plt.plot(N_max1, abs(disp_max4), 'g')

    plt.xlabel('N')
    plt.ylabel('Displacement [mm]')
    plt.title('creep sliding', fontsize=20)
    #plt.xlim(0, 0.8)

#     plt.subplot(234)
#
#     plt.plot(disp3_original, F_original, 'k')
#     plt.xlabel('Displacement [mm]')
#     plt.ylabel('kN')
#     plt.title('original data unloaded end', fontsize=20)
#     plt.xlim(-0.3, 0)
#     plt.ylim(-170, 0)
#
#     plt.subplot(235)
#
#     plt.plot(disp3, F1, 'k')
#     plt.xlabel('Displacement [mm]')
#     plt.ylabel('kN')
#     plt.title('filtered data unloaded end', fontsize=20)
#     plt.xlim(-0.3, 0)
#     plt.ylim(-170, 0)
#
#     plt.subplot(236)
#
#     plt.plot(N_min2, abs(disp_min3), 'k')
#     plt.plot(N_max2, abs(disp_max3), 'r')
#
#     plt.xlabel('N')
#     plt.ylabel('Displacement [mm]')
#     plt.title('creep sliding unloaded end', fontsize=20)
#     #plt.xlim(0, 0.8)

    plt.show()
