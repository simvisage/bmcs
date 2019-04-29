'''
Created on 05.03.2019

@author: Mario Aguilar Rueda
'''

import csv
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# obtaining data
name = 'CT80-42_3610_Zykl'
name2 = 'CT80-42_3610_Zykl'

name = 'CT80-39_6322_Zykl'
name2 = 'CT80-39_6322_Zykl'

home_dir = os.path.expanduser('~')
path_master = os.path.join(home_dir, 'Data Processing')
path_master = os.path.join(path_master, 'CT')
path_master = os.path.join(path_master, 'C80')
path_master = os.path.join(path_master, name)

path = os.path.join(path_master, name + '.csv')

path2 = os.path.join(path_master, name + '.hdf5')
f_1 = np.zeros((1, 1))
disp_1 = np.zeros((1, 1))
disp_2 = np.zeros((1, 1))
disp_3 = np.zeros((1, 1))
disp_4 = np.zeros((1, 1))
# disp_5 = np.zeros((1, 1))
f = open(path)
l = sum(1 for row in f)
# obtains number of data chunks, 1e5 rows per chunk
num_iter = np.int(np.floor(l / 10000))


# First chunk
disp1 = np.array(pd.read_hdf(
    path2, 'first', columns=['c']))
disp1 = disp1.astype(np.float)

disp2 = np.array(pd.read_hdf(
    path2, 'first', columns=['d']))
disp2 = disp2.astype(np.float)

disp3 = np.array(pd.read_hdf(
    path2, 'first', columns=['e']))
disp3 = disp3.astype(np.float)

disp4 = np.array(pd.read_hdf(
    path2, 'first', columns=['f']))
disp4 = disp4.astype(np.float)

# disp5 = np.array(pd.read_hdf(
#     path2, 'first', columns=['g']))
# disp5 = disp5.astype(np.float)

f = np.array(pd.read_hdf(path2, 'first', columns=['b']))
f = f.astype(np.float)

disp1_aux = np.reshape(
    (np.array(np.concatenate((max(disp1), min(disp1))))), (2, 1))
f_aux = np.reshape((np.array(np.concatenate((max(f), min(f))))), (2, 1))

f_1 = np.concatenate((f_1, f))
disp_1 = np.concatenate((disp_1, disp1))
disp_2 = np.concatenate((disp_2, disp2))
disp_3 = np.concatenate((disp_3, disp3))
disp_4 = np.concatenate((disp_4, disp4))
# disp_5 = np.concatenate((disp_5, disp5))


del f
del disp1
del disp2
del disp3
del disp4
# del disp5
del disp1_aux
del f_aux


# Intermidiate chunks, same procedure
for iter_num in range(num_iter - 1):

    disp1 = np.array(pd.read_hdf(
        path2, 'middle' + np.str(iter_num), columns=['c']))
    disp1 = disp1.astype(np.float)

    disp2 = np.array(pd.read_hdf(
        path2, 'middle' + np.str(iter_num), columns=['d']))
    disp2 = disp2.astype(np.float)

    disp3 = np.array(pd.read_hdf(
        path2, 'middle' + np.str(iter_num), columns=['e']))
    disp3 = disp3.astype(np.float)

    disp4 = np.array(pd.read_hdf(
        path2, 'middle' + np.str(iter_num), columns=['f']))
    disp4 = disp4.astype(np.float)

#     disp5 = np.array(pd.read_hdf(
#         path2, 'middle' + np.str(iter_num), columns=['g']))
#     disp5 = disp5.astype(np.float)

    f = np.array(pd.read_hdf(path2, 'middle' +
                             np.str(iter_num), columns=['b']))
    f = f.astype(np.float)

    disp1_aux = np.reshape(
        (np.array(np.concatenate((max(disp1), min(disp1))))), (2, 1))

    f_aux = np.reshape((np.array(np.concatenate((max(f), min(f))))), (2, 1))

    f_1 = np.concatenate((f_1, f))
    disp_1 = np.concatenate((disp_1, disp1))
    disp_2 = np.concatenate((disp_2, disp2))
    disp_3 = np.concatenate((disp_3, disp3))
    disp_4 = np.concatenate((disp_4, disp4))
#     disp_5 = np.concatenate((disp_5, disp5))

    del f
    del disp1
    del disp2
    del disp3
    del disp4
#     del disp5
    del disp1_aux
    del f_aux


# same procedure for last chunk
disp1 = np.array(pd.read_hdf(
    path2, 'last', columns=['c']))
print(disp1)
disp1 = disp1.astype(np.float)

disp2 = np.array(pd.read_hdf(
    path2, 'last', columns=['d']))
disp2 = disp2.astype(np.float)

disp3 = np.array(pd.read_hdf(
    path2, 'last', columns=['e']))
disp3 = disp3.astype(np.float)

disp4 = np.array(pd.read_hdf(
    path2, 'last', columns=['f']))
disp4 = disp4.astype(np.float)

# disp5 = np.array(pd.read_hdf(
#     path2, 'last', columns=['g']))
# disp5 = disp5.astype(np.float)


f = np.array(pd.read_hdf(
    path2, 'last', columns=['b']))
f = f.astype(np.float)

disp1_aux = np.reshape(
    (np.array(np.concatenate((max(disp1), min(disp1))))), (2, 1))
f_aux = np.reshape((np.array(np.concatenate((max(f), min(f))))), (2, 1))

f_1 = np.concatenate((f_1, f))
disp_1 = np.concatenate((disp_1, disp1))
disp_2 = np.concatenate((disp_2, disp2))
disp_3 = np.concatenate((disp_3, disp3))
disp_4 = np.concatenate((disp_4, disp4))
# disp_5 = np.concatenate((disp_5, disp5))


if name != name2:

    path = os.path.join(path_master, name2 + '.csv')

    path2 = os.path.join(path_master, name2 + '.hdf5')

    f = open(path)
    l = sum(1 for row in f)
    # obtains number of data chunks, 1e5 rows per chunk
    num_iter = np.int(np.floor(l / 1000000))

    # First chunk
    disp1 = np.array(pd.read_hdf(
        path2, 'first', columns=['c']))
    disp1 = disp1.astype(np.float)

    disp2 = np.array(pd.read_hdf(
        path2, 'first', columns=['d']))
    disp2 = disp2.astype(np.float)

    disp3 = np.array(pd.read_hdf(
        path2, 'first', columns=['e']))
    disp3 = disp3.astype(np.float)

    disp4 = np.array(pd.read_hdf(
        path2, 'first', columns=['f']))
    disp4 = disp4.astype(np.float)

#     disp5 = np.array(pd.read_hdf(
#         path2, 'first', columns=['g']))
#     disp5 = disp5.astype(np.float)

    f = np.array(pd.read_hdf(path2, 'first', columns=['b']))
    f = f.astype(np.float)

    disp1_aux = np.reshape(
        (np.array(np.concatenate((max(disp1), min(disp1))))), (2, 1))
    f_aux = np.reshape((np.array(np.concatenate((max(f), min(f))))), (2, 1))

    f_1 = np.concatenate((f_1, f))
    disp_1 = np.concatenate((disp_1, disp1))
    disp_2 = np.concatenate((disp_2, disp2))
    disp_3 = np.concatenate((disp_3, disp3))
    disp_4 = np.concatenate((disp_4, disp4))
#     disp_5 = np.concatenate((disp_5, disp5))

    del f
    del disp1
    del disp2
    del disp3
    del disp4
#     del disp5
    del disp1_aux
    del f_aux

    # Intermidiate chunks, same procedure
    for iter_num in range(num_iter - 1):
        disp1 = np.array(pd.read_hdf(
            path2, 'middle' + np.str(iter_num), columns=['c']))
        disp1 = disp1.astype(np.float)

        disp2 = np.array(pd.read_hdf(
            path2, 'middle' + np.str(iter_num), columns=['d']))
        disp2 = disp2.astype(np.float)

        disp3 = np.array(pd.read_hdf(
            path2, 'middle' + np.str(iter_num), columns=['e']))
        disp3 = disp3.astype(np.float)

        disp4 = np.array(pd.read_hdf(
            path2, 'middle' + np.str(iter_num), columns=['f']))
        disp4 = disp4.astype(np.float)

#         disp5 = np.array(pd.read_hdf(
#             path2, 'middle' + np.str(iter_num), columns=['g']))
#         disp5 = disp5.astype(np.float)

        f = np.array(pd.read_hdf(path2, 'middle' +
                                 np.str(iter_num), columns=['b']))
        f = f.astype(np.float)

        disp1_aux = np.reshape(
            (np.array(np.concatenate((max(disp1), min(disp1))))), (2, 1))

        f_aux = np.reshape(
            (np.array(np.concatenate((max(f), min(f))))), (2, 1))

        f_1 = np.concatenate((f_1, f))
        disp_1 = np.concatenate((disp_1, disp1))
        disp_2 = np.concatenate((disp_2, disp2))
        disp_3 = np.concatenate((disp_3, disp3))
        disp_4 = np.concatenate((disp_4, disp4))
#         disp_5 = np.concatenate((disp_5, disp5))

        del f
        del disp1
        del disp2
        del disp3
        del disp4
#         del disp5
        del disp1_aux
        del f_aux

    # same procedure for last chunk
    disp1 = np.array(pd.read_hdf(
        path2, 'last', columns=['c']))
    disp1 = disp1.astype(np.float)

    disp2 = np.array(pd.read_hdf(
        path2, 'last', columns=['d']))
    disp2 = disp2.astype(np.float)

    disp3 = np.array(pd.read_hdf(
        path2, 'last', columns=['e']))
    disp3 = disp3.astype(np.float)

    disp4 = np.array(pd.read_hdf(
        path2, 'last', columns=['f']))
    disp4 = disp4.astype(np.float)

#     disp5 = np.array(pd.read_hdf(
#         path2, 'last', columns=['g']))
#     disp5 = disp5.astype(np.float)

    f = np.array(pd.read_hdf(
        path2, 'last', columns=['b']))
    f = f.astype(np.float)

    disp1_aux = np.reshape(
        (np.array(np.concatenate((max(disp1), min(disp1))))), (2, 1))
    f_aux = np.reshape((np.array(np.concatenate((max(f), min(f))))), (2, 1))

    f_1 = np.concatenate((f_1, f))
    disp_1 = np.concatenate((disp_1, disp1))
    disp_2 = np.concatenate((disp_2, disp2))
    disp_3 = np.concatenate((disp_3, disp3))
    disp_4 = np.concatenate((disp_4, disp4))
#     disp_5 = np.concatenate((disp_5, disp5))

path_master = os.path.join(path_master, 'NPY')
if os.path.exists(path_master) == False:
    os.makedirs(path_master)
np.save(os.path.join(path_master, name + '_Force_nofilter.npy'), f_1)

np.save(os.path.join(path_master, name +
                     '_Displacement_machine_nofilter.npy'), disp_1)
np.save(os.path.join(path_master, name +
                     '_Displacement_sliding1_nofilter.npy'), disp_2)
np.save(os.path.join(path_master, name +
                     '_Displacement_sliding2_nofilter.npy'), disp_3)
np.save(os.path.join(path_master, name +
                     '_Displacement_crack1_nofilter.npy'), disp_4)
# np.save(os.path.join(path_master, name +
#                      '_Displacement_crack2_nofilter.npy'), disp_5)


mpl.rcParams['agg.path.chunksize'] = 10000
plt.subplot(111)
plt.xlabel('Displacement [mm]')
plt.ylabel('kN')
plt.title('original data', fontsize=20)
#plt.xlim(-0.1, 1.5)
plt.plot(disp_2, f_1, 'k')
# plt.plot(disp_3, f_1, 'r')
# plt.plot(disp_4, f_1, 'g')
# plt.plot(disp_2[:5000], f_1[:5000], 'k')
# plt.plot(disp_2[1000000:1005000], f_1[1000000:1005000], 'k')
# plt.plot(disp_2[2000000:2005000], f_1[2000000:2005000], 'k')
# plt.plot(disp_2[3000000:3005000], f_1[3000000:3005000], 'k')
# plt.plot(disp_2[4000000:4005000], f_1[4000000:4005000], 'k')
# plt.plot(disp_2[-5000:], f_1[-5000:], 'k')

# plt.subplot(122)
# plt.xlabel('Displacement [mm]')
# plt.ylabel('kN')
# plt.title('original data unloaded end', fontsize=20)
# plt.plot(disp_3, f_1, 'k')
#plt.plot(disp_3, f_1, 'r')
plt.show()
