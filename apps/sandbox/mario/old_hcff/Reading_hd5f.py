'''
Created on 05.03.2019

@author: Mario Aguilar Rueda
'''

import csv
import os

import h5py


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

# Tuning
max_min = 38
min_max = 44
initial_f = 30
final_disp = 1.4

# Tuning for LS3
max_min1 = 158.1
max_min2 = 400
max_min3 = 500
min_max1 = 646.7
min_max2 = 650
min_max3 = 750


f_1 = np.zeros((1, 1)).reshape(-1)
f_2 = np.zeros((1, 1)).reshape(-1)
disp_1 = np.zeros((1, 1)).reshape(-1)
disp_2 = np.zeros((1, 1)).reshape(-1)
disp_3 = np.zeros((1, 1)).reshape(-1)
disp_4 = np.zeros((1, 1)).reshape(-1)
# disp_5 = np.zeros((1, 1)).reshape(-1)
temp = np.empty((1, 1))
f = open(path)
l = sum(1 for row in f)
# obtains number of data chunks, 1e5 rows per chunk
num_iter = np.int(np.floor(l / 10000))


# First chunk
disp1 = np.array(pd.read_hdf(
    path2, 'first', columns=['c']))
disp2 = np.array(pd.read_hdf(
    path2, 'first', columns=['d']))
disp3 = np.array(pd.read_hdf(
    path2, 'first', columns=['e']))
disp4 = np.array(pd.read_hdf(
    path2, 'first', columns=['f']))
# disp5 = np.array(pd.read_hdf(
#     path2, 'first', columns=['g']))


# loads first chunk, skipping first loading branch
disp1 = disp1.astype(np.float)
disp2 = disp2.astype(np.float)
disp3 = disp3.astype(np.float)
disp4 = disp4.astype(np.float)
# disp5 = disp5.astype(np.float)

f = np.array(pd.read_hdf(path2, 'first', columns=['b']))
f = f.astype(np.float)


# deleting first loading branch, there is a bunch of noise
idx = np.where(np.abs(f) > initial_f)
a = idx[0]
idx = np.zeros((1, 1))

f_aux = f[np.int(a[0]):]
disp_aux1 = disp1[np.int(a[0]):]
disp_aux2 = disp2[np.int(a[0]):]
disp_aux3 = disp3[np.int(a[0]):]
disp_aux4 = disp4[np.int(a[0]):]
# disp_aux5 = disp5[np.int(a[0]):]


f_diff = np.abs(f_aux[1:]) - np.abs(f_aux[:-1])
g_diff = np.array(f_diff[1:] * f_diff[:-1])
idx1 = np.array(np.where((g_diff) <= 0))
idx1 = idx1[0] + 1

idx2 = idx1[1:] - idx1[0:-1]
idx3 = np.where(np.abs(idx2) < 1)
idx1 = list(idx1)

for index in sorted(idx3[0], reverse=True):
    del idx1[np.int(index)]

disp_aux1 = disp_aux1[idx1].reshape(-1)
disp_aux2 = disp_aux2[idx1].reshape(-1)
disp_aux3 = disp_aux3[idx1].reshape(-1)
disp_aux4 = disp_aux4[idx1].reshape(-1)
# disp_aux5 = disp_aux5[idx1].reshape(-1)
f_aux1 = f_aux[idx1].reshape(-1)
f_aux2 = f_aux[idx1].reshape(-1)

f_1 = np.concatenate((f_1, f_aux1))
f_2 = np.concatenate((f_2, f_aux2))
disp_1 = np.concatenate((disp_1, disp_aux1))
disp_2 = np.concatenate((disp_2, disp_aux2))
disp_3 = np.concatenate((disp_3, disp_aux3))
disp_4 = np.concatenate((disp_4, disp_aux4))
# disp_5 = np.concatenate((disp_5, disp_aux5))


del disp1
del disp2
del disp3
del disp4
# del disp5
del disp_aux1
del disp_aux2
del disp_aux3
del disp_aux4
# del disp_aux5
del f_aux1
del f_aux2
del idx1
del idx2
del idx3
del idx
del f


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

    f_diff = np.abs(f[1:]) - np.abs(f[:-1])
    g_diff = np.array(f_diff[1:] * f_diff[:-1])
    idx1 = np.array(np.where((g_diff) <= 0))
    idx1 = idx1[0] + 1
    idx2 = idx1[1:] - idx1[0:-1]
    idx3 = np.where(np.abs(idx2) < 1)
    idx1 = list(idx1)

    for index in sorted(idx3[0], reverse=True):
        del idx1[np.int(index)]

    idx = np.zeros((1, 1))
    temp = np.empty((1, 1))

    disp_aux1 = disp1[idx1].reshape(-1)
    disp_aux2 = disp2[idx1].reshape(-1)
    disp_aux3 = disp3[idx1].reshape(-1)
    disp_aux4 = disp4[idx1].reshape(-1)
#     disp_aux5 = disp5[idx1].reshape(-1)
    f_aux1 = f[idx1].reshape(-1)
    f_aux2 = f[idx1].reshape(-1)

    f_1 = np.concatenate((f_1, f_aux1))
    f_2 = np.concatenate((f_2, f_aux2))
    disp_1 = np.concatenate((disp_1, disp_aux1))
    disp_2 = np.concatenate((disp_2, disp_aux2))
    disp_3 = np.concatenate((disp_3, disp_aux3))
    disp_4 = np.concatenate((disp_4, disp_aux4))
#     disp_5 = np.concatenate((disp_5, disp_aux5))

    del disp_aux1
    del disp_aux2
    del disp_aux3
    del disp_aux4
#     del disp_aux5
    del f_aux1
    del f_aux2
    del idx1
    del idx2
    del idx3
    del idx
    del f
    del disp1
    del disp2
    del disp3
    del disp4
#     del disp5

# same procedure for last chunk
disp1 = np.array(pd.read_hdf(path2, 'last', columns=['c']))
disp1 = disp1.astype(np.float)

disp2 = np.array(pd.read_hdf(path2, 'last', columns=['d']))
disp2 = disp2.astype(np.float)

disp3 = np.array(pd.read_hdf(path2, 'last', columns=['e']))
disp3 = disp3.astype(np.float)

disp4 = np.array(pd.read_hdf(path2, 'last', columns=['f']))
disp4 = disp4.astype(np.float)

# disp5 = np.array(pd.read_hdf(path2, 'last', columns=['g']))
# disp5 = disp5.astype(np.float)

f = np.array(pd.read_hdf(
    path2, 'last', columns=['b']))
f = f.astype(np.float)

f_diff = np.abs(f[1:]) - np.abs(f[:-1])
g_diff = np.array(f_diff[1:] * f_diff[:-1])
idx1 = np.array(np.where((g_diff) < 0))
idx1 = idx1[0] + 1

idx2 = idx1[1:] - idx1[0:-1]
idx3 = np.where(np.abs(idx2) < 1)
idx1 = list(idx1)

for index in sorted(idx3[0], reverse=True):
    del idx1[np.int(index)]

disp_aux1 = disp1[idx1].reshape(-1)
disp_aux2 = disp2[idx1].reshape(-1)
disp_aux3 = disp3[idx1].reshape(-1)
disp_aux4 = disp4[idx1].reshape(-1)
# disp_aux5 = disp5[idx1].reshape(-1)
f_aux1 = f[idx1].reshape(-1)
f_aux2 = f[idx1].reshape(-1)


f_1 = np.concatenate((f_1, f_aux1))
f_2 = np.concatenate((f_2, f_aux2))
disp_1 = np.concatenate((disp_1, disp_aux1))
disp_2 = np.concatenate((disp_2, disp_aux2))
disp_3 = np.concatenate((disp_3, disp_aux3))
disp_4 = np.concatenate((disp_4, disp_aux4))
# disp_5 = np.concatenate((disp_5, disp_aux5))


del disp_aux1
del disp_aux2
del disp_aux3
del disp_aux4
del f_aux1
del f_aux2
del idx1
del idx2
del idx3
del f
del disp1
del disp2
del disp3
del disp4

if name2 != name:
    path = os.path.join(path_master, name2 + '.csv')
    path2 = os.path.join(path_master, name2 + '.hdf5')
    f = open(path)
    l = sum(1 for row in f)
    # obtains number of data chunks, 1e5 rows per chunk
    num_iter = np.int(np.floor(l / 1000000))

    # First chunk
    disp1 = np.array(pd.read_hdf(
        path2, 'first', columns=['c']))
    disp2 = np.array(pd.read_hdf(
        path2, 'first', columns=['d']))
    disp3 = np.array(pd.read_hdf(
        path2, 'first', columns=['e']))
    disp4 = np.array(pd.read_hdf(
        path2, 'first', columns=['f']))
#     disp5 = np.array(pd.read_hdf(
#         path2, 'first', columns=['g']))

    # loads first chunk, skipping first loading branch
    disp1 = disp1.astype(np.float)
    disp2 = disp2.astype(np.float)
    disp3 = disp3.astype(np.float)
    disp4 = disp4.astype(np.float)
#     disp5 = disp5.astype(np.float)

    f = np.array(pd.read_hdf(path2, 'first', columns=['b']))
    f = f.astype(np.float)

    f_aux = f
    disp_aux1 = disp1
    disp_aux2 = disp2
    disp_aux3 = disp3
    disp_aux4 = disp4
#     disp_aux5 = disp5

    f_diff = np.abs(f_aux[1:]) - np.abs(f_aux[:-1])
    g_diff = np.array(f_diff[1:] * f_diff[:-1])
    idx1 = np.array(np.where((g_diff) <= 0))
    idx1 = idx1[0] + 1

    idx2 = idx1[1:] - idx1[0:-1]
    idx3 = np.where(np.abs(idx2) == 1)
    idx1 = list(idx1)

    for index in sorted(idx3[0], reverse=True):
        del idx1[np.int(index)]

    disp_aux1 = disp_aux1[idx1].reshape(-1)
    disp_aux2 = disp_aux2[idx1].reshape(-1)
    disp_aux3 = disp_aux3[idx1].reshape(-1)
    disp_aux4 = disp_aux4[idx1].reshape(-1)
#     disp_aux5 = disp_aux5[idx1].reshape(-1)
    f_aux1 = f_aux[idx1].reshape(-1)
    f_aux2 = f_aux[idx1].reshape(-1)

    f_1 = np.concatenate((f_1, f_aux1))
    f_2 = np.concatenate((f_2, f_aux2))
    disp_1 = np.concatenate((disp_1, disp_aux1))
    disp_2 = np.concatenate((disp_2, disp_aux2))
    disp_3 = np.concatenate((disp_3, disp_aux3))
    disp_4 = np.concatenate((disp_4, disp_aux4))
#     disp_5 = np.concatenate((disp_5, disp_aux5))

    del disp1
    del disp2
    del disp3
    del disp4
#     del disp5
    del disp_aux1
    del disp_aux2
    del disp_aux3
    del disp_aux4
#     del disp_aux5
    del f_aux
    del f_aux1
    del f_aux2
    del idx1
    del idx2
    del idx3
    del f

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

        disp5 = np.array(pd.read_hdf(
            path2, 'middle' + np.str(iter_num), columns=['g']))
        disp5 = disp5.astype(np.float)

        f = np.array(pd.read_hdf(path2, 'middle' +
                                 np.str(iter_num), columns=['b']))
        f = f.astype(np.float)

        f_diff = np.abs(f[1:]) - np.abs(f[:-1])
        g_diff = np.array(f_diff[1:] * f_diff[:-1])
        idx1 = np.array(np.where((g_diff) <= 0))
        idx1 = idx1[0] + 1
        idx2 = idx1[1:] - idx1[0:-1]
        idx3 = np.where(np.abs(idx2) == 1)
        idx1 = list(idx1)

        for index in sorted(idx3[0], reverse=True):
            del idx1[np.int(index)]

        idx = np.zeros((1, 1))
        temp = np.empty((1, 1))

        disp_aux1 = disp1[idx1].reshape(-1)
        disp_aux2 = disp2[idx1].reshape(-1)
        disp_aux3 = disp3[idx1].reshape(-1)
        disp_aux4 = disp4[idx1].reshape(-1)
        disp_aux5 = disp5[idx1].reshape(-1)
        f_aux1 = f[idx1].reshape(-1)
        f_aux2 = f[idx1].reshape(-1)

        f_1 = np.concatenate((f_1, f_aux1))
        f_2 = np.concatenate((f_2, f_aux2))
        disp_1 = np.concatenate((disp_1, disp_aux1))
        disp_2 = np.concatenate((disp_2, disp_aux2))
        disp_3 = np.concatenate((disp_3, disp_aux3))
        disp_4 = np.concatenate((disp_4, disp_aux4))
#         disp_5 = np.concatenate((disp_5, disp_aux5))

        del disp_aux1
        del disp_aux2
        del disp_aux3
        del disp_aux4
        del disp_aux5
        del f_aux1
        del f_aux2
        del idx1
        del idx2
        del idx3
        del idx
        del f
        del disp1
        del disp2
        del disp3
        del disp4
        del disp5

    # same procedure for last chunk
    disp1 = np.array(pd.read_hdf(path2, 'last', columns=['c']))
    disp1 = disp1.astype(np.float)

    disp2 = np.array(pd.read_hdf(path2, 'last', columns=['d']))
    disp2 = disp2.astype(np.float)

    disp3 = np.array(pd.read_hdf(path2, 'last', columns=['e']))
    disp3 = disp3.astype(np.float)

    disp4 = np.array(pd.read_hdf(path2, 'last', columns=['f']))
    disp4 = disp4.astype(np.float)

#     disp5 = np.array(pd.read_hdf(path2, 'last', columns=['g']))
#     disp5 = disp5.astype(np.float)

    f = np.array(pd.read_hdf(
        path2, 'last', columns=['b']))
    f = f.astype(np.float)

    f_diff = np.abs(f[1:]) - np.abs(f[:-1])
    g_diff = np.array(f_diff[1:] * f_diff[:-1])
    idx1 = np.array(np.where((g_diff) < 0))
    idx1 = idx1[0] + 1

    idx2 = idx1[1:] - idx1[0:-1]
    idx3 = np.where(np.abs(idx2) == 1)
    idx1 = list(idx1)

    for index in sorted(idx3[0], reverse=True):
        del idx1[np.int(index)]

    disp_aux1 = disp1[idx1].reshape(-1)
    disp_aux2 = disp2[idx1].reshape(-1)
    disp_aux3 = disp3[idx1].reshape(-1)
    disp_aux4 = disp4[idx1].reshape(-1)
#     disp_aux5 = disp5[idx1].reshape(-1)
    f_aux1 = f[idx1].reshape(-1)
    f_aux2 = f[idx1].reshape(-1)

    f_1 = np.concatenate((f_1, f_aux1))
    f_2 = np.concatenate((f_2, f_aux2))
    disp_1 = np.concatenate((disp_1, disp_aux1))
    disp_2 = np.concatenate((disp_2, disp_aux2))
    disp_3 = np.concatenate((disp_3, disp_aux3))
    disp_4 = np.concatenate((disp_4, disp_aux4))
#     disp_5 = np.concatenate((disp_5, disp_aux5))

    del disp_aux1
    del disp_aux2
    del disp_aux3
    del disp_aux4
    del f_aux1
    del f_aux2
    del idx1
    del f
    del disp1
    del disp2
    del disp3
    del disp4


idx = np.where(np.abs(disp_2) > final_disp)
a = idx[0]
f_aux = f_1[:a[0]]
idx = np.zeros((1, 1))


# # filtering max and min for good quality tests
# idx_a = np.array(np.where(np.abs(f_aux) > min_max))
# idx = idx_a[0]
# idx = np.array(idx[0])
# start = np.int(idx)
#
# disp_max1 = disp_1[start:len(f_aux):2]
# disp_max2 = disp_2[start:len(f_aux):2]
# disp_max3 = disp_3[start:len(f_aux):2]
# disp_max4 = disp_4[start:len(f_aux):2]
# # disp_max5 = disp_5[start:len(f_aux):2]
# f_1_max = f_1[start:len(f_aux):2]
# f_2_max = f_2[start:len(f_aux):2]
# N_max1 = np.array(range(len(disp_max2)))
# N_max1 = N_max1 / N_max1[-1]
# N_max2 = np.array(range(len(disp_max3)))
# N_max2 = N_max2 / N_max2[-1]
# print(len(disp_max2))
#
# disp_min1 = disp_1[start + 1:len(f_aux):2]
# disp_min2 = disp_2[start + 1:len(f_aux):2]
# disp_min3 = disp_3[start + 1:len(f_aux):2]
# disp_min4 = disp_4[start + 1:len(f_aux):2]
# # disp_min5 = disp_5[start + 1:len(f_aux):2]
# f_1_min = f_1[start + 1:len(f_aux):2]
# f_2_min = f_2[start + 1:len(f_aux):2]
# N_min1 = np.array(range(len(disp_min2)))
# N_min1 = N_min1 / N_min1[-1]
# N_min2 = np.array(range(len(disp_min3)))
# N_min2 = N_min2 / N_min2[-1]
# del idx

# filtering max and min for bad quality tests

idx_a = np.array(np.where(np.abs(f_aux) > min_max1))
idx = idx_a[0]
disp_max1 = disp_1[idx]
disp_max2 = disp_2[idx]
disp_max3 = disp_3[idx]
disp_max4 = disp_4[idx]
# disp_max5 = disp_5[idx]
f_1_max = f_1[idx]
f_2_max = f_2[idx]
N_max1 = np.array(range(len(disp_max2)))
N_max1 = N_max1 / N_max1[-1]
N_max2 = np.array(range(len(disp_max3)))
N_max2 = N_max2 / N_max2[-1]
del idx

idx_a = np.array(np.where(np.abs(f_aux) < max_min1))
idx = idx_a[0]
disp_min1 = disp_1[idx]
disp_min2 = disp_2[idx]
disp_min3 = disp_3[idx]
disp_min4 = disp_4[idx]
# disp_min5 = disp_5[idx]
f_1_min = f_1[idx]
f_2_min = f_2[idx]
N_min1 = np.array(range(len(disp_min2)))
N_min1 = N_min1 / N_min1[-1]
N_min2 = np.array(range(len(disp_min3)))
N_min2 = N_min2 / N_min2[-1]
print(len(disp_min2))


# # filtering max and min for bad quality LS3 tests
# # First chunk
# idx = np.array(np.where(np.abs(f_1) > min_max2))
# idx = idx[0]
# end1 = idx[0]
#
# del idx
# idx = np.array(np.where(np.abs(f_1[0:end1]) > min_max1))
# disp1_max1 = disp_1[idx].reshape(-1)
# disp1_max2 = disp_2[idx].reshape(-1)
# disp1_max3 = disp_3[idx].reshape(-1)
# disp1_max4 = disp_4[idx].reshape(-1)
# # disp1_max5 = disp_5[idx].reshape(-1)
# f1_1_max = f_1[idx].reshape(-1)
# f1_2_max = f_2[idx].reshape(-1)
# del idx
# idx = np.array(np.where(abs(f_1[0:end1]) < max_min1))
# disp1_min1 = disp_1[idx].reshape(-1)
# disp1_min2 = disp_2[idx].reshape(-1)
# disp1_min3 = disp_3[idx].reshape(-1)
# disp1_min4 = disp_4[idx].reshape(-1)
# # disp1_min5 = disp_5[idx].reshape(-1)
# f1_1_min = f_1[idx].reshape(-1)
# f1_2_min = f_2[idx].reshape(-1)
# del idx
# print(disp1_max2[0])
# print(disp1_max2[-1])
#
#
# # second chunk
# idx = np.array(np.where(np.abs(f_1) > min_max3))
# idx = idx[0]
# end2 = idx[0]
#
#
# del idx
# idx = np.array(np.where(np.abs(f_1[end1:end2]) > min_max2))
# idx = idx + end1
# disp2_max1 = disp_1[idx].reshape(-1)
# disp2_max2 = disp_2[idx].reshape(-1)
# disp2_max3 = disp_3[idx].reshape(-1)
# disp2_max4 = disp_4[idx].reshape(-1)
# # disp2_max5 = disp_5[idx].reshape(-1)
# f2_1_max = f_1[idx].reshape(-1)
# f2_2_max = f_2[idx].reshape(-1)
# del idx
# idx = np.array(np.where(abs(f_1[end1:end2]) < max_min2))
# idx = idx + end1
# disp2_min1 = disp_1[idx].reshape(-1)
# disp2_min2 = disp_2[idx].reshape(-1)
# disp2_min3 = disp_3[idx].reshape(-1)
# disp2_min4 = disp_4[idx].reshape(-1)
# # disp2_min5 = disp_5[idx].reshape(-1)
# f2_1_min = f_1[idx].reshape(-1)
# f2_2_min = f_2[idx].reshape(-1)
# del idx
#
#
# print(disp2_max2[0])
# print(disp2_max2[-1])
#
#
# # third chunk
#
# idx = np.array(np.where(np.abs(f_1[end2:]) > min_max3))
# idx = idx + end2
# disp3_max1 = disp_1[idx].reshape(-1)
# disp3_max2 = disp_2[idx].reshape(-1)
# disp3_max3 = disp_3[idx].reshape(-1)
# disp3_max4 = disp_4[idx].reshape(-1)
# # disp3_max5 = disp_5[idx].reshape(-1)
# f3_1_max = f_1[idx].reshape(-1)
# f3_2_max = f_2[idx].reshape(-1)
# del idx
# idx = np.array(
#     np.where((max_min2 < abs(f_1[end2:])) & (abs(f_1[end2:]) < max_min3)))
# idx = idx + end2
# disp3_min1 = disp_1[idx].reshape(-1)
# disp3_min2 = disp_2[idx].reshape(-1)
# disp3_min3 = disp_3[idx].reshape(-1)
# disp3_min4 = disp_4[idx].reshape(-1)
# # disp3_min5 = disp_5[idx].reshape(-1)
# f3_1_min = f_1[idx].reshape(-1)
# f3_2_min = f_2[idx].reshape(-1)
# del idx
#
#
# disp_max1 = np.concatenate((disp1_max1, disp2_max1, disp3_max1))
# disp_max2 = np.concatenate((disp1_max2, disp2_max2, disp3_max2))
# disp_max3 = np.concatenate((disp1_max3, disp2_max3, disp3_max3))
# disp_max4 = np.concatenate((disp1_max4, disp2_max4, disp3_max4))
# # disp_max5 = np.concatenate((disp1_max5, disp2_max5, disp3_max5))
# f_1_max = np.concatenate((f1_1_max, f2_1_max, f3_1_max))
# f_2_max = np.concatenate((f1_2_max, f2_2_max, f3_2_max))
#
# disp_min1 = np.concatenate((disp1_min1, disp2_min1, disp3_min1))
# disp_min2 = np.concatenate((disp1_min2, disp2_min2, disp3_min2))
# disp_min3 = np.concatenate((disp1_min3, disp2_min3, disp3_min3))
# disp_min4 = np.concatenate((disp1_min4, disp2_min4, disp3_min4))
# # disp_min5 = np.concatenate((disp1_min5, disp2_min5, disp3_min5))
# f_1_min = np.concatenate((f1_1_min, f2_1_min, f3_1_min))
# f_2_min = np.concatenate((f1_2_min, f2_2_min, f3_2_min))
#
#
# N_max1 = np.array(range(len(disp_max2)))
# N_max1 = N_max1 / N_max1[-1]
# N_max2 = np.array(range(len(disp_max3)))
# N_max2 = N_max2 / N_max2[-1]
# N_min1 = np.array(range(len(disp_min2)))
# N_min1 = N_min1 / N_min1[-1]
# N_min2 = np.array(range(len(disp_min3)))
# N_min2 = N_min2 / N_min2[-1]
#
# print(disp2_max2[0])
# print(disp2_max2[-1])
# print(len(disp_max2))


path_master = os.path.join(path_master, 'NPY')
if os.path.exists(path_master) == False:
    os.makedirs(path_master)
np.save(os.path.join(path_master, name + '_Force1.npy'), f_1)
np.save(os.path.join(path_master, name + '_Force1_max.npy'), f_1_max)
np.save(os.path.join(path_master, name + '_Force1_min.npy'), f_1_min)
np.save(os.path.join(path_master, name + '_Force2.npy'), f_2)
np.save(os.path.join(path_master, name + '_Force2_max.npy'), f_2_max)
np.save(os.path.join(path_master, name + '_Force2_min.npy'), f_2_min)
np.save(os.path.join(path_master, name + '_Displacement_machine.npy'), disp_1)
np.save(os.path.join(path_master, name + '_Displacement_sliding1.npy'), disp_2)
np.save(os.path.join(path_master, name + '_Displacement_sliding2.npy'), disp_3)
np.save(os.path.join(path_master, name + '_Displacement_crack1.npy'), disp_4)
# np.save(os.path.join(path_master, name + '_Displacement_crack2.npy'), disp_5)
np.save(os.path.join(
    path_master, name + '_Creep_displacement_machine_max.npy'), disp_max1)
np.save(os.path.join(
    path_master, name + '_Creep_displacement_sliding1_max.npy'), disp_max2)
np.save(os.path.join(
    path_master, name + '_Creep_displacement_sliding2_max.npy'), disp_max3)
np.save(os.path.join(
    path_master, name + '_Creep_displacement_crack1_max.npy'), disp_max4)
# np.save(os.path.join(
#     path_master, name + '_Creep_displacement_crack2_max.npy'), disp_max5)
np.save(os.path.join(
    path_master, name + '_Creep_n_load_max1.npy'), N_max1)
np.save(os.path.join(
    path_master, name + '_Creep_n_load_max2.npy'), N_max2)
np.save(os.path.join(
    path_master, name + '_Creep_displacement_machine_min.npy'), disp_min1)
np.save(os.path.join(
    path_master, name + '_Creep_displacement_sliding1_min.npy'), disp_min2)
np.save(os.path.join(
    path_master, name + '_Creep_displacement_sliding2_min.npy'), disp_min3)
np.save(os.path.join(
    path_master, name + '_Creep_displacement_crack1_min.npy'), disp_min4)
# np.save(os.path.join(
#     path_master, name + '_Creep_displacement_crack2_min.npy'), disp_min5)
np.save(os.path.join(
    path_master, name + '_Creep_n_load_min1.npy'), N_min1)
np.save(os.path.join(
    path_master, name + '_Creep_n_load_min2.npy'), N_min2)


mpl.rcParams['agg.path.chunksize'] = 10000

plt.subplot(121)
plt.plot(disp_min2, f_1_min)
plt.plot(disp_max2, f_1_max)
plt.xlabel('Displacement [mm]')
plt.ylabel('kN')
plt.title('Sliding 1', fontsize=20)

plt.subplot(122)
plt.plot(disp_min3, f_1_min)
plt.plot(disp_max3, f_1_max)
plt.xlabel('Displacement [mm]')
plt.ylabel('kN')
plt.title('Sliding 2', fontsize=20)

plt.show()
