'''
Created on 15.02.2019

@author: Mario Aguilar Rueda
'''

import csv
import os

import h5py
from pykalman import KalmanFilter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#np.loadtxt("bio_1.asc", skiprows=6)

name = 'CT80-42_3610_Zykl'
name2 = 'CT80-42_3610_Zykl'
home_dir = os.path.expanduser('~')
path_master = os.path.join(home_dir, 'Data Processing')
path_master = os.path.join(path_master, 'CT')
path_master = os.path.join(path_master, 'C80')
path_master = os.path.join(path_master, name)

path = os.path.join(path_master, name2 + '.csv')

path2 = os.path.join(path_master, name2 + '.hdf5')

f = open(path)
l = sum(1 for row in f)
num_iter = np.int(np.floor(l / 100000))

f = pd.read_csv(path, sep=';', skiprows=4, nrows=99995)
nf = np.array(f)
df = pd.DataFrame(nf, columns=['a', 'b', 'c', 'd', 'e', 'f'])
print(df)

df['a'] = [x.replace(',', '.') for x in df['a']]
df['b'] = [x.replace(',', '.') for x in df['b']]
df['c'] = [x.replace(',', '.') for x in df['c']]
df['d'] = [x.replace(',', '.') for x in df['d']]
df['e'] = [x.replace(',', '.') for x in df['e']]
df['f'] = [x.replace(',', '.') for x in df['f']]
df['f'] = [x.replace(',', '.') for x in df['f']]
# df['g'] = [x.replace(',', '.') for x in df['g']]
df.to_hdf(path2, 'first', mode='w', format='table')
del df
del f

for iter_num in range(num_iter - 1):
    print(iter_num)
    f = np.array(pd.read_csv(path, skiprows=(
        iter_num + 1) * 100000 - 1, nrows=100000, sep=';'))
    nf = np.array(f)
    df = pd.DataFrame(f.astype(str), columns=[
                      'a', 'b', 'c', 'd', 'e', 'f'])
    df['a'] = [x.replace(',', '.') for x in df['a']]
    df['b'] = [x.replace(',', '.') for x in df['b']]
    df['c'] = [x.replace(',', '.') for x in df['c']]
    df['d'] = [x.replace(',', '.') for x in df['d']]
    df['e'] = [x.replace(',', '.') for x in df['e']]
    df['f'] = [x.replace(',', '.') for x in df['f']]
#     df['g'] = [x.replace(',', '.') for x in df['g']]
    df.to_hdf(path2, 'middle' + np.str(iter_num), append=True)
    del df
    del f

f = np.array(pd.read_csv(path, skiprows=num_iter *
                         100000 - 1, nrows=l - num_iter * 100000, sep=';'))
nf = np.array(f)
df = pd.DataFrame(nf, columns=['a', 'b', 'c', 'd', 'e', 'f'])
df['a'] = [x.replace(',', '.') for x in df['a']]
df['b'] = [x.replace(',', '.') for x in df['b']]
df['c'] = [x.replace(',', '.') for x in df['c']]
df['d'] = [x.replace(',', '.') for x in df['d']]
df['e'] = [x.replace(',', '.') for x in df['e']]
df['f'] = [x.replace(',', '.') for x in df['f']]
# df['g'] = [x.replace(',', '.') for x in df['g']]
df.to_hdf(path2, 'last', append=True)
del df
del f
