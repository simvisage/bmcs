'''
Created on 15.02.2019

@author: Mario Aguilar Rueda
'''

import os
import string

import numpy as np
import pandas as pd


def read_csv(path):
    '''Read the csv file and transform it to the hdf5 forma.
    The output file has the same name as the input csv file
    with an extension hdf5
    '''
    basename = path.split('.')
    path2 = ''.join(basename[:-1]) + '.hdf5'

    chunk_size = 10000
    skip_rows = 4
    n_rows = chunk_size - 1 - skip_rows

    f = open(path)
    l = sum(1 for row in f)
    n_chunks = np.int(np.floor(l / chunk_size))
    f = pd.read_csv(path, sep=';', skiprows=skip_rows, nrows=n_rows)
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

    for iter_num in range(n_chunks - 1):
        print(iter_num)
        f = np.array(pd.read_csv(path, skiprows=(
            iter_num + 1) * chunk_size - 1, nrows=chunk_size, sep=';'))
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

    f = np.array(pd.read_csv(path, skiprows=n_chunks *
                             chunk_size - 1, nrows=l - n_chunks * chunk_size, sep=';'))
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


if __name__ == '__main__':
    #np.loadtxt("bio_1.asc", skiprows=6)
    name = 'CT80-42_3610_Zykl'
    name2 = 'CT80-42_3610_Zykl'
    name = 'CT80-39_6322_Zykl'
    name2 = 'CT80-39_6322_Zykl'

    name = 'CT80-39_6322_Zykl'
    home_dir = os.path.expanduser('~')
    path_master = os.path.join(home_dir, 'Data Processing')
    path_master = os.path.join(path_master, 'CT')
    path_master = os.path.join(path_master, 'C80')
    path_master = os.path.join(path_master, name, name + '.csv')

    read_csv(path_master)
