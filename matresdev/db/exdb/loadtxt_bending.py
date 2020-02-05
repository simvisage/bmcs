'''
Created on Apr 9, 2010

@author: alexander
'''

import os

from matresdev.db.simdb.simdb import simdb
import numpy as np


def dot2comma(value):
    value = value.replace(',', '.')
    return float(value)


file_name = os.path.join(simdb.exdata_dir, 'bending_tests',
                         'ZiE_2011-06-08_BT-12c-6cm-0-TU', 'BT-12c-6cm-0-Tu-V4.raw')

# file contains both loading- and unloading path:
#
#file_name = '/home/alexander/simdb/exdata/bending_tests/ZiE_2011-06-08_BT-12c-6cm-0-TU/BT-12c-6cm-0-Tu-V2-converted.csv'


def loadtxt_bending(file_name):
    '''Return an data array of the bending test
    - first column: displacement [mm]
    - second column: compression strains at midfield [%]
    - third column: load [N]
    '''
    try:
        # Return an data array for the loading path (1 block).
        # load raw-data in case of loading path only
        # (no additional unloading path recorded below the first data block in the file)
        # in this case loadtxt works properly'''
        data_arr = np.loadtxt(file_name,
                              delimiter=';',
                              skiprows=41,
                              converters={
                                  1: dot2comma, 2: dot2comma, 3: dot2comma},
                              usecols=[1, 2, 3])
        print('loadtxt_bending: data_arr contains only loading path')

    except IndexError:
        print('loadtxt_bending: data_arr contains loading- and unloading path')
        data_arr = loadtxt_2blocks(file_name)

    return data_arr


def loadtxt_2blocks(file_name):
    '''Return an data array consisting of the loading AND unloading path (merge 2 blocks in the data file).
    in this case loadtxt doesn't work as the data file consits of 2 blocks'''
    file_ = open(file_name, 'r')
    lines = file_.readlines()

    data_arr_1 = np.zeros(3)
    data_arr_2 = np.zeros(3)

    start_n_blocks = []
    end_n_blocks = []

    # determine the starting number and end number of the data blocks 1 and 2:
    #
    n = 0
    for line in lines:
        line_split = line.split(';')
        if line_split[0] == '"Probe"':
            # first block normally starts with line 43
            # the starting line of the second block needs to be determined
            # 27 lines after the keyword "Probe" the data is recorded in both blocks
            #
            start_n_blocks.append(n + 28)
        if line_split[0] == '"Probe"':
            end_n_blocks.append(n)
        n += 1

    if len(end_n_blocks) != 1:
        # add the line number of the last line
        # this corresponds to the last line of block 2 if it is recorded
        #
        end_n_blocks.append(len(lines))
        end_n_blocks = end_n_blocks[1:]

#    print 'start_n_blocks', start_n_blocks
#    print 'end_n_blocks', end_n_blocks

    # convert data to array for blocks 1:
    #
    for line in lines[start_n_blocks[0]:end_n_blocks[0]]:
        line_split = line.split(';')
        line_arr = np.array([dot2comma(line_split[1]),
                             dot2comma(line_split[2]),
                             dot2comma(line_split[3])],
                            dtype=float)
        data_arr_1 = np.vstack([data_arr_1, line_arr])

    # convert data to array for blocks 2:
    #
    for line in lines[start_n_blocks[1]:end_n_blocks[1]]:
        line_split = line.split(';')
        line_arr = np.array([dot2comma(line_split[1]),
                             dot2comma(line_split[2]),
                             dot2comma(line_split[3])],
                            dtype=float)
        data_arr_2 = np.vstack([data_arr_2, line_arr])

    # remove line with zeros
    #
    data_arr = np.vstack([data_arr_1[1:], data_arr_2[1:]])
    return data_arr

if __name__ == '__main__':

    data_arr = loadtxt_bending(file_name)
    print('data_arr', data_arr)
