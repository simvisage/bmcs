'''
Created on Apr 9, 2010

@author: alexander
'''

from numpy import array, vstack, zeros

from os.path import \
    join

from matresdev.db.simdb.simdb import simdb

# The implementation works but is too time consuming.
# @todo: check for a faster and more simple solution!


def loadtxt_novalue(file_name):
    '''Return an data array similar to loadtxt. 
    "NOVALUE" entries are replaced by the value of the previous line.
    '''
    file = open(file_name, 'r')
    lines = file.readlines()
    n_columns = len(lines[0].split(';'))
    data_array = zeros(n_columns)
    n = 0
    for line in lines:
        line_split = lines[n].split(';')
        m = 0
        for value in line_split:
            if value == 'NOVALUE':
                print('---------------------------------------------------------------')
                print('NOVALUE entry in line', n, 'position', m, 'found')
                print('For faster processing replace values directly in the data file!')
                print('---------------------------------------------------------------')
                line_split[m] = lines[n - 1].split(';')[m]
            m += 1
        line_array = array(line_split, dtype=float)
        data_array = vstack([data_array, line_array])
        n += 1
    return data_array


if __name__ == '__main__':
    ex_path = join(
        simdb.exdata_dir, 'plate_tests', 'PT-10a', 'PT11-10a_original.ASC')
    data_array = loadtxt_novalue(ex_path)
    print('\n')
    print('data_array', data_array)
