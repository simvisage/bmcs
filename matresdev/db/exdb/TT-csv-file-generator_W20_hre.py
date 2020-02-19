'''
Created on Apr 9, 2010

Scipt generates a csv-file based on a ASC-file containing the test data of a
tensile test with four gauge displacements[mm].
The displacement gauge with name "W20_hre" needs to be multiplied by factor 2
due to a false machine setup.
@author: alexander
'''

from numpy import \
    array, shape, hstack, savetxt, loadtxt

from os.path import \
    join

from matresdev.db.simdb import \
    SimDB


# name of the tensile test / ASC-file with gauge displacements
#
TT_name = 'TT-12c-6cm-TU-SH1-V1'
TT_name = 'TT-12c-6cm-TU-SH1-V2'
TT_name = 'TT-12c-6cm-TU-SH1-V3'
TT_name = 'TT-12c-6cm-TU-SH1F-V1'
TT_name = 'TT-12c-6cm-TU-SH1F-V2'
TT_name = 'TT-12c-6cm-TU-SH1F-V3'


# file path in simdb
#
simdb = SimDB()
file_path = join( simdb.exdata_dir, 'tensile_tests', '2012-01-09_TT-12c-6cm-TU-SH1' )
file_name_ASC = join( file_path, TT_name + '.ASC' )

print('read input data from file ' + file_name_ASC + '\n')

# open file
#
file = open( file_name_ASC, 'r' )

data_arr = loadtxt( file_name_ASC,
                    delimiter = ";" )

# prepare outputarr multiplying the value of the displacement gauge 
# "W20_hre" by factor 2.
WA_hre = ( data_arr[:, -1] )[:, None]

output_arr = hstack( ( data_arr[:, 0:-1], 2.0 * WA_hre ) )
print('output_arr', output_arr)

file_name_csv = join( file_path, TT_name + '.csv' )

print('save output data to file ' + file_name_csv + '\n')

savetxt( file_name_csv, output_arr, delimiter = ';' )

