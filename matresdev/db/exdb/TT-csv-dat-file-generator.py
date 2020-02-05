'''
Created on Apr 9, 2010

Scipt generates a csv-file based on a xls-file containing the test data of a
tensile test with only the time[s], force[kN] and machine displacement[mm] recorded.
The returned .csv-file and .DAT-file allows to read in the test as if it has been recorded 
with 4 displacement gauges (WA_vli, WA_vre, WA_hli, WA_hre). 
The machine displacement starts at an arbitrary value and is reset to start with 0.

@author: alexander
'''

from numpy import \
    array, shape, hstack, savetxt

from os.path import \
    join

from matresdev.db.simdb import \
    SimDB


# name of the tensile tes / xls-file with the machine data
#
TT_name = 'TT-12c-6cm-0-TU-REF1'
#TT_name = 'TT-12c-6cm-0-TU-REF2'
#TT_name = 'TT-12c-6cm-0-TU-WM'
#TT_name = 'TT-12c-6cm-0-TU-WML'
#TT_name = 'TT-12c-6cm-0-TU-WMT'
#TT_name = 'TT-12c-6cm-0-TU-WMTC'

# file path in simdb
#
simdb = SimDB()
file_path = join( simdb.exdata_dir, 'tensile_tests', 'ZiE_2011-08-18_TT-12-6cm-TU_Laminiert' )
file_name_xls = join( file_path, TT_name + '.xls' )

print('read input data from file ' + file_name_xls + '\n')

# open file
#
file = open( file_name_xls, 'r' )

# read heading files without values
#
heading1 = file.readline()
heading2 = file.readline()

# number of columns
#
n_columns = 3

# replace comma with dot
#
file_str = file.read().replace( ",", "." )

# split string into separate values-strings
#
str_list = file_str.split()

# convert string-values into floats and convert to array
#
data_arr_1d = array( [float( val ) for val in str_list] )

# reshape 1d-array into 3-columns-array
#
n_rows = shape( data_arr_1d )[0] / n_columns
data_arr = data_arr_1d.reshape( n_rows, n_columns )

# prepare outputarr duplicating the displacement column 3 times
# and reseting the first value to 0.
# use machine displacement [mm] for displacement gauge ("WA")
WA = ( data_arr[:, 1] - data_arr[0, 1] )[:, None]

# reference time [s]
time = data_arr[:, 0][:, None]

# force [kN]
force = data_arr[:, 2][:, None]

output_arr = hstack( ( time, force, -WA, -WA, -WA, -WA ) )
print('output_arr', output_arr)

file_name_csv = join( file_path, TT_name + '.csv' )

print('save output data to file ' + file_name_csv + '\n')

savetxt( file_name_csv, output_arr, delimiter = ';' )



file_name_DAT = join( file_path, TT_name + '.DAT' )

print('generate .DAT-file ' + file_name_DAT + '\n')

DAT_txt = "#BEGINCHANNELHEADER\n\
200,Bezugskanal\n\
201,Erfassungskanal\n\
202,s (ab Start)\n\
\n\
#BEGINCHANNELHEADER\n\
200,Kraft\n\
201,Erfassungskanal\n\
202,kN\n\
\n\
#BEGINCHANNELHEADER\n\
200,W10_vli\n\
201,Erfassungskanal\n\
202,mm\n\
\n\
#BEGINCHANNELHEADER\n\
200,W10_vre\n\
201,Erfassungskanal\n\
202,mm\n\
\n\
#BEGINCHANNELHEADER\n\
200,W10_hli\n\
201,Erfassungskanal\n\
202,mm\n\
\n\
#BEGINCHANNELHEADER\n\
200,W20_hre\n\
201,Erfassungskanal\n\
202,mm\n\
"
file_DAT = open( file_name_DAT, 'w' )
file_DAT.write( DAT_txt )
