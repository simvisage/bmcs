'''
Created on Mar 6, 2012

@author: rch
'''

import fnmatch
import os.path

def replace_string_in_files(path,
                            replace_dict,
                            file_ext = '.pickle'
                            ):
    '''Walk through all files within the subdirectories of `path'
    and perform the replacement of strings specifified in `replace_dict\
    in all files ending with `file_ext'
    '''
    selected_files = []
    for path, dirs, files in os.walk(path):
        for f in fnmatch.filter(files, '*' + file_ext):
            selected_files.append(os.path.join(path, f))

    for file_name in selected_files:
        print('replacement applied to', file_name)
        fid = open(file_name, "r")
        text = fid.read()
        fid.close()
        # write the replacements back into the file.
        fid = open(file_name, "w")
        for old, new in list(replace_dict.items()):
            text = text.replace(old, new)
        fid.write(text)
        fid.close()
        print('finished')

def migrate_classes(migration_table):
    '''Walk through all files within the simdb storage'
    and perform the replacement of strings specifified in `replace_dict\
    in all files ending with pickle'
    '''
    from matresdev.db.simdb import SimDB
    simdb = SimDB()    

    replace_string_in_files(simdb.exdata_dir,
                            migration_table, '.pickle')
    replace_string_in_files(simdb.matdata_dir,
                            migration_table, '.pickle')
    replace_string_in_files(simdb.exdata_dir,
                            migration_table, 'ex_type.cls')
                    
if __name__ == '__main__':

    from matresdev.db.simdb import SimDB
    simdb = SimDB()

    migration_table = {'promod.matdb.trc' : 'matresdev.db.matdb.trc',

                       'promod.exdb.ex_composite_tensile_test' : 'quaducom.devproc.tensile_test.dog_bone.exp_tt_db',
                       'ExCompositeTensileTest' : 'ExpTensileTestDogBone',
#                       'ExCompositeTensileTest' : 'ExpDogBoneTensileTest',
#                       'promod.exdb.ex_bending_test' : 'quaducom.devproc.bt.p3.exp_bt_3pt',
                       'promod.exdb.ex_bending_test' : 'quaducom.devproc.bending_test.three_point.exp_bt_3pt',
#                       'ExBendingTest' : 'ExpBendingTest3Pt',
                       'ExBendingTest' : 'ExpBT3Pt',

#                       'promod.exdb.ex_plate_test' : 'quaducom.devproc.st.exp_st',
                       'promod.exdb.ex_plate_test' : 'quaducom.devproc.slab_test.exp_st',
#                       'ExPlateTest' : 'ExpSlabTest',
                       'ExPlateTest' : 'ExpST',
 
                       'exp_dbtt':'exp_tt_db',
#                       'promod.exdb.ex_composite_tensile_test' : 'quaducom.devproc.tt.dbtt.exp_dbtt',
#                       'quaducom.devproc.tt.dbtt' : 'quaducom.devproc.tensile_test.dog_bone',
                       'ExpDogBoneTensileTest' : 'ExpTTDB',
                       # also replace class name in the file "exp_tt_db.py" 

#                       'quaducom.devproc.bt.p4' : 'quaducom.devproc.bending_test.four_point',
#                       'quaducom.devproc.bt.p3' : 'quaducom.devproc.bending_test.three_point',
#                       'ExpBendingTest3Pt' : 'ExpBendingTestThreePoint',
#                       'ExpBendingTest4Pt' : 'ExpBendingTestFourPoint',

                       'quaducom.devproc.st' : 'quaducom.devproc.slab_test',
                       'quaducom.devproc.dt' : 'quaducom.devproc.disk_test',

                        'DiskTestSetup':'SimDT',
                        'ExpSlabTest':'ExpST',
                        'SimSlabTest':'SimpST',

                        'quaducom.devproc.tt.bctt' : 'quaducom.devproc.tensile_test.buttstrap_clamping',
#                        'ButtstrapClamping': 'ExpTTBC',
                        'ExpTensileTestButtstrapClamping':'ExpTTBC'

#                        'SimTreePointBending' : 'SimBT3PT'
#                        'SimTreePointBendingDB':'SimBT3PTDB'

    } 

    migration_table = {'enthought.traits' : 'traits'}

    migrate_classes(migration_table)
