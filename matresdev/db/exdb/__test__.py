'''
Created on Aug 8, 2009

@author: alex
'''
import unittest

from traits.api import \
    Instance

import os

from shutil import \
    copy

from .ex_run_view import \
    ExRunView

from matresdev.db.matdb.trc.composite_cross_section \
    import plain_concrete

from matresdev.db.simdb import \
    SimDB

# Access to the toplevel directory of the database
#
simdb = SimDB()


#--------------------------------------------------------------------------------
# run a test cycle of loading - saving - loading
#--------------------------------------------------------------------------------

#----------------------------------------------------------
# define the path to a tensile test:
#----------------------------------------------------------
# - two tests have the same cross section
# - one test has a different cross section and different layup

ex_path_TT_7a_V1 = os.path.join(simdb.exdata_dir, 'tensile_tests',
                                'butstrap_clamping', 'TT-7a',
                                'TT08-7a-V1.DAT')

ex_path_TT_7a_V2 = os.path.join(simdb.exdata_dir, 'tensile_tests', 'TT-7a',
                                'TT08-7a-V2.DAT')


class TestExRunView(unittest.TestCase):
    '''test for a cycle of different show cases.
    '''

    def setUp(self):
        '''open a file for testing and construct
        ExRunView as a class variable.
        '''
        print('\n')
        print('#------------------------------------------')
        print('# SetUp')
        print('#------------------------------------------')

        # define data file (original data file)
        #
        self.data_file_orig_1 = ex_path_TT_7a_V1
        self.data_file_orig_2 = ex_path_TT_7a_V2

        # make a copy of the original data files and get new path of test files
        #
        self.data_file_test_1 = self.copy_data_files(self.data_file_orig_1)
        self.data_file_test_2 = self.copy_data_files(self.data_file_orig_2)

        # reset/setup
        #
        self.exrv = Instance(ExRunView)
        self.delete_pickle_file(self.data_file_test_1)
        self.construct_exrv(self.data_file_test_1)

        # save the default settings
        #
        self.save_run()


    def test_ex_run_view_cycle(self):
        '''test the behavior of ExRunView with respect
        to changes of the input variables for the both cases
        that ExRunView was newly constructed or read in from
        a stored pickle file.
        '''

        # test changes in cm, flu_list and s_tex_z
        # before and after reloading from pickle file:
        #
        for attr in [ 'cm_key', 'age', 's_tex_z', 'flu_list' ]:
            print('\n')
            print('#------------------------------------------')
            print('# ', attr)
            print('#------------------------------------------')
            old_value = self.get_attr(attr)
            print('old_value', old_value)
            new_value = self.change_attr(attr)
            print('new_value', new_value)
#            self.assertNotEqual( old_value, new_value )
            self.assertEqual(self.exrv.unsaved, True)
            self.save_run()

            self.assertEqual(self.exrv.unsaved, False)
            self.reload_data_file(self.data_file_test_1, self.data_file_test_2)
            self.assertEqual(self.exrv.unsaved, False)
            old_value = self.get_attr(attr)
            new_value = self.change_attr(attr)
#            self.assertNotEqual( old_value, new_value )
            self.assertEqual(self.exrv.unsaved, True)
            self.save_run()

    #----------------------------------------------------------
    # copy the original data file:
    #----------------------------------------------------------
    def copy_data_files(self, data_file_orig):
        '''Copy the data files with ending .DAT and .ASC
        in order to run '__test__.py' without changing the
        original data files or associated pickle files with
        the same name. The new data files have the same name
        and path as the original data file extended with "_test".
        '''
        # get original dir path
        dir_path = os.path.dirname(data_file_orig)
        file_name = os.path.basename(data_file_orig)
        file_split = file_name.split('.')
        # copy the "*.ASC"-file
        file_name_orig_asc = os.path.join(dir_path, file_split[0] + '.ASC')
        file_name_test_asc = os.path.join(dir_path, file_split[0] + '_test.ASC')
        copy(file_name_orig_asc, file_name_test_asc)
        # copy the "*.DAT"-file
        file_name_orig_dat = os.path.join(dir_path, file_split[0] + '.DAT')
        file_name_test_dat = os.path.join(dir_path, file_split[0] + '_test.DAT')
        copy(file_name_orig_dat, file_name_test_dat)
        # return the path to the new "*_test.DAT"-file
        return file_name_test_dat

    #----------------------------------------------------------
    # delete the test data files after the test has been run:
    #----------------------------------------------------------
    def delete_test_data_files(self, data_file):
        print('XXX Delete the auxilary test data files if they exist --------------------')

        dir_path = os.path.dirname(data_file)
        file_name = os.path.basename(data_file)
        file_split = file_name.split('.')

        # delete the "*.ASC"-file
        file_name_orig_asc = os.path.join(dir_path, file_split[0] + '.ASC')
        file_name_test_asc = os.path.join(dir_path, file_split[0] + '_test.ASC')
        if os.path.exists(file_name_test_asc):
            print('--- test data file asc removed: ', file_name_test_asc, ' ---')
            os.remove(file_name_test_asc)
        else:
            print('--- test data file asc does not exist: ', file_name_test_asc, ' ---')
        # delete the "*.DAT"-file
        file_name_orig_dat = os.path.join(dir_path, file_split[0] + '.DAT')
        file_name_test_dat = os.path.join(dir_path, file_split[0] + '_test.DAT')
        if os.path.exists(file_name_test_dat):
            print('--- test data file dat removed: ', file_name_test_dat, ' ---')
            os.remove(file_name_test_dat)
        else:
            print('--- test data file dat does not exist: ', file_name_test_dat, ' ---')

    #----------------------------------------------------------
    # delete the pickle file if it exists:
    #----------------------------------------------------------
    def delete_pickle_file(self, data_file):
        print('XXX Delete the pickle file if it exists--------------------')

        dir_path = os.path.dirname(data_file)
        file_name = os.path.basename(data_file)
        file_split = file_name.split('.')
        pickle_file_name = os.path.join(dir_path, file_split[0] + '.pickle')
        if os.path.exists(pickle_file_name):
            print('--- pickle file removed: ', pickle_file_name, ' ---')
            os.remove(pickle_file_name)
        else:
            print('--- pickle file does not exist: ', pickle_file_name, ' ---')

    #----------------------------------------------------------
    # construct ExRunView and show attributes
    #----------------------------------------------------------
    def construct_exrv(self, data_file):
        print('XXX Construct ExRunView --------------------')
        self.exrv = ExRunView(data_file=data_file)
        print('--- GET: data_file = ', self.exrv.data_file, '---')
        print('--- GET: unsaved = ', self.exrv.unsaved, '---')

    #----------------------------------------------------------
    # get attributes of ExRunView
    #----------------------------------------------------------

    def get_attr(self, name):
        if name == 'age':
            return self.get_age()
        elif name == 'cm_key':
            return self.get_cm_key()
        elif name == 'flu_list':
            return self.get_flu_list()
        elif name == 's_tex_z':
            return self.get_s_tex_z()

    def get_age(self):
        print('XXX Get attributes of ExRunView --------------------')
        age = self.exrv.model.ex_type.age
        print('--- GET: age (old) = ', age , '---')
        return age

    def get_cm_key(self):
        print('XXX Get attributes of ExRunView --------------------')
        cm_key = self.exrv.model.ex_type.ccs.concrete_mixture_key
        print('--- GET: ccs.cm_key (old) = ', cm_key , '---')
        return cm_key

    def get_flu_list(self):
        print('XXX Get attributes of ExRunView --------------------')
        flu_list = self.exrv.model.ex_type.ccs.fabric_layup_list
        print('--- GET: ccs.flu_list (old)= ', flu_list , '---')
        return flu_list

    def get_s_tex_z(self):
        print('XXX Get attributes of ExRunView --------------------')
        s_tex_z = self.exrv.model.ex_type.ccs.fabric_layup_list[0].s_tex_z
        print('--- GET: ccs.flu_list[0].s_tex_z (old)= ', s_tex_z, '---')
        return s_tex_z

    #----------------------------------------------------------
    # save settings in pickle file
    #----------------------------------------------------------
    def save_run(self):
        print('XXX save settings in pickle file --------------------')
        self.exrv.save_run()
        print('--- SAVE RUN ---')
        print('--- pickle file saved to:', self.exrv.model.pickle_file_name, ' ---')
        print('--- GET: unsaved = ', self.exrv.unsaved, '---')

    #----------------------------------------------------------
    # change attribute of ExRunView
    #----------------------------------------------------------

    def change_attr(self, name):
        if name == 'age':
            return self.change_age()
        elif name == 'cm_key':
            return self.change_cm_key()
        elif name == 'flu_list':
            return self.change_flu_list()
        elif name == 's_tex_z':
            return self.change_s_tex_z()

    def change_age(self):
        print('XXX Change age --------------------')
        age_old = self.exrv.model.ex_type.age
        self.exrv.model.ex_type.age += 1
        print('--- SET: age (new)= ', self.exrv.model.ex_type.age , '---')
        print('--- GET: unsaved = ', self.exrv.unsaved, '---')
        return self.exrv.model.ex_type.age

    def change_cm_key(self):
        print('XXX Change concrete matrix in ccs --------------------')
        cm_key_old = self.exrv.model.ex_type.ccs.concrete_mixture_key
        if cm_key_old == 'FIL-10-09':
            self.exrv.model.ex_type.ccs.concrete_mixture_key = 'PZ-0708-1'
        elif cm_key_old == 'PZ-0708-1':
            self.exrv.model.ex_type.ccs.concrete_mixture_key = 'FIL-10-09'
        cm_key_new = self.exrv.model.ex_type.ccs.concrete_mixture_key
        print('--- SET: ccs.cm_key (new)= ', cm_key_new , '---')
        print('--- GET: unsaved = ', self.exrv.unsaved, '---')
        return cm_key_new

    def change_flu_list(self):
        print('XXX Append ccs.flu_list --------------------')
        self.exrv.model.ex_type.ccs.fabric_layup_list.append(plain_concrete(0.01))
        flu_list_new = self.exrv.model.ex_type.ccs.fabric_layup_list
        print('--- SET: ccs.fabric_layup_list (new)= ', flu_list_new , '---')
        print('--- GET: unsaved = ', self.exrv.unsaved, '---')
        return flu_list_new

    def change_s_tex_z(self):
        print('XXX Change s_tex_z in ccs.flu[0].s_tex_z --------------------')
        s_tex_z = self.exrv.model.ex_type.ccs.fabric_layup_list[0].s_tex_z
        self.exrv.model.ex_type.ccs.fabric_layup_list[0].s_tex_z += 1.0
        s_tex_z_new = self.exrv.model.ex_type.ccs.fabric_layup_list[0].s_tex_z
        print('--- SET: ccs.flu[0].s_tex_z (new)= ', s_tex_z_new , '---')
        print('--- GET: unsaved = ', self.exrv.unsaved, '---')
        return s_tex_z_new

    #----------------------------------------------------------
    # change data file to another file and then change it back
    #----------------------------------------------------------
    def reload_data_file(self, data_file_old, data_file_new):
        print('XXX Switch data file to another file and then back again --------------------')
        self.exrv.data_file = data_file_new
        print('--- SET: data file (new): ', self.exrv.data_file)
        if self.exrv.unsaved == True:
            self.exrv.save_run()
            print('--- SAVE RUN (new) ---')
        self.exrv.data_file = data_file_old
        print('--- SET: data file (back to old): ', self.exrv.data_file)


    def tearDown(self):
        print('\n')
        print('#------------------------------------------')
        print('# tearDown')
        print('#------------------------------------------')
        self.delete_test_data_files(self.data_file_orig_1)
        self.delete_test_data_files(self.data_file_orig_2)


if __name__ == "__main__":
    unittest.main()


