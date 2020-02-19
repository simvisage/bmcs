'''
Created on Apr 28, 2010

@author: alexander
'''
from traits.api import \
    HasTraits, Float, Property, cached_property, \
    Instance, File, List, on_trait_change, Int, Tuple, Bool, \
    DelegatesTo, Event, Str, Button, Dict, Array, Any, Enum, Callable
    
from traitsui.api import \
    View, Item, Tabbed, VGroup, HGroup, ModelView, HSplit, VSplit, \
    CheckListEditor, EnumEditor, TableEditor, TabularEditor, Handler, \
    Group, CancelButton, FileEditor

from etsproxy.traits.ui.menu import \
    Action, CloseAction, HelpAction, Menu, \
    MenuBar, NoButtons, Separator, ToolBar                    

from etsproxy.traits.ui.key_bindings import \
    KeyBinding, KeyBindings

from etsproxy.traits.ui.tabular_adapter \
    import TabularAdapter
    
from etsproxy.pyface.api import \
    ImageResource, confirm, error, information, warning, YES, NO, CANCEL
    
from etsproxy.traits.ui.menu import \
    OKButton

from numpy import \
    array, linspace, frompyfunc, zeros, column_stack, \
    log as ln, append, logspace, hstack, sign, trapz, mgrid, c_, \
    zeros
                    
from math import \
    exp, e, sqrt, log, pi
    
from .ex_run import \
    ExRun

from .ex_run_view import \
    ExRunView

from matresdev.db.simdb import \
    SimDB

import os

from matresdev.db.matdb.trc.concrete_mixture \
    import ConcreteMixture

from matresdev.db.matdb.trc.fabric_layout \
    import FabricLayOut

from matresdev.db.matdb.trc.fabric_layup \
    import FabricLayUp

from matresdev.db.matdb.trc.composite_cross_section \
    import CompositeCrossSection, plain_concrete

if __name__ == '__main__':
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
    
    ex_path_TT_7a_V1 = os.path.join(simdb.exdata_dir, 'tensile_tests', 'TT-7a',
                                'TT08-7a-V1.DAT')
    
    ex_path_TT_7a_V2 = os.path.join(simdb.exdata_dir, 'tensile_tests', 'TT-7a',
                                'TT08-7a-V2.DAT')
    
    ex_path_TT_9u = os.path.join(simdb.exdata_dir, 'tensile_tests', 'TT-9u',
                                'TT06-9u-V2all90.DAT')
    
    # define the path to a plate test:
    ex_path_PT_10a = os.path.join(simdb.exdata_dir, 'plate_tests', 'PT-10a',
                                'PT10-10a.DAT')
    
    #----------------------------------------------------------
    # delete the pickle file if it exists:
    #----------------------------------------------------------
    print('XXX Delete the pickle file if it exists--------------------')
    data_file = ex_path_TT_7a_V1
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
    print('XXX Construct ExRunView and show attributes--------------------')
    exrv = ExRunView(data_file = ex_path_TT_7a_V1)
    print('--- GET: data_file = ', exrv.data_file, '---')
    print('--- GET: unsaved = ', exrv.unsaved, '---')
    print('--- GET: ccs.flu_list[0].s_tex_z = ', exrv.model.ex_type.ccs.fabric_layup_list[0].s_tex_z, '---')
    print('--- GET: ccs.cm_key = ', exrv.model.ex_type.ccs.concrete_mixture_key, '---')
    
    #----------------------------------------------------------
    # save default settings in pickle file
    #----------------------------------------------------------
    print('XXX save default settings in pickle file --------------------')
    exrv.save_run()
    print('--- SAVE RUN ---')
    print('--- pickle file saved to:', pickle_file_name, ' ---')
    print('--- GET: unsaved = ', exrv.unsaved, '---')
    
    #----------------------------------------------------------
    # change attribute in 'ccs' and save changes in pickle file
    #----------------------------------------------------------
    print('XXX Change attribute in ccs and save changes in pickle file--------------------')
    exrv.model.ex_type.ccs.concrete_mixture_key = 'PZ-0708-1'
    print('--- SET: ccs.cm_key = ', exrv.model.ex_type.ccs.concrete_mixture_key, '---')
    print('--- GET: unsaved = ', exrv.unsaved, '---')
    exrv.save_run()
    print('--- SAVE RUN ---')
    print('--- pickle file saved to:', pickle_file_name, ' ---')
    print('--- GET: unsaved = ', exrv.unsaved, '---')
    
    #----------------------------------------------------------
    # change data file to another file and then change it back 
    #----------------------------------------------------------
    print('XXX Change data file to another file and then change it back--------------------')
    print('--- GET: data file (old): ', exrv.data_file)
    exrv.data_file = ex_path_TT_7a_V2
    print('--- SET: data file (new): ', exrv.data_file)
    print('--- GET: unsaved (new) = ', exrv.unsaved, '---')
    if exrv.unsaved == True:
        exrv.save_run()
        print('--- SAVE RUN (new) ---')
    exrv.data_file = ex_path_TT_7a_V1
    print('--- SET: data file (back to old): ', exrv.data_file)
    
    #----------------------------------------------------------
    # show attributes after reloading 
    #----------------------------------------------------------
    print('XXX Show attribute after reloading --------------------')
    print('--- GET: data_file = ', exrv.data_file, '---')
    print('--- GET: unsaved = ', exrv.unsaved, '---')
    print('--- GET: ccs.flu_list[0].s_tex_z = ', exrv.model.ex_type.ccs.fabric_layup_list[0].s_tex_z, '---')
    print('--- GET: ccs.cm_key = ', exrv.model.ex_type.ccs.concrete_mixture_key, '---')
    
    #----------------------------------------------------------
    # change attribute in 'ccs' 
    #----------------------------------------------------------
    print('XXX Change attribute in ccs --------------------')
    exrv.model.ex_type.ccs.concrete_mixture_key = 'FIL-10-09'
    print('--- SET: ccs.cm_key = ', exrv.model.ex_type.ccs.concrete_mixture_key, '---')
    print('--- GET: unsaved = ', exrv.unsaved, '---')
    exrv.save_run()
    print('--- SAVE RUN ---')
    print('--- pickle file saved to:', pickle_file_name, ' ---')
    print('--- GET: unsaved = ', exrv.unsaved, '---')
    
    #----------------------------------------------------------
    # change data file to another file and then change it back 
    #----------------------------------------------------------
    print('XXX Change data file to another file and then change it back--------------------')
    exrv.data_file = ex_path_TT_7a_V2
    print('--- SET: data file (new): ', exrv.data_file)
    exrv.data_file = ex_path_TT_7a_V1
    print('--- SET: data file (back to old): ', exrv.data_file)
    
    #----------------------------------------------------------
    # change attribute in 'ccs.fabric_layup_list' 
    #----------------------------------------------------------
    print('XXX change attribute in ccs.fabric_layup_list --------------------')
    exrv.model.ex_type.ccs.fabric_layup_list.append(plain_concrete(0.01))
    print('--- SET: ccs.fabric_layup_list = ', exrv.model.ex_type.ccs.fabric_layup_list, '---')
    print('--- GET: unsaved = ', exrv.unsaved, '---')
    exrv.save_run()
    
    #----------------------------------------------------------
    # change data file to another file and then change it back 
    #----------------------------------------------------------
    print('XXX Change data file to another file and then change it back--------------------')
    exrv.data_file = ex_path_TT_7a_V2
    print('--- SET: data file (new): ', exrv.data_file)
    exrv.data_file = ex_path_TT_7a_V1
    print('--- SET: data file (back to old): ', exrv.data_file)
    
    #----------------------------------------------------------
    # change attribute in 'ccs.fabric_layup_list[0].s_tex_z' 
    #----------------------------------------------------------
    print('XXX change attribute in ccs.fabric_layup_list[0].s_tex_z --------------------')
    exrv.model.ex_type.ccs.fabric_layup_list[0].s_tex_z = 0.02 
    print('--- SET: ccs.fabric_layup_list[0].s_tex_z = ', exrv.model.ex_type.ccs.fabric_layup_list[0].s_tex_z, '---')
    print('--- GET: unsaved = ', exrv.unsaved, '---')
    exrv.save_run()
