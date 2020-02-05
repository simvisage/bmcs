'''
Created on Apr 22, 2010

@author: alexander
'''
from traits.api import \
    HasTraits, Directory, List, Int, Float, Any, \
    on_trait_change, File, Constant, Instance, Trait, \
    Array, Str, Property, cached_property, WeakRef, \
    Dict, Button, Bool, Enum, Event, implements, \
    DelegatesTo, Regex

from traitsui.api import \
    View, Item, DirectoryEditor, TabularEditor, HSplit, VGroup, \
    TableEditor, EnumEditor, Handler, FileEditor, VSplit, Group, \
    InstanceEditor, HGroup, Spring
    

from etsproxy.units.ui.quantity_view import \
    QuantityView
    
from etsproxy.units import \
    *

from etsproxy.units.convert import \
    convert 

from etsproxy.units.quantity import \
    Quantity

from etsproxy.traits.ui.table_column import \
    ObjectColumn
    
from etsproxy.traits.ui.menu import \
    OKButton, CancelButton
    
from etsproxy.traits.ui.tabular_adapter \
    import TabularAdapter

from numpy import \
    array, fabs, where, copy, ones, linspace, ones_like, hstack, arange

from etsproxy.traits.ui.table_filter \
    import EvalFilterTemplate, MenuFilterTemplate, RuleFilterTemplate, \
           EvalTableFilter

from etsproxy.traits.ui.file_dialog  \
    import open_file, FileInfo, TextInfo, ImageInfo

from traitsui.api \
    import View,TabularEditor
    
from etsproxy.traits.ui.tabular_adapter \
    import TabularAdapter

from etsproxy.units.unit \
    import *

from util.traits.ui.item import \
    Item


#---------------------------------------------------------------------

def get_unit( trait ):
    '''Return the string stored in the trait metadata 
    for key 'unit'. If unspecified return 'None'.
    '''
    dict = trait.__dict__
    if 'unit' in dict:
        return dict['_metadata']['unit'] 

def get_convert_fact( unit_old, unit_new  ):
    '''Returns a factor for converting 'unit_old' to 'unit_new' 
    unit_old  - given unit to be converted
    unit_new - converted unit (target unit) 
    '''
    if unit_old == None:  
        return 1.
    else:
        m_dict = {'m':1, 'dm':10, 'cm':100, 'mm':1000,}
        return float(m_dict[ unit_new ]) / float( m_dict[ unit_old ] ) 

def unit2m( trait ):
    '''Convert trait to unit 'm'. Multiply value with a factor
    based on the unit specified in the trait's '_metadata'  
    '''
    unit = get_unit( trait )
    fact = get_convert_fact( unit, 'm')
    value = trait.__dict__['default_value']
    return fact * value


class Bcl( HasTraits ):
    '''Example class with a trait with metadata 'unit'
    '''
    #------------------------------
    # convert unit
    #------------------------------
    bb = Int(3, unit = 'cm')
    cc = Int(3 )


class Acl( HasTraits ):
    '''Example class with a trait with metadata 'unit'
    '''
    #------------------------------
    # convert unit
    #------------------------------
    b = Int(3, unit = 'cm')
    c = Int(3 )
    u = get_unit( b )
    f = get_convert_fact( u, 'm' )
    b_m = unit2m(b)

    #------------------------------
    # test indexing of Array object
    #------------------------------
#    B = Property( Array, unit = 'm' )
#    def _get_B(self):
#        return array([1,2,3])

#    bbb = arange(2)

    B = Array( unit = 'm' )
    def _B_default(self):
        print('default set')
        return array([9,9])    

    def ppp(self):
        print(self.B[0])
        return 1
    
    
#    B.default_value = array([22])    
    
    #------------------------------
    # set attributes
    #------------------------------
    factor_list = ['WA_1', 'WA_2', 'WA_3']
    units = ['s', 'kN', 'mm']
    processed_data_array = 1.0 * arange(9).reshape(3,3)

    def set_attr_type_and_unit(self):
        for i, factor in enumerate( self.factor_list ):
            setattr( self, factor, Array( unit = self.units[i] ))
            #attr = getattr( self, factor )
            #attr.default_value = self.processed_data_array[:,i] 
            setattr( self, factor+'.default_value', self.processed_data_array[:,i] )
            
    BI = Instance(Bcl)
    def _BI_default(self):
        return Bcl()
    
    traits_view = View(
                       Item('b', show_label = True,),
                       Item('c', show_label = True,),
                       Item('BI@', show_label = False,),
                       Item('BI@', show_label = True,),
                       scrollable = True,
                       resizable = True,
                       height = 0.4,
                       width = 0.4,
                       )

#---------------------------------

if __name__ == '__main__':

    A = Acl()
    A.set_attr_type_and_unit()
#    A.configure_traits()

    print('WA_1', A.WA_1)
    dict = A.WA_1.__dict__ 
    print('dict', dict)   
    
    B = A.B
    print('B',B[0])
    
    B = A.ppp()

