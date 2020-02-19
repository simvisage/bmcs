'''
Created on Apr 1, 2010

@author: alexander
'''
from traits.api import \
    HasTraits, Directory, List, Int, Float, Any, \
    on_trait_change, File, Constant, Instance, Trait, \
    Array, Str, Property, cached_property, WeakRef, \
    Dict, Button, Bool, Enum, Event, implements, \
    DelegatesTo, Callable

from traitsui.api import \
    View, Item, DirectoryEditor, TabularEditor, HSplit, VGroup, \
    TableEditor, EnumEditor, Handler, FileEditor, VSplit, Group, \
    InstanceEditor, HGroup, Spring

# # overload the 'get_label' method from 'Item' to display units in the label
from util.traits.ui.item import \
    Item

from numpy import \
    log

from matresdev.db.simdb.simdb_class import \
    SimDBClass, SimDBClassExt

class ConcreteMixture(SimDBClass):
    '''Describes the properties of the concrete matrix
    '''

    # E-modulus of the concrete after 28d
    E_m28 = Float(unit='MPa', simdb=True, input=False, auto_set=False, enter_set=False)

    # E-modulus of the concrete after 28d
    nu = Float(unit='-', simdb=True, input=False, auto_set=False, enter_set=False)

    # developement of the E-modulus depending on the age at the time of testing:
    get_E_m_time = Callable

    # view:
    traits_view = View(
                      Item('key'  , style='readonly'),
                      Item('E_m28', style='readonly', format_str="%.0f"),
                      Item('nu'   , style='readonly', format_str="%.2f"),
                      resizable=True,
                      scrollable=True
                      )

# Setup the database class extension
#
ConcreteMixture.db = SimDBClassExt(
            klass=ConcreteMixture,
            constants={
                # NOTE: log = natural logarithm  ("ln")

                'PZ-0708-1' : ConcreteMixture(
                                           E_m28=33036.,
                                           get_E_m_time=lambda t: 4665. * log(t + 0.024) + 17487.,
                                           nu=0.25
                                           ),
                'FIL-10-09' : ConcreteMixture(
                                           E_m28=28700.,
                                           # function for the evolution derived based on only
                                           # three values: Em0 = 0, Em7 = 23600, Em28 = 28700
                                           get_E_m_time=lambda t: 3682. * log(t + 0.012) + 16429.,
                                           nu=0.25
                                           ),
                'FIL-Standard-SF' : ConcreteMixture(
                                           E_m28=31000.,
                                           # function for the evolution unknown
                                           get_E_m_time=lambda t: 31000.,
                                           nu=0.2
                                           ),
                'C3-HF2-165-4' : ConcreteMixture(
                                           E_m28=44600.,  # @todo: approximation only
                                           # function for the evolution derived based on only
                                           # three values: Em0 = 0, Em7 = 23600, Em28 = 28700
                                           get_E_m_time=lambda t: 44600.,  # @todo: specify time function for evolution of E-modulus; so far value for 28d is used;
                                           nu=0.2  # @todo: approximation only
                                           ),
                'C3-HF2-155-5' : ConcreteMixture(
                                           E_m28=44600.,  # @todo: approximation only, no values determined yet
                                           # function for the evolution derived based on only
                                           # three values: Em0 = 0, Em7 = 23600, Em28 = 28700
                                           get_E_m_time=lambda t: 44600.,  # determine e-modulus;
                                           nu=0.2  # @todo: approximation only
                                           ),      
                'barrelshell' : ConcreteMixture(
#                                           E_m28 = 22721., #
                                           E_m28=19800.,  # based on cylinder tests
                                           get_E_m_time=lambda t: 19800.,  # @todo: specify time function for evolution of E-modulus; so far value for 28d is used;
                                           nu=0.25
                                           ),
                'shotcrete-4mm' : ConcreteMixture(
                                           E_m28=30000.,  # @todo: approximation only
                                           # @todo: function for the evolution derived based on only
                                           # three values: Em0 = 0, Em7 = 23600, Em28 = 28700
                                           get_E_m_time=lambda t: 30000.,  # @todo: specify time function for evolution of E-modulus; so far value for 28d is used;
                                           nu=0.2  # @todo: approximation only
                                           ),

                'sto-100' : ConcreteMixture(
                                           E_m28=30000.,  # @todo: approximation only
                                           # function for the evolution derived based on only
                                           # three values: Em0 = 0, Em7 = 23600, Em28 = 28700
                                           get_E_m_time=lambda t: 30000.,  # @todo: specify time function for evolution of E-modulus; so far value for 28d is used;
                                           nu=0.2  # @todo: approximation only
                                           ),

                'flowstone' : ConcreteMixture(
                                           E_m28=30000.,  # @todo: approximation only
                                           # function for the evolution derived based on only
                                           # three values: Em0 = 0, Em7 = 23600, Em28 = 28700
                                           get_E_m_time=lambda t: 30000.,  # @todo: specify time function for evolution of E-modulus; so far value for 28d is used;
                                           nu=0.2  # @todo: approximation only
                                           ),

                'Pagel_TF10' : ConcreteMixture(
                                           E_m28=30000.,  # @todo: approximation only
                                           # function for the evolution derived based on only
                                           # three values: Em0 = 0, Em7 = 23600, Em28 = 28700
                                           get_E_m_time=lambda t: 30000.,  # @todo: specify time function for evolution of E-modulus; so far value for 28d is used;
                                           nu=0.2  # @todo: approximation only
                                           ),
                'Pagel_TF10_Leipzig' : ConcreteMixture(
                                           E_m28=32700.,  # @todo: approximation only
                                           # function for the evolution derived based on only
                                           # three values: Em0 = 0, Em7 = 23600, Em28 = 28700
                                           get_E_m_time=lambda t: 32700.,  # @todo: specify time function for evolution of E-modulus; so far value for 28d is used;
                                           nu=0.2  # @todo: approximation only
                                           ),                       
                'HPC_TU_WIEN' : ConcreteMixture(
                                           E_m28=30000.,  # @todo: approximation only
                                           get_E_m_time=lambda t: 30000.,  # @todo: specify time function for evolution of E-modulus; so far value for 28d is used;
                                           nu=0.2  # @todo: approximation only
                                           ),
                'UHPC_TU_WIEN' : ConcreteMixture(
                                           E_m28=30000.,  # @todo: approximation only
                                           get_E_m_time=lambda t: 30000.,  # @todo: specify time function for evolution of E-modulus; so far value for 28d is used;
                                           nu=0.2  # @todo: approximation only
                                           ),
                'HPC_SF1.5_TU_WIEN' : ConcreteMixture(
                                           E_m28=30000.,  # @todo: approximation only
                                           get_E_m_time=lambda t: 30000.,  # @todo: specify time function for evolution of E-modulus; so far value for 28d is used;
                                           nu=0.2  # @todo: approximation only
                                           ),
                'UHPC_SF2.0_TU_WIEN' : ConcreteMixture(
                                           E_m28=30000.,  # @todo: approximation only
                                           get_E_m_time=lambda t: 30000.,  # @todo: specify time function for evolution of E-modulus; so far value for 28d is used;
                                           nu=0.2  # @todo: approximation only
                                           ),
                'T08_Boegel' : ConcreteMixture(
                                           E_m28=44886.,  # @todo: approximation only
                                           get_E_m_time=lambda t: 44886.,  # @todo: specify time function for evolution of E-modulus; so far value for 28d is used;
                                           nu=0.2  # @todo: approximation only
                                           ),                       
                'Hilo_Test_271216/2' : ConcreteMixture(
                                           E_m28=30000.,  # @todo: approximation only
                                           get_E_m_time=lambda t: 30000.,  # @todo: specify time function for evolution of E-modulus; so far value for 28d is used;
                                           nu=0.2  # @todo: approximation only
                                           ),
                                           }                                   
            )

if __name__ == '__main__':
    ConcreteMixture.db.configure_traits()
