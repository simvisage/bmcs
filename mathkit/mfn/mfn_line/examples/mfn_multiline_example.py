
from traits.api import \
    Array, Bool, Callable, Enum, Float, HasTraits, \
    Instance, Int, Trait, Str, Enum, Callable, List, TraitDict, Any, \
    Dict, Property, cached_property, WeakRef, Delegate, \
    ToolbarButton, on_trait_change, Code, Expression, Button
    
from traitsui.api import \
    Item, View, HGroup, ListEditor, VGroup, VSplit, Group, HSplit

from etsproxy.traits.ui.menu import \
    NoButtons, OKButton, CancelButton, Action, CloseAction, Menu, \
    MenuBar, Separator
                                     
from mathkit.mfn.mfn_line.mfn_multiline import MFnMultiLine
from mathkit.mfn import MFnLineArray
from mathkit.mfn.mfn_line.mfn_matplotlib_multiline_editor import MFnMatplotlibEditor
from mathkit.mfn.mfn_line.mfn_chaco_multiline_editor import MFnChacoEditor
from mathkit.mfn.mfn_line.mfn_plot_adapter import MFnMultiPlotAdapter
from numpy import linspace, frompyfunc, vstack, column_stack
from math import sin, cos

a = MFnMultiPlotAdapter( title = 'Force vs Displacement',
                        label_x = 'x axis',
                        label_y = 'y axis',
                        mline_color = ['green','blue', 'red'], 
                        mline_style = ['solid', 'dashed', 'dotted'],
                        legend_labels = ('1','2','3'),
                        mline_width = [3,2,1])

class AnalyticalFunction( HasTraits ):
    
    No_of_all_curves = Int(3)
    expression1 = Expression('x**2', auto_set = False, enter_set = True )
    expression2 = Expression('x**2.3', auto_set = False, enter_set = True )
    expression3 = Expression('x**2.6', auto_set = False, enter_set = True )
    refresh = Button('redraw')
    def _refresh_fired(self):
        
        #creates an empty plot data container as a list of MFnLineArray classes
        self.mfn.lines = self.No_of_all_curves * [MFnLineArray()]
        
        xdata = linspace(0,10,100)
        fneval1 = frompyfunc( lambda x: eval( self.expression1 ), 1, 1 )
        fneval2 = frompyfunc( lambda x: eval( self.expression2 ), 1, 1 )
        fneval3 = frompyfunc( lambda x: eval( self.expression3 ), 1, 1 )
        self.mfn.lines[0] = MFnLineArray(xdata = xdata, ydata = fneval1( xdata ))
        self.mfn.lines[1] = MFnLineArray(xdata = xdata, ydata = fneval2( xdata ))
        self.mfn.lines[2] = MFnLineArray(xdata = xdata, ydata = fneval3( xdata ))
        self.mfn.data_changed = True
        
    mfn = Instance( MFnMultiLine )
    def _mfn_default( self ):
        return MFnMultiLine()
    
    @on_trait_change('expression' )
    def update_mfn(self):
        self._refresh_fired()
    
    view_mpl = View( HGroup( Item( 'expression1' ),
                             Item( 'expression2' ),
                             Item( 'expression3' ),
                             Item('refresh' ) ),
                 Item( 'mfn', editor = MFnMatplotlibEditor(  
                        adapter = a), show_label = False ),
                 resizable = True,
                 scrollable = True,
                 height = 0.5, width = 0.5
                    )
    
    view_chaco = View( HGroup( Item( 'expression1' ),
                               Item( 'expression2' ),
                               Item( 'expression3' ), 
                               Item('refresh' ) ),
                               
                 Item( 'mfn', editor = MFnChacoEditor(adapter = a), 
                       resizable = True, show_label = False ),       
                 resizable = True,
                 scrollable = True,
                 height = 0.5, width = 0.5
                    )

if __name__ == '__main__':
    fn = AnalyticalFunction()
    fn._refresh_fired()
    fn.configure_traits( view = "view_mpl", kind = 'nonmodal' )
   # fn.configure_traits( view = "view_chaco", kind = 'nonmodal' )
    