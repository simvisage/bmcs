'''
Created on Mar 29, 2010

@author: rostislav
'''
from enthought.traits.api import Instance, Event, Button, Int
from enthought.traits.ui.api import View, Item, VGroup, HSplit, ModelView
from enthought.traits.ui.file_dialog import open_file, save_file, FileInfo, \
    TextInfo
from enthought.traits.ui.menu import Action, CloseAction, HelpAction, Menu, \
    MenuBar, NoButtons, Separator, ToolBar, OKButton
from matplotlib.figure import Figure
from util.traits.editors.mpl_figure_editor import MPLFigureEditor
from .resp_func_model import RespFunc
import pickle


class RespFuncView( ModelView ):
     
    def __init__( self, **kw ):
        super( RespFuncView, self ).__init__( **kw )
        self._redraw()
        self.on_trait_change( self._redraw, 'model.values' )
        
    model = Instance( RespFunc )
    def _model_default( self ):
        return RespFunc()
    
    figure = Instance( Figure )
    def _figure_default( self ):
        figure = Figure( facecolor='white' )
        figure.add_axes( [0.08, 0.13, 0.85, 0.74] )
        return figure
    
    data_changed = Event ( True )
    
    count = Int
    
    def _redraw( self ):
        self.count += 1
        print('redraw', self.count)
        
        # data post-processing
        #
        xdata, ydata = self.model.values
        if self.model.approach.plot.yvalues == 'forces':
            ylabel = 'pull-out force [N]'
        else:
            ydata = ydata / self.model.approach.Af
            ylabel = 'pull-out stress [N/m2]'
        
        title = self.model.boundary.type.BC
        if title[0] == 'd':
            xlabel = 'crack opening [m]'
        else:
            xlabel = 'displacement [m]'
        figure = self.figure
        axes = figure.axes[0]
        axes.clear()
        
        axes.plot( xdata, ydata, color='blue', linewidth=2, linestyle='solid' )     
        axes.set_xlabel( xlabel, weight='semibold' )
        axes.set_ylabel( ylabel, weight='semibold' )
        axes.set_title( title, \
                        size='large', color='black', \
                        weight='bold', position=( .5, 1.03 ) )
        axes.set_axis_bgcolor( color='white' )
        axes.ticklabel_format( scilimits=( -3., 4. ) )
        axes.grid( color='gray', linestyle='--', linewidth=0.1, alpha=0.4 )
    
        self.data_changed = True
    
    plot_button = Button( 'Plot pull-out' )
    def _plot_button_fired( self ):
        self._redraw()
    
    traits_view = View( HSplit( 
                       VGroup( Item( '@model.boundary',
                                     show_label=False ),
                                Item( '@model.approach', show_label=False ),
                               id='rf.model',
                               dock='tab',
                               label='pull-out model',
                               ),

                        VGroup( 
                                Item( 'figure', editor=MPLFigureEditor(),
                                resizable=True, show_label=False ),
                                #Item('plot_button', show_label = False),
                                label='plot sheet',
                                id='rf.figure_window',
                                dock='tab',
                                ),
                        id='rf.viewmodel.hsplit',
                        ),
                title='Response Function',
                id='rf.viewmodel',
                dock='tab',
                kind='live',
                resizable=True,
                height=0.8, width=0.8,
                buttons=[OKButton] )

            
    def open( self, uiinfo ):        
        file_name = open_file( filter=['*.pst'],
                               extensions=[FileInfo(), TextInfo()] )
        if file_name == '':
            return

        file = open( file_name, 'r' )
        self.model = pickle.load( file )
        file.close()

    def save( self, uiinfo ):
        file_name = save_file( filter=['*.pst'],
                               extensions=[FileInfo(), TextInfo()] )
        if file_name == '':
            return

        print('writing into', file_name)
        file = open( file_name, 'w' )
        pickle.dump( self.model, file )
        file.close()
    
if __name__ == "__main__":
    rfv = RespFuncView( model = RespFunc() )
    rfv.configure_traits()
