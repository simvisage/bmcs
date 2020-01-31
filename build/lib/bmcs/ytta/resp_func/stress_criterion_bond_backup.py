import wx
import matplotlib
# We want matplotlib to use a wxPython backend
matplotlib.use( 'WXAgg' )
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from enthought.traits.api import Instance, HasTraits, Event, Float, DelegatesTo, on_trait_change
from enthought.traits.ui.wx.editor import Editor
from enthought.traits.ui.wx.basic_editor_factory import BasicEditorFactory
from numpy import array
from enthought.traits.ui.api import View, Item
from matplotlib.figure import Figure
from .parameters import Material

class _BondLawEditor( Editor ):

    def init( self, parent ):
        self.control = self._create_canvas( parent )
        self.object.on_trait_change( self.update_editor, 'data_changed' )
        self.set_tooltip()

    def update_editor( self ):
        figure = self.value
        figure.canvas.draw()

    def _create_canvas( self, parent ):
        """ Create the MPL canvas. """
        # The panel lets us add additional controls.
        panel = wx.Panel( parent, -1, style = wx.CLIP_CHILDREN )
        sizer = wx.BoxSizer( wx.VERTICAL )
        panel.SetSizer( sizer )

        # matplotlib commands to create a canvas
        mpl_control = FigureCanvas( panel, -1, self.value )
        #toolbar = NavigationToolbar2Wx(mpl_control)
        #sizer.Add(toolbar, 0, wx.EXPAND)
        sizer.Add( mpl_control, 1, wx.LEFT | wx.TOP | wx.GROW )
        self.value.canvas.SetMinSize( ( 300, 200 ) )
        return panel

class BondLawEditor( BasicEditorFactory ):

    klass = _BondLawEditor


class StressCriterionBond ( HasTraits ):
    '''Response of an elastic brittle filament pulled out
    from the matrix with a bond represented by a simple
    constitutive law.'''

    def __init__( self, material, **kw ):
        super( StressCriterionBond, self ).__init__( **kw )
        self.material = material
        self._redraw()
        self.on_trait_change( self._redraw, 'qf,qy,k' )

    material = Instance( Material )

    k = DelegatesTo( 'material', modified = True )
    qf = DelegatesTo( 'material', modified = True )
    qy = DelegatesTo( 'material', modified = True )

    figure = Instance( Figure )
    def _figure_default( self ):
        figure = Figure( facecolor = 'white' )
        figure.add_axes( [0.17, 0.2, 0.8, 0.68] )
        return figure

    data_changed = Event( True )

    def _redraw( self ):
        slip_qy = self.qy / self.k
        xdata = array( [0, slip_qy, slip_qy, 3 * slip_qy] )
        ydata = [0, self.qy, self.qf , self.qf]
        figure = self.figure

        axes = figure.axes[0]
        axes.clear()

        axes.plot( xdata, ydata, color = 'black', linewidth = 2, linestyle = 'solid' )
        axes.set_xlabel( 'slip [m]', weight = 'semibold' )
        axes.set_ylabel( 'shear stress [N/m]', weight = 'semibold' )
        axes.set_title( 'bond law', \
                        size = 'large', color = 'black', \
                        weight = 'bold', position = ( .5, 1.03 ) )
        axes.set_axis_bgcolor( color = 'white' )
        axes.ticklabel_format( scilimits = ( -3., 4. ) )
        self.data_changed = True

    traits_view = View( Item( 'figure', editor = BondLawEditor(),
                            resizable = True, show_label = False ) )

if __name__ == '__main__':
    bond = StressCriterionBond( Material() )
    bond.configure_traits()
