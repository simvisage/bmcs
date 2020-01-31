
from traits.api import HasTraits, Instance, Delegate
from traitsui.api import View, Item, VSplit, Group
from .tloop import TLoop

class IBVPSolve( HasTraits ):
    ''' Manage the installation source tree
    DEPRECATED '''
    tloop = Instance( TLoop )
    def _tloop_default(self):
        ''' Default constructor'''
        return TLoop()
    
    tstepper   = Delegate( 'tloop' )
    rtrace_mngr = Delegate( 'tloop' )

    view = View( Group( Item( name   = 'tloop', style = 'custom', show_label = False ),
                        label = 'Sim-Control' ),
                 Group( Item( name   = 'tstepper', style = 'custom', show_label = False) ,
                        label = 'Sim-Model' ),
                 Group( Item( name   = 'rtrace_mngr', style = 'custom', show_label = False ),
                        label = 'Sim-Views' ),
                 title = 'IBVP-Solver',
                 buttons = ['OK'],
                 resizable = True,
                 scrollable = True,
                 style     = 'custom',
                 x         = 0.,
                 y         = 0.,
                 width     = .7,
                 height    = 0.8
                 )

if __name__ == '__main__':
    ibvpy = IBVPSolve()
    ibvpy.configure_traits()
    
    