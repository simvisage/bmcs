from etsproxy.traits.api import HasTraits, Str, Instance, Float, \
    Event, Int, Bool, Button
from etsproxy.traits.ui.api import View, Item, \
    HGroup, Tabbed, VGroup, ModelView, Group, HSplit

from stats.spirrid_bak.old.spirrid import SPIRRID
from math import pi

# -------------------------------
#        SPIRRID VIEW
# -------------------------------

class NoOfFibers( HasTraits ):
    Lx = Float( 16., auto_set = False, enter_set = True, desc = 'x-Length of the specimen in [cm]' )
    Ly = Float( 4., auto_set = False, enter_set = True, desc = 'y-Length of the specimen in [cm]' )
    Lz = Float( 4., auto_set = False, enter_set = True, desc = 'z-Length of the specimen in [cm]' )
    Fiber_volume_fraction = Float( 3.0 , auto_set = False, enter_set = True, desc = 'Fiber volume fraction in [%]' )
    Fiber_Length = Float( 17. , auto_set = False, enter_set = True, desc = 'Fiber length in [cm] ' )
    Fiber_diameter = Float( 0.15 , auto_set = False, enter_set = True, desc = 'Fiber diameter in [mm]' )


class SPIRRIDModelView( ModelView ):

    title = Str( 'spirrid exec ctrl' )

    model = Instance( SPIRRID )

    ins = Instance( NoOfFibers )

    def _ins_default( self ):
        return NoOfFibers()

    eval = Button

    def _eval_fired( self ):

        Specimen_Volume = self.ins.Lx * self.ins.Ly * self.ins.Lz
        self.no_of_fibers_in_specimen = ( Specimen_Volume * self.ins.Fiber_volume_fraction / 100 ) / ( pi * ( self.ins.Fiber_diameter / 20 ) ** 2 * self.ins.Fiber_Length / 10 )
        prob_crackbridging_fiber = ( self.ins.Fiber_Length / ( 10 * 2 ) ) / self.ins.Lx
        self.mean_parallel_links = prob_crackbridging_fiber * self.no_of_fibers_in_specimen
        self.stdev_parallel_links = ( prob_crackbridging_fiber * self.no_of_fibers_in_specimen * ( 1 - prob_crackbridging_fiber ) ) ** 0.5




    run = Button( desc = 'Run the computation' )
    def _run_fired( self ):
        self.evaluate()

    run_legend = Str( 'mean response',
                     desc = 'Legend to be added to the plot of the results' )

    min_eps = Float( 0.0,
                     desc = 'minimum value of the control variable' )

    max_eps = Float( 1.0,
                     desc = 'maximum value of the control variable' )

    n_eps = Int( 100,
                 desc = 'resolution of the control variable' )

    plot_title = Str( 'response',
                    desc = 'diagram title' )

    label_x = Str( 'epsilon',
                    desc = 'label of the horizontal axis' )

    label_y = Str( 'sigma',
                    desc = 'label of the vertical axis' )

    stdev = Bool( True )

    mean_parallel_links = Float( 1.,
                          desc = 'mean number of parallel links (fibers)' )
    stdev_parallel_links = Float( 0.,
                          desc = 'stdev of number of parallel links (fibers)' )
    no_of_fibers_in_specimen = Float( 0., desc = 'Number of Fibers in the specimen', )

    data_changed = Event( True )

    def evaluate( self ):
        self.model.set( 
                    min_eps = 0.00, max_eps = self.max_eps, n_eps = self.n_eps,
                )

        # evaluate the mean curve
        self.model.mean_curve

        # evaluate the variance if the stdev bool is True
        if self.stdev:
            self.model.var_curve
        self.data_changed = True

    traits_view = View( VGroup( 
                              HGroup( 
                                    Item( 'run_legend', resizable = False, label = 'Run label',
                                          width = 80, springy = False ),
                                    Item( 'run', show_label = False, resizable = False )
                                      ),
                               Tabbed( 
                                VGroup( 
                                       Item( 'model.cached_dG' , label = 'Cached weight factors',
                                             resizable = False,
                                             springy = False ),
                                       Item( 'model.compiled_QdG_loop' , label = 'Compiled loop over the integration product',
                                             springy = False ),
                                       Item( 'model.compiled_eps_loop' ,
                                             enabled_when = 'model.compiled_QdG_loop',
                                             label = 'Compiled loop over the control variable',
                                             springy = False ),
                                        scrollable = True,
                                       label = 'Execution configuration',
                                       id = 'spirrid.tview.exec_params',
                                       dock = 'tab',
                                     ),
                                VGroup( 
                                       HGroup( 
                                              Item( 'min_eps' , label = 'Min',
                                                       springy = False, resizable = False ),
                                              Item( 'max_eps' , label = 'Max',
                                                       springy = False, resizable = False ),
                                              Item( 'n_eps' , label = 'N',
                                                       springy = False, resizable = False ),
                                              label = 'Simulation range',
                                              show_border = True
                                             ),
                                        HGroup( 
                                               Item( 'stdev', label = 'plot standard deviation' ),
                                                    ),
                                               HSplit( 
                                                      HGroup( VGroup( Item( 'mean_parallel_links', label = 'mean No of fibers' ),
                                                                     Item( 'stdev_parallel_links', label = 'stdev No of fibers' ),
                                                                     )
                                                      ),
                                                      VGroup( Item( '@ins', label = 'evaluate No of fibers' , show_label = False ),
                                                              VGroup( HGroup( Item( 'eval', show_label = False, resizable = False, label = 'Evaluate No of Fibers' ),
                                                                     Item( 'no_of_fibers_in_specimen',
                                                                           label = 'No of Fibers in specimen',
                                                                           style = 'readonly' ) ) )
                                                      ),
                                                label = 'number of parralel fibers',
                                                show_border = True,
                                                scrollable = True, ),
                                          VGroup( 
                                                 Item( 'plot_title' , label = 'title', resizable = False,
                                                 springy = False ),
                                                 Item( 'label_x' , label = 'x', resizable = False,
                                                 springy = False ),
                                                 Item( 'label_y' , label = 'y', resizable = False,
                                                 springy = False ),
                                                 label = 'title and axes labels',
                                                 show_border = True,
                                                 scrollable = True,
                                                 ),
                                           label = 'Execution control',
                                           id = 'spirrid.tview.view_params',
                                           dock = 'tab',
                                 ),
                                scrollable = True,
                                id = 'spirrid.tview.tabs',
                                dock = 'tab',
                        ),
                        ),
                        title = 'SPIRRID',
                        id = 'spirrid.viewmodel',
                        dock = 'tab',
                        resizable = True,
                        height = 1.0, width = 1.0
                        )

if __name__ == '__main__':
    pass
