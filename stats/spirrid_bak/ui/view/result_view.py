'''
Created on 13.05.2011

@author: rrypl
'''

from etsproxy.traits.api import HasTraits, Str, Instance, Event, Button, on_trait_change, Int
from etsproxy.traits.ui.api import View, Item, ModelView, HGroup, VGroup
from matplotlib.figure import Figure
from util.traits.editors.mpl_figure_editor import MPLFigureEditor
from stats.spirrid_bak import SPIRRIDModelView
from numpy import repeat, array, sqrt

class ResultView( HasTraits ):

    spirrid_view = Instance( SPIRRIDModelView )

    title = Str( 'result plot' )

    n_samples = Int( 10 )

    figure = Instance( Figure )
    def _figure_default( self ):
        figure = Figure( facecolor = 'white' )
        #figure.add_axes( [0.08, 0.13, 0.85, 0.74] )
        return figure

    data_changed = Event( True )

    clear = Button
    def _clear_fired( self ):
        axes = self.figure.axes[0]
        axes.clear()
        self.data_changed = True

    def get_rvs_theta_arr( self, n_samples ):
        rvs_theta_arr = array( [ repeat( value, n_samples ) for value in self.spirrid_view.model.rf.param_values ] )
        for idx, name in enumerate( self.spirrid_view.model.rf.param_keys ):
            rv = self.spirrid_view.model.rv_dict.get( name, None )
            if rv:
                rvs_theta_arr[ idx, :] = rv.get_rvs_theta_arr( n_samples )
        return rvs_theta_arr

    sample = Button( desc = 'Show samples' )
    def _sample_fired( self ):
        n_samples = 20

        self.spirrid_view.model.set( 
                    min_eps = 0.00, max_eps = self.spirrid_view.max_eps, n_eps = self.spirrid_view.n_eps,
                    )

        # get the parameter combinations for plotting
        rvs_theta_arr = self.get_rvs_theta_arr( n_samples )

        eps_arr = self.spirrid_view.model.eps_arr

        figure = self.figure
        axes = figure.gca()

        for theta_arr in rvs_theta_arr.T:
            q_arr = self.spirrid_view.model.rf( eps_arr, *theta_arr )
            axes.plot( eps_arr, q_arr, color = 'grey' )

        self.data_changed = True

    @on_trait_change( 'spirrid_view.data_changed' )
    def _redraw( self ):

        figure = self.figure
        axes = figure.gca()

        mc = self.spirrid_view.model.mean_curve
        xdata = mc.xdata
        mean_per_fiber = mc.ydata
        # total expectation for independent variables = product of marginal expectations
        mean = mean_per_fiber * self.spirrid_view.mean_parallel_links

        axes.set_title( self.spirrid_view.plot_title, weight = 'bold' )
        axes.plot( xdata, mean,
                   linewidth = 2, label = self.spirrid_view.run_legend )

        if self.spirrid_view.stdev:
            # get the variance at x from SPIRRID
            variance = self.spirrid_view.model.var_curve.ydata

            # evaluate variance for the given mean and variance of parallel links
            # law of total variance D[xy] = E[x]*D[y] + D[x]*[E[y]]**2
            variance = self.spirrid_view.mean_parallel_links * variance + \
                    self.spirrid_view.stdev_parallel_links ** 2 * mean_per_fiber ** 2
            stdev = sqrt( variance )

            axes.plot( xdata, mean + stdev,
                       linewidth = 2, color = 'black', ls = 'dashed',
                       label = 'stdev' )
            axes.plot( xdata, mean - stdev,
                       linewidth = 2, ls = 'dashed', color = 'black' )
            axes.fill_between( xdata, mean + stdev, mean - stdev, color = 'lightgrey' )

        axes.set_xlabel( self.spirrid_view.label_x, weight = 'semibold' )
        axes.set_ylabel( self.spirrid_view.label_y, weight = 'semibold' )
        axes.legend( loc = 'best' )

        if xdata.any() == 0.:
            self.figure.clear()

        self.data_changed = True



    traits_view = View( HGroup( Item( 'n_samples', label = 'No of samples' ),
                                Item( 'sample', show_label = False, resizable = False ),
                                Item( 'clear', show_label = False, resizable = False,
                                    springy = False )
                              ),
                       Item( 'figure', show_label = False,
                            editor = MPLFigureEditor() )
                       )
