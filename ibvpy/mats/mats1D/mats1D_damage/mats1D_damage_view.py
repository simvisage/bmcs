#-------------------------------------------------------------------------------
#
# Copyright (c) 2009, IMB, RWTH Aachen.
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in simvisage/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.simvisage.com/licenses/BSD.txt
#
# Thanks for using Simvisage open source!
#
# Created on Mar 2, 2010 by: rch

from traits.api import HasTraits, Float, Property, cached_property, \
                                Instance, List, on_trait_change, Int, Tuple, Bool, \
                                DelegatesTo, Event, Str, Button, Dict, Array, Any
from traitsui.api import \
    View, Item, Tabbed, VGroup, HGroup, ModelView, HSplit, VSplit, \
    CheckListEditor, EnumEditor, TableEditor, TabularEditor,  Handler

from etsproxy.traits.ui.menu import Action, CloseAction, HelpAction, Menu, \
                                     MenuBar, NoButtons, Separator, ToolBar                    

from etsproxy.traits.ui.tabular_adapter \
    import TabularAdapter
    
from etsproxy.pyface.api import ImageResource
    
from etsproxy.traits.ui.menu import \
    OKButton

from numpy import array, linspace, frompyfunc, zeros, column_stack, \
                    log as ln, append, logspace, hstack, sign, trapz, mgrid, c_, \
                    zeros
                    
from math import exp, e, sqrt, log, pi
from scipy.special import erf, gamma
from scipy.stats import norm, weibull_min, uniform
from scipy.optimize import brentq, newton
from util.traits.editors.mpl_figure_editor import MPLFigureEditor
from matplotlib.figure import Figure
import string
from mats1D_damage import MATS1DDamage

class MATS1DDamageView( ModelView ):
    '''
    View into the parametric space constructed over the model.

    The is associated with the PStudySpace instance covering the
    factor ranges using an n-dimensional array.
    
    The view is responsible for transferring the response values
    into 2D and 3D plots. Depending on the current view specification
    it also initiates the calculation of response values in the 
    currently viewed subspace of the study. 
    '''

    model = Instance( MATS1DDamage )

    max_eps  = Float( 0.01, 
                      enter_set = True,
                      auto_set = False,                      
                      modified = True )

    n_points = Int( 100, 
                    enter_set = True,
                    auto_set = False,                    
                    modified = True )

    data_changed = Event( True )
    
    @on_trait_change('+modified,model.+modified')
    def _redraw(self):

        get_omega = frompyfunc( self.model._get_omega, 1, 1 )
        xdata = linspace( 0, self.max_eps, self.n_points )
        ydata = get_omega( xdata )
        
        
        axes = self.figure.axes[0]
        axes.clear()        

        axes.plot( xdata, ydata, color = 'blue', linewidth = 3 )

        self.data_changed = True
    
    #---------------------------------------------------------------
    # PLOT OBJECT
    #-------------------------------------------------------------------
    figure = Instance(Figure)
    def _figure_default(self):
        figure = Figure( facecolor = 'white' )
        figure.add_axes([0.12, 0.13, 0.85, 0.74])
        return figure

    traits_view = View( HSplit( 
                               VGroup(
                                     Item( 'model@', show_label = False, resizable = True ),
                                     label = 'material parameters',
                                     id = 'mats.viewmodel.model',
                                     dock = 'tab',
                                     ),                                 
                               VSplit( 
                                    VGroup(
                                        Item('figure',  editor=MPLFigureEditor(),
                                             resizable=True, show_label = False ),
                                             label = 'plot sheet',
                                            id = 'mats.viewmodel.figure_window',
                                            dock = 'tab',
                                        ),
                                    VGroup(
                                      HGroup(
                                       Item( 'max_eps' , label = 'maximum epsilon [-]',
                                             springy = True ),
                                       Item( 'n_points', label = 'plot points',
                                             springy = True ),
                                       ),
                                       label = 'plot parameters',
                                       id = 'mats.viewmodel.view_params',
                                       dock = 'tab',                                       
                                       ),
                                       id = 'mats.viewmodel.right',                                       
                                     ),
                            id = 'mats.viewmodel.splitter',        
                        ),
                        title = 'Yarn Size Effect',
                        id = 'yse.viewmodel',
                        dock = 'tab',                        
                        resizable = True,
                        height = 0.8, width = 0.8,
                        buttons = [OKButton]) 

if __name__ == '__main__':
    #-------------------------------------------------------------------------------
    # Example 
    #-------------------------------------------------------------------------------
    
    from ibvpy.mats.mats1D.mats1D_explore import MATS1DExplore
    from ibvpy.mats.mats_explore import MATSExplore
    from ibvpy.plugins.ibvpy_app import IBVPyApp
    
    mats_eval  = MATS1DDamageView( model = MATS1DDamage() )
    mats_eval.configure_traits()
    
