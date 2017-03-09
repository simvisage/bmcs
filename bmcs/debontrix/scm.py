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
# Created on Sep 21, 2009 by: rch

from enthought.traits.api import HasTraits, Float, Property, cached_property, Instance
from enthought.traits.ui.api import View, Item, Tabbed, VGroup, HGroup

from mathkit.mfn import MFnLineArray
from mathkit.mfn.mfn_line.mfn_matplotlib_editor import MFnMatplotlibEditor 
from mathkit.mfn.mfn_line.mfn_plot_adapter import MFnPlotAdapter
from numpy import linspace, frompyfunc, zeros, column_stack
from reinf_cross_section import SimplyRatio, GridReinforcement
from math import exp
from util.traits.either_type import EitherType

a = MFnPlotAdapter( max_size = (1200,800), 
                    padding = {'top': 0.95,
                               'left' : 0.1,
                               'bottom' : 0.1,
                               'right' : 0.95} )

class SCM( HasTraits ):
    '''
    Stochastic Cracking Theory
    '''
    E_f = Float( 70e+3, auto_set = False, enter_set = True, # [N/mm^2]
                 desc = 'E-Modulus of the fiber' )
    
    E_m = Float( 34.e+3, auto_set = False, enter_set = True, # [N/mm^2]
                 desc = 'E-Modulus of the matrix' )
    
    tau = Float( 8.0, auto_set = False, enter_set = True,  # [N/mm^2]
                 desc = 'Frictional stress' )
    
    r   = Float( 0.5, auto_set = False, enter_set = True, # [mm]
                 desc = 'Radius' )
    
    rho = Float( 0.03, auto_set = False, enter_set = True, # [-]
                 desc = 'Reinforcement ratio' )    

    reinf_cs = EitherType( klasses = [ SimplyRatio, GridReinforcement] )

    m   = Float( 4.0, auto_set = False, enter_set = True, # [-]
                 desc = 'Weibull modulus' )    

    sigma_mu = Float( 3.0, auto_set = False, enter_set = True, # [N/mm^2]
                       desc = 'Matrix tensional strength' )    

    sigma_fu = Float( 800.0, auto_set = False, enter_set = True, # [N/mm^2]
                      desc = 'Fiber tensional strength' )    
     
    V_f = Property( Float, depends_on = 'rho' )
    @cached_property
    def _get_V_f(self):
        return self.rho

    V_m = Property( Float, depends_on = 'rho' )
    @cached_property
    def _get_V_m(self):
        return 1 - self.rho
    
    alpha = Property( Float, depends_on = 'E_m,E_f,rho' )
    @cached_property
    def _get_alpha(self):
        return ( self.E_m * self.V_m ) / ( self.E_f * self.V_f )
    
    E_c1 = Property( Float, depends_on = 'E_m,E_f,rho' )
    @cached_property
    def _get_E_c1(self):
        return self.E_f * self.V_f + self.E_m * self.V_m

    delta_final = Property( Float, depends_on = 'E_m,E_f,rho,r,sigma_mu,tau' )
    @cached_property
    def _get_delta_final( self ):
        return self.sigma_mu * ( self.V_m * self.r ) / ( self.V_f * 2 * self.tau ) 

    cs_final = Property( Float )
    def _get_cs_final(self):
        return 1.337 * self.delta_final

    def _get_delta(self, sigma_c ):
        return sigma_c * ( self.V_m * self.r * self.E_m ) / ( self.V_f * 2 * self.tau * self.E_c1 ) 

    def _get_cs( self, sigma_c ):
        '''Get crack spacing for current composite stress.
        '''
        Pf = ( 1 - exp( -( (sigma_c * self.E_m) / (self.sigma_mu * self.E_c1 ) )**self.m ) )
        if Pf == 0:
            Pf = 1e-15
        return self.cs_final * 1.0 / Pf
    
    def _get_epsilon_c( self, sigma_c ):
        '''Get composite strain for current composite stress.
        '''
        cs = self._get_cs( sigma_c )
        delta = self._get_delta( sigma_c )
        if cs > 2 * delta:
            return sigma_c / self.E_c1 * ( 1 + self.alpha * delta / cs )
        else:
            return sigma_c * ( 1 / (self.E_f * self.V_f) - 
                               ( self.alpha * cs ) / (4 * delta * self.E_c1 ) )
    
    mfn_plot = Property( Instance( MFnLineArray ), 
                         depends_on = 'E_m,E_f,tau,r,rho,m,sigma_mu, sigma_fu' )
    @cached_property
    def _get_mfn_plot(self):
        n_points = 100
        sigma_max = self.sigma_fu * self.rho
        
        sigma_arr = linspace( 0, sigma_max, n_points )
        
        get_epsilon_f = frompyfunc( lambda sigma: sigma / self.E_f, 1, 1 ) 
        epsilon_f_arr  = get_epsilon_f( sigma_arr )
        
        get_epsilon_c = frompyfunc( self._get_epsilon_c, 1, 1 )
        epsilon_c_arr = get_epsilon_c( sigma_arr )
        
        return MFnLineArray( xdata = epsilon_c_arr, ydata = sigma_arr )
    
    traits_view = View( HGroup( 
                               VGroup(
                               Item( 'E_f' ),
                               Item( 'E_m' ),  
                               Item( 'tau' ),  
                               Item( 'r' ),  
                               Item( 'rho' ),  
                               Item( 'm' ),  
                               Item( 'sigma_mu' ),  
                               Item( 'sigma_fu' ),  
                                     label = 'Material parameters',
                                     id = 'scm.params',
                                     ),
                               VGroup(
                               Item( 'mfn_plot', resizable = True, show_label = False,
                                     editor = MFnMatplotlibEditor( adapter = a ) ),

                                     label = 'Strain hardening response',
                                     id = 'scm.plot',
                                     ),
                        ),
                        id = 'scm',
                        dock = 'horizontal',
                        resizable = True,
                        height = 0.8, width = 0.8 ) 
            
def run():
    scm = SCM()
    scm.configure_traits()
    
if __name__ == '__main__':
    run()
    
    