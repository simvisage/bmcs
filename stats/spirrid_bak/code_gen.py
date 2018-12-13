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
# Created on Sep 8, 2011 by: rch

from etsproxy.traits.api import HasStrictTraits, WeakRef, Bool, on_trait_change, \
    Event
from stats.spirrid_bak import RV
import numpy as np # import numpy package


#===============================================================================
# Generator of the integration code 
#===============================================================================
class CodeGen(HasStrictTraits):
    '''
        Base class for generators of the integration code.
        
        The code may be scripted or compiled depending on the choice 
        of the class. 
    '''
    # backward link to the spirrid object
    spirrid = WeakRef

    recalc = Event

    #===========================================================================
    # Consfiguration of the algorithm
    #===========================================================================
    implicit_var_eval = Bool(False, codegen_option = True)

    #===========================================================================
    # Propagate the change to the spirrid
    #===========================================================================
    @on_trait_change('+codegen_option')
    def set_codegen_option_changed(self):
        self.spirrid.codegen_option = True

if __name__ == '__main__':

    from stats.spirrid_bak import SPIRRID
    from spirrid_lab import Heaviside

    class fiber_tt_2p:

        def __call__(self, e, la, xi = 0.017):
            ''' Response function of a single fiber '''
            return la * e * Heaviside(xi - e)

        C_code = '''
                // Computation of the q( ... ) function
                if ( eps > xi ){
                    q = 0.0;
                }else{
                    q = la * eps;
                }
            '''

    s = SPIRRID(q = fiber_tt_2p(),
                 codegen_type = 'numpy',
                 sampling_type = 'LHS',
                 e_arr = [np.linspace(0, 1, 10)],
                 tvars = dict(la = 1.0, # RV( 'norm', 10.0, 1.0 ),
                               xi = RV('norm', 1.0, 0.1))
                )

    s.codegen.implicit_var_eval = True

    print(('var_names', s.var_names))
    print(('var_defaults', s.var_defaults))

    print((s.mu_q_arr))
    print((s.var_q_arr))

    #===========================================================================
    # Response  
    #===========================================================================

    from quaducom.resp_func.cb_clamped_fiber import \
        CBClampedFiberSP

    q = CBClampedFiberSP()

    s = SPIRRID(q = q,
                 sampling_type = 'LHS',
                 implicit_var_eval = False,
                 evars = dict(w = np.linspace(0.0, 0.4, 50),
                              x = np.linspace(-20.1, 20.5, 100),
                              Lr = np.linspace(0.1, 20.0, 50)
                              ),
                 tvars = dict(tau = RV('uniform', 0.7, 1.0),
                              l = RV('uniform', 5.0, 10.0),
                              D_f = 26e-3,
                               E_f = 72e3,
                               theta = 0.0,
                               xi = RV('weibull_min', scale = 0.017, shape = 8, n_int = 10),
                               phi = 1.0,
                               Ll = 50.0,
#                              Lr = 1.0
                               ),
                            n_int = 3)


    from stats.spirrid_bak import make_ogrid
    from time import sleep

    e_arr = make_ogrid(s.evar_lst)
    n_e_arr = [ e / np.max(np.fabs(e)) for e in e_arr ]

    max_mu_q = np.max(np.fabs(s.mu_q_arr))
    n_mu_q_arr = s.mu_q_arr / max_mu_q
    n_std_q_arr = np.sqrt(s.var_q_arr) / max_mu_q

    import etsproxy.mayavi.mlab as m

    f = m.figure(1, size = (1000, 500), fgcolor = (0, 0, 0),
                 bgcolor = (1., 1., 1.))

    s = m.surf(n_e_arr[1], n_e_arr[2], n_mu_q_arr[0, :, :])

    m.axes(s, color = (.7, .7, .7),
           extent = (-1, 1, 0, 1, 0, 1),
              ranges = (-0.21, 0.21, 0.1, 20, 0, max_mu_q),
              xlabel = 'x', ylabel = 'Lr',
              zlabel = 'Force',)

    m.view(-60.0, 70.0, focalpoint = [0., 0.45, 0.45])
    # Store the information
    view = m.view()
    roll = m.roll()
    print(('view', view))
    print(('roll', roll))
    print((n_mu_q_arr.shape[2]))

    ms = s.mlab_source
    for i in range(1, n_mu_q_arr.shape[0]):
        ms.scalars = n_mu_q_arr[i, :, :]
        fname = 'x%02d.jpg' % i
        print(('saving', fname))
        m.savefig(fname)
        sleep(0.1)
#    m.surf( n_e_arr[0], n_e_arr[1], n_mu_q_arr + n_std_q_arr )
#    m.surf( n_e_arr[0], n_e_arr[1], n_mu_q_arr - n_std_q_arr )

    m.show()
