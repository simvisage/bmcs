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
# Created on Aug 18, 2009 by: rch

from traits.api import HasTraits, Instance

import unittest

from ibvpy.fets.fets2D.fets2D4q import \
    FETS2D4Q
from ibvpy.util.simgrid import \
    simgrid
from ibvpy.mats.mats_eval import MATSEval

class TestMATS2D(unittest.TestCase, HasTraits):

    mats_eval = Instance( MATSEval )
    
    def setUp(self):

        self.fets_eval  = FETS2D4Q( mats_eval  = self.mats_eval )

        self.support_slices = [
                          [ (0   ,slice(None),0   ,slice(None)), # y axis 0
                            (0   ,0          ,0   ,0          ), # (0,0)   1
                          ],
                          [ 
                            (0   ,0   ,0   ,0      ), #  origin   0
                            (slice(None),0   ,slice(None),0   ), # xz plane  1
                          ],
                          ]
        self.support_dirs = [[0],[1]]
        
        self.loading_slices = [ 
                          (-1  ,slice(None),-1  ,slice(None)),  # loading in x dir
                          (slice(None),-1  ,slice(None),-1  ),  # loading in y dir
                        ]

    def assert_symmetry(self, load_dirs = [0,1] ):
        '''Assert that the symmetry is given for the applied loadings.
        '''
        load = 0.0001

        r = [ simgrid( self.fets_eval,  (1,1), (1,1),
                       support_slice, self.support_dirs,
                       loading_slice, load_dir, 
                       load, 1, vars = [] )
              for support_slice, loading_slice, load_dir 
              in zip( self.support_slices, self.loading_slices, load_dirs ) ]
        
        self.assertAlmostEqual( r[0][1][-1], r[1][1][-2] )

    def assert_stress_value(self,
                            sig_expected,  
                            n_steps = 3, 
                            load = 0.0001,
                            ):
        '''Assert that the symmetry is given for the applied loadings.
        '''
        vars = ['sig_app']

        r = simgrid( self.fets_eval,  (1,1), (1,1),
                       self.support_slices[0], self.support_dirs,
                       self.loading_slices[0], 0, 
                       load, n_steps, vars )

        sig_end = r[2][0]
        for sig, sig_exp in zip( sig_end.flatten(), sig_expected ):
            self.assertAlmostEqual( sig, sig_exp, 5 )

    def assert_total_energy_value(self,
                                         value_expected,  
                                         ivar = 'strain_energy',
                                         n_steps = 3, 
                                         load = 1.0,
                            ):
        '''Assert that the symmetry is given for the applied loadings.
        '''
        ivars = [ivar]
        r = simgrid( self.fets_eval,  (1,1), (1,1),
                     self.support_slices[0], self.support_dirs,
                     self.loading_slices[0], 0, 
                     load, n_steps, ivars = ivars )

        self.assertAlmostEqual( r[3][0][0], value_expected )
        
        