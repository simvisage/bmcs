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
# Created on Aug 19, 2009 by: rch


from traits.api import \
    HasTraits, Instance, List

import unittest

from ibvpy.fets.fets2D5.fets2D58h import \
    FETS2D58H
from ibvpy.util.simgrid import \
    simgrid
from ibvpy.mats.mats_eval import \
    MATSEval
from numpy import \
    array

class TestMATS2D5(unittest.TestCase, HasTraits):

    mats_eval = Instance( MATSEval )
    
    # values of displacements that must equal for the symmetry tests
    sym_assert_pattern = List( [( (0,0), (1,1) ),
                                ( (0,1), (1,0) ),
                                ( (0,2), (1,2) )] )
    
    def assert_2D_symmetry_clamped_cube(self, load_dirs, load = 0.001 ):
        '''Assert that the symmetry is given for the applied loadings.
        '''

        self.fets_eval  = FETS2D58H(mats_eval  = self.mats_eval)

        support_slices = [
                      [ 
                       (0   ,slice(None),slice(None),0   ,slice(None),slice(None)), #                             yz plane  0
                      ],
                      [ 
                       (slice(None),0   ,slice(None),slice(None),0   ,slice(None)), # xz plane  1
                      ],
                      ]
        support_dirs = [[0,1,2]]
        
        loading_slices = [
                      (-1, slice(None),slice(None),-1 ,slice(None),slice(None)),  # loading in x dir
                      (slice(None),-1 ,slice(None),slice(None),-1 ,slice(None)),  # loading in y dir
                        ]

        vars = [] 

        r = [ simgrid( self.fets_eval,  (1,1,1), (1,1,1),
                         support_slice, support_dirs,
                         loading_slice, load_dir, 
                         load, 1, vars )
                for support_slice, loading_slice, load_dir in zip( support_slices,
                                                                   loading_slices, 
                                                                   load_dirs ) ]

        u = array( [ r[i][1][-3:] for i in range(2) ] )
        for idx1, idx2 in self.sym_assert_pattern: 
            self.assertAlmostEqual( u[idx1], u[idx2] )

    def assert_stress_value(self,
                            sig_expected,  
                            n_steps = 3, 
                            load = 0.0001,
                            ):
        '''Assert that the symmetry is given for the applied loadings.
        '''
        self.fets_eval  = FETS2D58H(mats_eval  = self.mats_eval)        
        support_slices = [
                      [ 
                       (0   ,slice(None),slice(None),0   ,slice(None),slice(None)), # yz plane  0
                       (0   ,0          ,slice(None),0   ,0          ,slice(None)), # z plane  0
                       (0   ,0          ,0          ,0   ,0          ,0          ), # z plane  0
                      ],
                      ]
        support_dirs = [[0],[1],[2]]
        
        loading_slices = [
                      (-1, slice(None),slice(None),-1 ,slice(None),slice(None)),  # loading in x dir
                        ]

        load_dirs = [0]

        vars = [ 'sig_app' ]

        r = [ simgrid( self.fets_eval,  (1,1,1), (1,1,1),
                       support_slice, support_dirs,
                       loading_slice, load_dir, 
                       load, 1, vars )
                for support_slice, loading_slice, load_dir in zip( support_slices,
                                                                   loading_slices, 
                                                                   load_dirs ) ]

        for rr in r:
            sig_end = rr[2][0]
            # all material points must have the same value - uniform loading.
            for sig, sig_exp in zip( sig_end.flatten(), sig_expected ):
                self.assertAlmostEqual( sig, sig_exp, 5 )
