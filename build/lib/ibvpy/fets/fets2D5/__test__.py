'''
Created on Aug 8, 2009

@author: rch
'''
import unittest
from ibvpy.api import \
    TStepper as TS, RTDofGraph, RTraceDomainListField, TLoop, \
    TLine, BCDofGroup, IBVPSolve as IS, DOTSEval, BCSlice

from ibvpy.mesh.fe_grid import FEGrid

from ibvpy.mats.mats3D.mats3D_elastic.mats3D_elastic import MATS3DElastic
from ibvpy.mats.mats2D.mats2D_elastic.mats2D_elastic import MATS2DElastic

# linear
from ibvpy.fets.fets3D.fets3D8h import FETS3D8H
from ibvpy.fets.fets2D5.fets2D58h import FETS2D58H

# quadratic serendipity
from ibvpy.fets.fets3D.fets3D8h20u import FETS3D8H20U
from ibvpy.fets.fets2D5.fets2D58h20u import FETS2D58H20U

# quadratic/linear hybird
from ibvpy.fets.fets3D.fets3D8h16u import FETS3D8H16U
from ibvpy.fets.fets2D5.fets2D58h16u import FETS2D58H16U

# cubic serendipity
from ibvpy.fets.fets3D.fets3D8h32u import FETS3D8H32U
from ibvpy.fets.fets2D5.fets2D58h32u import FETS2D58H32U

# cubic/linear hybird
from ibvpy.fets.fets3D.fets3D8h24u import FETS3D8H24U
from ibvpy.fets.fets2D5.fets2D58h24u import FETS2D58H24U


from ibvpy.util.simgrid import simgrid

from ibvpy.mats.mats2D5.mats2D5_cmdm.mats2D5_cmdm import \
    MATS2D5MicroplaneDamage

from ibvpy.mats.matsXD.matsXD_cmdm.matsXD_cmdm_phi_fn import \
    PhiFnStrainSoftening, PhiFnGeneral


class TestFETS2D5( unittest.TestCase ):


    def setUp( self ):
        # test elastic case: compare values of MATS2D5 with no damage (phi_fn = 1 (const.)).
        phi_fn = PhiFnGeneral()
        print(phi_fn.mfn.xdata)
        print(phi_fn.mfn.ydata)

        # linear elements
        self.fets_eval3D = FETS3D8H( mats_eval = MATS3DElastic( E = 34000., nu = 0.25 ) )
        self.fets_eval2D5 = FETS2D58H( mats_eval = MATS2D5MicroplaneDamage( E = 34000., nu = 0.25,
                                                                          model_version = "compliance",
                                                                          #model_version   = "stiffness",
                                                                          phi_fn = phi_fn ) )

#        # quadratic serendipity elements
#        self.fets_eval3D = FETS3D8H20U( mats_eval = MATS3DElastic( E = 34000., nu = 0.25 ) )
#        self.fets_eval2D5 = FETS2D58H20U( mats_eval = MATS2D5MicroplaneDamage( E = 34000., nu = 0.25,
#                                                                          phi_fn = phi_fn ) )

#        # quadratic/linear hybrid elements
#        self.fets_eval3D = FETS3D8H16U( mats_eval = MATS3DElastic( E = 34000., nu = 0.25 ) )
#        self.fets_eval2D5 = FETS2D58H16U( mats_eval = MATS2D5MicroplaneDamage( E = 34000., nu = 0.25,
#                                                                          phi_fn = phi_fn ) )

#        # cubic serendipity elements
#        self.fets_eval3D = FETS3D8H32U( mats_eval = MATS3DElastic( E = 34000., nu = 0.25 ) )
#        self.fets_eval2D5 = FETS2D58H32U( mats_eval = MATS2D5MicroplaneDamage( E = 34000., nu = 0.25,
#                                                                          phi_fn = phi_fn ) )

#        # cubic/linear hybrid elements
#        self.fets_eval3D = FETS3D8H24U( mats_eval = MATS3DElastic( E = 34000., nu = 0.25 ) )
#        self.fets_eval2D5 = FETS2D58H24U( mats_eval = MATS2D5MicroplaneDamage( E = 34000., nu = 0.25,
#                                                                          phi_fn = phi_fn ) )


    def test_uniform_loading( self ):

        support_slices = [
                          [ ( 0   , slice( None ), slice( None ), 0   , slice( None ), slice( None ) ), # yz plane  0
                            ( 0   , 0   , slice( None ), 0   , 0   , slice( None ) ), #  z-axis   1
                            ( 0   , 0   , 0, 0   , 0   , 0 )  #  origin   2
                          ],
                          [
                            ( 0   , 0   , 0   , 0   , 0   , 0 ), #  origin   0
                            ( slice( None ), 0   , slice( None ), slice( None ), 0   , slice( None ) ), # xz plane  1
                            ( slice( None ), 0   , 0   , slice( None ), 0   , 0 ), #  x-axis   2
                          ],
                          [
                            ( 0   , slice( None ), 0   , 0   , slice( None ), 0 ), #  y-axis   0
                            ( 0   , 0   , 0   , 0   , 0   , 0 ), #  origin   1
                            ( slice( None ), slice( None ), 0   , slice( None ), slice( None ), 0 ), # xz plane  2
                          ],
                          ]
        support_dirs = [[0], [1], [2]]

        loading_slices = [
                          ( -1  , slice( None ), slice( None ), -1  , slice( None ), slice( None ) ), # loading in x dir
                          ( slice( None ), -1  , slice( None ), slice( None ), -1  , slice( None ) ), # loading in y dir
                          ( slice( None ), slice( None ), -1  , slice( None ), slice( None ), -1 )   # loading in z dir
                        ]

        load_dirs = [0, 1, 2]
        load = 0.01
        vars = ['eps_app']
        for support_slice, loading_slice in zip( support_slices, loading_slices ):

            for load_dir in load_dirs:
                tl, u3D, fields3D, integs, g = simgrid( self.fets_eval3D, ( 3, 3, 3 ), ( 1, 1, 1 ),
                                           support_slice, support_dirs,
                                           loading_slice, load_dir,
                                           load, 1, vars )

                tl, u2D5, fields2D5, integs, g = simgrid( self.fets_eval2D5, ( 3, 3, 3 ), ( 1, 1, 1 ),
                                           support_slice, support_dirs,
                                           loading_slice, load_dir,
                                           load, 1, vars )

                for u1_, u2_ in zip( u3D.flatten(), u2D5.flatten() ):
                    self.assertAlmostEqual( u1_, u2_ )

    def tearDown( self ):
        pass


if __name__ == "__main__":
    unittest.main()
