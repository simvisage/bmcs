'''
Created on May 11, 2009

@author: jakub
'''
import unittest
from ibvpy.mesh.fe_domain import FEDomain
from ibvpy.mesh.fe_refinement_grid import FERefinementGrid
from ibvpy.mesh.fe_grid import FEGrid
from ibvpy.fets.fets2D.fets2D4q import FETS2D4Q
from ibvpy.fets.fets3D.fets3D8h import FETS3D8H
from ibvpy.mats.mats2D.mats2D_elastic.mats2D_elastic import MATS2DElastic
from numpy import array

class FEDomainDynamicDofMap( unittest.TestCase ):
    '''
    Test class is to be used to test the sequence of changes of the FEDomain
    '''
    def setUp( self ):
        '''
        Construct the FEDomain with two FERefinementGrids (2,2) 
        '''
        self.domain1 = FEDomain()
        self.fets_eval = FETS2D4Q( mats_eval = MATS2DElastic() )
        self.d1 = FERefinementGrid( name = 'd1', domain = self.domain1 )
        g1 = FEGrid( coord_max = ( 1., 1., 0. ),
                     shape = ( 2, 2 ),
                     fets_eval = self.fets_eval,
                     level = self.d1 )
        self.d2 = FERefinementGrid( name = 'd2', domain = self.domain1 )
        g2 = FEGrid( coord_min = ( 1., 0., 0. ),
                     coord_max = ( 2., 1., 0. ),
                     shape = ( 2, 2 ),
                     fets_eval = self.fets_eval,
                     level = self.d2 )

    def test_n_dofs( self ):
        '''Verify the n_dofs (it implicitly verifies the dof_offset)
        '''
        n_dofs = self.domain1.n_dofs
        self.assertEqual( n_dofs, 36 )

    def test_rg_addition( self ):
        '''Check numbering after addition of FERefinementGrid
        Add another FERefinementGrid (2,2) as a child of grid 1
        Check the n_dofs of FEDomain to verify the re-enumeration 
        Check the elem_dof_map of the grid 3. 
        '''
        d3 = FERefinementGrid( name = 'd3', parent = self.d1 )
        g3 = FEGrid( coord_max = ( 1., 1., 0. ),
                     shape = ( 2, 2 ),
                     fets_eval = self.fets_eval,
                     level = d3 )
        n_dofs = self.domain1.n_dofs
        #check the n_dofs of the domain after addition
        self.assertEqual( n_dofs, 54 )
        #check elem_dof_map of added subdomain
        elem_dof_map = d3.elem_dof_map

        edm = [36, 37, 42, 43, 44, 45, 38, 39,
               38, 39, 44, 45, 46, 47, 40, 41,
               42, 43, 48, 49, 50, 51, 44, 45,
               44, 45, 50, 51, 52, 53, 46, 47]

        for e_, e_ex_ in zip( elem_dof_map.flatten() , edm ):
            self.assertEqual( e_, e_ex_ )

class FEDomainAdaptiveRefinement( unittest.TestCase ):
    '''
    Test class for adaptive refinement:

    Using the FEDomain with single parent FERefinementGrid and one 
    child FERefinementGrid perform refinement of elements within 
    the two grids and test the activation and deactivation. This 
    should be done by verifying the n_dofs and elem_dof map obtained 
    from both grids. 
    '''
    pass

class FEDomainGeoMap( unittest.TestCase ):
    '''
    Test the retrieval of geometric information of FEDomain.
    '''
    def setUp( self ):
        '''
        Construct the FEDomain with one FERefinementGrids (2,2) 
        '''
        self.domain1 = FEDomain()
        self.fets_eval = FETS2D4Q( mats_eval = MATS2DElastic() )
        self.d1 = FERefinementGrid( name = 'd1', domain = self.domain1 )
        self.g1 = FEGrid( coord_max = ( 1., 1., 0. ),
                     shape = ( 2, 2 ),
                     fets_eval = self.fets_eval,
                     level = self.d1 )

    def test_elem_X_map( self ):
        '''Test the retrieval of geometric information of FEDomain.
        '''
        elem_X_map = self.g1.elem_X_map
        egm = [ 0., 0., 0.5, 0., 0.5, 0.5, 0., 0.5, 0., 0.5, 0.5, 0.5, 0.5, 1., 0., 1., \
               0.5, 0., 1., 0., 1., 0.5, 0.5, 0.5, 0.5, 0.5, 1., 0.5, 1., 1., 0.5, 1.]
        for e_, e_ex_ in zip( elem_X_map.flatten() , egm ):
            self.assertAlmostEqual( e_, e_ex_ )


class FEDomainSliceTest( unittest.TestCase ):
    '''
    Test the slicing of a simple FEGrid.
    '''
    def setUp( self ):
        '''
        Construct the FEDomain with one FERefinementGrids (2,2) 
        '''
        self.fets_eval = FETS3D8H()
        self.grid = FEGrid( coord_max = ( 1., 1., 1. ),
                     shape = ( 1, 1, 1 ),
                     fets_eval = self.fets_eval )

    def test_dof_slices( self ):
        '''Test the retrieval of dofs on a slice. 
        Check the dofs retrieved on a cube with a single element on every
        axis plane. 
        '''

        result = array( [[[ 0, 1, 2],
                       [12, 13, 14],
                       [ 6, 7, 8],
                       [18, 19, 20]],
                      [[ 0, 1, 2],
                       [12, 13, 14],
                       [ 3, 4, 5],
                       [15, 16, 17]],
                      [[ 0, 1, 2],
                       [ 6, 7, 8],
                       [ 3, 4, 5],
                       [ 9, 10, 11]]], dtype = int )

        fe_grid_slice = self.grid[:, :, 0, :, :, 0 ] # xy plane  1
        for dof_, dof_ex_ in zip( result[0].flatten() , fe_grid_slice.dofs.flatten() ):
            self.assertAlmostEqual( dof_, dof_ex_ )

        fe_grid_slice = self.grid[:, 0   , :, :, 0   , :] # xz plane  1
        for dof_, dof_ex_ in zip( result[1].flatten() , fe_grid_slice.dofs.flatten() ):
            self.assertAlmostEqual( dof_, dof_ex_ )

        fe_grid_slice = self.grid[0, :, :, 0   , :, :] # yz plane  1
        for dof_, dof_ex_ in zip( result[2].flatten() , fe_grid_slice.dofs.flatten() ):
            self.assertAlmostEqual( dof_, dof_ex_ )
