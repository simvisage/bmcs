
#from sys_matrix import SysSparseMtx, SysDenseMtx
from numpy import array, zeros, arange, array_equal, hstack, dot, sqrt
from scipy.linalg import norm

from os import path
from mathkit.matrix_la.sys_mtx_assembly import SysMtxAssembly
from mathkit.matrix_la.coo_mtx import COOSparseMtx
from mathkit.matrix_la.dense_mtx import DenseMtx
import unittest
from tempfile import mkdtemp

from matresdev.simiter.sim_pstudy import SimPStudy, SimModel

class TestSimArray(unittest.TestCase):
    '''
    Test functionality connected with the application of
    boundary conditions.
    '''
    def setUp(self):

        self.sim_pstudy = SimPStudy( sim_model = SimModel() )    

        
    def test_save( self ):
        '''Calculate a study and save it 
        '''
        res = self.sim_pstudy.sim_array[0,:,:,:]
        print('fraction cached', self.sim_pstudy.sim_array.fraction_cached)
        
        filename = path.join( mkdtemp(), 'foo_file.pst' ) 
        self.sim_pstudy.save( filename )

        self.sim_pstudy.new()
        print('fraction cached', self.sim_pstudy.sim_array.fraction_cached)
        
        self.sim_pstudy.load( filename )
        print('fraction cached', self.sim_pstudy.sim_array.fraction_cached)

        print('FIRST BEFORE', self.sim_pstudy.sim_array.output_table[0])
        self.sim_pstudy.sim_array.clear_cache()
        print('fraction cached', self.sim_pstudy.sim_array.fraction_cached)
        print('FIRST AFTER', self.sim_pstudy.sim_array.output_table[0])
        
        #print res

        #self.assertAlmostEqual( F, -1 )
