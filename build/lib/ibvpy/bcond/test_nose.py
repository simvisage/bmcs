
#from sys_matrix import SysSparseMtx, SysDenseMtx
import unittest

from ibvpy.api import \
    TStepper as TS, RTDofGraph, RTraceDomainField, TLoop, \
    TLine, BCDof, DOTSEval
from ibvpy.fets.fets1D.fets1D2l import FETS1D2L
from ibvpy.mats.mats1D.mats1D_elastic.mats1D_elastic import MATS1DElastic
from ibvpy.mesh.fe_grid import FEGrid
from mathkit.matrix_la.coo_mtx import COOSparseMtx
from mathkit.matrix_la.dense_mtx import DenseMtx
from mathkit.matrix_la.sys_mtx_assembly import SysMtxAssembly
from numpy import array, zeros, arange, array_equal, hstack, dot, sqrt
from scipy.linalg import norm


class TestSysMtxConstraints(unittest.TestCase):
    '''
    Test functionality connected with the application of
    boundary conditions.
    '''

    def setUp(self):
        self.fets_eval = FETS1D2L(mats_eval=MATS1DElastic(E=10.))

        # Discretization
        self.domain = FEGrid(coord_max=(10., 0., 0.),
                             shape=(1,),
                             fets_eval=self.fets_eval)

        self.ts = TS(sdomain=self.domain,
                     dof_resultants=True
                     )
        self.tloop = TLoop(tstepper=self.ts,
                           tline=TLine(min=0.0,  step=1, max=1.0))

    def test_bar1(self):
        '''Clamped bar loaded at the right end with unit displacement
        [00]-[01]-[02]-[03]-[04]-[05]-[06]-[07]-[08]-[09]-[10]
        'u[0] = 0, u[10] = 1'''

        self.domain.coord_max = (10, 0, 0)
        self.domain.shape = (10,)
        self.ts.bcond_list = [BCDof(var='u', dof=0, value=0.),
                              BCDof(var='u', dof=10, value=1.)]
        self.ts.rtrace_list = [RTDofGraph(name='Fi,right over u_right (iteration)',
                                          var_y='F_int', idx_y=10,
                                          var_x='U_k', idx_x=10)]

        u = self.tloop.eval()
        # expected solution
        u_ex = array([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.],
                     dtype=float)
        difference = sqrt(norm(u - u_ex))
        self.assertAlmostEqual(difference, 0)
        # compare the reaction at the left end
        F = self.ts.F_int[0]
        self.assertAlmostEqual(F, -1)

    def test_bar3(self):
        '''Clamped bar with recursive constraints (load at right end)
        [0]-[1]-[2]-[3]
        u[1] = 0.2 * u[2], u[2] = 0.2 * u[3], R[3] = 10
        '''
        self.domain.coord_max = (3, 0, 0)
        self.domain.shape = (3,)
        self.ts.bcond_list = [BCDof(var='u', dof=0, value=0.),
                              BCDof(
                                  var='u', dof=1, link_dofs=[2], link_coeffs=[0.5]),
                              BCDof(
                                  var='u', dof=2, link_dofs=[3], link_coeffs=[1.]),
                              BCDof(var='f', dof=3, value=1)]
        # system solver
        u = self.tloop.eval()
        # expected solution
        u_ex = array([-0.,  0.1, 0.2, 0.2],
                     dtype=float)
        difference = sqrt(norm(u - u_ex))
        self.assertAlmostEqual(difference, 0)
        return
        #
        # '---------------------------------------------------------------'
        # 'Clamped bar with recursive constraints (displ at right end)'
        # 'u[1] = 0.5 * u[2], u[2] = 1.0 * u[3], u[3] = 1'
        self.ts.bcond_list = [BCDof(var='u', dof=0, value=0.),
                              BCDof(
                                  var='u', dof=1, link_dofs=[2], link_coeffs=[0.5]),
                              BCDof(
                                  var='u', dof=2, link_dofs=[3], link_coeffs=[1.]),
                              BCDof(var='u', dof=3, value=1)]
        u = self.tloop.eval()
        # expected solution
        u_ex = array([0.,  0.5, 1,  1], dtype=float)
        difference = sqrt(norm(u - u_ex))
        self.assertAlmostEqual(difference, 0)
        #


class TestMultiDomain(unittest.TestCase):
    '''
    Test functionality connected with kinematic constraints 
    on multiple domains.
    '''

    def setUp(self):

        self.fets_eval = FETS1D2L(mats_eval=MATS1DElastic(E=10.))

        # Discretization
        self.fe_domain1 = FEGrid(coord_max=(3., 0., 0.),
                                 shape=(3,),
                                 fets_eval=self.fets_eval)

        self.fe_domain2 = FEGrid(coord_min=(3., 0., 0.),
                                 coord_max=(6., 0., 0.),
                                 shape=(3,),
                                 fets_eval=self.fets_eval)

        self.fe_domain3 = FEGrid(coord_min=(3., 0., 0.),
                                 coord_max=(6., 0., 0.),
                                 shape=(3,),
                                 fets_eval=self.fets_eval)

        self.ts = TS(dof_resultants=True,
                     sdomain=[
                         self.fe_domain1, self.fe_domain2, self.fe_domain3],
                     bcond_list=[BCDof(var='u', dof=0, value=0.),
                                 BCDof(var='u', dof=4, link_dofs=[3], link_coeffs=[1.],
                                       value=0.),
                                 BCDof(var='f', dof=7, value=1,
                                       link_dofs=[2], link_coeffs=[2])],
                     rtrace_list=[RTDofGraph(name='Fi,right over u_right (iteration)',
                                             var_y='F_int', idx_y=0,
                                             var_x='U_k', idx_x=1),
                                  ]
                     )

        # Add the time-loop control
        self.tloop = TLoop(tstepper=self.ts,
                           tline=TLine(min=0.0,  step=1, max=1.0))

    def test_bar2(self):
        '''Clamped bar composed of two linked bars loaded at the right end
        [00]-[01]-[02]-[03]-[04]-[05]-[06]-[07]-[08]-[09]-[10]
        [11]-[12]-[13]-[14]-[15]-[16]-[17]-[18]-[19]-[20]-[21]
        u[0] = 0, u[5] = u[16], R[-1] = R[21] = 10
        '''

        self.fe_domain1.set(
            coord_min=(0, 0, 0), coord_max=(10, 0, 0), shape=(10,))
        self.fe_domain2.set(
            coord_min=(10, 0, 0), coord_max=(20, 0, 0), shape=(10,))
        self.ts.set(sdomain=[self.fe_domain1, self.fe_domain2],
                    bcond_list=[BCDof(var='u', dof=0, value=0.),
                                BCDof(
                                    var='u', dof=5, link_dofs=[16], link_coeffs=[1.], value=0.),
                                BCDof(var='f', dof=21, value=10)])

        u = self.tloop.eval()

        # expected solution
        u_ex = array([0., 1., 2., 3., 4., 5., 5., 5., 5., 5., 5.,
                      5., 5., 5., 5., 5., 5., 6., 7., 8., 9., 10.],
                     dtype=float)
        for u_, u_ex_ in zip(u, u_ex):
            self.assertAlmostEqual(u_, u_ex_)

        return
        # @todo - reactivate this test.
        #
        # '---------------------------------------------------------------'
        # 'Clamped bar composed of two linked bars control displ at right'
        # 'u[0] = 0, u[5] = u[16], u[21] = 1'
        # Remove the load and put a unit displacement at the right end
        # Note, the load is irrelevant in this case and will be rewritten
        #
        self.ts.bcond_list = [BCDof(var='u', dof=0, value=0.),
                              BCDof(var='u', dof=5, link_dofs=[16], link_coeffs=[1.],
                                    value=0.),
                              BCDof(var='u', dof=21, value=1.)]
        # system solver
        u = self.tloop.eval()
        # expected solution
        u_ex = array([0.,   1 / 10.,  2 / 10., 3 / 10., 4 / 10., 5 / 10., 5 / 10., 5 / 10.,
                      5 / 10., 5 / 10.,  5 / 10.,  5 /
                      10.,  5 / 10.,  5 / 10.,  5 / 10.,
                      5 / 10.,  5 / 10.,  6 / 10.,  7 / 10.,  8 / 10.,  9 / 10.,  1.],
                     dtype=float)
        for u_, u_ex_ in zip(u, u_ex):
            self.assertAlmostEqual(u_, u_ex_)
        #

    def test_bar4(self):
        '''Clamped bar 3 domains, each with 2 elems (displ at right end)
        [0]-[1]-[2] [3]-[4]-[5] [6]-[7]-[8]
        u[0] = 0, u[2] = u[3], u[5] = u[6], u[8] = 1'''

        self.fe_domain1.set(
            coord_min=(0, 0, 0), coord_max=(2, 0, 0), shape=(2,))
        self.fe_domain2.set(
            coord_min=(2, 0, 0), coord_max=(4, 0, 0), shape=(2,))
        self.fe_domain3.set(
            coord_min=(4, 0, 0), coord_max=(6, 0, 0), shape=(2,))

        self.ts.set(sdomain=[self.fe_domain1, self.fe_domain2, self.fe_domain3],
                    dof_resultants=True,
                    bcond_list=[BCDof(var='u', dof=0, value=0.),
                                BCDof(var='u', dof=2, link_dofs=[3], link_coeffs=[1.],
                                      value=0.),
                                BCDof(var='u', dof=5, link_dofs=[6], link_coeffs=[1.],
                                      value=0.),
                                BCDof(var='u', dof=8, value=1)],
                    rtrace_list=[RTDofGraph(name='Fi,right over u_right (iteration)',
                                            var_y='F_int', idx_y=0,
                                            var_x='U_k', idx_x=1),
                                 ]
                    )
        # system solver
        u = self.tloop.eval()
        # expected solution
        u_ex = array([0., 1 / 6.,  1 / 3., 1 / 3., 1 / 2., 2 / 3., 2 / 3.,  5 / 6.,  1.],
                     dtype=float)
        for u_, u_ex_ in zip(u, u_ex):
            self.assertAlmostEqual(u_, u_ex_)
        #

    def test_bar5(self):
        '''Clamped bar with 4 elements. Elements 2-4 are reinforced
        with another bar with 3 elements
        [0]-[1]-[2]-[3]-[4]
            [5]-[6]-[7]
        u[0] = 0, u[1] = u[5], u[3] = u[7], u[4] = 1
            '''
        self.fe_domain1.set(
            coord_min=(0, 0, 0), coord_max=(4, 0, 0), shape=(4,))
        self.fe_domain2.set(
            coord_min=(1, 0, 0), coord_max=(3, 0, 0), shape=(2,))
        self.ts.set(sdomain=[self.fe_domain1, self.fe_domain2],
                    bcond_list=[BCDof(var='u', dof=0, value=0.),
                                BCDof(
                                    var='u', dof=5, link_dofs=[1], link_coeffs=[1.], value=0.),
                                BCDof(
                                    var='u', dof=7, link_dofs=[3], link_coeffs=[1.], value=0.),
                                BCDof(var='u', dof=4, value=1)])

        u = self.tloop.eval()
        # expected solution
        u_ex = array([0., 1 / 3.,  0.5,
                      2 / 3.,  1.,
                      1 / 3.,  0.5,         2 / 3.],
                     dtype=float)
        for u_, u_ex_ in zip(u, u_ex):
            self.assertAlmostEqual(u_, u_ex_)

    def test_bar6(self):
        '''Clamped bar with 4 elements. Elements 2-4 are reinforced 
        with another bar with 1 element linked proportianally
        [0]-[1]-[2]-[3]-[4]
              [5]-[6]
        u[0] = 0, u[1] = u[5], u[3] = u[7], u[4] = 1'''
        self.fe_domain1.set(
            coord_min=(0, 0, 0), coord_max=(4, 0, 0), shape=(4,))
        self.fe_domain2.set(
            coord_min=(1.5, 0, 0), coord_max=(2.5, 0, 0), shape=(1,))
        self.ts.set(sdomain=[self.fe_domain1, self.fe_domain2],
                    bcond_list=[BCDof(var='u', dof=0, value=0.),
                                BCDof(
                                    var='u', dof=5, link_dofs=[1, 2], link_coeffs=[.5, .5]),
                                BCDof(
                                    var='u', dof=6, link_dofs=[2, 3], link_coeffs=[.5, .5]),
                                BCDof(var='u', dof=4, value=1)])
        u = self.tloop.eval()
        # expected solution
        u_ex = array([-0.,  0.3, 0.5, 0.7, 1.,  0.4,  0.6], dtype=float)
        for u_, u_ex_ in zip(u, u_ex):
            self.assertAlmostEqual(u_, u_ex_)

    def test_bar7(self):
        '''Two clamped beams link in parallel
        and loaded by force at right end
        [5]-[6]-[7]-[8]-[9]
        [0]-[1]-[2]-[3]-[4]
        u[5] = u[0], u[0] = 0, u[4] = u[9], R[4] = 1'''
        self.fe_domain1.set(
            coord_min=(0, 0, 0), coord_max=(4, 0, 0), shape=(4,))
        self.fe_domain2.set(
            coord_min=(0, 0, 0), coord_max=(4, 0, 0), shape=(4,))
        self.ts.set(sdomain=[self.fe_domain1, self.fe_domain2],
                    bcond_list=[BCDof(var='u', dof=0, value=0.),
                                BCDof(
                                    var='u', dof=5, link_dofs=[0], link_coeffs=[1.]),
                                BCDof(
                                    var='u', dof=4, link_dofs=[9], link_coeffs=[0.5]),
                                BCDof(var='f', dof=4, value=1),
                                BCDof(var='f', dof=9, value=1)])

        u = self.tloop.eval()
        # expected solution
        u_ex = array([-0., 0.06, 0.12,  0.18,  0.24,  0.,
                      0.12,  0.24, 0.36,  0.48],
                     dtype=float)

        for u_, u_ex_ in zip(u, u_ex):
            self.assertAlmostEqual(u_, u_ex_)
