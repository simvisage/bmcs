'''
Created on Mar 24, 2011

@author: jakub
'''

from numpy import \
    array, zeros,  dot, hstack, \
    identity
from scipy.linalg import \
    inv
from traits.api import \
    Instance, Int, Trait, Dict, \
    DelegatesTo, Property
from ibvpy.fets.fets_eval import FETSEval, RTraceEvalElemFieldVar

#-------------------------------------------------------------------------
# FETS2D4Q - 4 nodes iso-parametric quadrilateral element (2D, linear, Lagrange family)
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
# Element Information:
#-------------------------------------------------------------------------
#
# Here an isoparametric element formulation is applied.
# The implemented shape functions are derived based on the
# ordering of the nodes of the parent element defined in
# '_node_coord_map' (see below)
#
#-------------------------------------------------------------------------


class FETS2DTF(FETSEval):

    debug_on = True
    parent_fets = Instance(FETSEval)

    # Dimensional mapping
    dim_slice = slice(0, 2)
    dof_r = DelegatesTo('parent_fets')
    geo_r = DelegatesTo('parent_fets')

    get_dNr_geo_mtx = DelegatesTo('parent_fets')

    get_N_geo_mtx = DelegatesTo('parent_fets')

    n_nodal_dofs = Int(4)

    n_e_dofs = Property(Int)

    def _get_n_e_dofs(self):
        return self.parent_fets.n_e_dofs * 2

    ngp_r = Property

    def _get_ngp_r(self):
        return self.parent_fets.ngp_r

    ngp_s = Property

    def _get_ngp_s(self):
        return self.parent_fets.ngp_s

    vtk_cell_types = Property

    def _get_vtk_cell_types(self):
        return self.parent_fets.vtk_cell_types

    vtk_cells = Property

    def _get_vtk_cells(self):
        return self.parent_fets.vtk_cells

    vtk_r = Property

    def _get_vtk_r(self):
        return self.parent_fets.vtk_r

    #---------------------------------------------------------------------
    # Method required to represent the element geometry
    #---------------------------------------------------------------------

    #---------------------------------------------------------------------
    # Method delivering the shape functions for the field variables and their derivatives
    #---------------------------------------------------------------------
    def get_N_mtx(self, r_pnt):
        '''
        Returns the matrix of the shape functions used for the field approximation
        containing zero entries. The number of rows corresponds to the number of nodal
        dofs. The matrix is evaluated for the specified local coordinate r.
        '''
        n_nodes = self.n_e_dofs / self.n_nodal_dofs
        p_N_mtx = self.parent_fets.get_N_mtx(r_pnt)[0, ::2]
        I_mtx = identity(self.n_nodal_dofs, float)
        N_mtx_list = [I_mtx * p_N_mtx[i] for i in range(0, n_nodes)]
        N_mtx = hstack(N_mtx_list)
        return N_mtx

    def get_dNr_mtx(self, r_pnt):
        '''
        Return the derivatives of the shape functions
        '''
        return self.parent_fets.get_dNr_mtx(r_pnt)

    def get_B_mtx(self, r_pnt, X_mtx):
        J_mtx = self.get_J_mtx(r_pnt, X_mtx)
        dNr_mtx = self.get_dNr_mtx(r_pnt)
        dNx_mtx = dot(inv(J_mtx), dNr_mtx)
        N_mtx = self.get_N_mtx(r_pnt)
        N_mtx_red = N_mtx[:2] - N_mtx[2:]
        Bx_mtx = zeros((8, self.n_e_dofs), dtype='float_')
        Bx_mtx[[0, 2], ::4] = dNx_mtx[[0, 1]]
        Bx_mtx[[3, 5], 2::4] = dNx_mtx[[0, 1]]
        Bx_mtx[[1, 2], 1::4] = dNx_mtx[[1, 0]]
        Bx_mtx[[4, 5], 3::4] = dNx_mtx[[1, 0]]
        Bx_mtx[6:] = N_mtx_red
        return Bx_mtx

    def get_eps_m(self, sctx, u):
        X_mtx = sctx.X
        r_pnt = sctx.loc
        B_mtx = self.get_B_mtx(r_pnt, X_mtx)
        eps = dot(B_mtx, u)
        return array([[eps[0], eps[2]], [eps[2], eps[1]]])

    def get_eps_f(self, sctx, u):
        X_mtx = sctx.X
        r_pnt = sctx.loc
        B_mtx = self.get_B_mtx(r_pnt, X_mtx)
        eps = dot(B_mtx, u)
        return array([[eps[3], eps[5]], [eps[5], eps[4]]])

    def get_eps_b(self, sctx, u):
        X_mtx = sctx.X
        r_pnt = sctx.loc
        B_mtx = self.get_B_mtx(r_pnt, X_mtx)
        eps = dot(B_mtx, u)
        return array([[eps[6], 0.], [0., eps[7]]])

    def get_u_m(self, sctx, u):
        N_mtx = self.get_N_mtx(sctx.loc)
        return dot(N_mtx, u)[:2]

    def get_u_f(self, sctx, u):
        N_mtx = self.get_N_mtx(sctx.loc)
        return dot(N_mtx, u)[2:]

    rte_dict = Trait(Dict)

    def _rte_dict_default(self):
        '''
        RTraceEval dictionary with standard field variables.
        '''
        rte_dict = self._debug_rte_dict()
        for key, v_eval in list(self.mats_eval.rte_dict.items()):

            # add the eval into the loop.
            #
            rte_dict[key] = RTraceEvalElemFieldVar(name=key,
                                                   u_mapping=self.map_eps,
                                                   eval=v_eval)

        rte_dict.update({'eps_m': RTraceEvalElemFieldVar(eval=self.get_eps_m),
                         'u_m': RTraceEvalElemFieldVar(eval=self.get_u_m),
                         'eps_f': RTraceEvalElemFieldVar(eval=self.get_eps_f),
                         'u_f': RTraceEvalElemFieldVar(eval=self.get_u_f),
                         'eps_b': RTraceEvalElemFieldVar(eval=self.get_eps_b)})

        return rte_dict


#----------------------- example --------------------

def example_with_new_domain():
    from ibvpy.api import \
        TStepper as TS, RTraceDomainListField, TLoop, TLine

    from ibvpy.mats.mats2D5.mats2D5_bond.mats2D_bond import MATS2D5Bond
    from ibvpy.api import BCDofGroup
    from ibvpy.fets.fets2D.fets2D4q import FETS2D4Q
    fets_eval = FETS2DTF(parent_fets=FETS2D4Q(),
                         mats_eval=MATS2D5Bond(E_m=30, nu_m=0.2,
                                               E_f=10, nu_f=0.1,
                                               G=10.))

    from ibvpy.mesh.fe_grid import FEGrid
    from mathkit.mfn import MFnLineArray

    # Discretization
    fe_grid = FEGrid(coord_max=(10., 4., 0.),
                     n_elems=(10, 3),
                     fets_eval=fets_eval)

    mf = MFnLineArray(  # xdata = arange(10),
        ydata=array([0, 1, 2, 3]))

    tstepper = TS(sdomain=fe_grid,
                  bcond_list=[BCDofGroup(var='u', value=0., dims=[0, 1],
                                         get_dof_method=fe_grid.get_left_dofs),
                              #                                   BCDofGroup( var='u', value = 0., dims = [1],
                              # get_dof_method = fe_grid.get_bottom_dofs ),
                              BCDofGroup(var='u', value=.005, dims=[0],
                                         time_function=mf.get_value,
                                         get_dof_method=fe_grid.get_right_dofs)],
                  rtrace_list=[
                      #                     RTDofGraph(name = 'Fi,right over u_right (iteration)' ,
                      #                               var_y = 'F_int', idx_y = right_dof,
                      #                               var_x = 'U_k', idx_x = right_dof,
                      #                               record_on = 'update'),
                      #                         RTraceDomainListField(name = 'Stress' ,
                      #                         var = 'sig_app', idx = 0,
                      #                         #position = 'int_pnts',
                      #                         record_on = 'update'),
                      #                     RTraceDomainListField(name = 'Damage' ,
                      #                                    var = 'omega', idx = 0,
                      #                                    record_on = 'update',
                      #                                    warp = True),
                      RTraceDomainListField(name='Displ matrix',
                                            var='u_m', idx=0,
                                            record_on='update',
                                            warp=True),
                      RTraceDomainListField(name='Displ reinf',
                                            var='u_f', idx=0,
                                            record_on='update',
                                            warp=True),

                      #                    RTraceDomainListField(name = 'N0' ,
                      #                                      var = 'N_mtx', idx = 0,
                      # record_on = 'update')
                  ]
                  )

    # Add the time-loop control
    #global tloop
    tloop = TLoop(tstepper=tstepper, KMAX=300, tolerance=1e-4,
                  tline=TLine(min=0.0, step=1.0, max=1.0))

    #import cProfile
    #cProfile.run('tloop.eval()', 'tloop_prof' )
    print(tloop.eval())
    #import pstats
    #p = pstats.Stats('tloop_prof')
    # p.strip_dirs()
    # print 'cumulative'
    # p.sort_stats('cumulative').print_stats(20)
    # print 'time'
    # p.sort_stats('time').print_stats(20)

    # Put the whole thing into the simulation-framework to map the
    # individual pieces of definition into the user interface.
    #
    from ibvpy.plugins.ibvpy_app import IBVPyApp
    app = IBVPyApp(ibv_resource=tloop)
    app.main()

if __name__ == '__main__':
    example_with_new_domain()
