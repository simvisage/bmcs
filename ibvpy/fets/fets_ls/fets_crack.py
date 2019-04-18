
from numpy import \
     array, zeros, int_, float_, ix_, dot, linspace, hstack, vstack, arange, \
     identity, unique, average, frompyfunc, linalg, sign, eye
from scipy.linalg import \
     inv, norm
from traits.api import \
     Array, Bool, Callable, Enum, Float, HasTraits, Interface, \
     Instance, Int, Trait, Str, Enum, Callable, List, TraitDict, Any, \
     on_trait_change, Tuple, WeakRef, Delegate, Property, cached_property

from .fets_ls_eval import FETSLSEval


class FETSCrack(FETSLSEval):

    # identify the enriched dofs within the parent element
    # by default a two-dimensional element with tdo-dimensional 
    # discontinuity is assumed.
    x_slice = slice(0, 2)

    n_nodal_dofs = Property(Int, depends_on='parent_fets.n_nodal_dofs')

    @cached_property
    def _get_n_nodal_dofs(self):
        return self.parent_fets.n_nodal_dofs * 2

    n_e_dofs = Property(Int, depends_on='parent_fets.n_e_dofs')

    @cached_property
    def _get_n_e_dofs(self):
        return self.parent_fets.n_e_dofs * 2

    #---------------------------------------------------------------------
    # Method required to represent the element geometry
    #---------------------------------------------------------------------

    #---------------------------------------------------------------------
    # Method delivering the shape functions for the field variables and their derivatives
    #---------------------------------------------------------------------

    def get_N_mtx(self, r_pnt, node_ls_values, r_ls_value):
        '''
        Returns the matrix of the shape functions used for the field approximation
        containing zero entries. The number of rows corresponds to the number of nodal
        dofs. The matrix is evaluated for the specified local coordinate r.
        '''
        p_N_mtx = self.parent_fets.get_N_mtx(r_pnt)
        p_nodal_dofs = self.parent_fets.n_nodal_dofs
        p_value = sign(r_ls_value)
        N_e_list = [ p_N_mtx[:, i * p_nodal_dofs:i * p_nodal_dofs + p_nodal_dofs] * \
                    (p_value - sign(node_ls_values[i]))\
                    for i in range(0, self.n_nodes) ]
        N_e_mtx = hstack(N_e_list)
        N_enr_mtx = hstack((p_N_mtx, N_e_mtx))
        return N_enr_mtx

    def get_dNr_psi_mtx(self, r_pnt, node_ls_values, r_ls_value):
        '''
        Return the derivatives of the shape functions
        '''
        p_dNr_mtx = self.parent_fets.get_dNr_mtx(r_pnt)
        if r_ls_value == 0:  # todo:this could be done nicer
            psi = (sign(node_ls_values) + 1) * (-1.)
            # psi = array([0.,-2.])
            dNr_mtx = array([p_dNr_mtx[:, i] * \
                             psi[i] for i in range(0, self.n_nodes) ])
            return dNr_mtx.T
        p_value = sign(r_ls_value)
        dNr_mtx = array([ p_dNr_mtx[:, i] * \
                           (p_value\
                           -sign(node_ls_values[i])) for i in range(0, self.n_nodes) ])
        return dNr_mtx.T

    def get_dNr_dd_mtx(self, r_pnt, r_norm):
        '''
        Return the derivatives of the shape functions
        '''
        p_N_mtx = self.parent_fets.get_N_mtx(r_pnt)
        print('parent N at', r_pnt, p_N_mtx)
        return p_N_mtx  # r_norm* p_N_mtx * 2.#size of the jump

    def get_B_mtx(self, r_pnt, X_mtx, node_ls_values, r_ls_value):
        B_FEM = self.parent_fets.get_B_mtx(r_pnt, X_mtx)
        # B_H = self.get_B_H(r_pnt, X_mtx, node_ls_values, r_ls_value)
        J_mtx = self.get_J_mtx(r_pnt, X_mtx)
        dNr_psi_mtx = self.get_dNr_psi_mtx(r_pnt, node_ls_values, r_ls_value)
        dNx_psi_mtx = dot(inv(J_mtx), dNr_psi_mtx)
        n_p_nodal_dofs = self.parent_fets.n_nodal_dofs
        if dNr_psi_mtx.shape[0] == 1:  # 1D
            BHx_mtx = zeros((1, self.parent_fets.n_e_dofs), dtype='float_')
            for i in range(0, self.n_nodes):
                BHx_mtx[0, i] = dNx_psi_mtx[0, i]

        elif dNr_psi_mtx.shape[0] == 2 and n_p_nodal_dofs == 2:  # 2D-displ
            BHx_mtx = zeros((3, self.parent_fets.n_e_dofs), dtype='float_')
            for i in range(0, self.n_nodes):
                BHx_mtx[0, i * 2] = dNx_psi_mtx[0, i]
                BHx_mtx[1, i * 2 + 1] = dNx_psi_mtx[1, i]
                BHx_mtx[2, i * 2] = dNx_psi_mtx[1, i]
                BHx_mtx[2, i * 2 + 1] = dNx_psi_mtx[0, i]

        elif dNr_psi_mtx.shape[0] == 2 and n_p_nodal_dofs == 1:  # 2D-conduction
            BHx_mtx = dNx_psi_mtx
        B_mtx = hstack((B_FEM, BHx_mtx))
        return B_mtx

    def get_B_H(self, r_pnt, X_mtx, node_ls_values, r_ls_value):
        J_mtx = self.get_J_mtx(r_pnt, X_mtx)
        dNr_psi_mtx = self.get_dNr_psi_mtx(r_pnt, node_ls_values, r_ls_value)
        dNx_psi_mtx = dot(inv(J_mtx), dNr_psi_mtx)
        n_p_dofs = self.parent_fets.n_e_dofs
        n_p_nodal_dofs = self.parent_fets.n_nodal_dofs
        if dNr_psi_mtx.shape[0] == 1:  # 1D
            Bx_mtx = zeros((1, self.parent_fets.n_e_dofs), dtype='float_')
            for i in range(0, self.n_nodes):
                Bx_mtx[0, i] = dNx_psi_mtx[0, i]
            print('B_H\n', Bx_mtx)

        elif dNr_psi_mtx.shape[0] == 2 and n_p_nodal_dofs == 2:  # 2D-displ
            Bx_mtx = zeros((3, self.parent_fets.n_e_dofs), dtype='float_')
            for i in range(0, self.n_nodes):
                Bx_mtx[0, i * 2] = dNx_psi_mtx[0, i]
                Bx_mtx[1, i * 2 + 1] = dNx_psi_mtx[1, i]
                Bx_mtx[2, i * 2] = dNx_psi_mtx[1, i]
                Bx_mtx[2, i * 2 + 1] = dNx_psi_mtx[0, i]

        elif dNr_psi_mtx.shape[0] == 2 and n_p_nodal_dofs == 1:  # 2D-conduction
            Bx_mtx = dNx_psi_mtx
        return Bx_mtx

    def get_B_D(self, r_pnt, r_norm):
        dNr_dd_mtx = self.get_dNr_dd_mtx(r_pnt, r_norm)  # no Jacobi transformation is needed - nature of the dirac delta
        dNx_dd_mtx = dNr_dd_mtx
        n_p_dofs = self.parent_fets.n_e_dofs
        n_p_nodal_dofs = self.parent_fets.n_nodal_dofs
#        if dNr_dd_mtx.shape[0] == 1:#1D
#            Bx_mtx = zeros( (1, self.parent_fets.n_e_dofs ), dtype = 'float_' )
#            for i in range(0,self.n_nodes):            
#                Bx_mtx[0,i] = dNx_dd_mtx[0,i]

        if dNr_dd_mtx.shape[0] == 2 and n_p_nodal_dofs == 2:  # 2D-displ
            Bx_mtx = zeros((3, self.parent_fets.n_e_dofs), dtype='float_')
            for i in range(0, self.n_nodes):
                Bx_mtx[0, i * 2] = dNx_dd_mtx[0, i]
                Bx_mtx[1, i * 2 + 1] = dNx_dd_mtx[1, i]
                Bx_mtx[2, i * 2] = dNx_dd_mtx[1, i]
                Bx_mtx[2, i * 2 + 1] = dNx_dd_mtx[0, i]

        elif dNr_dd_mtx.shape[0] == 1 and n_p_nodal_dofs == 1:  # 2D-conduction
            Bx_mtx = zeros((2, self.parent_fets.n_e_dofs), dtype='float_')
            Bx_mtx[0, :] = dNx_dd_mtx  # works just fot this case with vertical crack
        return Bx_mtx

    def get_B_disc(self, r_pnt, X_disc, X_mtx, node_ls_values, r_norm):
        '''
        B_fem matrix cannot be used directly, because the Jacobi transformation
        it has to be constructed here
        @param r_pnt:
        @param X_mtx:
        @param node_ls_values:
        @param r_value:
        '''
#        J_mtx = array([[1.]])#self.get_J_mtx_disc(r_pnt, r_mtx, X_mtx)
        B_FEM = self.parent_fets.get_B_mtx(r_pnt, X_mtx)
#        dNFEMr_mtx = self.parent_fets.get_dNr_mtx( r_pnt )
#        dNFEMx_mtx = dot( inv( J_mtx ), dNFEMr_mtx  )
#        
#        dNr_psi_mtx = self.get_dNr_psi_mtx( r_pnt, node_ls_values, r_ls_value = 0. )
#        dNx_psi_mtx = dot( inv( J_mtx ), dNr_psi_mtx  )
#        
#        dNr_dd_mtx = self.get_dNr_dd_mtx( r_pnt, r_norm )# no Jacobi transformation is needed - nature of the dirac delta
#        dNx_dd_mtx = dNr_dd_mtx
#        
#        n_p_dofs = self.parent_fets.n_e_dofs
#        if dNFEMr_mtx.shape[0] == 1:#1D
#            B_mtx = zeros( (1, self.parent_fets.n_e_dofs*3 ), dtype = 'float_' )
#            for i in range(0,self.n_nodes):   
#                B_mtx[0,i] = dNFEMx_mtx[0,i]    
#                B_mtx[0,i+self.n_nodes] = dNx_psi_mtx[0,i]
#                B_mtx[0,i+self.n_nodes*2] = dNx_dd_mtx[0,i]    
#        elif dNFEMr_mtx.shape[0] == 2:#2D
#            Bx_mtx = zeros( (3, self.parent_fets.n_e_dofs*3), dtype = 'float_' )
#            for i in range(0,self.n_nodes):                
#                Bx_mtx[0,i*2]   = dNx_psi_mtx[0,i]
#                Bx_mtx[1,i*2+1] = dNx_psi_mtx[1,i]
#                Bx_mtx[2,i*2]   = dNx_psi_mtx[1,i]
#                Bx_mtx[2,i*2+1] = dNx_psi_mtx[0,i]  

        B_H = self.get_B_H(r_pnt, X_mtx, node_ls_values, r_ls_value=0.)  # r_ls value is overwritten here!!
        B_D = self.get_B_D(r_pnt, r_norm)
        print('BFEM\n', B_FEM)
        print('BH\n', B_H)
        print('BD\n', B_D)
        return hstack((B_FEM, B_H, B_D))
        # return B_mtx

    def get_corr_pred_disc(self, sctx, u,
                                B_mtx_grid=None,
                                J_det_grid=None,
                                ip_coords=None,
                                ip_weights=None):
        n_e_dofs = self.n_e_dofs
        K = zeros((n_e_dofs, n_e_dofs))
        F = zeros(n_e_dofs)
        ip = 0
        for r_pnt, wt in zip(ip_coords, ip_weights):
            B_cached = B_mtx_grid[ip, ... ]
            J_det = J_det_grid[ip, ... ]

            B_shape = B_cached.shape[1] / 3  # equal number of standard and enriched dofs 

            B_FEM = B_cached[:, :B_shape]
            B_H = B_cached[:, B_shape:2 * B_shape]
            B_D = B_cached[:, 2 * B_shape:]

            B_mtx = hstack((B_FEM, (B_H + B_D)))
            eps_mtx = dot(B_mtx, u)

#            sctx.r_ls = 1.
#            sig_mtx_pos, D_mtx_pos = self.get_mtrl_corr_pred(sctx, eps_mtx, 0., 0., 0.)
#            sctx.r_ls = -1.
#            sig_mtx_neg, D_mtx_neg = self.get_mtrl_corr_pred(sctx, eps_mtx, 0., 0., 0.)
#            sctx.r_ls = 0.
            sig_mtx_disc, D_mtx_disc = self.get_mtrl_corr_pred(sctx, eps_mtx, 0., 0., 0.)

            # D_mtx_jump = D_mtx_disc - D_mtx_neg# D_mtx_disc - (D_mtx_pos+D_mtx_neg)/2.
            D_mtx_jump = eye(2)
            # print 'sig_mtx_disc, D_mtx_disc ', sig_mtx_disc, D_mtx_disc 

            k = zeros((2 * B_shape, 2 * B_shape), dtype=float)  # 1D
            k[:B_shape, B_shape:] = dot(B_FEM.T, dot((D_mtx_jump), B_D))
            k[B_shape:, :B_shape] = k[:B_shape, B_shape:].T
            k[B_shape:, B_shape:] = dot(B_D.T, dot(D_mtx_disc, B_D))

            k *= (wt * J_det)
            K += k
            f = zeros(2 * B_shape, dtype=float)
            f[B_shape:] = dot(B_D.T, sig_mtx_disc) * wt * J_det
            F += f
            ip += 1
        return K, F

#    def _get_mtrl_disc_corr_pred(self, sctx, eps_mtx, d_eps, tn, tn1):
#        sig_mtx, D_mtx = self.mats_eval_disc.get_corr_pred(sctx, eps_mtx, d_eps, tn, tn1)
#        return sig_mtx, D_mtx

    def get_J_det_disc(self, r_pnt, X_d, X_mtx, ls_nodes, ls_r):
        '''
        unified interface for caching
        has to account for geometry of the discontinuity
        @param r_pnt:
        @param X_mtx:
        @param ls_nodes:
        @param ls_r:
        '''
        shape = X_d.shape[0]
        if shape == 1:  # 1D
            J_det_ip = 1.
        elif shape == 2:  # 2D
            J_det_ip = norm(X_d[1] - X_d[0]) * 0.5
        return array(J_det_ip)

    def get_u(self, sctx, u):
        e_id = sctx.e_id
        p_id = sctx.p_id
        N_mtx = self.get_N_mtx(sctx.loc,
                                sctx.dots.dof_node_ls_values[e_id],
                                sctx.dots.vtk_ls_values[e_id][p_id])
#        print "N ",N_mtx
#        print "u ",u
#        print "x u",dot( N_mtx, u )
        return dot(N_mtx, u)

    def get_eps_eng(self, sctx, u):
        e_id = sctx.e_id
        p_id = sctx.p_id
        B_mtx = self.get_B_mtx(sctx.loc,
                               sctx.X,
                               sctx.dots.dof_node_ls_values[e_id],
                               sctx.dots.vtk_ls_values[e_id][p_id])
        return dot(B_mtx, u)


if __name__ == '__main__':
        from ibvpy.api import FEDomain, FERefinementGrid, FEGrid, TStepper as TS, \
            BCDofGroup, BCDof, RTraceDomainListField
        from ibvpy.core.tloop import TLoop, TLine
        from ibvpy.mesh.xfe_subdomain import XFESubDomain
        from ibvpy.mats.mats2D.mats2D_elastic.mats2D_elastic import MATS2DElastic
        from ibvpy.fets.fets2D.fets2D4q import FETS2D4Q
        from ibvpy.fets.fets2D.fets2D4q8u import FETS2D4Q8U
        from ibvpy.fets.fets2D.fets2D4q9u import FETS2D4Q9U
        from ibvpy.fets.fets2D.fets2D9q import FETS2D9Q
        fets_eval = FETS2D4Q(mats_eval=MATS2DElastic(E=1., nu=0.))
        xfets_eval = FETSCrack(parent_fets=fets_eval, int_order=5)

        # Discretization

        fe_domain = FEDomain()
        fe_level1 = FERefinementGrid(domain=fe_domain, fets_eval=fets_eval)
        fe_grid1 = FEGrid(coord_max=(1., 1., 0.),
                           shape=(1, 1),
                           rt_tol=0.1,
                           fets_eval=fets_eval,
                           level=fe_level1)
#        fe_grid1.deactivate( (1,0) )
#        fe_grid1.deactivate( (1,1) )

        fe_xdomain = XFESubDomain(domain=fe_domain,
                                   fets_eval=xfets_eval,
                                   # fe_grid_idx_slice = fe_grid1[1,0],
                                   fe_grid_slice=fe_grid1['X  -  0.5  -0.1*Y'])

        ts = TS(dof_resultants=True,
                 sdomain=fe_domain,
                 bcond_list=[BCDofGroup(var='u', value=0., dims=[0, 1],
                                          get_dof_method=fe_grid1.get_right_dofs),
                                BCDofGroup(var='u', value=0., dims=[1],
                                          get_dof_method=fe_grid1.get_left_dofs),
                                BCDofGroup(var='u', value=-1., dims=[0],
                                           get_dof_method=fe_grid1.get_left_dofs),
                                           ],
                 rtrace_list=[
#                                 RTDofGraph(name = 'Fi,right over u_right (iteration)' ,
#                                       var_y = 'F_int', idx_y = 0,
#                                       var_x = 'U_k', idx_x = 1),
#                            RTraceDomainListField(name = 'Stress' ,
#                                 var = 'sig_app', idx = 0, warp = True ),
                             RTraceDomainListField(name='Displacement' ,
                                            var='u', idx=0,
                                            warp=True),
#                                     RTraceDomainField(name = 'N0' ,
#                                                  var = 'N_mtx', idx = 0,
#                                                  record_on = 'update')
                        ]
                    )
#        
#        # Add the time-loop control
        tloop = TLoop(tstepper=ts,
#                       tolerance = 1e-4, KMAX = 4,
#                       debug = True, RESETMAX = 2,
                       tline=TLine(min=0.0, step=1., max=1.0))

        # print "elements ",fe_xdomain.elements[0]
        fe_xdomain.deactivate_sliced_elems()
        print('parent elems ', fe_xdomain.fe_grid_slice.elems)
        print('parent dofs ', fe_xdomain.fe_grid_slice.dofs)
        print("dofmap ", fe_xdomain.elem_dof_map)
        print("ls_values ", fe_xdomain.dots.dof_node_ls_values)
        print('intersection points ', fe_xdomain.fe_grid_slice.r_i)
        print("triangles ", fe_xdomain.dots.rt_triangles)
        print("vtk points ", fe_xdomain.dots.vtk_X)
        print("vtk data ", fe_xdomain.dots.get_vtk_cell_data('blabla', 0, 0))
        print('ip_triangles', fe_xdomain.dots.int_division)
        print('ip_coords', fe_xdomain.dots.ip_coords)
        print('ip_weigths', fe_xdomain.dots.ip_weights)
        print('ip_offset', fe_xdomain.dots.ip_offset)
        print('ip_X_coords', fe_xdomain.dots.ip_X)
        print('ip_ls', fe_xdomain.dots.ip_ls_values)
        print('vtk_ls', fe_xdomain.dots.vtk_ls_values)
        print('J_det ', fe_xdomain.dots.J_det_grid)

        print(tloop.eval())
#        #ts.setup()
        from ibvpy.plugins.ibvpy_app import IBVPyApp
        ibvpy_app = IBVPyApp(ibv_resource=ts)
        ibvpy_app.main()
