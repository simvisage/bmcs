
from ibvpy.fets.fets2D.fets2D4q import FETS2D4Q
from ibvpy.fets.fets2D.fets2D4q8u import FETS2D4Q8U
from ibvpy.fets.fets2D.fets2D4q9u import FETS2D4Q9U
from ibvpy.fets.fets2D.fets2D9q import FETS2D9Q
from numpy import \
     array, zeros, int_, float_, ix_, dot, linspace, hstack, vstack, arange, \
     identity, unique, average, frompyfunc, linalg, sign
from scipy.linalg import \
     inv
from traits.api import \
     Array, Bool, Callable, Enum, Float, HasTraits, Interface, \
     Instance, Int, Trait, Str, Enum, Callable, List, TraitDict, Any, \
     on_trait_change, Tuple, WeakRef, Delegate, Property, cached_property
from traitsui.api import \
     Item, View, HGroup, ListEditor, VGroup, Group

from .fets_ls_eval import FETSLSEval


class FETSBimaterial(FETSLSEval):
       
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
        # print "in N ",r_pnt
        p_N_mtx = self.parent_fets.get_N_mtx(r_pnt)
        p_nodal_dofs = self.parent_fets.n_nodal_dofs
                
        N_e_mtx = p_N_mtx * self._get_Psi(node_ls_values, p_N_mtx)
        N_enr_mtx = hstack((p_N_mtx, N_e_mtx))      
        return N_enr_mtx

    def get_dNr_psi_mtx(self, r_pnt, node_ls_values, r_ls_value):
        '''
        Return the derivatives of the shape functions
        '''
        print("in dN ", r_pnt)
        p_N_mtx = self.parent_fets.get_N_mtx(r_pnt)
        p_dNr_mtx = self.get_parent_dNr_mtx(r_pnt)
        sh = p_dNr_mtx.shape
        p_N_red = ((sum(p_N_mtx, 0)).reshape(sh[1], sh[0])).transpose()
                        
        first_mtx = p_N_red * self._get_dPsir(node_ls_values, p_N_mtx, p_dNr_mtx)
        print("first_mtx ", first_mtx)
        second_mtx = p_dNr_mtx * self._get_Psi(node_ls_values, p_N_mtx)  # ??
        
        dNr_e_mtx = first_mtx + second_mtx
        print("dNr_e_mtx ", dNr_e_mtx)
        return dNr_e_mtx
        
    def _get_Psi(self, node_ls_values, p_N_mtx):
        p_nodal_dofs = self.parent_fets.n_nodal_dofs
        first = (hstack([node_ls_values[i] * \
                   p_N_mtx[:, i * p_nodal_dofs:i * p_nodal_dofs + p_nodal_dofs] \
                   for i in range(0, self.n_nodes)])).sum()
        
        third = (hstack([p_N_mtx[:, i * p_nodal_dofs:i * p_nodal_dofs + p_nodal_dofs] * \
                     abs(node_ls_values[i])\
                     for i in range(0, self.n_nodes)])).sum()
        # print "Psi ", third - abs(first)
        return third - abs(first)
    
    def _get_dPsir(self, node_ls_values, p_N_mtx, p_dNr_mtx):
        sh = p_dNr_mtx.shape
        p_N_red = ((sum(p_N_mtx, 0)).reshape(sh[1], sh[0])).transpose()
        
        first = sum(hstack([node_ls_values[i] * \
                   p_N_red[:, i] \
                   for i in range(0, self.n_nodes) ]))
        second = sum(hstack([abs(node_ls_values[i]) * \
                    p_dNr_mtx[:, i] \
                    for i in range(0, self.n_nodes) ]))
        fourth = sum(hstack([ p_dNr_mtx[:, i] * \
                    node_ls_values[i]\
                    for i in range(0, self.n_nodes) ]))
        # print "dPsi ", second - sign(first)*fourth
        return second - sign(first) * fourth
        
    def get_B_mtx(self, r_pnt, X_mtx, node_ls_values, r_ls_value):
        J_mtx = self.get_J_mtx(r_pnt, X_mtx)
        dNr_mtx = self.get_parent_dNr_mtx(r_pnt)
        dNr_psi_mtx = self.get_dNr_psi_mtx(r_pnt, node_ls_values, r_ls_value)
        dNx_mtx = dot(inv(J_mtx), dNr_mtx)
        dNx_psi_mtx = dot(inv(J_mtx), dNr_psi_mtx)
        Bx_mtx = zeros((3, self.parent_fets.n_e_dofs * 2), dtype='float_')
        n_p_dofs = self.parent_fets.n_e_dofs
        if dNr_mtx.shape[0] == 1:  # 1D
            Bx_mtx = zeros((1, self.parent_fets.n_e_dofs * 2), dtype='float_')
            for i in range(0, self.n_nodes):
                Bx_mtx[0, i] = dNx_mtx[0, i]               
                Bx_mtx[0, i + n_p_dofs] = dNx_psi_mtx[0, i]
 
        elif dNr_mtx.shape[0] == 2:  # 2D
            Bx_mtx = zeros((3, self.parent_fets.n_e_dofs * 2), dtype='float_')
            for i in range(0, (self.parent_fets.n_e_dofs / self.parent_fets.n_nodal_dofs)):
                Bx_mtx[0, i * 2] = dNx_mtx[0, i]
                Bx_mtx[1, i * 2 + 1] = dNx_mtx[1, i]
                Bx_mtx[2, i * 2] = dNx_mtx[1, i]
                Bx_mtx[2, i * 2 + 1] = dNx_mtx[0, i]
                
                Bx_mtx[0, i * 2 + n_p_dofs] = dNx_psi_mtx[0, i]
                Bx_mtx[1, i * 2 + n_p_dofs + 1] = dNx_psi_mtx[1, i]
                Bx_mtx[2, i * 2 + n_p_dofs] = dNx_psi_mtx[1, i]
                Bx_mtx[2, i * 2 + n_p_dofs + 1] = dNx_psi_mtx[0, i]  
              
# below version for the alternating standard and enriched dofs    
#            Bx_mtx[0,i*4]   = dNx_mtx[0,i]
#            Bx_mtx[1,i*4+1] = dNx_mtx[1,i]
#            Bx_mtx[2,i*4]   = dNx_mtx[1,i]
#            Bx_mtx[2,i*4+1] = dNx_mtx[0,i]
#            
#            Bx_mtx[0,i*4+2] = dNx_psi_mtx[0,i]
#            Bx_mtx[1,i*4+3] = dNx_psi_mtx[1,i]
#            Bx_mtx[2,i*4+2] = dNx_psi_mtx[1,i]
#            Bx_mtx[2,i*4+3] = dNx_psi_mtx[0,i]           
        return Bx_mtx
    
#    def get_mtrl_corr_pred(self, sctx, eps_mtx, d_eps, tn, tn1, eps_avg = None):
#        r_pnt = sctx.r_pnt
#     
#        if r_pnt[0] >= 0.:#hack, should be done per proxy
#            sig_mtx, D_mtx = self.mats_eval2.get_corr_pred(sctx, eps_mtx, d_eps, tn, tn1,)
#        else:
#            sig_mtx, D_mtx = self.mats_eval.get_corr_pred(sctx, eps_mtx, d_eps, tn, tn1,) 
#        return sig_mtx, D_mtx

    
if __name__ == '__main__':
        from ibvpy.api import FEDomain, FERefinementGrid, FEGrid, TStepper as TS, \
            BCDofGroup, RTraceDomainListField
        from ibvpy.core.tloop import TLoop, TLine
        from ibvpy.mesh.xfe_subdomain import XFESubDomain

        def example_1d():
            from ibvpy.mats.mats1D.mats1D_elastic.mats1D_elastic import MATS1DElastic
            from ibvpy.fets.fets1D.fets1D2l import FETS1D2L
            from ibvpy.fets.fets1D.fets1D2l3u import FETS1D2L3U
            from ibvpy.fets.fets_ls.fets_crack import FETSCrack
            fets_eval = FETS1D2L(mats_eval=MATS1DElastic(E=1.))
            xfets_eval = FETSBimaterial(mats_eval=MATS1DElastic(E=1.),
                                         mats_eval2=MATS1DElastic(E=2.),
                                         parent_fets=fets_eval, int_order=1)
        
            # Discretization
            
            fe_domain = FEDomain()
            fe_level1 = FERefinementGrid(domain=fe_domain, fets_eval=fets_eval)
            fe_grid1 = FEGrid(coord_max=(2., 0., 0.),
                               shape=(1,),
                               fets_eval=fets_eval,
                               level=fe_level1)
            
            enr = True
            if enr:
                fe_xdomain = XFESubDomain(domain=fe_domain,
                                           fets_eval=xfets_eval,
                                           # fe_grid_idx_slice = fe_grid1[1,0],
                                           fe_grid_slice=fe_grid1['X  - 0.'])
    
            ts = TS(dof_resultants=True,
                     sdomain=fe_domain,
                     bcond_list=[BCDofGroup(var='u', value=1., dims=[0],
                                              get_dof_method=fe_grid1.get_right_dofs),
                                    BCDofGroup(var='u', value=0., dims=[0],
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
                                RTraceDomainListField(name='Strain' ,
                                                var='eps', idx=0,
                                                warp=True),
    #                                     RTraceDomainField(name = 'N0' ,
    #                                                  var = 'N_mtx', idx = 0,
    #                                                  record_on = 'update')
                            ]             
                        )
    #        
    #        # Add the time-loop control
            tloop = TLoop(tstepper=ts,
                           # tolerance = 1e-4, KMAX = 2,
                           # debug = True, RESETMAX = 2,
                           tline=TLine(min=0.0, step=1, max=1.0))
            
            # print "elements ",fe_xdomain.elements[0]
            if enr:
                fe_xdomain.deactivate_sliced_elems()
                print('parent elems ', fe_xdomain.fe_grid_slice.elems)
                print('parent dofs ', fe_xdomain.fe_grid_slice.dofs)
                print("dofmap ", fe_xdomain.elem_dof_map)
                print("ls_values ", fe_xdomain.dots.dof_node_ls_values)
                print('intersection points ', fe_xdomain.fe_grid_slice.r_i)  #
                print("triangles ", fe_xdomain.dots.int_division)
                print('ip_coords', fe_xdomain.dots.ip_coords)
                print('ip_weigths', fe_xdomain.dots.ip_weights)
                print('ip_offset ', fe_xdomain.dots.ip_offset)
                print('ip_X_coords', fe_xdomain.dots.ip_X)
                print('ip_ls', fe_xdomain.dots.ip_ls_values)
                print('vtk_X ', fe_xdomain.dots.vtk_X)
                print('vtk triangles ', fe_xdomain.dots.rt_triangles)
                print("vtk data ", fe_xdomain.dots.get_vtk_cell_data('blabla', 0, 0))
                print('vtk_ls', fe_xdomain.dots.vtk_ls_values)
                print('J_det ', fe_xdomain.dots.J_det_grid)
            
            print(tloop.eval())
    #        #ts.setup()
            from ibvpy.plugins.ibvpy_app import IBVPyApp
            ibvpy_app = IBVPyApp(ibv_resource=ts)
            ibvpy_app.main()
        
        def example_2d():
            from ibvpy.mats.mats2D.mats2D_elastic.mats2D_elastic import MATS2DElastic
            from ibvpy.fets.fets2D.fets2D4q import FETS2D4Q
            from ibvpy.fets.fets2D.fets2D4q8u import FETS2D4Q8U
            from ibvpy.fets.fets2D.fets2D4q9u import FETS2D4Q9U
            from ibvpy.fets.fets2D.fets2D9q import FETS2D9Q
            fets_eval = FETS2D4Q(mats_eval=MATS2DElastic(E=1., nu=0.))
            xfets_eval = FETSBimaterial(parent_fets=fets_eval, int_order=3 ,
                                         mats_eval=MATS2DElastic(E=1., nu=0.),
                                         mats_eval2=MATS2DElastic(E=5., nu=0.))
        
            # Discretization
            
            fe_domain = FEDomain()
            fe_level1 = FERefinementGrid(domain=fe_domain, fets_eval=fets_eval)
            fe_grid1 = FEGrid(coord_max=(3., 1., 0.),
                               shape=(3, 1),
                               fets_eval=fets_eval,
                               level=fe_level1)
           
            fe_xdomain = XFESubDomain(domain=fe_domain,
                                       fets_eval=xfets_eval,
                                       # fe_grid_idx_slice = fe_grid1[1,0],
                                       fe_grid_slice=fe_grid1['X   - 1.5'])
    
            ts = TS(dof_resultants=True,
                     sdomain=fe_domain,
                     bcond_list=[BCDofGroup(var='u', value=1., dims=[0],
                                              get_dof_method=fe_grid1.get_right_dofs),
                                    BCDofGroup(var='u', value=0., dims=[1],
                                              get_dof_method=fe_grid1.get_right_dofs),
                                    BCDofGroup(var='u', value=0., dims=[0, 1],
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
                                RTraceDomainListField(name='Strain' ,
                                                var='eps', idx=0,
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

        example_1d()
