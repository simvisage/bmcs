
from numpy import array, sign, argwhere

from ibvpy.api import FEDomain, FERefinementGrid, FEGrid, TStepper as TS, \
    RTraceDomainListField, BCDof, BCSlice
from ibvpy.core.tloop import TLoop, TLine
from ibvpy.mesh.fe_ls_domain import FELSDomain
from ibvpy.mats.mats1D.mats1D_elastic.mats1D_elastic import MATS1DElastic
from ibvpy.fets.fets1D.fets1D2l import FETS1D2L
from ibvpy.fets.fets1D.fets1D2l3u import FETS1D2L3U
from ibvpy.fets.fets_ls.fets_crack import FETSCrack
from numpy import ma

if __name__ == '__main__':
    def example_1d():
        fets_eval = FETS1D2L3U( mats_eval = MATS1DElastic( E = 20. ) )
        xfets_eval = FETSCrack( parent_fets = fets_eval,
                                int_order = 2 )

        # Discretization

        fe_domain = FEDomain()
        fe_level1 = FERefinementGrid( domain = fe_domain, fets_eval = fets_eval )
        fe_grid1 = FEGrid( coord_max = ( 2., 0., 0. ),
                           shape = ( 2, ),
                           fets_eval = fets_eval,
                           level = fe_level1 )


        enr = True
        if enr:
            fe_xdomain = XFESubDomain( domain = fe_domain,
                                       fets_eval = xfets_eval,
                                       #fe_grid_idx_slice = fe_grid1[1,0],
                                       fe_grid_slice = fe_grid1['X  - .75'] )
            fe_xdomain.deactivate_sliced_elems()

        ts = TS( dof_resultants = True,
                 sdomain = fe_domain,
                 bcond_list = [BCSlice( var = 'u', value = -1. / 2., dims = [0],
                                        slice = fe_grid1[ 0, 0 ] ),
                                BCSlice( var = 'u', value = 0., dims = [0],
                                        slice = fe_grid1[ -1, -1 ] ),
                                        ],
                 rtrace_list = [
#                                 RTDofGraph(name = 'Fi,right over u_right (iteration)' ,
#                                       var_y = 'F_int', idx_y = 0,
#                                       var_x = 'U_k', idx_x = 1),
                            RTraceDomainListField( name = 'Stress' ,
                                 var = 'eps', idx = 0, warp = True ),
                             RTraceDomainListField( name = 'Displacement' ,
                                            var = 'u', idx = 0,
                                            warp = True ),
#                                     RTraceDomainField(name = 'N0' ,
#                                                  var = 'N_mtx', idx = 0,
#                                                  record_on = 'update')
                        ]
                    )
#        
#        # Add the time-loop control
        tloop = TLoop( tstepper = ts,
                       debug = True,
                       tolerance = 1e-4, RESETMAX = 0,
                       tline = TLine( min = 0.0, step = 1, max = 1.0 ) )

        #print "elements ",fe_xdomain.elements[0]
        if enr:
            print('parent elems ', fe_xdomain.fe_grid_slice.elems)
            print('parent dofs ', fe_xdomain.fe_grid_slice.dofs)
            print("dofmap ", fe_xdomain.elem_dof_map)
            print("ls_values ", fe_xdomain.dots.dof_node_ls_values)
            print('intersection points ', fe_xdomain.fe_grid_slice.r_i)#
            print("triangles ", fe_xdomain.dots.int_division)
            print('ip_coords', fe_xdomain.dots.ip_coords)
            print('ip_weigths', fe_xdomain.dots.ip_weights)
            print('ip_offset ', fe_xdomain.dots.ip_offset)
            print('ip_X_coords', fe_xdomain.dots.ip_X)
            print('ip_ls', fe_xdomain.dots.ip_ls_values)
            print('vtk_X ', fe_xdomain.dots.vtk_X)
            print('vtk triangles ', fe_xdomain.dots.rt_triangles)
            print("vtk data ", fe_xdomain.dots.get_vtk_cell_data( 'blabla', 0, 0 ))
            print('vtk_ls', fe_xdomain.dots.vtk_ls_values)
            print('J_det ', fe_xdomain.dots.J_det_grid)

        tloop.eval()

        from ibvpy.plugins.ibvpy_app import IBVPyApp
        ibvpy_app = IBVPyApp( ibv_resource = ts )
        ibvpy_app.main()

    def example_2d():
        from ibvpy.api import FEDomain, FERefinementGrid, FEGrid, TStepper as TS, \
            BCDofGroup, RTraceDomainListField
        from ibvpy.core.tloop import TLoop, TLine
        from ibvpy.mesh.xfe_subdomain import XFESubDomain
        from ibvpy.mats.mats2D.mats2D_elastic.mats2D_elastic import MATS2DElastic
        from ibvpy.mats.mats2D import MATS2DPlastic
        from ibvpy.fets.fets2D.fets2D4q import FETS2D4Q
        from ibvpy.fets.fets2D import FETS2D9Q
        from ibvpy.fets.fets2D.fets2D4q8u import FETS2D4Q8U
        from ibvpy.fets.fets_ls.fets_crack import FETSCrack
        #fets_eval = FETS2D4Q( mats_eval = MATS2DPlastic( E = 1., nu = 0. ) )
        fets_eval = FETS2D4Q8U( mats_eval = MATS2DPlastic( E = 1., nu = 0. ) )
        xfets_eval = FETSCrack( parent_fets = fets_eval,
                                int_order = 5,
                                tri_subdivision = 1 )

        # Discretization

        fe_domain = FEDomain()
        fe_level1 = FERefinementGrid( domain = fe_domain, fets_eval = fets_eval )
        fe_grid1 = FEGrid( coord_max = ( 1., 1. ),
                           shape = ( 8, 8 ),
                           fets_eval = fets_eval,
                           level = fe_level1 )

        #ls_function = lambda X, Y: X - Y - 0.13
        ls_function = lambda X, Y: ( X - 0.52 ) ** 2 + ( Y - 0.72 ) ** 2 - 0.51 ** 2
        bls_function = lambda X, Y:-( ( X - 0.5 ) ** 2 + ( Y - 0.21 ) ** 2 - 0.28 ** 2 )
        bls_function2 = lambda X, Y:-( ( X - 0.5 ) ** 2 + ( Y - 0.21 ) ** 2 - 0.38 ** 2 )

        # design deficits:
        # - How to define a level set spanned over several fe_grids
        #   (i.e. it is defined over the hierarchy of FESubDomains)
        # - Patching of subdomains within the FEPatchedGrid (FERefinementGrid)
        # - What are the compatibility conditions?
        # - What is the difference between FEGridLeveSetSlice
        #   and FELSDomain? 
        #   FELSDomain is associated with a DOTS - Slice is not.

        # FEGrid has a multidimensional array - elem_grid
        # it can be accessed through this index.
        # it is masked by the activity map. The activity map can 
        # be defined using slices and level sets.
        # the elems array enumerates the elements using the activity map.
        # in this way, the specialization of grids is available implicitly.
        #
        fe_xdomain = FELSDomain( domain = fe_domain,
                                 fets_eval = xfets_eval,
                                 fe_grid = fe_grid1,
                                 ls_function = ls_function,
                                 bls_function = bls_function,
                                 )

        fe_tip_xdomain = FELSDomain( domain = fe_domain,
                                     fets_eval = xfets_eval,
                                     fe_grid = fe_xdomain,
                                     ls_function = bls_function,
                                     )

        # deactivation must be done only after the dof enumeration has been completed
        fe_xdomain.deactivate_intg_elems_in_parent()
        fe_tip_xdomain.deactivate_intg_elems_in_parent()

        fe_xdomain.bls_function = bls_function2
        fe_tip_xdomain.ls_function = bls_function2

        # deactivation must be done only after the dof enumeration has been completed
        fe_xdomain.deactivate_intg_elems_in_parent()
        fe_tip_xdomain.deactivate_intg_elems_in_parent()

        #
        # General procedure:
        # 1) define the level sets with the boundaries
        # 2) use the bls to identify the tips of the level set
        # 3) use independent level sets to introduce indpendently junctions.
        #
        # get the extended dofs of the bls_elems and constrain it
        #
        cdofs = fe_tip_xdomain.elem_xdof_map.flatten()
        bc_list = [ BCDof( var = 'u', dof = dof, value = 0.0 )
                    for                   dof in cdofs ]

        # construct the time stepper

        ts = TS( dof_resultants = True,
                 sdomain = fe_domain,
                 bcond_list = [BCSlice( var = 'u', value = -0.1, dims = [1],
                                        slice = fe_grid1[:, 0, :, 0] ),
                                BCSlice( var = 'u', value = 0., dims = [0],
                                          slice = fe_grid1[:, 0, :, 0] ),
                                BCSlice( var = 'u', value = 0., dims = [0, 1],
                                         slice = fe_grid1[:, -1, :, -1] )
                                ] + bc_list,
                 rtrace_list = [
#                                 RTDofGraph(name = 'Fi,right over u_right (iteration)' ,
#                                       var_y = 'F_int', idx_y = 0,
#                                       var_x = 'U_k', idx_x = 1),
                            RTraceDomainListField( name = 'Stress' ,
                                 var = 'sig_app', idx = 0, warp = True ),
                             RTraceDomainListField( name = 'Displacement' ,
                                            var = 'u', idx = 0,
                                            warp = True ),
#                                     RTraceDomainField(name = 'N0' ,
#                                                  var = 'N_mtx', idx = 0,
#                                                  record_on = 'update')
                        ]
                    )
#        

        do = 'print'

        if do == 'print':

            p = 'state'

            if p == 'grids':
                print('fe_xdomain.ls mask')
                print(fe_xdomain.ls_mask)
                print('fe_xdomain.idx mask')
                print(fe_xdomain.idx_mask)
                print('fe_xdomain.intg mask')
                print(fe_xdomain.intg_mask)
                print('fe_xdomain.xelems_mask')
                print(fe_xdomain.xelems_mask)
                print('fe_xdomain.xelems_grid_ix')
                print(fe_xdomain.xelems_grid_ix)
                print('fe_xdomain.ls_elem_grid')
                print(fe_xdomain.ls_elem_grid)
                print('fe_xdomain.ls_ielem_grid')
                print(fe_xdomain.ls_ielem_grid)
                print('fe_xdomain.intg_elem_grid')
                print(fe_xdomain.intg_elem_grid)

                print('fe_tip_xdomain.ls_mask`')
                print(fe_tip_xdomain.ls_mask)
                print('fe_tip_xdomain.intg_mask`')
                print(fe_tip_xdomain.intg_mask)
                print('fe_tip_xdomain.idx_mask`')
                print(fe_tip_xdomain.idx_mask)
                print('fe_tip_xdomain.xelems_mask')
                print(fe_tip_xdomain.xelems_mask)
                print('fe_tip_xdomain.xelems_grid_ix')
                print(fe_tip_xdomain.xelems_grid_ix)
                print('fe_tip_xdomain.ls_elem_grid')
                print(fe_tip_xdomain.ls_elem_grid)
                print('fe_tip_xdomain.ls_ielems_grid')
                print(fe_tip_xdomain.ls_ielem_grid)
                print('fe_tip_xdomain.intg_elem_grid')
                print(fe_tip_xdomain.intg_elem_grid)

            if p == 'maps':

                print('fe_xdomain.elem_dof_map')
                print(fe_xdomain.elem_dof_map)
                print('fe_tip_xdomain.elem_dof_map')
                print(fe_tip_xdomain.elem_dof_map)

                print('fe_xdomain.elems')
                print(fe_xdomain.elems)
                print('fe_tip_xdomain.elems')
                print(fe_tip_xdomain.elems)

                print('fe_xdomain.elem_X_map')
                print(fe_xdomain.elem_X_map)
                print('fe_tip_xdomain.elem_X_map')
                print(fe_tip_xdomain.elem_X_map)

            if p == 'fields':

                print("ls_values ", fe_xdomain.dots.dof_node_ls_values)
                print("tip ls_values ", fe_tip_xdomain.dots.dof_node_ls_values)

                print('intersection points ', fe_xdomain.ls_intersection_r)
                print('tip intersection points ', fe_tip_xdomain.ls_intersection_r)

                print("triangles ", fe_xdomain.dots.rt_triangles)
                print("vtk points ", fe_xdomain.dots.vtk_X)
                print("vtk data ", fe_xdomain.dots.get_vtk_cell_data( 'blabla', 0, 0 ))

                print('ip_triangles', fe_xdomain.dots.int_division)
                print('ip_coords', fe_xdomain.dots.ip_coords)
                print('ip_weigths', fe_xdomain.dots.ip_weights)
                print('ip_offset', fe_xdomain.dots.ip_offset)
                print('ip_X_coords', fe_xdomain.dots.ip_X)
                print('ip_ls', fe_xdomain.dots.ip_ls_values)
                print('vtk_ls', fe_xdomain.dots.vtk_ls_values)
                print('J_det ', fe_xdomain.dots.J_det_grid)

            if p == 'state':

                # Add the time-loop control
                print('STATE: initial')

                print('fe_xdomain.dots.state_elem grid')
                print(fe_xdomain.dots.state_start_elem_grid)
                print('fe_tip_xdomain.dots.state_elem grid')
                print(fe_tip_xdomain.dots.state_start_elem_grid)
                print('fe_xdomain.dots.state_end_elem grid')
                print(fe_xdomain.dots.state_end_elem_grid)
                print('fe_tip_xdomain.dots.state_end_elem grid')
                print(fe_tip_xdomain.dots.state_end_elem_grid)

                fe_xdomain.dots.state_array[:] = 25.5
                print('state_array 25', fe_xdomain.dots.state_array)
                fe_tip_xdomain.dots.state_array[:] = 58

                bls_function3 = lambda X, Y:-( ( X - 0.5 ) ** 2 + ( Y - 0.21 ) ** 2 - 0.58 ** 2 )

                fe_xdomain.bls_function = bls_function3
                fe_tip_xdomain.ls_function = bls_function3

                print('STATE: changed')

                print('fe_xdomain.dots.state_elem grid')
                print(fe_xdomain.dots.state_start_elem_grid)
                print('fe_tip_xdomain.dots.state_elem grid')
                print(fe_tip_xdomain.dots.state_start_elem_grid)
                print('fe_xdomain.dots.state_end_elem grid')
                print(fe_xdomain.dots.state_end_elem_grid)
                print('fe_tip_xdomain.dots.state_end_elem grid')
                print(fe_tip_xdomain.dots.state_end_elem_grid)

                print('state_array 25', fe_xdomain.dots.state_array.shape)
                print('state_array 25', fe_xdomain.dots.state_array[570:])
                print('state_array 58', fe_tip_xdomain.dots.state_array.shape)


        elif do == 'ui':

            tloop = TLoop( tstepper = ts,
                           debug = False,
                           tolerance = 1e-4, KMAX = 3, RESETMAX = 0,
                           tline = TLine( min = 0.0, step = 1, max = 1.0 ) )

            tloop.eval()
            from ibvpy.plugins.ibvpy_app import IBVPyApp
            ibvpy_app = IBVPyApp( ibv_resource = ts )
            ibvpy_app.main()
    #            
    #    #    print ts.F_int
    #    #    print ts.rtrace_list[0].trace.ydata

    example_2d()
