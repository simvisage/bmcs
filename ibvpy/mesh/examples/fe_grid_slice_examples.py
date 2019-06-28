'''
Created on May 18, 2009

@author: jakub
'''
from ibvpy.fets.fets2D.fets2D4q import FETS2D4Q
from ibvpy.fets.fets3D.fets3D8h20u import FETS3D8H20U
from ibvpy.fets.fets_eval import FETSEval
from ibvpy.mesh.fe_grid import FEGrid

if __name__ == '__main__':

    example = '2D'

    if example == '1D':
        fets_sample = FETSEval( dof_r = [[-1], [0], [1]],
                               geo_r = [[-1], [1]],
                               n_nodal_dofs = 1 )

        fe_domain = FEGrid( coord_max = ( 2., ),
                            shape = ( 2, ),
        #                              inactive_elems = [3],
                            fets_eval = fets_sample )
        print("n_e ", fe_domain.shape)
        #second element,right node 
        fe_slice = fe_domain[1, -1]
        #second element
        #fe_slice = fe_domain[1] 
    elif example == '2D':
        fets_sample = FETS2D4Q()

        fe_domain = FEGrid( coord_max = ( 3., 2. ),
                            shape = ( 3, 2 ),
        #                              inactive_elems = [3],
                            fets_eval = fets_sample )
        #first element right top node 
        fe_slice = fe_domain[2, 1, -1, -1]

    elif example == '3D':
        fets_sample = FETS3D8H20U()

        fe_domain = FEGrid( coord_max = ( 3., 2., 4 ),
                            shape = ( 3, 2, 4 ),
        #                              inactive_elems = [3],
                            fets_eval = fets_sample )

        # first element, right front edge                  
        fe_slice = fe_domain[0, 0, 0, -1, :, -1]

    print('elems')
    print(fe_slice.elems)
    print('dof_nodes')
    print(fe_slice.dof_nodes)
    print('dofs')
    print(fe_slice.dofs)
    print('dof coords')
    print(fe_slice.dof_X)
    print('geo_nodes')
    print(fe_slice.geo_nodes)
    print('geo_r')
    print(fe_slice.geo_X)
    print('points? what are they for?')
