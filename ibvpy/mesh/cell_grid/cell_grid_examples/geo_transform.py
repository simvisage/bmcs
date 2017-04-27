
from ibvpy.mesh.cell_grid.cell_spec import CellSpec
from ibvpy.mesh.cell_grid.cell_grid import CellGrid
from numpy import frompyfunc, c_, array, cos, sin
from math import pi

#def shell_stb( x, y ):
#    return x, y, x**2 - y**2
#    return x, y, 4e-5*x**2 - 0.2495*x - 1.4554 + -1e-6*y**2 + 0.0297*y -374.84
#
#vfn_shell_stb = frompyfunc( shell_stb, 2, 3 )
#
#def shell_stb_arr( points ):
#    x,y,z = vfn_shell_stb( points[:,0], points[:,1] )
#    points = c_[ array(x,dtype='float_'),
#                 array(y,dtype='float_'),
#                 array(z,dtype='float_') ]
#    return points
#
#mgd = CellGrid( shape   = (2,2),
#                coord_min = (-3,-10, 0),
#                coord_max = ( 3, 10, 0),
#                geo_transform  = shell_stb_arr,
#                grid_cell_spec = CellSpec( node_coords = [[-1,-1],[1,-1],[1,1],[-1,1]] ) )
#
from ibvpy.mesh.fe_grid import FEGrid


#def screwed_arch_3d( points ):
#    x = points[:,0]
#    y = points[:,1]
#    z = points[:,2]
#    l = x[-1] - x[0]
#    R = 10.
#    phi = x / l * pi
#    r = R + y
#    x,y,z = - r * cos( phi ), r * sin( phi ), z 
#    return c_[ x,y,z ]

def arch_3d( points ):
    x = points[:,0]
    y = points[:,1]
    z = points[:,2]
    D_z = 3.
    R = 30.
    alpha = pi/2.
    len_x = x[-1] - x[0]
    len_z = z[-1] - z[0]
    zz = z - len_z / 2.
    yy = y + 4 * D_z / len_z**2 * zz**2 + D_z
    
    phi_min, phi_max = pi/2. - alpha/2., pi/2. + alpha/2. 
    delta_phi = phi_max - phi_min 
    phi = phi_min + (x / len_x) * delta_phi
    r = R + yy
    #return c_[ x, yy, zz ]
    x,y,z = - r * cos( phi ), r * sin( phi ), zz * ( (phi - delta_phi) **2 + 2. ) 
    return c_[ x,y, z ]

#def arch_2d( points ):
#    x = points[:,0]
#    y = points[:,1]
#    l = x[-1] - x[0]
#    R = 10.
#    phi = x / l * pi
#    r = R + y
#    x,y = - r * cos( phi ), r * sin( phi ) 
#    return c_[ x,y ]
#
#fe_domain_arch_2d = FEGrid( geo_r = [[-1,-1 ],
#                                        [-1, 1 ],
#                                        [ 1,-1 ],
#                                        [ 1, 1 ]],                            
#                          dof_r = [[-1,-1 ],
#                                        [-1, 1 ],
#                                        [ 1,-1 ],
#                                        [ 1, 1 ]],
#                          shape = ( 10, 3 ),
#                          geo_transform = arch_2d )

if __name__ == '__main__':

    from ibvpy.fets.fets_eval import FETSEval
    fets_eval = FETSEval( geo_r = [[-1,-1,-1],
                                                [-1,-1, 1],
                                                [-1, 1,-1],
                                                [-1, 1, 1],
                                                [ 1,-1,-1],
                                                [ 1,-1, 1],
                                                [ 1, 1,-1],
                                                [ 1, 1, 1]],                            
                          dof_r = [[-1, 0, 0],
                                        [ 1, 0, 0],
                                        [ 0, 0,-1],
                                        [ 0, 0, 1],
                                        [ 0,-1, 0],
                                        [ 0, 1, 0]] )
    fe_domain_arch_3d = FEGrid( fets_eval = fets_eval,
                                      coord_max = ( 10., 1, 10. ),
                                      shape = ( 10, 3, 10 ),
                                      geo_transform = arch_3d )

    from ibvpy.plugins.ibvpy_app import IBVPyApp
    ibvpy_app = IBVPyApp( ibv_resource = fe_domain_arch_3d )
    ibvpy_app.main()
