from traits.api import \
    Instance, Int, Property, Array, cached_property

from numpy import \
     zeros, dot, hstack, identity

from scipy.linalg import \
     inv

from .fets3D import FETS3D

#-----------------------------------------------------------------------------------
# FETS3D8H16U - 16 nodes Subparametric 3D volume element: rs-direction: quadratic (serendipity)
#                                                          t-direction: linear
#-----------------------------------------------------------------------------------

class FETS3D8H16U( FETS3D ):
    '''
    quadratic/linear hybrid serendipity volume element.
    '''
    debug_on = False

    # Dimensional mapping
    dim_slice = slice( 0, 3 )

    # number of nodal degrees of freedom
    #
    n_nodal_dofs = Int( 3 )

    # number of degrees of freedom of each element
    #
    n_e_dofs = Int( 16 * 3 )

    # Integration parameters
    # NOTE: reduced integration order for t
    #       due to linear formulation direction
    #
    ngp_r = 3
    ngp_s = 3
    ngp_t = 2

    geo_r = \
             Array(value = [[-1., -1., -1.],
                            [  1., -1., -1.],
                            [-1., 1., -1.],
                            [  1., 1., -1.],
                            [-1., -1., 1.],
                            [  1., -1., 1.],
                            [-1., 1., 1.],
                            [  1., 1., 1.]])

    dof_r = \
            Array(value = [[-1., -1., -1.],
                           [ 1., -1., -1.],
                           [ 1., 1., -1.],
                           [-1., 1., -1.],
                           [-1., -1., 1.],
                           [ 1., -1., 1.],
                           [ 1., 1., 1.],
                           [-1., 1., 1.],
                           [ 0., -1., -1.],
                           [ 1., 0., -1.],
                           [ 0., 1., -1.],
                           [-1., 0., -1.],
                           [ 0., -1., 1.],
                           [ 1., 0., 1.],
                           [ 0., 1., 1.],
                           [-1., 0., 1.]])
            

    # Used for Visualization 
    vtk_r = Array(value = [[-1., -1., -1.],
                           [  1., -1., -1.],
                           [  1., 1., -1.],
                           [-1., 1., -1.],
                           [-1., -1., 1.],
                           [  1., -1., 1.],
                           [  1., 1., 1.],
                           [-1., 1., 1.],
                           [  0., -1., -1.],
                           [  1., 0., -1.],
                           [  0., 1., -1.],
                           [-1., 0., -1.],
                           [  0., -1., 1.],
                           [  1., 0., 1.],
                           [  0., 1., 1.],
                           [-1., 0., 1.],
                           [-1., -1., 0.],
                           [  1., -1., 0.],
                           [  1., 1., 0.],
                           [-1., 1., 0.]])

    vtk_cells = [[ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, \
                  10, 11, 12, 13, 14, 15, 16, 17, 18, 19]]

    vtk_cell_types = 'QuadraticHexahedron'

    #---------------------------------------------------------------------
    # Method required to represent the element geometry
    #---------------------------------------------------------------------
    def get_N_geo_mtx( self, r_pnt ):
        '''
        Return the value of shape functions (derived in femple) for the 
        specified local coordinate r_pnt
        '''
        N_geo_mtx = zeros( ( 1, 8 ), dtype = 'float_' )
        N_geo_mtx[0, 0] = -( ( -1 + r_pnt[2] ) * ( -1 + r_pnt[1] ) * \
                             ( -1 + r_pnt[0] ) ) / 8.0
        N_geo_mtx[0, 1] = ( ( -1 + r_pnt[2] ) * ( -1 + r_pnt[1] ) * \
                             ( 1 + r_pnt[0] ) ) / 8.0
        N_geo_mtx[0, 2] = ( ( -1 + r_pnt[2] ) * ( 1 + r_pnt[1] ) * \
                             ( -1 + r_pnt[0] ) ) / 8.0
        N_geo_mtx[0, 3] = -( ( -1 + r_pnt[2] ) * ( 1 + r_pnt[1] ) * \
                             ( 1 + r_pnt[0] ) ) / 8.0
        N_geo_mtx[0, 4] = ( ( 1 + r_pnt[2] ) * ( -1 + r_pnt[1] ) * \
                             ( -1 + r_pnt[0] ) ) / 8.0
        N_geo_mtx[0, 5] = -( ( 1 + r_pnt[2] ) * ( -1 + r_pnt[1] ) * \
                             ( 1 + r_pnt[0] ) ) / 8.0
        N_geo_mtx[0, 6] = -( ( 1 + r_pnt[2] ) * ( 1 + r_pnt[1] ) * \
                             ( -1 + r_pnt[0] ) ) / 8.0
        N_geo_mtx[0, 7] = ( ( 1 + r_pnt[2] ) * ( 1 + r_pnt[1] ) * \
                             ( 1 + r_pnt[0] ) ) / 8.0
        return N_geo_mtx

    def get_dNr_geo_mtx( self, r_pnt ):
        '''
        Return the matrix of shape function derivatives (derived in femple).
        Used for the construction of the Jacobi matrix.
        '''
        dNr_geo_mtx = zeros( ( 3, 8 ), dtype = 'float_' )
        dNr_geo_mtx[0, 0] = -( ( -1 + r_pnt[2] ) * ( -1 + r_pnt[1] ) ) / 8.0
        dNr_geo_mtx[0, 1] = ( ( -1 + r_pnt[2] ) * ( -1 + r_pnt[1] ) ) / 8.0
        dNr_geo_mtx[0, 2] = ( ( -1 + r_pnt[2] ) * ( 1 + r_pnt[1] ) ) / 8.0
        dNr_geo_mtx[0, 3] = -( ( -1 + r_pnt[2] ) * ( 1 + r_pnt[1] ) ) / 8.0
        dNr_geo_mtx[0, 4] = ( ( 1 + r_pnt[2] ) * ( -1 + r_pnt[1] ) ) / 8.0
        dNr_geo_mtx[0, 5] = -( ( 1 + r_pnt[2] ) * ( -1 + r_pnt[1] ) ) / 8.0
        dNr_geo_mtx[0, 6] = -( ( 1 + r_pnt[2] ) * ( 1 + r_pnt[1] ) ) / 8.0
        dNr_geo_mtx[0, 7] = ( ( 1 + r_pnt[2] ) * ( 1 + r_pnt[1] ) ) / 8.0
        dNr_geo_mtx[1, 0] = -( ( -1 + r_pnt[2] ) * ( -1 + r_pnt[0] ) ) / 8.0
        dNr_geo_mtx[1, 1] = ( ( -1 + r_pnt[2] ) * ( 1 + r_pnt[0] ) ) / 8.0
        dNr_geo_mtx[1, 2] = ( ( -1 + r_pnt[2] ) * ( -1 + r_pnt[0] ) ) / 8.0
        dNr_geo_mtx[1, 3] = -( ( -1 + r_pnt[2] ) * ( 1 + r_pnt[0] ) ) / 8.0
        dNr_geo_mtx[1, 4] = ( ( 1 + r_pnt[2] ) * ( -1 + r_pnt[0] ) ) / 8.0
        dNr_geo_mtx[1, 5] = -( ( 1 + r_pnt[2] ) * ( 1 + r_pnt[0] ) ) / 8.0
        dNr_geo_mtx[1, 6] = -( ( 1 + r_pnt[2] ) * ( -1 + r_pnt[0] ) ) / 8.0
        dNr_geo_mtx[1, 7] = ( ( 1 + r_pnt[2] ) * ( 1 + r_pnt[0] ) ) / 8.0
        dNr_geo_mtx[2, 0] = -( ( -1 + r_pnt[1] ) * ( -1 + r_pnt[0] ) ) / 8.0
        dNr_geo_mtx[2, 1] = ( ( -1 + r_pnt[1] ) * ( 1 + r_pnt[0] ) ) / 8.0
        dNr_geo_mtx[2, 2] = ( ( 1 + r_pnt[1] ) * ( -1 + r_pnt[0] ) ) / 8.0
        dNr_geo_mtx[2, 3] = -( ( 1 + r_pnt[1] ) * ( 1 + r_pnt[0] ) ) / 8.0
        dNr_geo_mtx[2, 4] = ( ( -1 + r_pnt[1] ) * ( -1 + r_pnt[0] ) ) / 8.0
        dNr_geo_mtx[2, 5] = -( ( -1 + r_pnt[1] ) * ( 1 + r_pnt[0] ) ) / 8.0
        dNr_geo_mtx[2, 6] = -( ( 1 + r_pnt[1] ) * ( -1 + r_pnt[0] ) ) / 8.0
        dNr_geo_mtx[2, 7] = ( ( 1 + r_pnt[1] ) * ( 1 + r_pnt[0] ) ) / 8.0
        return dNr_geo_mtx

    #---------------------------------------------------------------------
    # Method delivering the shape functions for the field variables and their
    # derivatives
    #---------------------------------------------------------------------
    def get_N_mtx( self, r_pnt ):
        '''
        Returns the matrix of the shape functions used for the field approximation
        containing zero entries. The number of rows corresponds to the number of nodal
        dofs. The matrix is evaluated for the specified local coordinate r_pnt.
        '''
        r = r_pnt[0]
        s = r_pnt[1]
        t = r_pnt[2]
        N_mtx = zeros( ( 3, 48 ), dtype = 'float_' )
        N_mtx[0, 0] = -0.125 + 0.125 * r * s + 0.125 * t - 0.125 * r * s * t + 0.125 * r * r - 0.125 * r * r * s - 0.125 * r * r * t + 0.125 * r * r * s * t + 0.125 * s * s - 0.125 * r * s * s - 0.125 * s * s * t + 0.125 * r * s * s * t
        N_mtx[0, 3] = -0.125 - 0.125 * r * s + 0.125 * t + 0.125 * r * s * t + 0.125 * r * r - 0.125 * r * r * s - 0.125 * r * r * t + 0.125 * r * r * s * t + 0.125 * s * s + 0.125 * r * s * s - 0.125 * s * s * t - 0.125 * r * s * s * t
        N_mtx[0, 6] = -0.125 + 0.125 * r * s + 0.125 * t - 0.125 * r * s * t + 0.125 * s * s + 0.125 * r * s * s - 0.125 * s * s * t - 0.125 * r * s * s * t + 0.125 * r * r + 0.125 * r * r * s - 0.125 * r * r * t - 0.125 * r * r * s * t
        N_mtx[0, 9] = -0.125 - 0.125 * r * s + 0.125 * t + 0.125 * r * s * t + 0.125 * r * r + 0.125 * r * r * s - 0.125 * r * r * t - 0.125 * r * r * s * t + 0.125 * s * s - 0.125 * r * s * s - 0.125 * s * s * t + 0.125 * r * s * s * t
        N_mtx[0, 12] = -0.125 + 0.125 * r * s - 0.125 * t + 0.125 * r * s * t + 0.125 * r * r - 0.125 * r * r * s + 0.125 * r * r * t - 0.125 * r * r * s * t + 0.125 * s * s - 0.125 * r * s * s + 0.125 * s * s * t - 0.125 * r * s * s * t
        N_mtx[0, 15] = -0.125 - 0.125 * r * s - 0.125 * t - 0.125 * r * s * t + 0.125 * r * r - 0.125 * r * r * s + 0.125 * r * r * t - 0.125 * r * r * s * t + 0.125 * s * s + 0.125 * r * s * s + 0.125 * s * s * t + 0.125 * r * s * s * t
        N_mtx[0, 18] = -0.125 + 0.125 * r * s - 0.125 * t + 0.125 * r * s * t + 0.125 * s * s + 0.125 * r * s * s + 0.125 * s * s * t + 0.125 * r * s * s * t + 0.125 * r * r + 0.125 * r * r * s + 0.125 * r * r * t + 0.125 * r * r * s * t
        N_mtx[0, 21] = -0.125 - 0.125 * r * s - 0.125 * t - 0.125 * r * s * t + 0.125 * r * r + 0.125 * r * r * s + 0.125 * r * r * t + 0.125 * r * r * s * t + 0.125 * s * s - 0.125 * r * s * s + 0.125 * s * s * t - 0.125 * r * s * s * t
        N_mtx[0, 24] = 0.250 - 0.250 * r * r - 0.250 * s + 0.250 * r * r * s - 0.250 * t + 0.250 * r * r * t + 0.250 * s * t - 0.250 * r * r * s * t
        N_mtx[0, 27] = 0.250 + 0.250 * r - 0.250 * s * s - 0.250 * r * s * s - 0.250 * t - 0.250 * r * t + 0.250 * s * s * t + 0.250 * r * s * s * t
        N_mtx[0, 30] = 0.250 - 0.250 * r * r + 0.250 * s - 0.250 * r * r * s - 0.250 * t + 0.250 * r * r * t - 0.250 * s * t + 0.250 * r * r * s * t
        N_mtx[0, 33] = 0.250 - 0.250 * r - 0.250 * s * s + 0.250 * r * s * s - 0.250 * t + 0.250 * r * t + 0.250 * s * s * t - 0.250 * r * s * s * t
        N_mtx[0, 36] = 0.250 - 0.250 * r * r - 0.250 * s + 0.250 * r * r * s + 0.250 * t - 0.250 * r * r * t - 0.250 * s * t + 0.250 * r * r * s * t
        N_mtx[0, 39] = 0.250 + 0.250 * r - 0.250 * s * s - 0.250 * r * s * s + 0.250 * t + 0.250 * r * t - 0.250 * s * s * t - 0.250 * r * s * s * t
        N_mtx[0, 42] = 0.250 - 0.250 * r * r + 0.250 * s - 0.250 * r * r * s + 0.250 * t - 0.250 * r * r * t + 0.250 * s * t - 0.250 * r * r * s * t
        N_mtx[0, 45] = 0.250 - 0.250 * r - 0.250 * s * s + 0.250 * r * s * s + 0.250 * t - 0.250 * r * t - 0.250 * s * s * t + 0.250 * r * s * s * t
        N_mtx[1, 1] = -0.125 + 0.125 * r * s + 0.125 * t - 0.125 * r * s * t + 0.125 * r * r - 0.125 * r * r * s - 0.125 * r * r * t + 0.125 * r * r * s * t + 0.125 * s * s - 0.125 * r * s * s - 0.125 * s * s * t + 0.125 * r * s * s * t
        N_mtx[1, 4] = -0.125 - 0.125 * r * s + 0.125 * t + 0.125 * r * s * t + 0.125 * r * r - 0.125 * r * r * s - 0.125 * r * r * t + 0.125 * r * r * s * t + 0.125 * s * s + 0.125 * r * s * s - 0.125 * s * s * t - 0.125 * r * s * s * t
        N_mtx[1, 7] = -0.125 + 0.125 * r * s + 0.125 * t - 0.125 * r * s * t + 0.125 * s * s + 0.125 * r * s * s - 0.125 * s * s * t - 0.125 * r * s * s * t + 0.125 * r * r + 0.125 * r * r * s - 0.125 * r * r * t - 0.125 * r * r * s * t
        N_mtx[1, 10] = -0.125 - 0.125 * r * s + 0.125 * t + 0.125 * r * s * t + 0.125 * r * r + 0.125 * r * r * s - 0.125 * r * r * t - 0.125 * r * r * s * t + 0.125 * s * s - 0.125 * r * s * s - 0.125 * s * s * t + 0.125 * r * s * s * t
        N_mtx[1, 13] = -0.125 + 0.125 * r * s - 0.125 * t + 0.125 * r * s * t + 0.125 * r * r - 0.125 * r * r * s + 0.125 * r * r * t - 0.125 * r * r * s * t + 0.125 * s * s - 0.125 * r * s * s + 0.125 * s * s * t - 0.125 * r * s * s * t
        N_mtx[1, 16] = -0.125 - 0.125 * r * s - 0.125 * t - 0.125 * r * s * t + 0.125 * r * r - 0.125 * r * r * s + 0.125 * r * r * t - 0.125 * r * r * s * t + 0.125 * s * s + 0.125 * r * s * s + 0.125 * s * s * t + 0.125 * r * s * s * t
        N_mtx[1, 19] = -0.125 + 0.125 * r * s - 0.125 * t + 0.125 * r * s * t + 0.125 * s * s + 0.125 * r * s * s + 0.125 * s * s * t + 0.125 * r * s * s * t + 0.125 * r * r + 0.125 * r * r * s + 0.125 * r * r * t + 0.125 * r * r * s * t
        N_mtx[1, 22] = -0.125 - 0.125 * r * s - 0.125 * t - 0.125 * r * s * t + 0.125 * r * r + 0.125 * r * r * s + 0.125 * r * r * t + 0.125 * r * r * s * t + 0.125 * s * s - 0.125 * r * s * s + 0.125 * s * s * t - 0.125 * r * s * s * t
        N_mtx[1, 25] = 0.250 - 0.250 * r * r - 0.250 * s + 0.250 * r * r * s - 0.250 * t + 0.250 * r * r * t + 0.250 * s * t - 0.250 * r * r * s * t
        N_mtx[1, 28] = 0.250 + 0.250 * r - 0.250 * s * s - 0.250 * r * s * s - 0.250 * t - 0.250 * r * t + 0.250 * s * s * t + 0.250 * r * s * s * t
        N_mtx[1, 31] = 0.250 - 0.250 * r * r + 0.250 * s - 0.250 * r * r * s - 0.250 * t + 0.250 * r * r * t - 0.250 * s * t + 0.250 * r * r * s * t
        N_mtx[1, 34] = 0.250 - 0.250 * r - 0.250 * s * s + 0.250 * r * s * s - 0.250 * t + 0.250 * r * t + 0.250 * s * s * t - 0.250 * r * s * s * t
        N_mtx[1, 37] = 0.250 - 0.250 * r * r - 0.250 * s + 0.250 * r * r * s + 0.250 * t - 0.250 * r * r * t - 0.250 * s * t + 0.250 * r * r * s * t
        N_mtx[1, 40] = 0.250 + 0.250 * r - 0.250 * s * s - 0.250 * r * s * s + 0.250 * t + 0.250 * r * t - 0.250 * s * s * t - 0.250 * r * s * s * t
        N_mtx[1, 43] = 0.250 - 0.250 * r * r + 0.250 * s - 0.250 * r * r * s + 0.250 * t - 0.250 * r * r * t + 0.250 * s * t - 0.250 * r * r * s * t
        N_mtx[1, 46] = 0.250 - 0.250 * r - 0.250 * s * s + 0.250 * r * s * s + 0.250 * t - 0.250 * r * t - 0.250 * s * s * t + 0.250 * r * s * s * t
        N_mtx[2, 2] = -0.125 + 0.125 * r * s + 0.125 * t - 0.125 * r * s * t + 0.125 * r * r - 0.125 * r * r * s - 0.125 * r * r * t + 0.125 * r * r * s * t + 0.125 * s * s - 0.125 * r * s * s - 0.125 * s * s * t + 0.125 * r * s * s * t
        N_mtx[2, 5] = -0.125 - 0.125 * r * s + 0.125 * t + 0.125 * r * s * t + 0.125 * r * r - 0.125 * r * r * s - 0.125 * r * r * t + 0.125 * r * r * s * t + 0.125 * s * s + 0.125 * r * s * s - 0.125 * s * s * t - 0.125 * r * s * s * t
        N_mtx[2, 8] = -0.125 + 0.125 * r * s + 0.125 * t - 0.125 * r * s * t + 0.125 * s * s + 0.125 * r * s * s - 0.125 * s * s * t - 0.125 * r * s * s * t + 0.125 * r * r + 0.125 * r * r * s - 0.125 * r * r * t - 0.125 * r * r * s * t
        N_mtx[2, 11] = -0.125 - 0.125 * r * s + 0.125 * t + 0.125 * r * s * t + 0.125 * r * r + 0.125 * r * r * s - 0.125 * r * r * t - 0.125 * r * r * s * t + 0.125 * s * s - 0.125 * r * s * s - 0.125 * s * s * t + 0.125 * r * s * s * t
        N_mtx[2, 14] = -0.125 + 0.125 * r * s - 0.125 * t + 0.125 * r * s * t + 0.125 * r * r - 0.125 * r * r * s + 0.125 * r * r * t - 0.125 * r * r * s * t + 0.125 * s * s - 0.125 * r * s * s + 0.125 * s * s * t - 0.125 * r * s * s * t
        N_mtx[2, 17] = -0.125 - 0.125 * r * s - 0.125 * t - 0.125 * r * s * t + 0.125 * r * r - 0.125 * r * r * s + 0.125 * r * r * t - 0.125 * r * r * s * t + 0.125 * s * s + 0.125 * r * s * s + 0.125 * s * s * t + 0.125 * r * s * s * t
        N_mtx[2, 20] = -0.125 + 0.125 * r * s - 0.125 * t + 0.125 * r * s * t + 0.125 * s * s + 0.125 * r * s * s + 0.125 * s * s * t + 0.125 * r * s * s * t + 0.125 * r * r + 0.125 * r * r * s + 0.125 * r * r * t + 0.125 * r * r * s * t
        N_mtx[2, 23] = -0.125 - 0.125 * r * s - 0.125 * t - 0.125 * r * s * t + 0.125 * r * r + 0.125 * r * r * s + 0.125 * r * r * t + 0.125 * r * r * s * t + 0.125 * s * s - 0.125 * r * s * s + 0.125 * s * s * t - 0.125 * r * s * s * t
        N_mtx[2, 26] = 0.250 - 0.250 * r * r - 0.250 * s + 0.250 * r * r * s - 0.250 * t + 0.250 * r * r * t + 0.250 * s * t - 0.250 * r * r * s * t
        N_mtx[2, 29] = 0.250 + 0.250 * r - 0.250 * s * s - 0.250 * r * s * s - 0.250 * t - 0.250 * r * t + 0.250 * s * s * t + 0.250 * r * s * s * t
        N_mtx[2, 32] = 0.250 - 0.250 * r * r + 0.250 * s - 0.250 * r * r * s - 0.250 * t + 0.250 * r * r * t - 0.250 * s * t + 0.250 * r * r * s * t
        N_mtx[2, 35] = 0.250 - 0.250 * r - 0.250 * s * s + 0.250 * r * s * s - 0.250 * t + 0.250 * r * t + 0.250 * s * s * t - 0.250 * r * s * s * t
        N_mtx[2, 38] = 0.250 - 0.250 * r * r - 0.250 * s + 0.250 * r * r * s + 0.250 * t - 0.250 * r * r * t - 0.250 * s * t + 0.250 * r * r * s * t
        N_mtx[2, 41] = 0.250 + 0.250 * r - 0.250 * s * s - 0.250 * r * s * s + 0.250 * t + 0.250 * r * t - 0.250 * s * s * t - 0.250 * r * s * s * t
        N_mtx[2, 44] = 0.250 - 0.250 * r * r + 0.250 * s - 0.250 * r * r * s + 0.250 * t - 0.250 * r * r * t + 0.250 * s * t - 0.250 * r * r * s * t
        N_mtx[2, 47] = 0.250 - 0.250 * r - 0.250 * s * s + 0.250 * r * s * s + 0.250 * t - 0.250 * r * t - 0.250 * s * s * t + 0.250 * r * s * s * t
        return N_mtx

    def get_dNr_mtx( self, r_pnt ):
        '''
        Return the derivatives of the shape functions used for the field approximation
        '''
        r = r_pnt[0]
        s = r_pnt[1]
        t = r_pnt[2]
        dNr_mtx = zeros( ( 3, 16 ), dtype = 'float_' )
        dNr_mtx[0, 0] = 0.125 * s - 0.125 * s * t + 0.250 * r - 0.250 * r * s - 0.250 * r * t + 0.250 * r * s * t - 0.125 * s * s + 0.125 * s * s * t
        dNr_mtx[0, 1] = -0.125 * s + 0.125 * s * t + 0.250 * r - 0.250 * r * s - 0.250 * r * t + 0.250 * r * s * t + 0.125 * s * s - 0.125 * s * s * t
        dNr_mtx[0, 2] = 0.125 * s - 0.125 * s * t + 0.125 * s * s - 0.125 * s * s * t + 0.250 * r + 0.250 * r * s - 0.250 * r * t - 0.250 * r * s * t
        dNr_mtx[0, 3] = -0.125 * s + 0.125 * s * t + 0.250 * r + 0.250 * r * s - 0.250 * r * t - 0.250 * r * s * t - 0.125 * s * s + 0.125 * s * s * t
        dNr_mtx[0, 4] = 0.125 * s + 0.125 * s * t + 0.250 * r - 0.250 * r * s + 0.250 * r * t - 0.250 * r * s * t - 0.125 * s * s - 0.125 * s * s * t
        dNr_mtx[0, 5] = -0.125 * s - 0.125 * s * t + 0.250 * r - 0.250 * r * s + 0.250 * r * t - 0.250 * r * s * t + 0.125 * s * s + 0.125 * s * s * t
        dNr_mtx[0, 6] = 0.125 * s + 0.125 * s * t + 0.125 * s * s + 0.125 * s * s * t + 0.250 * r + 0.250 * r * s + 0.250 * r * t + 0.250 * r * s * t
        dNr_mtx[0, 7] = -0.125 * s - 0.125 * s * t + 0.250 * r + 0.250 * r * s + 0.250 * r * t + 0.250 * r * s * t - 0.125 * s * s - 0.125 * s * s * t
        dNr_mtx[0, 8] = -0.500 * r + 0.500 * r * s + 0.500 * r * t - 0.500 * r * s * t
        dNr_mtx[0, 9] = 0.250 - 0.250 * s * s - 0.250 * t + 0.250 * s * s * t
        dNr_mtx[0, 10] = -0.500 * r - 0.500 * r * s + 0.500 * r * t + 0.500 * r * s * t
        dNr_mtx[0, 11] = -0.250 + 0.250 * s * s + 0.250 * t - 0.250 * s * s * t
        dNr_mtx[0, 12] = -0.500 * r + 0.500 * r * s - 0.500 * r * t + 0.500 * r * s * t
        dNr_mtx[0, 13] = 0.250 - 0.250 * s * s + 0.250 * t - 0.250 * s * s * t
        dNr_mtx[0, 14] = -0.500 * r - 0.500 * r * s - 0.500 * r * t - 0.500 * r * s * t
        dNr_mtx[0, 15] = -0.250 + 0.250 * s * s - 0.250 * t + 0.250 * s * s * t
        dNr_mtx[1, 0] = 0.125 * r - 0.125 * r * t - 0.125 * r * r + 0.125 * r * r * t + 0.250 * s - 0.250 * r * s - 0.250 * s * t + 0.250 * r * s * t
        dNr_mtx[1, 1] = -0.125 * r + 0.125 * r * t - 0.125 * r * r + 0.125 * r * r * t + 0.250 * s + 0.250 * r * s - 0.250 * s * t - 0.250 * r * s * t
        dNr_mtx[1, 2] = 0.125 * r - 0.125 * r * t + 0.250 * s + 0.250 * r * s - 0.250 * s * t - 0.250 * r * s * t + 0.125 * r * r - 0.125 * r * r * t
        dNr_mtx[1, 3] = -0.125 * r + 0.125 * r * t + 0.125 * r * r - 0.125 * r * r * t + 0.250 * s - 0.250 * r * s - 0.250 * s * t + 0.250 * r * s * t
        dNr_mtx[1, 4] = 0.125 * r + 0.125 * r * t - 0.125 * r * r - 0.125 * r * r * t + 0.250 * s - 0.250 * r * s + 0.250 * s * t - 0.250 * r * s * t
        dNr_mtx[1, 5] = -0.125 * r - 0.125 * r * t - 0.125 * r * r - 0.125 * r * r * t + 0.250 * s + 0.250 * r * s + 0.250 * s * t + 0.250 * r * s * t
        dNr_mtx[1, 6] = 0.125 * r + 0.125 * r * t + 0.250 * s + 0.250 * r * s + 0.250 * s * t + 0.250 * r * s * t + 0.125 * r * r + 0.125 * r * r * t
        dNr_mtx[1, 7] = -0.125 * r - 0.125 * r * t + 0.125 * r * r + 0.125 * r * r * t + 0.250 * s - 0.250 * r * s + 0.250 * s * t - 0.250 * r * s * t
        dNr_mtx[1, 8] = -0.250 + 0.250 * r * r + 0.250 * t - 0.250 * r * r * t
        dNr_mtx[1, 9] = -0.500 * s - 0.500 * r * s + 0.500 * s * t + 0.500 * r * s * t
        dNr_mtx[1, 10] = 0.250 - 0.250 * r * r - 0.250 * t + 0.250 * r * r * t
        dNr_mtx[1, 11] = -0.500 * s + 0.500 * r * s + 0.500 * s * t - 0.500 * r * s * t
        dNr_mtx[1, 12] = -0.250 + 0.250 * r * r - 0.250 * t + 0.250 * r * r * t
        dNr_mtx[1, 13] = -0.500 * s - 0.500 * r * s - 0.500 * s * t - 0.500 * r * s * t
        dNr_mtx[1, 14] = 0.250 - 0.250 * r * r + 0.250 * t - 0.250 * r * r * t
        dNr_mtx[1, 15] = -0.500 * s + 0.500 * r * s - 0.500 * s * t + 0.500 * r * s * t
        dNr_mtx[2, 0] = 0.125 - 0.125 * r * s - 0.125 * r * r + 0.125 * r * r * s - 0.125 * s * s + 0.125 * r * s * s
        dNr_mtx[2, 1] = 0.125 + 0.125 * r * s - 0.125 * r * r + 0.125 * r * r * s - 0.125 * s * s - 0.125 * r * s * s
        dNr_mtx[2, 2] = 0.125 - 0.125 * r * s - 0.125 * s * s - 0.125 * r * s * s - 0.125 * r * r - 0.125 * r * r * s
        dNr_mtx[2, 3] = 0.125 + 0.125 * r * s - 0.125 * r * r - 0.125 * r * r * s - 0.125 * s * s + 0.125 * r * s * s
        dNr_mtx[2, 4] = -0.125 + 0.125 * r * s + 0.125 * r * r - 0.125 * r * r * s + 0.125 * s * s - 0.125 * r * s * s
        dNr_mtx[2, 5] = -0.125 - 0.125 * r * s + 0.125 * r * r - 0.125 * r * r * s + 0.125 * s * s + 0.125 * r * s * s
        dNr_mtx[2, 6] = -0.125 + 0.125 * r * s + 0.125 * s * s + 0.125 * r * s * s + 0.125 * r * r + 0.125 * r * r * s
        dNr_mtx[2, 7] = -0.125 - 0.125 * r * s + 0.125 * r * r + 0.125 * r * r * s + 0.125 * s * s - 0.125 * r * s * s
        dNr_mtx[2, 8] = -0.250 + 0.250 * r * r + 0.250 * s - 0.250 * r * r * s
        dNr_mtx[2, 9] = -0.250 - 0.250 * r + 0.250 * s * s + 0.250 * r * s * s
        dNr_mtx[2, 10] = -0.250 + 0.250 * r * r - 0.250 * s + 0.250 * r * r * s
        dNr_mtx[2, 11] = -0.250 + 0.250 * r + 0.250 * s * s - 0.250 * r * s * s
        dNr_mtx[2, 12] = 0.250 - 0.250 * r * r - 0.250 * s + 0.250 * r * r * s
        dNr_mtx[2, 13] = 0.250 + 0.250 * r - 0.250 * s * s - 0.250 * r * s * s
        dNr_mtx[2, 14] = 0.250 - 0.250 * r * r + 0.250 * s - 0.250 * r * r * s
        dNr_mtx[2, 15] = 0.250 - 0.250 * r - 0.250 * s * s + 0.250 * r * s * s
        return dNr_mtx

#----------------------- example --------------------

if __name__ == '__main__':
    from ibvpy.api import \
        TStepper as TS, RTDofGraph, RTraceDomainListField, TLoop, \
        TLine, BCDofGroup, IBVPSolve as IS, DOTSEval

    #from lib.mats.mats2D.mats_cmdm2D.mats_mdm2d import MACMDM
    from ibvpy.mats.mats2D.mats2D_sdamage.mats2D_sdamage import MATS2DScalarDamage
    from ibvpy.mats.mats2D.mats2D_sdamage.strain_norm2d import *
    from ibvpy.mats.mats3D.mats3D_elastic.mats3D_elastic import MATS3DElastic

    fets_eval = FETS3D8H16U( mats_eval = MATS3DElastic(), )

    from ibvpy.mesh.fe_grid import FEGrid

    # Discretization
    domain = FEGrid( coord_max = ( 3., 3., 3. ),
                           shape = ( 3, 3, 3 ),
                           fets_eval = fets_eval )

    # Put the tseval (time-stepper) into the spatial context of the
    # discretization and specify the response tracers to evaluate there.
    right_dof = 2
    ts = TS( 
            sdomain = domain,
             # conversion to list (square brackets) is only necessary for slicing of 
             # single dofs, e.g "get_left_dofs()[0,1]" which elsewise retuns an integer only
             bcond_list = [ BCDofGroup( var = 'u', value = 0., dims = [0],
                                        get_dof_method = domain.get_left_dofs ),
                        BCDofGroup( var = 'u', value = 0., dims = [1, 2],
                                  get_dof_method = domain.get_bottom_left_dofs ),
                        BCDofGroup( var = 'u', value = 0.002, dims = [0],
                                  get_dof_method = domain.get_right_dofs ) ],
             rtrace_list = [
#                        RTDofGraph(name = 'Fi,right over u_right (iteration)' ,
#                                  var_y = 'F_int', idx_y = right_dof,
#                                  var_x = 'U_k', idx_x = right_dof,
#                                  record_on = 'update'),
#                        RTraceDomainListField(name = 'Deformation' ,
#                                       var = 'eps', idx = 0,
#                                       record_on = 'update'),
#                         RTraceDomainListField(name = 'Displacement_ip' ,
#                                        var = 'u', idx = 0,
#                                        position = 'int_pnts'),
                          RTraceDomainListField( name = 'Displacement' ,
                                        var = 'u', idx = 0 ),
#                         RTraceDomainListField(name = 'Stress' ,
#                                        var = 'sig', idx = 0,
#                                        record_on = 'update'),
#                        RTraceDomainListField(name = 'N0' ,
#                                       var = 'N_mtx', idx = 0,
#                                       record_on = 'update')
                        ]
            )

    # Add the time-loop control
    #
    tloop = TLoop( tstepper = ts,
         tline = TLine( min = 0.0, step = 0.5, max = 1.0 ) )

    tloop.eval()

    # Put the whole stuff into the simulation-framework to map the
    # individual pieces of definition into the user interface.
    #
    from ibvpy.plugins.ibvpy_app import IBVPyApp
    app = IBVPyApp( ibv_resource = tloop )
    app.main()
