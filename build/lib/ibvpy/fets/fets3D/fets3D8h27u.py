from traits.api import \
    Instance, Int, Property, Array, cached_property

from numpy import \
     zeros, dot, hstack, identity

from scipy.linalg import \
     inv

from .fets3D import FETS3D

#-----------------------------------------------------------------------------------
# FETS3D8H27U - 27 nodes subparametric volume element (3D, quadratic, Lagrange family)    
#-----------------------------------------------------------------------------------

class FETS3D8H27U( FETS3D ):
    '''
    eight nodes volume element
    '''
    debug_on = True

    # Dimensional mapping
    dim_slice = slice( 0, 3 )

    # number of nodal degrees of freedom
    # number of degrees of freedom of each element
    n_nodal_dofs = Int( 3 )
    n_e_dofs = Int( 27 * 3 )

    # Integration parameters
    #
    ngp_r = 3
    ngp_s = 3
    ngp_t = 3

    dof_r = \
                Array(value = [[-1, -1, -1],
                               [ 1, -1, -1],
                               [ 1, 1, -1],
                               [-1, 1, -1],
                               [ 0, -1, -1],
                               [ 1, 0, -1],
                               [ 0, 1, -1],
                               [-1, 0, -1],
                               [-1, -1, 0],
                               [ 1, -1, 0],
                               [ 1, 1, 0],
                               [-1, 1, 0],
                               [-1, -1, 1],
                               [ 1, -1, 1],
                               [ 1, 1, 1],
                               [-1, 1, 1],
                               [ 0, -1, 1],
                               [ 1, 0, 1],
                               [ 0, 1, 1],
                               [-1, 0, 1],
                               [ 0, 0, -1],
                               [ 0, -1, 0],
                               [ 1, 0, 0],
                               [ 0, 1, 0],
                               [-1, 0, 0],
                               [ 0, 0, 1],
                               [ 0, 0, 0]])
    geo_r = \
         Array(value = [[-1., -1., -1.],
                        [  1., -1., -1.],
                        [-1., 1., -1.],
                        [  1., 1., -1.],
                        [-1., -1., 1.],
                        [  1., -1., 1.],
                        [-1., 1., 1.],
                        [  1., 1., 1.]])

    # Used for Visualization 
    vtk_cell_types = 'TriQuadraticHexahedron'
    vtk_r = Array(value = [[-1., -1., -1.], #bottom
                           [  1., -1., -1.],
                           [  1., 1., -1.],
                           [-1., 1., -1.],
                           [-1., -1., 1.], #top
                           [  1., -1., 1.],
                           [  1., 1., 1.],
                           [-1., 1., 1.],
                           [  0., -1., -1.], #bottom midside
                           [  1., 0., -1.],
                           [  0., 1., -1.],
                           [-1., 0., -1.],
                           [  0., -1., 1.], #top midside
                           [  1., 0., 1.],
                           [  0., 1., 1.],
                           [-1., 0., 1.],
                           [-1., -1., 0.], #middle
                           [  1., -1., 0.],
                           [  1., 1., 0.],
                           [-1., 1., 0.],
                           [  0., -1., 0.], #middle midside (different order)
                           [  1., 0., 0.],
                           [  0., 1., 0.],
                           [-1., 0., 0.],
                           [  0., 0., -1.], #bottom centre
                           [  0., 0., 1.], #top centre
                           [  0., 0., 0.]])#middle centre

    vtk_cells = [[ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, \
                    10, 11, 12, 13, 14, 15, 16, 17, 18, 19, \
                    23, 21, 20, 22, 24, 25, 26]]

    #---------------------------------------------------------------------
    # Method required to represent the element geometry
    #---------------------------------------------------------------------
    def get_N_geo_mtx( self, r_pnt ):
        '''
        Return the value of shape functions (derived in femple) for the 
        specified local coordinate r_pnt
        '''
        N_geo_mtx = zeros( ( 1, 8 ), dtype = 'float_' )
        N_geo_mtx[0, 0] = -( ( -1 + r_pnt[2] ) * ( -1 + r_pnt[1] ) * ( -1 + r_pnt[0] ) ) / 8.0
        N_geo_mtx[0, 1] = ( ( -1 + r_pnt[2] ) * ( -1 + r_pnt[1] ) * ( 1 + r_pnt[0] ) ) / 8.0
        N_geo_mtx[0, 2] = ( ( -1 + r_pnt[2] ) * ( 1 + r_pnt[1] ) * ( -1 + r_pnt[0] ) ) / 8.0
        N_geo_mtx[0, 3] = -( ( -1 + r_pnt[2] ) * ( 1 + r_pnt[1] ) * ( 1 + r_pnt[0] ) ) / 8.0
        N_geo_mtx[0, 4] = ( ( 1 + r_pnt[2] ) * ( -1 + r_pnt[1] ) * ( -1 + r_pnt[0] ) ) / 8.0
        N_geo_mtx[0, 5] = -( ( 1 + r_pnt[2] ) * ( -1 + r_pnt[1] ) * ( 1 + r_pnt[0] ) ) / 8.0
        N_geo_mtx[0, 6] = -( ( 1 + r_pnt[2] ) * ( 1 + r_pnt[1] ) * ( -1 + r_pnt[0] ) ) / 8.0
        N_geo_mtx[0, 7] = ( ( 1 + r_pnt[2] ) * ( 1 + r_pnt[1] ) * ( 1 + r_pnt[0] ) ) / 8.0
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
        N_mtx = zeros( ( 3, 81 ), dtype = 'float_' )
        N_mtx[0, 0] = ( r * t * s * ( -1 + s ) * ( -1 + t ) * ( -1 + r ) ) / 8.
        N_mtx[0, 3] = ( r * t * s * ( -1 + s ) * ( -1 + t ) * ( 1 + r ) ) / 8.
        N_mtx[0, 6] = ( r * t * s * ( 1 + s ) * ( -1 + t ) * ( 1 + r ) ) / 8.
        N_mtx[0, 9] = ( r * t * s * ( 1 + s ) * ( -1 + t ) * ( -1 + r ) ) / 8.
        N_mtx[0, 12] = -( t * s * ( -1 + r ) * ( 1 + r ) * ( -1 + s ) * ( -1 + t ) ) / 4.
        N_mtx[0, 15] = -( r * t * ( -1 + s ) * ( 1 + s ) * ( -1 + t ) * ( 1 + r ) ) / 4.
        N_mtx[0, 18] = -( t * s * ( -1 + r ) * ( 1 + r ) * ( 1 + s ) * ( -1 + t ) ) / 4.
        N_mtx[0, 21] = -( r * t * ( -1 + s ) * ( 1 + s ) * ( -1 + t ) * ( -1 + r ) ) / 4.
        N_mtx[0, 24] = -( r * s * ( -1 + t ) * ( 1 + t ) * ( -1 + s ) * ( -1 + r ) ) / 4.
        N_mtx[0, 27] = -( r * s * ( -1 + t ) * ( 1 + t ) * ( -1 + s ) * ( 1 + r ) ) / 4.
        N_mtx[0, 30] = -( r * s * ( -1 + t ) * ( 1 + t ) * ( 1 + s ) * ( 1 + r ) ) / 4.
        N_mtx[0, 33] = -( r * s * ( -1 + t ) * ( 1 + t ) * ( 1 + s ) * ( -1 + r ) ) / 4.
        N_mtx[0, 36] = ( r * t * s * ( -1 + s ) * ( 1 + t ) * ( -1 + r ) ) / 8.
        N_mtx[0, 39] = ( r * t * s * ( -1 + s ) * ( 1 + t ) * ( 1 + r ) ) / 8.
        N_mtx[0, 42] = ( r * t * s * ( 1 + s ) * ( 1 + t ) * ( 1 + r ) ) / 8.
        N_mtx[0, 45] = ( r * t * s * ( 1 + s ) * ( 1 + t ) * ( -1 + r ) ) / 8.
        N_mtx[0, 48] = -( t * s * ( -1 + r ) * ( 1 + r ) * ( -1 + s ) * ( 1 + t ) ) / 4.
        N_mtx[0, 51] = -( r * t * ( -1 + s ) * ( 1 + s ) * ( 1 + t ) * ( 1 + r ) ) / 4.
        N_mtx[0, 54] = -( t * s * ( -1 + r ) * ( 1 + r ) * ( 1 + s ) * ( 1 + t ) ) / 4.
        N_mtx[0, 57] = -( r * t * ( -1 + s ) * ( 1 + s ) * ( 1 + t ) * ( -1 + r ) ) / 4.
        N_mtx[0, 60] = ( t * ( -1 + s ) * ( 1 + s ) * ( -1 + r ) * ( 1 + r ) * ( -1 + t ) ) / 2.
        N_mtx[0, 63] = ( s * ( -1 + t ) * ( 1 + t ) * ( -1 + r ) * ( 1 + r ) * ( -1 + s ) ) / 2.
        N_mtx[0, 66] = ( r * ( -1 + s ) * ( 1 + s ) * ( -1 + t ) * ( 1 + t ) * ( 1 + r ) ) / 2.
        N_mtx[0, 69] = ( s * ( -1 + t ) * ( 1 + t ) * ( -1 + r ) * ( 1 + r ) * ( 1 + s ) ) / 2.
        N_mtx[0, 72] = ( r * ( -1 + s ) * ( 1 + s ) * ( -1 + t ) * ( 1 + t ) * ( -1 + r ) ) / 2.
        N_mtx[0, 75] = ( t * ( -1 + s ) * ( 1 + s ) * ( -1 + r ) * ( 1 + r ) * ( 1 + t ) ) / 2.
        N_mtx[0, 78] = -( -1 + s ) * ( 1 + s ) * ( -1 + t ) * ( 1 + t ) * ( -1 + r ) * ( 1 + r )
        N_mtx[1, 1] = ( r * t * s * ( -1 + s ) * ( -1 + t ) * ( -1 + r ) ) / 8.
        N_mtx[1, 4] = ( r * t * s * ( -1 + s ) * ( -1 + t ) * ( 1 + r ) ) / 8.
        N_mtx[1, 7] = ( r * t * s * ( 1 + s ) * ( -1 + t ) * ( 1 + r ) ) / 8.
        N_mtx[1, 10] = ( r * t * s * ( 1 + s ) * ( -1 + t ) * ( -1 + r ) ) / 8.
        N_mtx[1, 13] = -( t * s * ( -1 + r ) * ( 1 + r ) * ( -1 + s ) * ( -1 + t ) ) / 4.
        N_mtx[1, 16] = -( r * t * ( -1 + s ) * ( 1 + s ) * ( -1 + t ) * ( 1 + r ) ) / 4.
        N_mtx[1, 19] = -( t * s * ( -1 + r ) * ( 1 + r ) * ( 1 + s ) * ( -1 + t ) ) / 4.
        N_mtx[1, 22] = -( r * t * ( -1 + s ) * ( 1 + s ) * ( -1 + t ) * ( -1 + r ) ) / 4.
        N_mtx[1, 25] = -( r * s * ( -1 + t ) * ( 1 + t ) * ( -1 + s ) * ( -1 + r ) ) / 4.
        N_mtx[1, 28] = -( r * s * ( -1 + t ) * ( 1 + t ) * ( -1 + s ) * ( 1 + r ) ) / 4.
        N_mtx[1, 31] = -( r * s * ( -1 + t ) * ( 1 + t ) * ( 1 + s ) * ( 1 + r ) ) / 4.
        N_mtx[1, 34] = -( r * s * ( -1 + t ) * ( 1 + t ) * ( 1 + s ) * ( -1 + r ) ) / 4.
        N_mtx[1, 37] = ( r * t * s * ( -1 + s ) * ( 1 + t ) * ( -1 + r ) ) / 8.
        N_mtx[1, 40] = ( r * t * s * ( -1 + s ) * ( 1 + t ) * ( 1 + r ) ) / 8.
        N_mtx[1, 43] = ( r * t * s * ( 1 + s ) * ( 1 + t ) * ( 1 + r ) ) / 8.
        N_mtx[1, 46] = ( r * t * s * ( 1 + s ) * ( 1 + t ) * ( -1 + r ) ) / 8.
        N_mtx[1, 49] = -( t * s * ( -1 + r ) * ( 1 + r ) * ( -1 + s ) * ( 1 + t ) ) / 4.
        N_mtx[1, 52] = -( r * t * ( -1 + s ) * ( 1 + s ) * ( 1 + t ) * ( 1 + r ) ) / 4.
        N_mtx[1, 55] = -( t * s * ( -1 + r ) * ( 1 + r ) * ( 1 + s ) * ( 1 + t ) ) / 4.
        N_mtx[1, 58] = -( r * t * ( -1 + s ) * ( 1 + s ) * ( 1 + t ) * ( -1 + r ) ) / 4.
        N_mtx[1, 61] = ( t * ( -1 + s ) * ( 1 + s ) * ( -1 + r ) * ( 1 + r ) * ( -1 + t ) ) / 2.
        N_mtx[1, 64] = ( s * ( -1 + t ) * ( 1 + t ) * ( -1 + r ) * ( 1 + r ) * ( -1 + s ) ) / 2.
        N_mtx[1, 67] = ( r * ( -1 + s ) * ( 1 + s ) * ( -1 + t ) * ( 1 + t ) * ( 1 + r ) ) / 2.
        N_mtx[1, 70] = ( s * ( -1 + t ) * ( 1 + t ) * ( -1 + r ) * ( 1 + r ) * ( 1 + s ) ) / 2.
        N_mtx[1, 73] = ( r * ( -1 + s ) * ( 1 + s ) * ( -1 + t ) * ( 1 + t ) * ( -1 + r ) ) / 2.
        N_mtx[1, 76] = ( t * ( -1 + s ) * ( 1 + s ) * ( -1 + r ) * ( 1 + r ) * ( 1 + t ) ) / 2.
        N_mtx[1, 79] = -( -1 + s ) * ( 1 + s ) * ( -1 + t ) * ( 1 + t ) * ( -1 + r ) * ( 1 + r )
        N_mtx[2, 2] = ( r * t * s * ( -1 + s ) * ( -1 + t ) * ( -1 + r ) ) / 8.
        N_mtx[2, 5] = ( r * t * s * ( -1 + s ) * ( -1 + t ) * ( 1 + r ) ) / 8.
        N_mtx[2, 8] = ( r * t * s * ( 1 + s ) * ( -1 + t ) * ( 1 + r ) ) / 8.
        N_mtx[2, 11] = ( r * t * s * ( 1 + s ) * ( -1 + t ) * ( -1 + r ) ) / 8.
        N_mtx[2, 14] = -( t * s * ( -1 + r ) * ( 1 + r ) * ( -1 + s ) * ( -1 + t ) ) / 4.
        N_mtx[2, 17] = -( r * t * ( -1 + s ) * ( 1 + s ) * ( -1 + t ) * ( 1 + r ) ) / 4.
        N_mtx[2, 20] = -( t * s * ( -1 + r ) * ( 1 + r ) * ( 1 + s ) * ( -1 + t ) ) / 4.
        N_mtx[2, 23] = -( r * t * ( -1 + s ) * ( 1 + s ) * ( -1 + t ) * ( -1 + r ) ) / 4.
        N_mtx[2, 26] = -( r * s * ( -1 + t ) * ( 1 + t ) * ( -1 + s ) * ( -1 + r ) ) / 4.
        N_mtx[2, 29] = -( r * s * ( -1 + t ) * ( 1 + t ) * ( -1 + s ) * ( 1 + r ) ) / 4.
        N_mtx[2, 32] = -( r * s * ( -1 + t ) * ( 1 + t ) * ( 1 + s ) * ( 1 + r ) ) / 4.
        N_mtx[2, 35] = -( r * s * ( -1 + t ) * ( 1 + t ) * ( 1 + s ) * ( -1 + r ) ) / 4.
        N_mtx[2, 38] = ( r * t * s * ( -1 + s ) * ( 1 + t ) * ( -1 + r ) ) / 8.
        N_mtx[2, 41] = ( r * t * s * ( -1 + s ) * ( 1 + t ) * ( 1 + r ) ) / 8.
        N_mtx[2, 44] = ( r * t * s * ( 1 + s ) * ( 1 + t ) * ( 1 + r ) ) / 8.
        N_mtx[2, 47] = ( r * t * s * ( 1 + s ) * ( 1 + t ) * ( -1 + r ) ) / 8.
        N_mtx[2, 50] = -( t * s * ( -1 + r ) * ( 1 + r ) * ( -1 + s ) * ( 1 + t ) ) / 4.
        N_mtx[2, 53] = -( r * t * ( -1 + s ) * ( 1 + s ) * ( 1 + t ) * ( 1 + r ) ) / 4.
        N_mtx[2, 56] = -( t * s * ( -1 + r ) * ( 1 + r ) * ( 1 + s ) * ( 1 + t ) ) / 4.
        N_mtx[2, 59] = -( r * t * ( -1 + s ) * ( 1 + s ) * ( 1 + t ) * ( -1 + r ) ) / 4.
        N_mtx[2, 62] = ( t * ( -1 + s ) * ( 1 + s ) * ( -1 + r ) * ( 1 + r ) * ( -1 + t ) ) / 2.
        N_mtx[2, 65] = ( s * ( -1 + t ) * ( 1 + t ) * ( -1 + r ) * ( 1 + r ) * ( -1 + s ) ) / 2.
        N_mtx[2, 68] = ( r * ( -1 + s ) * ( 1 + s ) * ( -1 + t ) * ( 1 + t ) * ( 1 + r ) ) / 2.
        N_mtx[2, 71] = ( s * ( -1 + t ) * ( 1 + t ) * ( -1 + r ) * ( 1 + r ) * ( 1 + s ) ) / 2.
        N_mtx[2, 74] = ( r * ( -1 + s ) * ( 1 + s ) * ( -1 + t ) * ( 1 + t ) * ( -1 + r ) ) / 2.
        N_mtx[2, 77] = ( t * ( -1 + s ) * ( 1 + s ) * ( -1 + r ) * ( 1 + r ) * ( 1 + t ) ) / 2.
        N_mtx[2, 80] = -( -1 + s ) * ( 1 + s ) * ( -1 + t ) * ( 1 + t ) * ( -1 + r ) * ( 1 + r )
        return N_mtx

    def get_dNr_mtx( self, r_pnt ):
        '''
        Return the derivatives of the shape functions used for the field approximation
        '''
        r = r_pnt[0]
        s = r_pnt[1]
        t = r_pnt[2]
        dNr = zeros( ( 3, 27 ), dtype = 'float_' )
        dNr[0, 0] = ( t * s * ( -1 + r ) * ( -1 + s ) * ( -1 + t ) ) / 8. + ( t * s * r * ( -1 + s ) * ( -1 + t ) ) / 8.
        dNr[0, 1] = ( t * s * ( 1 + r ) * ( -1 + s ) * ( -1 + t ) ) / 8. + ( t * s * r * ( -1 + s ) * ( -1 + t ) ) / 8.
        dNr[0, 2] = ( t * s * ( 1 + r ) * ( 1 + s ) * ( -1 + t ) ) / 8. + ( t * s * r * ( 1 + s ) * ( -1 + t ) ) / 8.
        dNr[0, 3] = ( t * s * ( -1 + r ) * ( 1 + s ) * ( -1 + t ) ) / 8. + ( t * s * r * ( 1 + s ) * ( -1 + t ) ) / 8.
        dNr[0, 4] = -( t * s * ( 1 + r ) * ( -1 + s ) * ( -1 + t ) ) / 4. - ( t * s * ( -1 + r ) * ( -1 + s ) * ( -1 + t ) ) / 4.
        dNr[0, 5] = -( t * ( -1 + s ) * ( 1 + s ) * ( 1 + r ) * ( -1 + t ) ) / 4. - ( t * r * ( -1 + s ) * ( 1 + s ) * ( -1 + t ) ) / 4.
        dNr[0, 6] = -( t * s * ( 1 + r ) * ( 1 + s ) * ( -1 + t ) ) / 4. - ( t * s * ( -1 + r ) * ( 1 + s ) * ( -1 + t ) ) / 4.
        dNr[0, 7] = -( t * ( -1 + s ) * ( 1 + s ) * ( -1 + r ) * ( -1 + t ) ) / 4. - ( t * r * ( -1 + s ) * ( 1 + s ) * ( -1 + t ) ) / 4.
        dNr[0, 8] = -( s * ( -1 + t ) * ( 1 + t ) * ( -1 + r ) * ( -1 + s ) ) / 4. - ( s * r * ( -1 + t ) * ( 1 + t ) * ( -1 + s ) ) / 4.
        dNr[0, 9] = -( s * ( -1 + t ) * ( 1 + t ) * ( 1 + r ) * ( -1 + s ) ) / 4. - ( s * r * ( -1 + t ) * ( 1 + t ) * ( -1 + s ) ) / 4.
        dNr[0, 10] = -( s * ( -1 + t ) * ( 1 + t ) * ( 1 + r ) * ( 1 + s ) ) / 4. - ( s * r * ( -1 + t ) * ( 1 + t ) * ( 1 + s ) ) / 4.
        dNr[0, 11] = -( s * ( -1 + t ) * ( 1 + t ) * ( -1 + r ) * ( 1 + s ) ) / 4. - ( s * r * ( -1 + t ) * ( 1 + t ) * ( 1 + s ) ) / 4.
        dNr[0, 12] = ( t * s * ( -1 + r ) * ( -1 + s ) * ( 1 + t ) ) / 8. + ( t * s * r * ( -1 + s ) * ( 1 + t ) ) / 8.
        dNr[0, 13] = ( t * s * ( 1 + r ) * ( -1 + s ) * ( 1 + t ) ) / 8. + ( t * s * r * ( -1 + s ) * ( 1 + t ) ) / 8.
        dNr[0, 14] = ( t * s * ( 1 + r ) * ( 1 + s ) * ( 1 + t ) ) / 8. + ( t * s * r * ( 1 + s ) * ( 1 + t ) ) / 8.
        dNr[0, 15] = ( t * s * ( -1 + r ) * ( 1 + s ) * ( 1 + t ) ) / 8. + ( t * s * r * ( 1 + s ) * ( 1 + t ) ) / 8.
        dNr[0, 16] = -( t * s * ( 1 + r ) * ( -1 + s ) * ( 1 + t ) ) / 4. - ( t * s * ( -1 + r ) * ( -1 + s ) * ( 1 + t ) ) / 4.
        dNr[0, 17] = -( t * ( -1 + s ) * ( 1 + s ) * ( 1 + r ) * ( 1 + t ) ) / 4. - ( t * r * ( -1 + s ) * ( 1 + s ) * ( 1 + t ) ) / 4.
        dNr[0, 18] = -( t * s * ( 1 + r ) * ( 1 + s ) * ( 1 + t ) ) / 4. - ( t * s * ( -1 + r ) * ( 1 + s ) * ( 1 + t ) ) / 4.
        dNr[0, 19] = -( t * ( -1 + s ) * ( 1 + s ) * ( -1 + r ) * ( 1 + t ) ) / 4. - ( t * r * ( -1 + s ) * ( 1 + s ) * ( 1 + t ) ) / 4.
        dNr[0, 20] = ( t * ( -1 + s ) * ( 1 + s ) * ( 1 + r ) * ( -1 + t ) ) / 2. + ( t * ( -1 + s ) * ( 1 + s ) * ( -1 + r ) * ( -1 + t ) ) / 2.
        dNr[0, 21] = ( s * ( -1 + t ) * ( 1 + t ) * ( 1 + r ) * ( -1 + s ) ) / 2. + ( s * ( -1 + t ) * ( 1 + t ) * ( -1 + r ) * ( -1 + s ) ) / 2.
        dNr[0, 22] = ( ( -1 + s ) * ( 1 + s ) * ( -1 + t ) * ( 1 + t ) * ( 1 + r ) ) / 2. + ( r * ( -1 + s ) * ( 1 + s ) * ( -1 + t ) * ( 1 + t ) ) / 2.
        dNr[0, 23] = ( s * ( -1 + t ) * ( 1 + t ) * ( 1 + r ) * ( 1 + s ) ) / 2. + ( s * ( -1 + t ) * ( 1 + t ) * ( -1 + r ) * ( 1 + s ) ) / 2.
        dNr[0, 24] = ( ( -1 + s ) * ( 1 + s ) * ( -1 + t ) * ( 1 + t ) * ( -1 + r ) ) / 2. + ( r * ( -1 + s ) * ( 1 + s ) * ( -1 + t ) * ( 1 + t ) ) / 2.
        dNr[0, 25] = ( t * ( -1 + s ) * ( 1 + s ) * ( 1 + r ) * ( 1 + t ) ) / 2. + ( t * ( -1 + s ) * ( 1 + s ) * ( -1 + r ) * ( 1 + t ) ) / 2.
        dNr[0, 26] = -( -1 + s ) * ( 1 + s ) * ( -1 + t ) * ( 1 + t ) * ( 1 + r ) - ( -1 + s ) * ( 1 + s ) * ( -1 + t ) * ( 1 + t ) * ( -1 + r )
        dNr[1, 0] = ( t * r * ( -1 + r ) * ( -1 + s ) * ( -1 + t ) ) / 8. + ( t * s * r * ( -1 + r ) * ( -1 + t ) ) / 8.
        dNr[1, 1] = ( t * r * ( 1 + r ) * ( -1 + s ) * ( -1 + t ) ) / 8. + ( t * s * r * ( 1 + r ) * ( -1 + t ) ) / 8.
        dNr[1, 2] = ( t * r * ( 1 + r ) * ( 1 + s ) * ( -1 + t ) ) / 8. + ( t * s * r * ( 1 + r ) * ( -1 + t ) ) / 8.
        dNr[1, 3] = ( t * r * ( -1 + r ) * ( 1 + s ) * ( -1 + t ) ) / 8. + ( t * s * r * ( -1 + r ) * ( -1 + t ) ) / 8.
        dNr[1, 4] = -( t * ( -1 + r ) * ( 1 + r ) * ( -1 + s ) * ( -1 + t ) ) / 4. - ( t * s * ( -1 + r ) * ( 1 + r ) * ( -1 + t ) ) / 4.
        dNr[1, 5] = -( t * r * ( 1 + r ) * ( 1 + s ) * ( -1 + t ) ) / 4. - ( t * r * ( 1 + r ) * ( -1 + s ) * ( -1 + t ) ) / 4.
        dNr[1, 6] = -( t * ( -1 + r ) * ( 1 + r ) * ( 1 + s ) * ( -1 + t ) ) / 4. - ( t * s * ( -1 + r ) * ( 1 + r ) * ( -1 + t ) ) / 4.
        dNr[1, 7] = -( t * r * ( -1 + r ) * ( 1 + s ) * ( -1 + t ) ) / 4. - ( t * r * ( -1 + r ) * ( -1 + s ) * ( -1 + t ) ) / 4.
        dNr[1, 8] = -( r * ( -1 + t ) * ( 1 + t ) * ( -1 + r ) * ( -1 + s ) ) / 4. - ( s * r * ( -1 + t ) * ( 1 + t ) * ( -1 + r ) ) / 4.
        dNr[1, 9] = -( r * ( -1 + t ) * ( 1 + t ) * ( 1 + r ) * ( -1 + s ) ) / 4. - ( s * r * ( -1 + t ) * ( 1 + t ) * ( 1 + r ) ) / 4.
        dNr[1, 10] = -( r * ( -1 + t ) * ( 1 + t ) * ( 1 + r ) * ( 1 + s ) ) / 4. - ( s * r * ( -1 + t ) * ( 1 + t ) * ( 1 + r ) ) / 4.
        dNr[1, 11] = -( r * ( -1 + t ) * ( 1 + t ) * ( -1 + r ) * ( 1 + s ) ) / 4. - ( s * r * ( -1 + t ) * ( 1 + t ) * ( -1 + r ) ) / 4.
        dNr[1, 12] = ( t * r * ( -1 + r ) * ( -1 + s ) * ( 1 + t ) ) / 8. + ( t * s * r * ( -1 + r ) * ( 1 + t ) ) / 8.
        dNr[1, 13] = ( t * r * ( 1 + r ) * ( -1 + s ) * ( 1 + t ) ) / 8. + ( t * s * r * ( 1 + r ) * ( 1 + t ) ) / 8.
        dNr[1, 14] = ( t * r * ( 1 + r ) * ( 1 + s ) * ( 1 + t ) ) / 8. + ( t * s * r * ( 1 + r ) * ( 1 + t ) ) / 8.
        dNr[1, 15] = ( t * r * ( -1 + r ) * ( 1 + s ) * ( 1 + t ) ) / 8. + ( t * s * r * ( -1 + r ) * ( 1 + t ) ) / 8.
        dNr[1, 16] = -( t * ( -1 + r ) * ( 1 + r ) * ( -1 + s ) * ( 1 + t ) ) / 4. - ( t * s * ( -1 + r ) * ( 1 + r ) * ( 1 + t ) ) / 4.
        dNr[1, 17] = -( t * r * ( 1 + r ) * ( 1 + s ) * ( 1 + t ) ) / 4. - ( t * r * ( 1 + r ) * ( -1 + s ) * ( 1 + t ) ) / 4.
        dNr[1, 18] = -( t * ( -1 + r ) * ( 1 + r ) * ( 1 + s ) * ( 1 + t ) ) / 4. - ( t * s * ( -1 + r ) * ( 1 + r ) * ( 1 + t ) ) / 4.
        dNr[1, 19] = -( t * r * ( -1 + r ) * ( 1 + s ) * ( 1 + t ) ) / 4. - ( t * r * ( -1 + r ) * ( -1 + s ) * ( 1 + t ) ) / 4.
        dNr[1, 20] = ( t * ( -1 + r ) * ( 1 + r ) * ( 1 + s ) * ( -1 + t ) ) / 2. + ( t * ( -1 + r ) * ( 1 + r ) * ( -1 + s ) * ( -1 + t ) ) / 2.
        dNr[1, 21] = ( ( -1 + r ) * ( 1 + r ) * ( -1 + t ) * ( 1 + t ) * ( -1 + s ) ) / 2. + ( s * ( -1 + r ) * ( 1 + r ) * ( -1 + t ) * ( 1 + t ) ) / 2.
        dNr[1, 22] = ( r * ( -1 + t ) * ( 1 + t ) * ( 1 + r ) * ( 1 + s ) ) / 2. + ( r * ( -1 + t ) * ( 1 + t ) * ( 1 + r ) * ( -1 + s ) ) / 2.
        dNr[1, 23] = ( ( -1 + r ) * ( 1 + r ) * ( -1 + t ) * ( 1 + t ) * ( 1 + s ) ) / 2. + ( s * ( -1 + r ) * ( 1 + r ) * ( -1 + t ) * ( 1 + t ) ) / 2.
        dNr[1, 24] = ( r * ( -1 + t ) * ( 1 + t ) * ( -1 + r ) * ( 1 + s ) ) / 2. + ( r * ( -1 + t ) * ( 1 + t ) * ( -1 + r ) * ( -1 + s ) ) / 2.
        dNr[1, 25] = ( t * ( -1 + r ) * ( 1 + r ) * ( 1 + s ) * ( 1 + t ) ) / 2. + ( t * ( -1 + r ) * ( 1 + r ) * ( -1 + s ) * ( 1 + t ) ) / 2.
        dNr[1, 26] = -( -1 + r ) * ( 1 + r ) * ( -1 + t ) * ( 1 + t ) * ( 1 + s ) - ( -1 + r ) * ( 1 + r ) * ( -1 + t ) * ( 1 + t ) * ( -1 + s )
        dNr[2, 0] = ( s * r * ( -1 + r ) * ( -1 + s ) * ( -1 + t ) ) / 8. + ( t * s * r * ( -1 + r ) * ( -1 + s ) ) / 8.
        dNr[2, 1] = ( s * r * ( 1 + r ) * ( -1 + s ) * ( -1 + t ) ) / 8. + ( t * s * r * ( 1 + r ) * ( -1 + s ) ) / 8.
        dNr[2, 2] = ( s * r * ( 1 + r ) * ( 1 + s ) * ( -1 + t ) ) / 8. + ( t * s * r * ( 1 + r ) * ( 1 + s ) ) / 8.
        dNr[2, 3] = ( s * r * ( -1 + r ) * ( 1 + s ) * ( -1 + t ) ) / 8. + ( t * s * r * ( -1 + r ) * ( 1 + s ) ) / 8.
        dNr[2, 4] = -( s * ( -1 + r ) * ( 1 + r ) * ( -1 + s ) * ( -1 + t ) ) / 4. - ( t * s * ( -1 + r ) * ( 1 + r ) * ( -1 + s ) ) / 4.
        dNr[2, 5] = -( r * ( -1 + s ) * ( 1 + s ) * ( 1 + r ) * ( -1 + t ) ) / 4. - ( t * r * ( -1 + s ) * ( 1 + s ) * ( 1 + r ) ) / 4.
        dNr[2, 6] = -( s * ( -1 + r ) * ( 1 + r ) * ( 1 + s ) * ( -1 + t ) ) / 4. - ( t * s * ( -1 + r ) * ( 1 + r ) * ( 1 + s ) ) / 4.
        dNr[2, 7] = -( r * ( -1 + s ) * ( 1 + s ) * ( -1 + r ) * ( -1 + t ) ) / 4. - ( t * r * ( -1 + s ) * ( 1 + s ) * ( -1 + r ) ) / 4.
        dNr[2, 8] = -( s * r * ( 1 + t ) * ( -1 + r ) * ( -1 + s ) ) / 4. - ( s * r * ( -1 + r ) * ( -1 + s ) * ( -1 + t ) ) / 4.
        dNr[2, 9] = -( s * r * ( 1 + t ) * ( 1 + r ) * ( -1 + s ) ) / 4. - ( s * r * ( 1 + r ) * ( -1 + s ) * ( -1 + t ) ) / 4.
        dNr[2, 10] = -( s * r * ( 1 + t ) * ( 1 + r ) * ( 1 + s ) ) / 4. - ( s * r * ( 1 + r ) * ( 1 + s ) * ( -1 + t ) ) / 4.
        dNr[2, 11] = -( s * r * ( 1 + t ) * ( -1 + r ) * ( 1 + s ) ) / 4. - ( s * r * ( -1 + r ) * ( 1 + s ) * ( -1 + t ) ) / 4.
        dNr[2, 12] = ( s * r * ( 1 + t ) * ( -1 + r ) * ( -1 + s ) ) / 8. + ( t * s * r * ( -1 + r ) * ( -1 + s ) ) / 8.
        dNr[2, 13] = ( s * r * ( 1 + t ) * ( 1 + r ) * ( -1 + s ) ) / 8. + ( t * s * r * ( 1 + r ) * ( -1 + s ) ) / 8.
        dNr[2, 14] = ( s * r * ( 1 + t ) * ( 1 + r ) * ( 1 + s ) ) / 8. + ( t * s * r * ( 1 + r ) * ( 1 + s ) ) / 8.
        dNr[2, 15] = ( s * r * ( 1 + t ) * ( -1 + r ) * ( 1 + s ) ) / 8. + ( t * s * r * ( -1 + r ) * ( 1 + s ) ) / 8.
        dNr[2, 16] = -( s * ( -1 + r ) * ( 1 + r ) * ( -1 + s ) * ( 1 + t ) ) / 4. - ( t * s * ( -1 + r ) * ( 1 + r ) * ( -1 + s ) ) / 4.
        dNr[2, 17] = -( r * ( -1 + s ) * ( 1 + s ) * ( 1 + r ) * ( 1 + t ) ) / 4. - ( t * r * ( -1 + s ) * ( 1 + s ) * ( 1 + r ) ) / 4.
        dNr[2, 18] = -( s * ( -1 + r ) * ( 1 + r ) * ( 1 + s ) * ( 1 + t ) ) / 4. - ( t * s * ( -1 + r ) * ( 1 + r ) * ( 1 + s ) ) / 4.
        dNr[2, 19] = -( r * ( -1 + s ) * ( 1 + s ) * ( -1 + r ) * ( 1 + t ) ) / 4. - ( t * r * ( -1 + s ) * ( 1 + s ) * ( -1 + r ) ) / 4.
        dNr[2, 20] = ( ( -1 + r ) * ( 1 + r ) * ( -1 + s ) * ( 1 + s ) * ( -1 + t ) ) / 2. + ( t * ( -1 + r ) * ( 1 + r ) * ( -1 + s ) * ( 1 + s ) ) / 2.
        dNr[2, 21] = ( s * ( -1 + r ) * ( 1 + r ) * ( -1 + s ) * ( 1 + t ) ) / 2. + ( s * ( -1 + r ) * ( 1 + r ) * ( -1 + s ) * ( -1 + t ) ) / 2.
        dNr[2, 22] = ( r * ( -1 + s ) * ( 1 + s ) * ( 1 + r ) * ( 1 + t ) ) / 2. + ( r * ( -1 + s ) * ( 1 + s ) * ( 1 + r ) * ( -1 + t ) ) / 2.
        dNr[2, 23] = ( s * ( -1 + r ) * ( 1 + r ) * ( 1 + s ) * ( 1 + t ) ) / 2. + ( s * ( -1 + r ) * ( 1 + r ) * ( 1 + s ) * ( -1 + t ) ) / 2.
        dNr[2, 24] = ( r * ( -1 + s ) * ( 1 + s ) * ( -1 + r ) * ( 1 + t ) ) / 2. + ( r * ( -1 + s ) * ( 1 + s ) * ( -1 + r ) * ( -1 + t ) ) / 2.
        dNr[2, 25] = ( ( -1 + r ) * ( 1 + r ) * ( -1 + s ) * ( 1 + s ) * ( 1 + t ) ) / 2. + ( t * ( -1 + r ) * ( 1 + r ) * ( -1 + s ) * ( 1 + s ) ) / 2.
        dNr[2, 26] = -( -1 + r ) * ( 1 + r ) * ( -1 + s ) * ( 1 + s ) * ( 1 + t ) - ( -1 + r ) * ( 1 + r ) * ( -1 + s ) * ( 1 + s ) * ( -1 + t )
        return dNr

#----------------------- example --------------------

if __name__ == '__main__':
    from ibvpy.api import \
        TStepper as TS, RTDofGraph, RTraceDomainListField, TLoop, \
        TLine, BCDofGroup, IBVPSolve as IS, DOTSEval

    #from lib.mats.mats2D.mats_cmdm2D.mats_mdm2d import MACMDM
    from ibvpy.mats.mats2D.mats2D_sdamage.mats2D_sdamage import MATS2DScalarDamage
    from ibvpy.mats.mats2D.mats2D_sdamage.strain_norm2d import *
    from ibvpy.mats.mats3D.mats3D_elastic.mats3D_elastic import MATS3DElastic

#    fets_eval = FETS2D9Q(mats_eval = MA2DSca    larDamage(strain_norm = Euclidean())) 
    fets_eval = FETS3D8H27U( mats_eval = MATS3DElastic() )

    from ibvpy.mesh.fe_grid import FEGrid

    # Discretization
    domain = FEGrid( coord_max = ( 3., 3., 3. ),
                           shape = ( 3, 3, 3 ),
                           fets_eval = fets_eval )

    # Put the tseval (time-stepper) into the spatial context of the
    # discretization and specify the response tracers to evaluate there.
    #

    right_dof = 2
    ts = TS( 
            sdomain = domain,
             # conversion to list (square brackets) is only necessary for slicing of 
             # single dofs, e.g "get_left_dofs()[0,1]" which elsewise retuns an integer only
             bcond_list = [ BCDofGroup( var = 'u', value = 0., dims = [0],
                                  get_dof_method = domain.get_left_dofs ),
                        BCDofGroup( var = 'u', value = 0., dims = [1, 2],
                                  get_dof_method = domain.get_bottom_left_dofs ),
                        BCDofGroup( var = 'u', value = 0.002, dims = [1],
                                  get_dof_method = domain.get_right_dofs ) ],
             rtrace_list = [
#                        RTDofGraph(name = 'Fi,right over u_right (iteration)' ,
#                                  var_y = 'F_int', idx_y = right_dof,
#                                  var_x = 'U_k', idx_x = right_dof,
#                                  record_on = 'update'),
#                        RTraceDomainListField(name = 'Deformation' ,
#                                       var = 'eps', idx = 0,
#                                       record_on = 'update'),
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
    global tloop
    tloop = TLoop( tstepper = ts,
         tline = TLine( min = 0.0, step = 0.5, max = .5 ) )

    import cProfile
    cProfile.run( 'tloop.eval()', 'tloop_prof' )

#    import pstats
#    p = pstats.Stats('tloop_prof')
#    p.strip_dirs()
#    print 'cumulative'
#    p.sort_stats('cumulative').print_stats(20)
#    print 'time'
#    p.sort_stats('time').print_stats(20)

    # Put the whole stuff into the simulation-framework to map the
    # individual pieces of definition into the user interface.
    #
    from ibvpy.plugins.ibvpy_app import IBVPyApp
    app = IBVPyApp( ibv_resource = tloop )
    app.main()
