from numpy import \
    zeros, dot, hstack, identity
from scipy.linalg import \
    inv
from traits.api import \
    Instance, Int, Property, Array, cached_property

from .fets3D import FETS3D


#-------------------------------------------------------------------------
# FETS3D8H24U - 24 nodes subparametric 3D volume element: rs-direction: cubic (serendipity)
#                                                          t-direction: linear
#-------------------------------------------------------------------------
class FETS3D8H24U(FETS3D):
    '''
    cubic/linear hybrid serendipity volume element
    '''
    debug_on = True

    # Dimensional mapping
    #
    dim_slice = slice(0, 3)

    # number of nodal degrees of freedom
    # number of degrees of freedom of each element
    n_nodal_dofs = Int(3)
    n_e_dofs = Int(24 * 3)

    # Integration parameters
    # NOTE: reduced integration order for t
    #       due to linear formulation direction
    #
    ngp_r = 4
    ngp_s = 4
    ngp_t = 2

    dof_r = \
        Array(value=[
            # lower side (t=-1)
            #
            [-1., -1., -1.],
            [-1. / 3., -1., -1.],
            [1. / 3., -1., -1.],
            [1., -1., -1.],
            #
            [-1., -1. / 3., -1.],
            [1., -1. / 3., -1.],
            #
            [-1., 1. / 3., -1.],
            [1., 1. / 3., -1.],
            #
            [-1., 1., -1.],
            [-1. / 3., 1., -1.],
            [1. / 3., 1., -1.],
            [1., 1., -1.],

            # upper side (t=1)
            #
            [-1., -1., 1.],
            [-1. / 3., -1., 1.],
            [1. / 3., -1., 1.],
            [1., -1., 1.],
            #
            [-1., -1. / 3., 1.],
            [1., -1. / 3., 1.],
            #
            [-1., 1. / 3., 1.],
            [1., 1. / 3., 1.],
            #
            [-1., 1., 1.],
            [-1. / 3., 1., 1.],
            [1. / 3., 1., 1.],
            [1., 1., 1.]
        ])

    geo_r = \
        Array(value=[[-1., -1., -1.],
                     [1., -1., -1.],
                     [-1., 1., -1.],
                     [1., 1., -1.],
                     [-1., -1., 1.],
                     [1., -1., 1.],
                     [-1., 1., 1.],
                     [1., 1., 1.]])

#    # Used for Visualization
#    vtk_cell_types = 'QuadraticHexahedron'
#    vtk_r = Array(value = [[-1., -1., -1.],
#                        [  1., -1., -1.],
#                        [  1., 1., -1.],
#                        [-1., 1., -1.],
#                        [-1., -1., 1.],
#                        [  1., -1., 1.],
#                        [  1., 1., 1.],
#                        [-1., 1., 1.],
#                        [  0., -1., -1.],
#                        [  1., 0., -1.],
#                        [  0., 1., -1.],
#                        [-1., 0., -1.],
#                        [  0., -1., 1.],
#                        [  1., 0., 1.],
#                        [  0., 1., 1.],
#                        [-1., 0., 1.],
#                        [-1., -1., 0.],
#                        [  1., -1., 0.],
#                        [  1., 1., 0.],
#                        [-1., 1., 0.]])
#    vtk_cells = [[ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, \
#                  10, 11, 12, 13, 14, 15, 16, 17, 18, 19]]

    # Used for Visualization
    vtk_cell_types = 'TriQuadraticHexahedron'
    vtk_r = Array(value=[[-1., -1., -1.],  # bottom
                         [1., -1., -1.],
                         [1., 1., -1.],
                         [-1., 1., -1.],
                         [-1., -1., 1.],  # top
                         [1., -1., 1.],
                         [1., 1., 1.],
                         [-1., 1., 1.],
                         [0., -1., -1.],  # bottom midside
                         [1., 0., -1.],
                         [0., 1., -1.],
                         [-1., 0., -1.],
                         [0., -1., 1.],  # top midside
                         [1., 0., 1.],
                         [0., 1., 1.],
                         [-1., 0., 1.],
                         [-1., -1., 0.],  # middle
                         [1., -1., 0.],
                         [1., 1., 0.],
                         [-1., 1., 0.],
                         [0., -1., 0.],  # middle midside (different order)
                         [1., 0., 0.],
                         [0., 1., 0.],
                         [-1., 0., 0.],
                         [0., 0., -1.],  # bottom centre
                         [0., 0., 1.],  # top centre
                         [0., 0., 0.]])  # middle centre

    vtk_cells = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                  10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                  23, 21, 20, 22, 24, 25, 26]]

    #---------------------------------------------------------------------
    # Method required to represent the element geometry
    #---------------------------------------------------------------------
    def get_N_geo_mtx(self, r_pnt):
        '''
        Return the value of shape functions (derived in femple) for the 
        specified local coordinate r_pnt
        '''
        N_geo_mtx = zeros((1, 8), dtype='float_')
        N_geo_mtx[0, 0] = - \
            ((-1 + r_pnt[2]) * (-1 + r_pnt[1]) * (-1 + r_pnt[0])) / 8.0
        N_geo_mtx[0, 1] = (
            (-1 + r_pnt[2]) * (-1 + r_pnt[1]) * (1 + r_pnt[0])) / 8.0
        N_geo_mtx[0, 2] = (
            (-1 + r_pnt[2]) * (1 + r_pnt[1]) * (-1 + r_pnt[0])) / 8.0
        N_geo_mtx[0, 3] = - \
            ((-1 + r_pnt[2]) * (1 + r_pnt[1]) * (1 + r_pnt[0])) / 8.0
        N_geo_mtx[0, 4] = (
            (1 + r_pnt[2]) * (-1 + r_pnt[1]) * (-1 + r_pnt[0])) / 8.0
        N_geo_mtx[0, 5] = - \
            ((1 + r_pnt[2]) * (-1 + r_pnt[1]) * (1 + r_pnt[0])) / 8.0
        N_geo_mtx[0, 6] = - \
            ((1 + r_pnt[2]) * (1 + r_pnt[1]) * (-1 + r_pnt[0])) / 8.0
        N_geo_mtx[0, 7] = (
            (1 + r_pnt[2]) * (1 + r_pnt[1]) * (1 + r_pnt[0])) / 8.0
        return N_geo_mtx

    def get_dNr_geo_mtx(self, r_pnt):
        '''
        Return the matrix of shape function derivatives (derived in femple).
        Used for the construction of the Jacobi matrix.
        '''
        dNr_geo_mtx = zeros((3, 8), dtype='float_')
        dNr_geo_mtx[0, 0] = -((-1 + r_pnt[2]) * (-1 + r_pnt[1])) / 8.0
        dNr_geo_mtx[0, 1] = ((-1 + r_pnt[2]) * (-1 + r_pnt[1])) / 8.0
        dNr_geo_mtx[0, 2] = ((-1 + r_pnt[2]) * (1 + r_pnt[1])) / 8.0
        dNr_geo_mtx[0, 3] = -((-1 + r_pnt[2]) * (1 + r_pnt[1])) / 8.0
        dNr_geo_mtx[0, 4] = ((1 + r_pnt[2]) * (-1 + r_pnt[1])) / 8.0
        dNr_geo_mtx[0, 5] = -((1 + r_pnt[2]) * (-1 + r_pnt[1])) / 8.0
        dNr_geo_mtx[0, 6] = -((1 + r_pnt[2]) * (1 + r_pnt[1])) / 8.0
        dNr_geo_mtx[0, 7] = ((1 + r_pnt[2]) * (1 + r_pnt[1])) / 8.0
        dNr_geo_mtx[1, 0] = -((-1 + r_pnt[2]) * (-1 + r_pnt[0])) / 8.0
        dNr_geo_mtx[1, 1] = ((-1 + r_pnt[2]) * (1 + r_pnt[0])) / 8.0
        dNr_geo_mtx[1, 2] = ((-1 + r_pnt[2]) * (-1 + r_pnt[0])) / 8.0
        dNr_geo_mtx[1, 3] = -((-1 + r_pnt[2]) * (1 + r_pnt[0])) / 8.0
        dNr_geo_mtx[1, 4] = ((1 + r_pnt[2]) * (-1 + r_pnt[0])) / 8.0
        dNr_geo_mtx[1, 5] = -((1 + r_pnt[2]) * (1 + r_pnt[0])) / 8.0
        dNr_geo_mtx[1, 6] = -((1 + r_pnt[2]) * (-1 + r_pnt[0])) / 8.0
        dNr_geo_mtx[1, 7] = ((1 + r_pnt[2]) * (1 + r_pnt[0])) / 8.0
        dNr_geo_mtx[2, 0] = -((-1 + r_pnt[1]) * (-1 + r_pnt[0])) / 8.0
        dNr_geo_mtx[2, 1] = ((-1 + r_pnt[1]) * (1 + r_pnt[0])) / 8.0
        dNr_geo_mtx[2, 2] = ((1 + r_pnt[1]) * (-1 + r_pnt[0])) / 8.0
        dNr_geo_mtx[2, 3] = -((1 + r_pnt[1]) * (1 + r_pnt[0])) / 8.0
        dNr_geo_mtx[2, 4] = ((-1 + r_pnt[1]) * (-1 + r_pnt[0])) / 8.0
        dNr_geo_mtx[2, 5] = -((-1 + r_pnt[1]) * (1 + r_pnt[0])) / 8.0
        dNr_geo_mtx[2, 6] = -((1 + r_pnt[1]) * (-1 + r_pnt[0])) / 8.0
        dNr_geo_mtx[2, 7] = ((1 + r_pnt[1]) * (1 + r_pnt[0])) / 8.0
        return dNr_geo_mtx

    #---------------------------------------------------------------------
    # Method delivering the shape functions for the field variables and their
    # derivatives
    #---------------------------------------------------------------------
    def get_N_mtx(self, r_pnt):
        '''
        Returns the matrix of the shape functions used for the field approximation
        containing zero entries. The number of rows corresponds to the number of nodal
        dofs. The matrix is evaluated for the specified local coordinate r_pnt.
        '''
        r = r_pnt[0]
        s = r_pnt[1]
        t = r_pnt[2]
        N_mtx = zeros((3, 72), dtype='float_')
        N_mtx[
            0, 0] = -((-1 + s) * (-1 + r) * (9 * r * r + 9 * s * s - 10) * (-1 + t)) / 64.
        N_mtx[0, 3] = 9. / 64. * \
            (-1 + r) * (3 * r - 1) * (1 + r) * (-1 + t) * (-1 + s)
        N_mtx[0, 6] = -9. / 64. * \
            (-1 + r) * (3 * r + 1) * (1 + r) * (-1 + t) * (-1 + s)
        N_mtx[0, 9] = (
            (-1 + s) * (1 + r) * (9 * r * r + 9 * s * s - 10) * (-1 + t)) / 64.
        N_mtx[0, 12] = 9. / 64. * \
            (-1 + s) * (3 * s - 1) * (1 + s) * (-1 + t) * (-1 + r)
        N_mtx[0, 15] = -9. / 64. * \
            (-1 + s) * (3 * s - 1) * (1 + s) * (-1 + t) * (1 + r)
        N_mtx[0, 18] = -9. / 64. * \
            (-1 + s) * (3 * s + 1) * (1 + s) * (-1 + t) * (-1 + r)
        N_mtx[0, 21] = 9. / 64. * \
            (-1 + s) * (3 * s + 1) * (1 + s) * (-1 + t) * (1 + r)
        N_mtx[0, 24] = (
            (1 + s) * (-1 + r) * (9 * r * r + 9 * s * s - 10) * (-1 + t)) / 64.
        N_mtx[0, 27] = -9. / 64. * \
            (-1 + r) * (3 * r - 1) * (1 + r) * (-1 + t) * (1 + s)
        N_mtx[0, 30] = 9. / 64. * \
            (-1 + r) * (3 * r + 1) * (1 + r) * (-1 + t) * (1 + s)
        N_mtx[0, 33] = - \
            ((1 + s) * (1 + r) * (9 * r * r + 9 * s * s - 10) * (-1 + t)) / 64.
        N_mtx[0, 36] = (
            (-1 + s) * (-1 + r) * (9 * r * r + 9 * s * s - 10) * (1 + t)) / 64.
        N_mtx[0, 39] = -9. / 64. * \
            (-1 + r) * (3 * r - 1) * (1 + r) * (1 + t) * (-1 + s)
        N_mtx[0, 42] = 9. / 64. * \
            (-1 + r) * (3 * r + 1) * (1 + r) * (1 + t) * (-1 + s)
        N_mtx[0, 45] = - \
            ((-1 + s) * (1 + r) * (9 * r * r + 9 * s * s - 10) * (1 + t)) / 64.
        N_mtx[0, 48] = -9. / 64. * \
            (-1 + s) * (3 * s - 1) * (1 + s) * (1 + t) * (-1 + r)
        N_mtx[0, 51] = 9. / 64. * \
            (-1 + s) * (3 * s - 1) * (1 + s) * (1 + t) * (1 + r)
        N_mtx[0, 54] = 9. / 64. * \
            (-1 + s) * (3 * s + 1) * (1 + s) * (1 + t) * (-1 + r)
        N_mtx[0, 57] = -9. / 64. * \
            (-1 + s) * (3 * s + 1) * (1 + s) * (1 + t) * (1 + r)
        N_mtx[0, 60] = - \
            ((1 + s) * (-1 + r) * (9 * r * r + 9 * s * s - 10) * (1 + t)) / 64.
        N_mtx[0, 63] = 9. / 64. * \
            (-1 + r) * (3 * r - 1) * (1 + r) * (1 + t) * (1 + s)
        N_mtx[0, 66] = -9. / 64. * \
            (-1 + r) * (3 * r + 1) * (1 + r) * (1 + t) * (1 + s)
        N_mtx[0, 69] = (
            (1 + s) * (1 + r) * (9 * r * r + 9 * s * s - 10) * (1 + t)) / 64.
        N_mtx[
            1, 1] = -((-1 + s) * (-1 + r) * (9 * r * r + 9 * s * s - 10) * (-1 + t)) / 64.
        N_mtx[1, 4] = 9. / 64. * \
            (-1 + r) * (3 * r - 1) * (1 + r) * (-1 + t) * (-1 + s)
        N_mtx[1, 7] = -9. / 64. * \
            (-1 + r) * (3 * r + 1) * (1 + r) * (-1 + t) * (-1 + s)
        N_mtx[1, 10] = (
            (-1 + s) * (1 + r) * (9 * r * r + 9 * s * s - 10) * (-1 + t)) / 64.
        N_mtx[1, 13] = 9. / 64. * \
            (-1 + s) * (3 * s - 1) * (1 + s) * (-1 + t) * (-1 + r)
        N_mtx[1, 16] = -9. / 64. * \
            (-1 + s) * (3 * s - 1) * (1 + s) * (-1 + t) * (1 + r)
        N_mtx[1, 19] = -9. / 64. * \
            (-1 + s) * (3 * s + 1) * (1 + s) * (-1 + t) * (-1 + r)
        N_mtx[1, 22] = 9. / 64. * \
            (-1 + s) * (3 * s + 1) * (1 + s) * (-1 + t) * (1 + r)
        N_mtx[1, 25] = (
            (1 + s) * (-1 + r) * (9 * r * r + 9 * s * s - 10) * (-1 + t)) / 64.
        N_mtx[1, 28] = -9. / 64. * \
            (-1 + r) * (3 * r - 1) * (1 + r) * (-1 + t) * (1 + s)
        N_mtx[1, 31] = 9. / 64. * \
            (-1 + r) * (3 * r + 1) * (1 + r) * (-1 + t) * (1 + s)
        N_mtx[1, 34] = - \
            ((1 + s) * (1 + r) * (9 * r * r + 9 * s * s - 10) * (-1 + t)) / 64.
        N_mtx[1, 37] = (
            (-1 + s) * (-1 + r) * (9 * r * r + 9 * s * s - 10) * (1 + t)) / 64.
        N_mtx[1, 40] = -9. / 64. * \
            (-1 + r) * (3 * r - 1) * (1 + r) * (1 + t) * (-1 + s)
        N_mtx[1, 43] = 9. / 64. * \
            (-1 + r) * (3 * r + 1) * (1 + r) * (1 + t) * (-1 + s)
        N_mtx[1, 46] = - \
            ((-1 + s) * (1 + r) * (9 * r * r + 9 * s * s - 10) * (1 + t)) / 64.
        N_mtx[1, 49] = -9. / 64. * \
            (-1 + s) * (3 * s - 1) * (1 + s) * (1 + t) * (-1 + r)
        N_mtx[1, 52] = 9. / 64. * \
            (-1 + s) * (3 * s - 1) * (1 + s) * (1 + t) * (1 + r)
        N_mtx[1, 55] = 9. / 64. * \
            (-1 + s) * (3 * s + 1) * (1 + s) * (1 + t) * (-1 + r)
        N_mtx[1, 58] = -9. / 64. * \
            (-1 + s) * (3 * s + 1) * (1 + s) * (1 + t) * (1 + r)
        N_mtx[1, 61] = - \
            ((1 + s) * (-1 + r) * (9 * r * r + 9 * s * s - 10) * (1 + t)) / 64.
        N_mtx[1, 64] = 9. / 64. * \
            (-1 + r) * (3 * r - 1) * (1 + r) * (1 + t) * (1 + s)
        N_mtx[1, 67] = -9. / 64. * \
            (-1 + r) * (3 * r + 1) * (1 + r) * (1 + t) * (1 + s)
        N_mtx[1, 70] = (
            (1 + s) * (1 + r) * (9 * r * r + 9 * s * s - 10) * (1 + t)) / 64.
        N_mtx[
            2, 2] = -((-1 + s) * (-1 + r) * (9 * r * r + 9 * s * s - 10) * (-1 + t)) / 64.
        N_mtx[2, 5] = 9. / 64. * \
            (-1 + r) * (3 * r - 1) * (1 + r) * (-1 + t) * (-1 + s)
        N_mtx[2, 8] = -9. / 64. * \
            (-1 + r) * (3 * r + 1) * (1 + r) * (-1 + t) * (-1 + s)
        N_mtx[2, 11] = (
            (-1 + s) * (1 + r) * (9 * r * r + 9 * s * s - 10) * (-1 + t)) / 64.
        N_mtx[2, 14] = 9. / 64. * \
            (-1 + s) * (3 * s - 1) * (1 + s) * (-1 + t) * (-1 + r)
        N_mtx[2, 17] = -9. / 64. * \
            (-1 + s) * (3 * s - 1) * (1 + s) * (-1 + t) * (1 + r)
        N_mtx[2, 20] = -9. / 64. * \
            (-1 + s) * (3 * s + 1) * (1 + s) * (-1 + t) * (-1 + r)
        N_mtx[2, 23] = 9. / 64. * \
            (-1 + s) * (3 * s + 1) * (1 + s) * (-1 + t) * (1 + r)
        N_mtx[2, 26] = (
            (1 + s) * (-1 + r) * (9 * r * r + 9 * s * s - 10) * (-1 + t)) / 64.
        N_mtx[2, 29] = -9. / 64. * \
            (-1 + r) * (3 * r - 1) * (1 + r) * (-1 + t) * (1 + s)
        N_mtx[2, 32] = 9. / 64. * \
            (-1 + r) * (3 * r + 1) * (1 + r) * (-1 + t) * (1 + s)
        N_mtx[2, 35] = - \
            ((1 + s) * (1 + r) * (9 * r * r + 9 * s * s - 10) * (-1 + t)) / 64.
        N_mtx[2, 38] = (
            (-1 + s) * (-1 + r) * (9 * r * r + 9 * s * s - 10) * (1 + t)) / 64.
        N_mtx[2, 41] = -9. / 64. * \
            (-1 + r) * (3 * r - 1) * (1 + r) * (1 + t) * (-1 + s)
        N_mtx[2, 44] = 9. / 64. * \
            (-1 + r) * (3 * r + 1) * (1 + r) * (1 + t) * (-1 + s)
        N_mtx[2, 47] = - \
            ((-1 + s) * (1 + r) * (9 * r * r + 9 * s * s - 10) * (1 + t)) / 64.
        N_mtx[2, 50] = -9. / 64. * \
            (-1 + s) * (3 * s - 1) * (1 + s) * (1 + t) * (-1 + r)
        N_mtx[2, 53] = 9. / 64. * \
            (-1 + s) * (3 * s - 1) * (1 + s) * (1 + t) * (1 + r)
        N_mtx[2, 56] = 9. / 64. * \
            (-1 + s) * (3 * s + 1) * (1 + s) * (1 + t) * (-1 + r)
        N_mtx[2, 59] = -9. / 64. * \
            (-1 + s) * (3 * s + 1) * (1 + s) * (1 + t) * (1 + r)
        N_mtx[2, 62] = - \
            ((1 + s) * (-1 + r) * (9 * r * r + 9 * s * s - 10) * (1 + t)) / 64.
        N_mtx[2, 65] = 9. / 64. * \
            (-1 + r) * (3 * r - 1) * (1 + r) * (1 + t) * (1 + s)
        N_mtx[2, 68] = -9. / 64. * \
            (-1 + r) * (3 * r + 1) * (1 + r) * (1 + t) * (1 + s)
        N_mtx[2, 71] = (
            (1 + s) * (1 + r) * (9 * r * r + 9 * s * s - 10) * (1 + t)) / 64.
        return N_mtx

    def get_dNr_mtx(self, r_pnt):
        '''
        Return the derivatives of the shape functions used for the field approximation
        '''
        r = r_pnt[0]
        s = r_pnt[1]
        t = r_pnt[2]
        dNr_mtx = zeros((3, 24), dtype='float_')
        dNr_mtx[0, 0] = -((-1 + s) * (9 * r * r + 9 * s * s - 10) *
                          (-1 + t)) / 64. - 9. / 32. * (-1 + s) * (-1 + r) * r * (-1 + t)
        dNr_mtx[0, 1] = 9. / 64. * (3 * r - 1) * (1 + r) * (-1 + t) * (-1 + s) + 27. / 64. * (-1 + r) * (
            1 + r) * (-1 + t) * (-1 + s) + 9. / 64. * (-1 + r) * (3 * r - 1) * (-1 + t) * (-1 + s)
        dNr_mtx[0, 2] = -9. / 64. * (3 * r + 1) * (1 + r) * (-1 + t) * (-1 + s) - 27. / 64. * (-1 + r) * (
            1 + r) * (-1 + t) * (-1 + s) - 9. / 64. * (-1 + r) * (3 * r + 1) * (-1 + t) * (-1 + s)
        dNr_mtx[0, 3] = ((-1 + s) * (9 * r * r + 9 * s * s - 10) *
                         (-1 + t)) / 64. + 9. / 32. * (-1 + s) * (1 + r) * r * (-1 + t)
        dNr_mtx[0, 4] = 9. / 64. * (-1 + s) * (3 * s - 1) * (1 + s) * (-1 + t)
        dNr_mtx[0, 5] = -9. / 64. * (-1 + s) * (3 * s - 1) * (1 + s) * (-1 + t)
        dNr_mtx[0, 6] = -9. / 64. * (-1 + s) * (3 * s + 1) * (1 + s) * (-1 + t)
        dNr_mtx[0, 7] = 9. / 64. * (-1 + s) * (3 * s + 1) * (1 + s) * (-1 + t)
        dNr_mtx[0, 8] = ((1 + s) * (9 * r * r + 9 * s * s - 10) *
                         (-1 + t)) / 64. + 9. / 32. * (1 + s) * (-1 + r) * r * (-1 + t)
        dNr_mtx[0, 9] = -9. / 64. * (3 * r - 1) * (1 + r) * (-1 + t) * (1 + s) - 27. / 64. * (-1 + r) * (
            1 + r) * (-1 + t) * (1 + s) - 9. / 64. * (-1 + r) * (3 * r - 1) * (-1 + t) * (1 + s)
        dNr_mtx[0, 10] = 9. / 64. * (3 * r + 1) * (1 + r) * (-1 + t) * (1 + s) + 27. / 64. * (-1 + r) * (
            1 + r) * (-1 + t) * (1 + s) + 9. / 64. * (-1 + r) * (3 * r + 1) * (-1 + t) * (1 + s)
        dNr_mtx[0, 11] = -((1 + s) * (9 * r * r + 9 * s * s - 10) *
                           (-1 + t)) / 64. - 9. / 32. * (1 + s) * (1 + r) * r * (-1 + t)
        dNr_mtx[0, 12] = ((-1 + s) * (9 * r * r + 9 * s * s - 10) *
                          (1 + t)) / 64. + 9. / 32. * (-1 + s) * (-1 + r) * r * (1 + t)
        dNr_mtx[0, 13] = -9. / 64. * (3 * r - 1) * (1 + r) * (1 + t) * (-1 + s) - 27. / 64. * (-1 + r) * (
            1 + r) * (1 + t) * (-1 + s) - 9. / 64. * (-1 + r) * (3 * r - 1) * (1 + t) * (-1 + s)
        dNr_mtx[0, 14] = 9. / 64. * (3 * r + 1) * (1 + r) * (1 + t) * (-1 + s) + 27. / 64. * (-1 + r) * (
            1 + r) * (1 + t) * (-1 + s) + 9. / 64. * (-1 + r) * (3 * r + 1) * (1 + t) * (-1 + s)
        dNr_mtx[0, 15] = -((-1 + s) * (9 * r * r + 9 * s * s - 10)
                           * (1 + t)) / 64. - 9. / 32. * (-1 + s) * (1 + r) * r * (1 + t)
        dNr_mtx[0, 16] = -9. / 64. * (-1 + s) * (3 * s - 1) * (1 + s) * (1 + t)
        dNr_mtx[0, 17] = 9. / 64. * (-1 + s) * (3 * s - 1) * (1 + s) * (1 + t)
        dNr_mtx[0, 18] = 9. / 64. * (-1 + s) * (3 * s + 1) * (1 + s) * (1 + t)
        dNr_mtx[0, 19] = -9. / 64. * (-1 + s) * (3 * s + 1) * (1 + s) * (1 + t)
        dNr_mtx[0, 20] = -((1 + s) * (9 * r * r + 9 * s * s - 10) *
                           (1 + t)) / 64. - 9. / 32. * (1 + s) * (-1 + r) * r * (1 + t)
        dNr_mtx[0, 21] = 9. / 64. * (3 * r - 1) * (1 + r) * (1 + t) * (1 + s) + 27. / 64. * (-1 + r) * (
            1 + r) * (1 + t) * (1 + s) + 9. / 64. * (-1 + r) * (3 * r - 1) * (1 + t) * (1 + s)
        dNr_mtx[0, 22] = -9. / 64. * (3 * r + 1) * (1 + r) * (1 + t) * (1 + s) - 27. / 64. * (-1 + r) * (
            1 + r) * (1 + t) * (1 + s) - 9. / 64. * (-1 + r) * (3 * r + 1) * (1 + t) * (1 + s)
        dNr_mtx[0, 23] = ((1 + s) * (9 * r * r + 9 * s * s - 10) *
                          (1 + t)) / 64. + 9. / 32. * (1 + s) * (1 + r) * r * (1 + t)
        dNr_mtx[1, 0] = -((-1 + r) * (9 * r * r + 9 * s * s - 10) *
                          (-1 + t)) / 64. - 9. / 32. * (-1 + s) * (-1 + r) * s * (-1 + t)
        dNr_mtx[1, 1] = 9. / 64. * (-1 + r) * (3 * r - 1) * (1 + r) * (-1 + t)
        dNr_mtx[1, 2] = -9. / 64. * (-1 + r) * (3 * r + 1) * (1 + r) * (-1 + t)
        dNr_mtx[1, 3] = ((1 + r) * (9 * r * r + 9 * s * s - 10) *
                         (-1 + t)) / 64. + 9. / 32. * (-1 + s) * (1 + r) * s * (-1 + t)
        dNr_mtx[1, 4] = 9. / 64. * (3 * s - 1) * (1 + s) * (-1 + t) * (-1 + r) + 27. / 64. * (-1 + s) * (
            1 + s) * (-1 + t) * (-1 + r) + 9. / 64. * (-1 + s) * (3 * s - 1) * (-1 + t) * (-1 + r)
        dNr_mtx[1, 5] = -9. / 64. * (3 * s - 1) * (1 + s) * (-1 + t) * (1 + r) - 27. / 64. * (-1 + s) * (
            1 + s) * (-1 + t) * (1 + r) - 9. / 64. * (-1 + s) * (3 * s - 1) * (-1 + t) * (1 + r)
        dNr_mtx[1, 6] = -9. / 64. * (3 * s + 1) * (1 + s) * (-1 + t) * (-1 + r) - 27. / 64. * (-1 + s) * (
            1 + s) * (-1 + t) * (-1 + r) - 9. / 64. * (-1 + s) * (3 * s + 1) * (-1 + t) * (-1 + r)
        dNr_mtx[1, 7] = 9. / 64. * (3 * s + 1) * (1 + s) * (-1 + t) * (1 + r) + 27. / 64. * (-1 + s) * (
            1 + s) * (-1 + t) * (1 + r) + 9. / 64. * (-1 + s) * (3 * s + 1) * (-1 + t) * (1 + r)
        dNr_mtx[1, 8] = ((-1 + r) * (9 * r * r + 9 * s * s - 10) *
                         (-1 + t)) / 64. + 9. / 32. * (1 + s) * (-1 + r) * s * (-1 + t)
        dNr_mtx[1, 9] = -9. / 64. * (-1 + r) * (3 * r - 1) * (1 + r) * (-1 + t)
        dNr_mtx[1, 10] = 9. / 64. * (-1 + r) * (3 * r + 1) * (1 + r) * (-1 + t)
        dNr_mtx[1, 11] = -((1 + r) * (9 * r * r + 9 * s * s - 10) *
                           (-1 + t)) / 64. - 9. / 32. * (1 + s) * (1 + r) * s * (-1 + t)
        dNr_mtx[1, 12] = ((-1 + r) * (9 * r * r + 9 * s * s - 10) *
                          (1 + t)) / 64. + 9. / 32. * (-1 + s) * (-1 + r) * s * (1 + t)
        dNr_mtx[1, 13] = -9. / 64. * (-1 + r) * (3 * r - 1) * (1 + r) * (1 + t)
        dNr_mtx[1, 14] = 9. / 64. * (-1 + r) * (3 * r + 1) * (1 + r) * (1 + t)
        dNr_mtx[1, 15] = -((1 + r) * (9 * r * r + 9 * s * s - 10) *
                           (1 + t)) / 64. - 9. / 32. * (-1 + s) * (1 + r) * s * (1 + t)
        dNr_mtx[1, 16] = -9. / 64. * (3 * s - 1) * (1 + s) * (1 + t) * (-1 + r) - 27. / 64. * (-1 + s) * (
            1 + s) * (1 + t) * (-1 + r) - 9. / 64. * (-1 + s) * (3 * s - 1) * (1 + t) * (-1 + r)
        dNr_mtx[1, 17] = 9. / 64. * (3 * s - 1) * (1 + s) * (1 + t) * (1 + r) + 27. / 64. * (-1 + s) * (
            1 + s) * (1 + t) * (1 + r) + 9. / 64. * (-1 + s) * (3 * s - 1) * (1 + t) * (1 + r)
        dNr_mtx[1, 18] = 9. / 64. * (3 * s + 1) * (1 + s) * (1 + t) * (-1 + r) + 27. / 64. * (-1 + s) * (
            1 + s) * (1 + t) * (-1 + r) + 9. / 64. * (-1 + s) * (3 * s + 1) * (1 + t) * (-1 + r)
        dNr_mtx[1, 19] = -9. / 64. * (3 * s + 1) * (1 + s) * (1 + t) * (1 + r) - 27. / 64. * (-1 + s) * (
            1 + s) * (1 + t) * (1 + r) - 9. / 64. * (-1 + s) * (3 * s + 1) * (1 + t) * (1 + r)
        dNr_mtx[1, 20] = -((-1 + r) * (9 * r * r + 9 * s * s - 10)
                           * (1 + t)) / 64. - 9. / 32. * (1 + s) * (-1 + r) * s * (1 + t)
        dNr_mtx[1, 21] = 9. / 64. * (-1 + r) * (3 * r - 1) * (1 + r) * (1 + t)
        dNr_mtx[1, 22] = -9. / 64. * (-1 + r) * (3 * r + 1) * (1 + r) * (1 + t)
        dNr_mtx[1, 23] = ((1 + r) * (9 * r * r + 9 * s * s - 10) *
                          (1 + t)) / 64. + 9. / 32. * (1 + s) * (1 + r) * s * (1 + t)
        dNr_mtx[2, 0] = - \
            ((-1 + s) * (-1 + r) * (9 * r * r + 9 * s * s - 10)) / 64.
        dNr_mtx[2, 1] = 9. / 64. * (-1 + r) * (3 * r - 1) * (1 + r) * (-1 + s)
        dNr_mtx[2, 2] = -9. / 64. * (-1 + r) * (3 * r + 1) * (1 + r) * (-1 + s)
        dNr_mtx[2, 3] = (
            (-1 + s) * (1 + r) * (9 * r * r + 9 * s * s - 10)) / 64.
        dNr_mtx[2, 4] = 9. / 64. * (-1 + s) * (3 * s - 1) * (1 + s) * (-1 + r)
        dNr_mtx[2, 5] = -9. / 64. * (-1 + s) * (3 * s - 1) * (1 + s) * (1 + r)
        dNr_mtx[2, 6] = -9. / 64. * (-1 + s) * (3 * s + 1) * (1 + s) * (-1 + r)
        dNr_mtx[2, 7] = 9. / 64. * (-1 + s) * (3 * s + 1) * (1 + s) * (1 + r)
        dNr_mtx[2, 8] = (
            (1 + s) * (-1 + r) * (9 * r * r + 9 * s * s - 10)) / 64.
        dNr_mtx[2, 9] = -9. / 64. * (-1 + r) * (3 * r - 1) * (1 + r) * (1 + s)
        dNr_mtx[2, 10] = 9. / 64. * (-1 + r) * (3 * r + 1) * (1 + r) * (1 + s)
        dNr_mtx[2, 11] = - \
            ((1 + s) * (1 + r) * (9 * r * r + 9 * s * s - 10)) / 64.
        dNr_mtx[2, 12] = (
            (-1 + s) * (-1 + r) * (9 * r * r + 9 * s * s - 10)) / 64.
        dNr_mtx[2, 13] = -9. / 64. * \
            (-1 + r) * (3 * r - 1) * (1 + r) * (-1 + s)
        dNr_mtx[2, 14] = 9. / 64. * (-1 + r) * (3 * r + 1) * (1 + r) * (-1 + s)
        dNr_mtx[2, 15] = - \
            ((-1 + s) * (1 + r) * (9 * r * r + 9 * s * s - 10)) / 64.
        dNr_mtx[2, 16] = -9. / 64. * \
            (-1 + s) * (3 * s - 1) * (1 + s) * (-1 + r)
        dNr_mtx[2, 17] = 9. / 64. * (-1 + s) * (3 * s - 1) * (1 + s) * (1 + r)
        dNr_mtx[2, 18] = 9. / 64. * (-1 + s) * (3 * s + 1) * (1 + s) * (-1 + r)
        dNr_mtx[2, 19] = -9. / 64. * (-1 + s) * (3 * s + 1) * (1 + s) * (1 + r)
        dNr_mtx[2, 20] = - \
            ((1 + s) * (-1 + r) * (9 * r * r + 9 * s * s - 10)) / 64.
        dNr_mtx[2, 21] = 9. / 64. * (-1 + r) * (3 * r - 1) * (1 + r) * (1 + s)
        dNr_mtx[2, 22] = -9. / 64. * (-1 + r) * (3 * r + 1) * (1 + r) * (1 + s)
        dNr_mtx[2, 23] = (
            (1 + s) * (1 + r) * (9 * r * r + 9 * s * s - 10)) / 64.
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

    fets_eval = FETS3D8H24U(mats_eval=MATS3DElastic())

    from ibvpy.mesh.fe_grid import FEGrid

    # Discretization
    domain = FEGrid(coord_max=(3., 3., 3.),
                    shape=(3, 3, 3),
                    fets_eval=fets_eval)

    # Put the tseval (time-stepper) into the spatial context of the
    # discretization and specify the response tracers to evaluate there.
    #

    right_dof = 2
    ts = TS(
        sdomain=domain,
        # conversion to list (square brackets) is only necessary for slicing of
        # single dofs, e.g "get_left_dofs()[0,1]" which elsewise retuns an
        # integer only
        bcond_list=[BCDofGroup(var='u', value=0., dims=[0],
                               get_dof_method=domain.get_left_dofs),
                    BCDofGroup(var='u', value=0., dims=[1, 2],
                               get_dof_method=domain.get_bottom_left_dofs),
                    BCDofGroup(var='u', value=0.002, dims=[1],
                               get_dof_method=domain.get_right_dofs)],
        rtrace_list=[
            #                        RTDofGraph(name = 'Fi,right over u_right (iteration)' ,
            #                                  var_y = 'F_int', idx_y = right_dof,
            #                                  var_x = 'U_k', idx_x = right_dof,
            #                                  record_on = 'update'),
            #                        RTraceDomainListField(name = 'Deformation' ,
            #                                       var = 'eps', idx = 0,
            #                                       record_on = 'update'),
            RTraceDomainListField(name='Displacement',
                                       var='u', idx=0),
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
    tloop = TLoop(tstepper=ts,
                  tline=TLine(min=0.0, step=0.5, max=.5))

    import cProfile
    cProfile.run('tloop.eval()', 'tloop_prof')

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
    app = IBVPyApp(ibv_resource=tloop)
    app.main()
