from traits.api import \
    Instance, Int, Property, Array, cached_property

from numpy import \
     zeros, dot, hstack, identity

from scipy.linalg import \
     inv

from .fets3D import FETS3D

#-----------------------------------------------------------------------------------
# FETS3D8H20U - 20 nodes Subparametric volume element (3D, quadratic, serendipity family)
#-----------------------------------------------------------------------------------

class FETS3D8H20U(FETS3D):
    '''
    quadratic serendipity volume element.
    '''
    debug_on = False

    # Dimensional mapping
    dim_slice = slice(0, 3)

    # number of nodal degrees of freedom
    # number of degrees of freedom of each element
    n_nodal_dofs = Int(3)
    n_e_dofs = Int(20 * 3)

    # Integration parameters
    #
    ngp_r = 2
    ngp_s = 2
    ngp_t = 2

    geo_r = \
             Array(value=[[-1., -1., -1.],
                            [  1., -1., -1.],
                            [-1., 1., -1.],
                            [  1., 1., -1.],
                            [-1., -1., 1.],
                            [  1., -1., 1.],
                            [-1., 1., 1.],
                            [  1., 1., 1.]])

    dof_r = \
            Array(value=[  [-1, -1, -1],
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
                           [-1, 0, 1]])

    # Used for Visualization
    #
    vtk_cell_types = 'QuadraticHexahedron'
    vtk_r = Array(value=[[-1., -1., -1.],
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

    #---------------------------------------------------------------------
    # Method required to represent the element geometry
    #---------------------------------------------------------------------
    def get_N_geo_mtx(self, r_pnt):
        '''
        Return the value of shape functions (derived in femple) for the
        specified local coordinate r_pnt
        '''
        N_geo_mtx = zeros((1, 8), dtype='float_')
        N_geo_mtx[0, 0] = -((-1 + r_pnt[2]) * (-1 + r_pnt[1]) * \
                             (-1 + r_pnt[0])) / 8.0
        N_geo_mtx[0, 1] = ((-1 + r_pnt[2]) * (-1 + r_pnt[1]) * \
                             (1 + r_pnt[0])) / 8.0
        N_geo_mtx[0, 2] = ((-1 + r_pnt[2]) * (1 + r_pnt[1]) * \
                             (-1 + r_pnt[0])) / 8.0
        N_geo_mtx[0, 3] = -((-1 + r_pnt[2]) * (1 + r_pnt[1]) * \
                             (1 + r_pnt[0])) / 8.0
        N_geo_mtx[0, 4] = ((1 + r_pnt[2]) * (-1 + r_pnt[1]) * \
                             (-1 + r_pnt[0])) / 8.0
        N_geo_mtx[0, 5] = -((1 + r_pnt[2]) * (-1 + r_pnt[1]) * \
                             (1 + r_pnt[0])) / 8.0
        N_geo_mtx[0, 6] = -((1 + r_pnt[2]) * (1 + r_pnt[1]) * \
                             (-1 + r_pnt[0])) / 8.0
        N_geo_mtx[0, 7] = ((1 + r_pnt[2]) * (1 + r_pnt[1]) * \
                             (1 + r_pnt[0])) / 8.0
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
        N_mtx = zeros((3, 60), dtype='float_')
        N_mtx[0, 0] = ((-1 + s) * (-1 + r) * (-1 + t) * (t + r + s + 2)) / 8.
        N_mtx[0, 3] = -((-1 + s) * (1 + r) * (-1 + t) * (t - r + s + 2)) / 8.
        N_mtx[0, 6] = ((1 + s) * (1 + r) * (-1 + t) * (t - r - s + 2)) / 8.
        N_mtx[0, 9] = -((1 + s) * (-1 + r) * (-1 + t) * (t + r - s + 2)) / 8.
        N_mtx[0, 12] = -((-1 + r) * (1 + r) * (-1 + s) * (-1 + t)) / 4.
        N_mtx[0, 15] = ((-1 + s) * (1 + s) * (1 + r) * (-1 + t)) / 4.
        N_mtx[0, 18] = ((-1 + r) * (1 + r) * (1 + s) * (-1 + t)) / 4.
        N_mtx[0, 21] = -((-1 + s) * (1 + s) * (-1 + r) * (-1 + t)) / 4.
        N_mtx[0, 24] = -((-1 + t) * (1 + t) * (-1 + s) * (-1 + r)) / 4.
        N_mtx[0, 27] = ((-1 + t) * (1 + t) * (-1 + s) * (1 + r)) / 4.
        N_mtx[0, 30] = -((-1 + t) * (1 + t) * (1 + s) * (1 + r)) / 4.
        N_mtx[0, 33] = ((-1 + t) * (1 + t) * (1 + s) * (-1 + r)) / 4.
        N_mtx[0, 36] = ((-1 + s) * (-1 + r) * (1 + t) * (t - r - s - 2)) / 8.
        N_mtx[0, 39] = -((-1 + s) * (1 + r) * (1 + t) * (t + r - 2 - s)) / 8.
        N_mtx[0, 42] = ((1 + s) * (1 + r) * (1 + t) * (t + r + s - 2)) / 8.
        N_mtx[0, 45] = -((1 + s) * (-1 + r) * (1 + t) * (t - r - 2 + s)) / 8.
        N_mtx[0, 48] = ((-1 + r) * (1 + r) * (-1 + s) * (1 + t)) / 4.
        N_mtx[0, 51] = -((-1 + s) * (1 + s) * (1 + r) * (1 + t)) / 4.
        N_mtx[0, 54] = -((-1 + r) * (1 + r) * (1 + s) * (1 + t)) / 4.
        N_mtx[0, 57] = ((-1 + s) * (1 + s) * (-1 + r) * (1 + t)) / 4.
        N_mtx[1, 1] = ((-1 + s) * (-1 + r) * (-1 + t) * (t + r + s + 2)) / 8.
        N_mtx[1, 4] = -((-1 + s) * (1 + r) * (-1 + t) * (t - r + s + 2)) / 8.
        N_mtx[1, 7] = ((1 + s) * (1 + r) * (-1 + t) * (t - r - s + 2)) / 8.
        N_mtx[1, 10] = -((1 + s) * (-1 + r) * (-1 + t) * (t + r - s + 2)) / 8.
        N_mtx[1, 13] = -((-1 + r) * (1 + r) * (-1 + s) * (-1 + t)) / 4.
        N_mtx[1, 16] = ((-1 + s) * (1 + s) * (1 + r) * (-1 + t)) / 4.
        N_mtx[1, 19] = ((-1 + r) * (1 + r) * (1 + s) * (-1 + t)) / 4.
        N_mtx[1, 22] = -((-1 + s) * (1 + s) * (-1 + r) * (-1 + t)) / 4.
        N_mtx[1, 25] = -((-1 + t) * (1 + t) * (-1 + s) * (-1 + r)) / 4.
        N_mtx[1, 28] = ((-1 + t) * (1 + t) * (-1 + s) * (1 + r)) / 4.
        N_mtx[1, 31] = -((-1 + t) * (1 + t) * (1 + s) * (1 + r)) / 4.
        N_mtx[1, 34] = ((-1 + t) * (1 + t) * (1 + s) * (-1 + r)) / 4.
        N_mtx[1, 37] = ((-1 + s) * (-1 + r) * (1 + t) * (t - r - s - 2)) / 8.
        N_mtx[1, 40] = -((-1 + s) * (1 + r) * (1 + t) * (t + r - 2 - s)) / 8.
        N_mtx[1, 43] = ((1 + s) * (1 + r) * (1 + t) * (t + r + s - 2)) / 8.
        N_mtx[1, 46] = -((1 + s) * (-1 + r) * (1 + t) * (t - r - 2 + s)) / 8.
        N_mtx[1, 49] = ((-1 + r) * (1 + r) * (-1 + s) * (1 + t)) / 4.
        N_mtx[1, 52] = -((-1 + s) * (1 + s) * (1 + r) * (1 + t)) / 4.
        N_mtx[1, 55] = -((-1 + r) * (1 + r) * (1 + s) * (1 + t)) / 4.
        N_mtx[1, 58] = ((-1 + s) * (1 + s) * (-1 + r) * (1 + t)) / 4.
        N_mtx[2, 2] = ((-1 + s) * (-1 + r) * (-1 + t) * (t + r + s + 2)) / 8.
        N_mtx[2, 5] = -((-1 + s) * (1 + r) * (-1 + t) * (t - r + s + 2)) / 8.
        N_mtx[2, 8] = ((1 + s) * (1 + r) * (-1 + t) * (t - r - s + 2)) / 8.
        N_mtx[2, 11] = -((1 + s) * (-1 + r) * (-1 + t) * (t + r - s + 2)) / 8.
        N_mtx[2, 14] = -((-1 + r) * (1 + r) * (-1 + s) * (-1 + t)) / 4.
        N_mtx[2, 17] = ((-1 + s) * (1 + s) * (1 + r) * (-1 + t)) / 4.
        N_mtx[2, 20] = ((-1 + r) * (1 + r) * (1 + s) * (-1 + t)) / 4.
        N_mtx[2, 23] = -((-1 + s) * (1 + s) * (-1 + r) * (-1 + t)) / 4.
        N_mtx[2, 26] = -((-1 + t) * (1 + t) * (-1 + s) * (-1 + r)) / 4.
        N_mtx[2, 29] = ((-1 + t) * (1 + t) * (-1 + s) * (1 + r)) / 4.
        N_mtx[2, 32] = -((-1 + t) * (1 + t) * (1 + s) * (1 + r)) / 4.
        N_mtx[2, 35] = ((-1 + t) * (1 + t) * (1 + s) * (-1 + r)) / 4.
        N_mtx[2, 38] = ((-1 + s) * (-1 + r) * (1 + t) * (t - r - s - 2)) / 8.
        N_mtx[2, 41] = -((-1 + s) * (1 + r) * (1 + t) * (t + r - 2 - s)) / 8.
        N_mtx[2, 44] = ((1 + s) * (1 + r) * (1 + t) * (t + r + s - 2)) / 8.
        N_mtx[2, 47] = -((1 + s) * (-1 + r) * (1 + t) * (t - r - 2 + s)) / 8.
        N_mtx[2, 50] = ((-1 + r) * (1 + r) * (-1 + s) * (1 + t)) / 4.
        N_mtx[2, 53] = -((-1 + s) * (1 + s) * (1 + r) * (1 + t)) / 4.
        N_mtx[2, 56] = -((-1 + r) * (1 + r) * (1 + s) * (1 + t)) / 4.
        N_mtx[2, 59] = ((-1 + s) * (1 + s) * (-1 + r) * (1 + t)) / 4.
        return N_mtx

    def get_dNr_mtx(self, r_pnt):
        '''
        Return the derivatives of the shape functions used for the field approximation
        '''
        r = r_pnt[0]
        s = r_pnt[1]
        t = r_pnt[2]
        dNr = zeros((3, 20), dtype='float_')
        dNr[0, 0] = ((-1 + s) * (-1 + t) * (r + t + 2 + s)) / 8. + ((-1 + s) * (-1 + t) * (-1 + r)) / 8.
        dNr[0, 1] = ((-1 + s) * (-1 + t) * (r - t - s - 2)) / 8. + ((-1 + s) * (-1 + t) * (1 + r)) / 8.
        dNr[0, 2] = -((1 + s) * (-1 + t) * (r - t - 2 + s)) / 8. - ((1 + s) * (-1 + t) * (1 + r)) / 8.
        dNr[0, 3] = -((1 + s) * (-1 + t) * (r + t - s + 2)) / 8. - ((1 + s) * (-1 + t) * (-1 + r)) / 8.
        dNr[0, 4] = -((-1 + s) * (-1 + t) * (1 + r)) / 4. - ((-1 + s) * (-1 + t) * (-1 + r)) / 4.
        dNr[0, 5] = ((-1 + s) * (1 + s) * (-1 + t)) / 4.
        dNr[0, 6] = ((1 + s) * (-1 + t) * (1 + r)) / 4. + ((1 + s) * (-1 + t) * (-1 + r)) / 4.
        dNr[0, 7] = -((-1 + s) * (1 + s) * (-1 + t)) / 4.
        dNr[0, 8] = -((-1 + t) * (1 + t) * (-1 + s)) / 4.
        dNr[0, 9] = ((-1 + t) * (1 + t) * (-1 + s)) / 4.
        dNr[0, 10] = -((-1 + t) * (1 + t) * (1 + s)) / 4.
        dNr[0, 11] = ((-1 + t) * (1 + t) * (1 + s)) / 4.
        dNr[0, 12] = -((-1 + s) * (1 + t) * (r - t + s + 2)) / 8. - ((1 + t) * (-1 + s) * (-1 + r)) / 8.
        dNr[0, 13] = -((-1 + s) * (1 + t) * (r + t - 2 - s)) / 8. - ((1 + t) * (-1 + s) * (1 + r)) / 8.
        dNr[0, 14] = ((1 + s) * (1 + t) * (r + t + s - 2)) / 8. + ((1 + t) * (1 + s) * (1 + r)) / 8.
        dNr[0, 15] = ((1 + s) * (1 + t) * (r - t + 2 - s)) / 8. + ((1 + t) * (1 + s) * (-1 + r)) / 8.
        dNr[0, 16] = ((1 + t) * (-1 + s) * (1 + r)) / 4. + ((1 + t) * (-1 + s) * (-1 + r)) / 4.
        dNr[0, 17] = -((-1 + s) * (1 + s) * (1 + t)) / 4.
        dNr[0, 18] = -((1 + t) * (1 + s) * (1 + r)) / 4. - ((1 + t) * (1 + s) * (-1 + r)) / 4.
        dNr[0, 19] = ((-1 + s) * (1 + s) * (1 + t)) / 4.
        dNr[1, 0] = ((-1 + t) * (-1 + r) * (r + t + 2 + s)) / 8. + ((-1 + s) * (-1 + t) * (-1 + r)) / 8.
        dNr[1, 1] = ((-1 + t) * (1 + r) * (r - t - s - 2)) / 8. - ((-1 + s) * (-1 + t) * (1 + r)) / 8.
        dNr[1, 2] = -((-1 + t) * (1 + r) * (r - t - 2 + s)) / 8. - ((1 + s) * (-1 + t) * (1 + r)) / 8.
        dNr[1, 3] = -((-1 + t) * (-1 + r) * (r + t - s + 2)) / 8. + ((1 + s) * (-1 + t) * (-1 + r)) / 8.
        dNr[1, 4] = -((-1 + r) * (1 + r) * (-1 + t)) / 4.
        dNr[1, 5] = ((1 + s) * (-1 + t) * (1 + r)) / 4. + ((-1 + s) * (-1 + t) * (1 + r)) / 4.
        dNr[1, 6] = ((-1 + r) * (1 + r) * (-1 + t)) / 4.
        dNr[1, 7] = -((1 + s) * (-1 + t) * (-1 + r)) / 4. - ((-1 + s) * (-1 + t) * (-1 + r)) / 4.
        dNr[1, 8] = -((-1 + t) * (1 + t) * (-1 + r)) / 4.
        dNr[1, 9] = ((-1 + t) * (1 + t) * (1 + r)) / 4.
        dNr[1, 10] = -((-1 + t) * (1 + t) * (1 + r)) / 4.
        dNr[1, 11] = ((-1 + t) * (1 + t) * (-1 + r)) / 4.
        dNr[1, 12] = -((1 + t) * (-1 + r) * (r - t + s + 2)) / 8. - ((1 + t) * (-1 + s) * (-1 + r)) / 8.
        dNr[1, 13] = -((1 + t) * (1 + r) * (r + t - 2 - s)) / 8. + ((1 + t) * (-1 + s) * (1 + r)) / 8.
        dNr[1, 14] = ((1 + t) * (1 + r) * (r + t + s - 2)) / 8. + ((1 + t) * (1 + s) * (1 + r)) / 8.
        dNr[1, 15] = ((1 + t) * (-1 + r) * (r - t + 2 - s)) / 8. - ((1 + t) * (1 + s) * (-1 + r)) / 8.
        dNr[1, 16] = ((-1 + r) * (1 + r) * (1 + t)) / 4.
        dNr[1, 17] = -((1 + t) * (1 + s) * (1 + r)) / 4. - ((1 + t) * (-1 + s) * (1 + r)) / 4.
        dNr[1, 18] = -((-1 + r) * (1 + r) * (1 + t)) / 4.
        dNr[1, 19] = ((1 + t) * (1 + s) * (-1 + r)) / 4. + ((1 + t) * (-1 + s) * (-1 + r)) / 4.
        dNr[2, 0] = ((-1 + s) * (-1 + r) * (r + t + 2 + s)) / 8. + ((-1 + s) * (-1 + t) * (-1 + r)) / 8.
        dNr[2, 1] = ((-1 + s) * (1 + r) * (r - t - s - 2)) / 8. - ((-1 + s) * (-1 + t) * (1 + r)) / 8.
        dNr[2, 2] = -((1 + s) * (1 + r) * (r - t - 2 + s)) / 8. + ((1 + s) * (-1 + t) * (1 + r)) / 8.
        dNr[2, 3] = -((1 + s) * (-1 + r) * (r + t - s + 2)) / 8. - ((1 + s) * (-1 + t) * (-1 + r)) / 8.
        dNr[2, 4] = -((-1 + r) * (1 + r) * (-1 + s)) / 4.
        dNr[2, 5] = ((-1 + s) * (1 + s) * (1 + r)) / 4.
        dNr[2, 6] = ((-1 + r) * (1 + r) * (1 + s)) / 4.
        dNr[2, 7] = -((-1 + s) * (1 + s) * (-1 + r)) / 4.
        dNr[2, 8] = -((1 + t) * (-1 + s) * (-1 + r)) / 4. - ((-1 + s) * (-1 + t) * (-1 + r)) / 4.
        dNr[2, 9] = ((1 + t) * (-1 + s) * (1 + r)) / 4. + ((-1 + s) * (-1 + t) * (1 + r)) / 4.
        dNr[2, 10] = -((1 + t) * (1 + s) * (1 + r)) / 4. - ((1 + s) * (-1 + t) * (1 + r)) / 4.
        dNr[2, 11] = ((1 + t) * (1 + s) * (-1 + r)) / 4. + ((1 + s) * (-1 + t) * (-1 + r)) / 4.
        dNr[2, 12] = -((-1 + s) * (-1 + r) * (r - t + s + 2)) / 8. + ((1 + t) * (-1 + s) * (-1 + r)) / 8.
        dNr[2, 13] = -((-1 + s) * (1 + r) * (r + t - 2 - s)) / 8. - ((1 + t) * (-1 + s) * (1 + r)) / 8.
        dNr[2, 14] = ((1 + s) * (1 + r) * (r + t + s - 2)) / 8. + ((1 + t) * (1 + s) * (1 + r)) / 8.
        dNr[2, 15] = ((1 + s) * (-1 + r) * (r - t + 2 - s)) / 8. - ((1 + t) * (1 + s) * (-1 + r)) / 8.
        dNr[2, 16] = ((-1 + r) * (1 + r) * (-1 + s)) / 4.
        dNr[2, 17] = -((-1 + s) * (1 + s) * (1 + r)) / 4.
        dNr[2, 18] = -((-1 + r) * (1 + r) * (1 + s)) / 4.
        dNr[2, 19] = ((-1 + s) * (1 + s) * (-1 + r)) / 4.
        return dNr

#----------------------- example --------------------

if __name__ == '__main__':
    from ibvpy.api import \
        TStepper as TS, RTDofGraph, RTraceDomainListField, TLoop, \
        TLine, BCDofGroup, IBVPSolve as IS, DOTSEval

    # from lib.mats.mats2D.mats_cmdm2D.mats_mdm2d import MACMDM
    from ibvpy.mats.mats2D.mats2D_sdamage.mats2D_sdamage import MATS2DScalarDamage
    from ibvpy.mats.mats2D.mats2D_sdamage.strain_norm2d import *
    from ibvpy.mats.mats3D.mats3D_elastic.mats3D_elastic import MATS3DElastic

#    fets_eval = FETS2D9Q(mats_eval = MA2DSca    larDamage(strain_norm = Euclidean()))
    fets_eval = FETS3D8H20U(mats_eval=MATS3DElastic(),)

    from ibvpy.mesh.fe_grid import FEGrid

    # Discretization
    domain = FEGrid(coord_max=(3., 3., 3.),
                           shape=(3, 3, 3),
                           fets_eval=fets_eval)

    # Put the tseval (time-stepper) into the spatial context of the
    # discretization and specify the response tracers to evaluate there.
    right_dof = 2
    ts = TS(
            sdomain=domain,
             # conversion to list (square brackets) is only necessary for slicing of
             # single dofs, e.g "get_left_dofs()[0,1]" which elsewise retuns an integer only
             bcond_list=[ BCDofGroup(var='u', value=0., dims=[0],
                                        get_dof_method=domain.get_left_dofs),
                        BCDofGroup(var='u', value=0., dims=[1, 2],
                                  get_dof_method=domain.get_bottom_left_dofs),
                        BCDofGroup(var='u', value=0.002, dims=[0],
                                  get_dof_method=domain.get_right_dofs) ],
             rtrace_list=[
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
                          RTraceDomainListField(name='Displacement' ,
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
    tloop = TLoop(tstepper=ts,
         tline=TLine(min=0.0, step=0.5, max=1.0))

    tloop.eval()

    # Put the whole stuff into the simulation-framework to map the
    # individual pieces of definition into the user interface.
    #
    from ibvpy.plugins.ibvpy_app import IBVPyApp
    app = IBVPyApp(ibv_resource=tloop)
    app.main()
