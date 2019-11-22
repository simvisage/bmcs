
from math  import \
    pow, fabs
import time

from ibvpy.fets.fets_eval import FETSEval
from ibvpy.mats.mats_eval import MATSEval
from numpy import \
    array, zeros, int_, float_, ix_, dot, linspace, hstack, vstack, arange, \
    identity
from scipy.linalg import \
    inv, det
from traits.api import \
    Array, Bool, Callable, Enum, Float, HasTraits, Interface, \
    Instance, Int, Trait, Str, Enum, Callable, List, TraitDict, Any, \
    on_trait_change, Tuple, WeakRef, Delegate, Property, cached_property
from traitsui.api import \
    Item, View, HGroup, ListEditor, VGroup, Group
from traitsui.menu import \
    NoButtons, OKButton, CancelButton, Action, CloseAction, Menu, \
    MenuBar, Separator
from tvtk.api import tvtk


#-----------------------------------------------------------------------------
# FEQ16sub - 16 nodes subparametric quadrilateral (2D, cubic, Lagrange family)
#-----------------------------------------------------------------------------
#-------------------------------------------------------------------------
# Element Information:
#-------------------------------------------------------------------------
#
# The order of the field approximation is higher then the order of the geometry
# approximation (subparametric element).
# The implemented shape functions are derived (in femple) based
# on the following ordering of the nodes of the parent element.
#    _node_coord_map_dof = Array( Float, (16,2),
#                            [[ -1,  -1   ],
#                             [  1,  -1   ],
#                             [  1,   1   ],
#                             [ -1,   1   ],
#                             [ -1/3,-1   ],
#                             [  1/3,-1   ],
#                             [  1,  -1/3 ],
#                             [  1,   1/3 ],
#                             [  1/3, 1   ],
#                             [ -1/3, 1   ],
#                             [ -1,   1/3 ],
#                             [ -1,  -1/3 ],
#                             [ -1/3,-1/3 ],
#                             [  1/3,-1/3 ],
#                             [ -1/3, 1/3 ],
#                             [  1/3, 1/3 ]])
#
#-------------------------------------------------------------------------
class FETS2D4Q16U(FETSEval):
    debug_on = True

    mats_eval = Instance(MATSEval)

    # Dimensional mapping
    dim_slice = slice(0, 2)

    n_e_dofs = Int(2 * 16)
    t = Float(1.0, label='thickness')

    # Integration parameters
    #
    ngp_r = 4
    ngp_s = 4

    dof_r = Array(value=[[-1., -1.],
                         [1., -1.],
                         [1., 1.],
                         [-1., 1.],
                         [-1. / 3, -1.],
                         [1. / 3, -1.],
                         [1., -1. / 3],
                         [1., 1. / 3],
                         [1. / 3, 1.],
                         [-1. / 3, 1.],
                         [-1., 1. / 3],
                         [-1., -1. / 3],
                         [-1. / 3, -1. / 3],
                         [1. / 3, -1. / 3],
                         [-1. / 3, 1. / 3],
                         [1. / 3, 1. / 3]])

    geo_r = Array(value=[[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
    #
    vtk_r = Array(value=[[-1., -1.],
                         [0., -1.],
                         [1., -1.],
                         [-1., 0.],
                         [0., 0.],
                         [1., 0.],
                         [-1., 1.],
                         [0., 1.],
                         [1., 1.]])
    vtk_cells = [[0, 2, 8, 6, 1, 5, 7, 3, 4]]
    vtk_cell_types = ['QuadraticQuad']  # Biquadratic Quad

    n_nodal_dofs = Int(2)

    # Ordering of the nodes of the parent element used for the geometry
    # approximation
    _node_coord_map_geo = Array(Float, (4, 2),
                                [[-1., -1.],
                                 [1., -1.],
                                 [1., 1.],
                                 [-1., 1.]])

    #---------------------------------------------------------------------
    # Method required to represent the element geometry
    #---------------------------------------------------------------------
    def get_N_geo_mtx(self, r_pnt):
        '''
        Return the value of shape functions for the specified local coordinate r
        '''
        cx = self._node_coord_map_geo
        N_geo_mtx = array(
            [[1 / 4. * (1 + r_pnt[0] * cx[i, 0]) * (1 + r_pnt[1] * cx[i, 1]) for i in range(0, 4)]])
        return N_geo_mtx

    def get_dNr_geo_mtx(self, r_pnt):
        '''
        Return the matrix of shape function derivatives.
        Used for the conrcution of the Jacobi matrix.

        @TODO - the B matrix is used
        just for uniaxial bar here with a trivial differential
        operator.
        '''
        cx = self._node_coord_map_geo
        dNr_geo_mtx = array([[1 / 4. * cx[i, 0] * (1 + r_pnt[1] * cx[i, 1]) for i in range(0, 4)],
                             [1 / 4. * cx[i, 1] * (1 + r_pnt[0] * cx[i, 0]) for i in range(0, 4)]])
        return dNr_geo_mtx

    #---------------------------------------------------------------------
    # Method delivering the shape functions for the field variables and their derivatives
    #---------------------------------------------------------------------
    def get_N_mtx(self, r_pnt):
        '''
        Returns the matrix of the shape functions used for the field approximation
        containing zero entries. The number of rows corresponds to the number of nodal
        dofs. The matrix is evaluated for the specified local coordinate r.
        '''
        r = r_pnt[0]
        s = r_pnt[1]
        N_mtx = zeros((2, 32), dtype='float_')
        N_mtx[0, 0] = ((-1 + s) * (3 * s - 1) * (3 * s + 1)
                       * (-1 + r) * (3 * r - 1) * (3 * r + 1)) / 0.256e3
        N_mtx[0, 2] = -((-1 + s) * (3 * s - 1) * (3 * s + 1)
                        * (3 * r - 1) * (3 * r + 1) * (1 + r)) / 0.256e3
        N_mtx[0, 4] = ((3 * s - 1) * (3 * s + 1) * (1 + s)
                       * (3 * r - 1) * (3 * r + 1) * (1 + r)) / 0.256e3
        N_mtx[0, 6] = -((3 * s - 1) * (3 * s + 1) * (1 + s)
                        * (-1 + r) * (3 * r - 1) * (3 * r + 1)) / 0.256e3
        N_mtx[0, 8] = -0.9e1 / 0.256e3 * \
            (-1 + s) * (3 * s - 1) * (3 * s + 1) * \
            (-1 + r) * (3 * r - 1) * (1 + r)
        N_mtx[0, 10] = 0.9e1 / 0.256e3 * \
            (-1 + s) * (3 * s - 1) * (3 * s + 1) * \
            (-1 + r) * (3 * r + 1) * (1 + r)
        N_mtx[0, 12] = 0.9e1 / 0.256e3 * \
            (-1 + s) * (3 * s - 1) * (1 + s) * \
            (3 * r - 1) * (3 * r + 1) * (1 + r)
        N_mtx[0, 14] = -0.9e1 / 0.256e3 * \
            (-1 + s) * (3 * s + 1) * (1 + s) * \
            (3 * r - 1) * (3 * r + 1) * (1 + r)
        N_mtx[0, 16] = -0.9e1 / 0.256e3 * \
            (3 * s - 1) * (3 * s + 1) * (1 + s) * \
            (-1 + r) * (3 * r + 1) * (1 + r)
        N_mtx[0, 18] = 0.9e1 / 0.256e3 * \
            (3 * s - 1) * (3 * s + 1) * (1 + s) * \
            (-1 + r) * (3 * r - 1) * (1 + r)
        N_mtx[0, 20] = 0.9e1 / 0.256e3 * \
            (-1 + s) * (3 * s + 1) * (1 + s) * \
            (-1 + r) * (3 * r - 1) * (3 * r + 1)
        N_mtx[0, 22] = -0.9e1 / 0.256e3 * \
            (-1 + s) * (3 * s - 1) * (1 + s) * \
            (-1 + r) * (3 * r - 1) * (3 * r + 1)
        N_mtx[0, 24] = 0.81e2 / 0.256e3 * \
            (-1 + s) * (3 * s - 1) * (1 + s) * (-1 + r) * (3 * r - 1) * (1 + r)
        N_mtx[0, 26] = -0.81e2 / 0.256e3 * \
            (-1 + s) * (3 * s - 1) * (1 + s) * (-1 + r) * (3 * r + 1) * (1 + r)
        N_mtx[0, 28] = -0.81e2 / 0.256e3 * \
            (-1 + s) * (3 * s + 1) * (1 + s) * (-1 + r) * (3 * r - 1) * (1 + r)
        N_mtx[0, 30] = 0.81e2 / 0.256e3 * \
            (-1 + s) * (3 * s + 1) * (1 + s) * (-1 + r) * (3 * r + 1) * (1 + r)
        N_mtx[1, 1] = ((-1 + s) * (3 * s - 1) * (3 * s + 1)
                       * (-1 + r) * (3 * r - 1) * (3 * r + 1)) / 0.256e3
        N_mtx[1, 3] = -((-1 + s) * (3 * s - 1) * (3 * s + 1)
                        * (3 * r - 1) * (3 * r + 1) * (1 + r)) / 0.256e3
        N_mtx[1, 5] = ((3 * s - 1) * (3 * s + 1) * (1 + s)
                       * (3 * r - 1) * (3 * r + 1) * (1 + r)) / 0.256e3
        N_mtx[1, 7] = -((3 * s - 1) * (3 * s + 1) * (1 + s)
                        * (-1 + r) * (3 * r - 1) * (3 * r + 1)) / 0.256e3
        N_mtx[1, 9] = -0.9e1 / 0.256e3 * \
            (-1 + s) * (3 * s - 1) * (3 * s + 1) * \
            (-1 + r) * (3 * r - 1) * (1 + r)
        N_mtx[1, 11] = 0.9e1 / 0.256e3 * \
            (-1 + s) * (3 * s - 1) * (3 * s + 1) * \
            (-1 + r) * (3 * r + 1) * (1 + r)
        N_mtx[1, 13] = 0.9e1 / 0.256e3 * \
            (-1 + s) * (3 * s - 1) * (1 + s) * \
            (3 * r - 1) * (3 * r + 1) * (1 + r)
        N_mtx[1, 15] = -0.9e1 / 0.256e3 * \
            (-1 + s) * (3 * s + 1) * (1 + s) * \
            (3 * r - 1) * (3 * r + 1) * (1 + r)
        N_mtx[1, 17] = -0.9e1 / 0.256e3 * \
            (3 * s - 1) * (3 * s + 1) * (1 + s) * \
            (-1 + r) * (3 * r + 1) * (1 + r)
        N_mtx[1, 19] = 0.9e1 / 0.256e3 * \
            (3 * s - 1) * (3 * s + 1) * (1 + s) * \
            (-1 + r) * (3 * r - 1) * (1 + r)
        N_mtx[1, 21] = 0.9e1 / 0.256e3 * \
            (-1 + s) * (3 * s + 1) * (1 + s) * \
            (-1 + r) * (3 * r - 1) * (3 * r + 1)
        N_mtx[1, 23] = -0.9e1 / 0.256e3 * \
            (-1 + s) * (3 * s - 1) * (1 + s) * \
            (-1 + r) * (3 * r - 1) * (3 * r + 1)
        N_mtx[1, 25] = 0.81e2 / 0.256e3 * \
            (-1 + s) * (3 * s - 1) * (1 + s) * (-1 + r) * (3 * r - 1) * (1 + r)
        N_mtx[1, 27] = -0.81e2 / 0.256e3 * \
            (-1 + s) * (3 * s - 1) * (1 + s) * (-1 + r) * (3 * r + 1) * (1 + r)
        N_mtx[1, 29] = -0.81e2 / 0.256e3 * \
            (-1 + s) * (3 * s + 1) * (1 + s) * (-1 + r) * (3 * r - 1) * (1 + r)
        N_mtx[1, 31] = 0.81e2 / 0.256e3 * \
            (-1 + s) * (3 * s + 1) * (1 + s) * (-1 + r) * (3 * r + 1) * (1 + r)
        return N_mtx

    def get_dNr_mtx(self, r_pnt):
        '''
        Return the derivatives of the shape functions used for the field approximation
        '''
        r = r_pnt[0]
        s = r_pnt[1]
        dNr = zeros((2, 16), dtype='float_')
        dNr[0, 0] = ((3 * r - 1) * (3 * r + 1) * (-1 + s) * (3 * s - 1) * (3 * s + 1)) / 0.256e3 + 0.3e1 / 0.256e3 * (-1 + r) * (3 * 
                                                                                                                                 r + 1) * (-1 + s) * (3 * s - 1) * (3 * s + 1) + 0.3e1 / 0.256e3 * (-1 + r) * (3 * r - 1) * (-1 + s) * (3 * s - 1) * (3 * s + 1)
        dNr[0, 1] = -0.3e1 / 0.256e3 * (3 * r + 1) * (1 + r) * (-1 + s) * (3 * s - 1) * (3 * s + 1) - 0.3e1 / 0.256e3 * (3 * r - 1) * (
            1 + r) * (-1 + s) * (3 * s - 1) * (3 * s + 1) - ((3 * r - 1) * (3 * r + 1) * (-1 + s) * (3 * s - 1) * (3 * s + 1)) / 0.256e3
        dNr[0, 2] = 0.3e1 / 0.256e3 * (3 * r + 1) * (1 + r) * (3 * s - 1) * (3 * s + 1) * (1 + s) + 0.3e1 / 0.256e3 * (3 * r - 1) * (
            1 + r) * (3 * s - 1) * (3 * s + 1) * (1 + s) + ((3 * r - 1) * (3 * r + 1) * (3 * s - 1) * (3 * s + 1) * (1 + s)) / 0.256e3
        dNr[0, 3] = -((3 * r - 1) * (3 * r + 1) * (3 * s - 1) * (3 * s + 1) * (1 + s)) / 0.256e3 - 0.3e1 / 0.256e3 * (-1 + r) * \
            (3 * r + 1) * (3 * s - 1) * (3 * s + 1) * (1 + s) - 0.3e1 / \
            0.256e3 * (-1 + r) * (3 * r - 1) * \
            (3 * s - 1) * (3 * s + 1) * (1 + s)
        dNr[0, 4] = -0.9e1 / 0.256e3 * (3 * r - 1) * (1 + r) * (-1 + s) * (3 * s - 1) * (3 * s + 1) - 0.27e2 / 0.256e3 * (-1 + r) * (
            1 + r) * (-1 + s) * (3 * s - 1) * (3 * s + 1) - 0.9e1 / 0.256e3 * (-1 + r) * (3 * r - 1) * (-1 + s) * (3 * s - 1) * (3 * s + 1)
        dNr[0, 5] = 0.9e1 / 0.256e3 * (3 * r + 1) * (1 + r) * (-1 + s) * (3 * s - 1) * (3 * s + 1) + 0.27e2 / 0.256e3 * (-1 + r) * (
            1 + r) * (-1 + s) * (3 * s - 1) * (3 * s + 1) + 0.9e1 / 0.256e3 * (-1 + r) * (3 * r + 1) * (-1 + s) * (3 * s - 1) * (3 * s + 1)
        dNr[0, 6] = 0.27e2 / 0.256e3 * (3 * r + 1) * (1 + r) * (-1 + s) * (3 * s - 1) * (1 + s) + 0.27e2 / 0.256e3 * (3 * r - 1) * (
            1 + r) * (-1 + s) * (3 * s - 1) * (1 + s) + 0.9e1 / 0.256e3 * (3 * r - 1) * (3 * r + 1) * (-1 + s) * (3 * s - 1) * (1 + s)
        dNr[0, 7] = -0.27e2 / 0.256e3 * (3 * r + 1) * (1 + r) * (-1 + s) * (3 * s + 1) * (1 + s) - 0.27e2 / 0.256e3 * (3 * r - 1) * (
            1 + r) * (-1 + s) * (3 * s + 1) * (1 + s) - 0.9e1 / 0.256e3 * (3 * r - 1) * (3 * r + 1) * (-1 + s) * (3 * s + 1) * (1 + s)
        dNr[0, 8] = -0.9e1 / 0.256e3 * (3 * r + 1) * (1 + r) * (3 * s - 1) * (3 * s + 1) * (1 + s) - 0.27e2 / 0.256e3 * (-1 + r) * (
            1 + r) * (3 * s - 1) * (3 * s + 1) * (1 + s) - 0.9e1 / 0.256e3 * (-1 + r) * (3 * r + 1) * (3 * s - 1) * (3 * s + 1) * (1 + s)
        dNr[0, 9] = 0.9e1 / 0.256e3 * (3 * r - 1) * (1 + r) * (3 * s - 1) * (3 * s + 1) * (1 + s) + 0.27e2 / 0.256e3 * (-1 + r) * (
            1 + r) * (3 * s - 1) * (3 * s + 1) * (1 + s) + 0.9e1 / 0.256e3 * (-1 + r) * (3 * r - 1) * (3 * s - 1) * (3 * s + 1) * (1 + s)
        dNr[0, 10] = 0.9e1 / 0.256e3 * (3 * r - 1) * (3 * r + 1) * (-1 + s) * (3 * s + 1) * (1 + s) + 0.27e2 / 0.256e3 * (-1 + r) * (
            3 * r + 1) * (-1 + s) * (3 * s + 1) * (1 + s) + 0.27e2 / 0.256e3 * (-1 + r) * (3 * r - 1) * (-1 + s) * (3 * s + 1) * (1 + s)
        dNr[0, 11] = -0.9e1 / 0.256e3 * (3 * r - 1) * (3 * r + 1) * (-1 + s) * (3 * s - 1) * (1 + s) - 0.27e2 / 0.256e3 * (-1 + r) * (
            3 * r + 1) * (-1 + s) * (3 * s - 1) * (1 + s) - 0.27e2 / 0.256e3 * (-1 + r) * (3 * r - 1) * (-1 + s) * (3 * s - 1) * (1 + s)
        dNr[0, 12] = 0.81e2 / 0.256e3 * (3 * r - 1) * (1 + r) * (-1 + s) * (3 * s - 1) * (1 + s) + 0.243e3 / 0.256e3 * (-1 + r) * (
            1 + r) * (-1 + s) * (3 * s - 1) * (1 + s) + 0.81e2 / 0.256e3 * (-1 + r) * (3 * r - 1) * (-1 + s) * (3 * s - 1) * (1 + s)
        dNr[0, 13] = -0.81e2 / 0.256e3 * (3 * r + 1) * (1 + r) * (-1 + s) * (3 * s - 1) * (1 + s) - 0.243e3 / 0.256e3 * (-1 + r) * (
            1 + r) * (-1 + s) * (3 * s - 1) * (1 + s) - 0.81e2 / 0.256e3 * (-1 + r) * (3 * r + 1) * (-1 + s) * (3 * s - 1) * (1 + s)
        dNr[0, 14] = -0.81e2 / 0.256e3 * (3 * r - 1) * (1 + r) * (-1 + s) * (3 * s + 1) * (1 + s) - 0.243e3 / 0.256e3 * (-1 + r) * (
            1 + r) * (-1 + s) * (3 * s + 1) * (1 + s) - 0.81e2 / 0.256e3 * (-1 + r) * (3 * r - 1) * (-1 + s) * (3 * s + 1) * (1 + s)
        dNr[0, 15] = 0.81e2 / 0.256e3 * (3 * r + 1) * (1 + r) * (-1 + s) * (3 * s + 1) * (1 + s) + 0.243e3 / 0.256e3 * (-1 + r) * (
            1 + r) * (-1 + s) * (3 * s + 1) * (1 + s) + 0.81e2 / 0.256e3 * (-1 + r) * (3 * r + 1) * (-1 + s) * (3 * s + 1) * (1 + s)
        dNr[1, 0] = ((-1 + r) * (3 * r - 1) * (3 * r + 1) * (3 * s - 1) * (3 * s + 1)) / 0.256e3 + 0.3e1 / 0.256e3 * (-1 + r) * (3 * 
                                                                                                                                 r - 1) * (3 * r + 1) * (-1 + s) * (3 * s + 1) + 0.3e1 / 0.256e3 * (-1 + r) * (3 * r - 1) * (3 * r + 1) * (-1 + s) * (3 * s - 1)
        dNr[1, 1] = -((3 * r - 1) * (3 * r + 1) * (1 + r) * (3 * s - 1) * (3 * s + 1)) / 0.256e3 - 0.3e1 / 0.256e3 * (3 * r - 1) * \
            (3 * r + 1) * (1 + r) * (-1 + s) * (3 * s + 1) - 0.3e1 / 0.256e3 * \
            (3 * r - 1) * (3 * r + 1) * (1 + r) * (-1 + s) * (3 * s - 1)
        dNr[1, 2] = 0.3e1 / 0.256e3 * (3 * r - 1) * (3 * r + 1) * (1 + r) * (3 * s + 1) * (1 + s) + 0.3e1 / 0.256e3 * (3 * r - 1) * (
            3 * r + 1) * (1 + r) * (3 * s - 1) * (1 + s) + ((3 * r - 1) * (3 * r + 1) * (1 + r) * (3 * s - 1) * (3 * s + 1)) / 0.256e3
        dNr[1, 3] = -0.3e1 / 0.256e3 * (-1 + r) * (3 * r - 1) * (3 * r + 1) * (3 * s + 1) * (1 + s) - 0.3e1 / 0.256e3 * (-1 + r) * (
            3 * r - 1) * (3 * r + 1) * (3 * s - 1) * (1 + s) - ((-1 + r) * (3 * r - 1) * (3 * r + 1) * (3 * s - 1) * (3 * s + 1)) / 0.256e3
        dNr[1, 4] = -0.9e1 / 0.256e3 * (-1 + r) * (3 * r - 1) * (1 + r) * (3 * s - 1) * (3 * s + 1) - 0.27e2 / 0.256e3 * (-1 + r) * (
            3 * r - 1) * (1 + r) * (-1 + s) * (3 * s + 1) - 0.27e2 / 0.256e3 * (-1 + r) * (3 * r - 1) * (1 + r) * (-1 + s) * (3 * s - 1)
        dNr[1, 5] = 0.9e1 / 0.256e3 * (-1 + r) * (3 * r + 1) * (1 + r) * (3 * s - 1) * (3 * s + 1) + 0.27e2 / 0.256e3 * (-1 + r) * (
            3 * r + 1) * (1 + r) * (-1 + s) * (3 * s + 1) + 0.27e2 / 0.256e3 * (-1 + r) * (3 * r + 1) * (1 + r) * (-1 + s) * (3 * s - 1)
        dNr[1, 6] = 0.9e1 / 0.256e3 * (3 * r - 1) * (3 * r + 1) * (1 + r) * (3 * s - 1) * (1 + s) + 0.27e2 / 0.256e3 * (3 * r - 1) * (
            3 * r + 1) * (1 + r) * (-1 + s) * (1 + s) + 0.9e1 / 0.256e3 * (3 * r - 1) * (3 * r + 1) * (1 + r) * (-1 + s) * (3 * s - 1)
        dNr[1, 7] = -0.9e1 / 0.256e3 * (3 * r - 1) * (3 * r + 1) * (1 + r) * (3 * s + 1) * (1 + s) - 0.27e2 / 0.256e3 * (3 * r - 1) * (
            3 * r + 1) * (1 + r) * (-1 + s) * (1 + s) - 0.9e1 / 0.256e3 * (3 * r - 1) * (3 * r + 1) * (1 + r) * (-1 + s) * (3 * s + 1)
        dNr[1, 8] = -0.27e2 / 0.256e3 * (-1 + r) * (3 * r + 1) * (1 + r) * (3 * s + 1) * (1 + s) - 0.27e2 / 0.256e3 * (-1 + r) * (
            3 * r + 1) * (1 + r) * (3 * s - 1) * (1 + s) - 0.9e1 / 0.256e3 * (-1 + r) * (3 * r + 1) * (1 + r) * (3 * s - 1) * (3 * s + 1)
        dNr[1, 9] = 0.27e2 / 0.256e3 * (-1 + r) * (3 * r - 1) * (1 + r) * (3 * s + 1) * (1 + s) + 0.27e2 / 0.256e3 * (-1 + r) * (
            3 * r - 1) * (1 + r) * (3 * s - 1) * (1 + s) + 0.9e1 / 0.256e3 * (-1 + r) * (3 * r - 1) * (1 + r) * (3 * s - 1) * (3 * s + 1)
        dNr[1, 10] = 0.9e1 / 0.256e3 * (-1 + r) * (3 * r - 1) * (3 * r + 1) * (3 * s + 1) * (1 + s) + 0.27e2 / 0.256e3 * (-1 + r) * (
            3 * r - 1) * (3 * r + 1) * (-1 + s) * (1 + s) + 0.9e1 / 0.256e3 * (-1 + r) * (3 * r - 1) * (3 * r + 1) * (-1 + s) * (3 * s + 1)
        dNr[1, 11] = -0.9e1 / 0.256e3 * (-1 + r) * (3 * r - 1) * (3 * r + 1) * (3 * s - 1) * (1 + s) - 0.27e2 / 0.256e3 * (-1 + r) * (
            3 * r - 1) * (3 * r + 1) * (-1 + s) * (1 + s) - 0.9e1 / 0.256e3 * (-1 + r) * (3 * r - 1) * (3 * r + 1) * (-1 + s) * (3 * s - 1)
        dNr[1, 12] = 0.81e2 / 0.256e3 * (-1 + r) * (3 * r - 1) * (1 + r) * (3 * s - 1) * (1 + s) + 0.243e3 / 0.256e3 * (-1 + r) * (
            3 * r - 1) * (1 + r) * (-1 + s) * (1 + s) + 0.81e2 / 0.256e3 * (-1 + r) * (3 * r - 1) * (1 + r) * (-1 + s) * (3 * s - 1)
        dNr[1, 13] = -0.81e2 / 0.256e3 * (-1 + r) * (3 * r + 1) * (1 + r) * (3 * s - 1) * (1 + s) - 0.243e3 / 0.256e3 * (-1 + r) * (
            3 * r + 1) * (1 + r) * (-1 + s) * (1 + s) - 0.81e2 / 0.256e3 * (-1 + r) * (3 * r + 1) * (1 + r) * (-1 + s) * (3 * s - 1)
        dNr[1, 14] = -0.81e2 / 0.256e3 * (-1 + r) * (3 * r - 1) * (1 + r) * (3 * s + 1) * (1 + s) - 0.243e3 / 0.256e3 * (-1 + r) * (
            3 * r - 1) * (1 + r) * (-1 + s) * (1 + s) - 0.81e2 / 0.256e3 * (-1 + r) * (3 * r - 1) * (1 + r) * (-1 + s) * (3 * s + 1)
        dNr[1, 15] = 0.81e2 / 0.256e3 * (-1 + r) * (3 * r + 1) * (1 + r) * (3 * s + 1) * (1 + s) + 0.243e3 / 0.256e3 * (-1 + r) * (
            3 * r + 1) * (1 + r) * (-1 + s) * (1 + s) + 0.81e2 / 0.256e3 * (-1 + r) * (3 * r + 1) * (1 + r) * (-1 + s) * (3 * s + 1)
        return dNr

    def get_B_mtx(self, r_pnt, X_mtx):

        J_mtx = self.get_J_mtx(r_pnt, X_mtx)
        dNr_mtx = self.get_dNr_mtx(r_pnt)
        dNx_mtx = dot(inv(J_mtx), dNr_mtx)
        Bx_mtx = zeros((3, 32), dtype='float_')
        for i in range(0, 16):
            Bx_mtx[0, i * 2] = dNx_mtx[0, i]
            Bx_mtx[1, i * 2 + 1] = dNx_mtx[1, i]
            Bx_mtx[2, i * 2] = dNx_mtx[1, i]
            Bx_mtx[2, i * 2 + 1] = dNx_mtx[0, i]
        return Bx_mtx


#----------------------- example --------------------
if __name__ == '__main__':
    from ibvpy.api import \
        TStepper as TS, RTDofGraph, RTraceDomainListField, TLoop, \
        TLine, BCDofGroup, IBVPSolve as IS

    # from lib.mats.mats2D.mats_cmdm2D.mats_mdm2d import MACMDM
    # from lib.mats.mats2D.mats2D_sdamage.mats2D_sdamage import MATS2DScalarDamage
    # from lib.mats.mats2D.mats2D_sdamage.strain_norm2d import *
    from ibvpy.mats.mats2D.mats2D_elastic.mats2D_elastic import MATS2DElastic

    # fets_eval = FETS2D4Q16U(mats_eval = MATS2DScalarDamage(strain_norm = Euclidean()))
    # fets_eval = FEQ16U(mats_eval = MACMDM())
    fets_eval = FETS2D4Q16U(mats_eval=MATS2DElastic())

    from ibvpy.mesh.fe_grid import FEGrid

    # Discretization
    domain = FEGrid(coord_max=(3., 3., 0.),
                    shape=(2, 2),
                    fets_eval=fets_eval)

    right_dof = 2
    ts = TS(
        sdomain=domain,
        # conversion to list (square brackets) is only necessary for slicing of
        # single dofs, e.g "get_left_dofs()[0,1]"
        bcond_list=[BCDofGroup(var='u', value=0., dims=[0],
                               get_dof_method=domain.get_left_dofs),
                    BCDofGroup(var='u', value=0., dims=[1],
                               get_dof_method=domain.get_bottom_left_dofs),
                    BCDofGroup(var='u', value=0.002, dims=[0],
                               get_dof_method=domain.get_right_dofs)],

        rtrace_list=[RTDofGraph(name='Fi,right over u_right (iteration)',
                                 var_y='F_int', idx_y=right_dof,
                                 var_x='U_k', idx_x=right_dof),
                     #                         RTraceDomainField(name = 'Stress' ,
                     #                         var = 'sig_app', idx = 0,
                     #                         record_on = 'update'),
                     RTraceDomainListField(name='Displacement',
                                           var='u', idx=0),
                     #                             RTraceDomainField(name = 'N0' ,
                     #                                          var = 'N_mtx', idx = 0,
                     # record_on = 'update')

                     ]
    )

    # Add the time-loop control
    #
    tl = TLoop(tstepper=ts,
               DT=0.5,
               tline=TLine(min=0.0, max=1.0))

    tl.eval()
    # Put the whole stuff into the simulation-framework to map the
    # individual pieces of definition into the user interface.
    #
    from ibvpy.plugins.ibvpy_app import IBVPyApp
    app = IBVPyApp(ibv_resource=tl)
    app.main()
