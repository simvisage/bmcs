'''
Created on May 25, 2009

@author: jakub
'''
from scipy.optimize import \
    brentq
from traits.api import \
    Array, Int, Property, cached_property, \
    HasTraits, provides, WeakRef, Float, \
    Str, Tuple
import numpy as np
from .i_fe_grid_slice import \
    IFEGridSlice


@provides(IFEGridSlice)
class FEGridLevelSetSlice(HasTraits):

    # Link to the source grid to slice
    #
    fe_grid = WeakRef('ibvpy.mesh.fe_grid.FEGrid')

    # Specification of further methods to be provided by the fe_grid_cell
    #
    # @todo -
    # 1) describe the functionality of the methods
    # 2) decide where to implement the details of the method
    #    Generally, the lookup for elements should be directed
    #    to the geo_grid, while the lookup for individual degrees
    #    of freedom should go into the dof_grid
    #
    ls_function_eval = Str

    ls_spec_list = Property(Tuple, depends_on='ls_function_eval')

    @cached_property
    def _get_ls_spec_list(self):
        fn_list = self.ls_function_eval.split(';')
        fns = []
        lim = []
        for fn in fn_list:
            try:
                afn, limits = fn.split('@')
            except:  # convenience - of there is just one fn, user does not have to add region
                afn = fn
                limits = '1'
            fns.append(afn.upper())  # transform to uppercase
            lim.append(limits)  # limits.upper() )
        return fns, lim

    ls_fns = Property

    def _get_ls_fns(self):
        return self.ls_spec_list[0]

    ls_lims = Property

    def _get_ls_lims(self):
        return self.ls_spec_list[1]

    def ls_function(self, X=None, Y=None, Z=None):
        '''
        get_value() evaluates the value for all functions as defined in set_fns, 
        will be 0 if grid point is not defined in any function
        '''
        val = 0.0
        for fn, limit in zip(self.ls_fns, self.ls_lims):
            if eval(limit):
                val = eval(fn)
        return val

    def ls_mask_function(self, X=None, Y=None, Z=None):
        '''Return True if the point is within one of the ranges of the 
        supplied ls functions.
        '''
        masked = True
        for limit in self.ls_lims:
            if eval(limit):
                if masked == False:
                    raise ValueError(
                        'overlapping level set limits - two level sets defined')
                masked = False
        return masked

    #
    tip_elems = Property(Array(int))

    def _get_tip_elems(self):
        '''Get the elements at the tip of the level set function

        The current implementation does not consider the association
        between the ls_mask function and the ls_function.

        This is probably not correct.

        The tip elems should be removed from the level set alone.

        '''
        ls_mask = self.fe_grid.get_ls_mask(self.ls_mask_function)
        # get the nodes having True value

        masked_nodes = np.argwhere(ls_mask)
        # find out the element associated with these nodes

        neighbour_idx = np.array([[-1, -1],
                                  [-1, 0],
                                  [0, -1],
                                  [0, 0]], dtype=int)

        # get for each node the neighbors
        e_arr = masked_nodes[:, None, :] + neighbour_idx[None, :, :]

        # get for each node the corresponding neighbors
        e_flattened = [e_arr[:, :, idx].flatten()
                       for idx in range(e_arr.shape[2])]
        e_masked = np.c_[tuple(e_flattened)]

        # find the intersection with the level set
        intersected_elem_offsets = self.fe_grid.get_intersected_elems(
            self.ls_function)

        # get the indices of the intersected by the level set
        # this should be done more efficiently
        e_ls = np.array([self.fe_grid.geo_grid.cell_grid.get_cell_idx(offset)
                         for offset in intersected_elem_offsets], dtype=int)

        # make a broadcasted comparison between the masked elements
        # (elements containing a node outside outside the level set domain)
        # and elements intersected by the level set.
        #
        broadcasted_compare = (e_ls == e_masked[:, None, :])

        # along the axis 2 - there is a comparison of the coordinates
        # - if all are True, then all coordinates are equal - this
        # is detected by the boolean product (logical and along axis 2)
        #
        logical_and_along_ax2 = broadcasted_compare.prod(axis=2, dtype=bool)

        # along the axis 0 - there is a each-with-each comparison
        # - if one of the entries is True, then the second list contains
        #   the same entry as well. This is detected by the sum operator
        #  (logical or along axis 0)
        logical_or_along_ax0 = logical_and_along_ax2.sum(axis=0, dtype=bool)

        #res = ( e_ls == e_masked[:, None, :] ).prod( axis = 2, dtype = bool ).sum( axis = 0, dtype = bool )

        return e_ls[logical_or_along_ax0]

    elems = Property(Array(int))

    def _get_elems(self):
        # 1. for each ls_function identify the intersected elements
        # 2. choose those lying within the ls_domain - including the tip
        # elements.
        return self.fe_grid.get_intersected_elems(self.ls_function,
                                                  self.ls_mask_function)

    neg_elems = Property(Array(Int))

    def _get_neg_elems(self):
        return self.fe_grid.get_negative_elems(self.ls_function)

    pos_elems = Property(Array(Int))

    def _get_pos_elems(self):
        # return self.fe_grid.get_positive_elems()
        raise NotImplementedError

    dof_nodes_values = Property(Array(Float))

    @cached_property
    def _get_dof_nodes_values(self):
        #nodes = self.dof_grid.cell_node_map[ self.get_intersected_elems() ]
        # print "map ", nodes
        #coords = self.dof_grid.get_cell_points(self.get_intersected_elems())
        coords = self.dof_X

        # @todo - make the dimensional dependency
        #
        ls_fn = np.frompyfunc(self.ls_function, 2, 1)
        o_shape = coords.shape  # save shape for later
        if coords.shape[2] == 1:
            X = coords.reshape(o_shape[0] * o_shape[1], o_shape[2]).T
            Y = np.zeros_like(X)
        elif coords.shape[2] == 2:
            X, Y = coords.reshape(o_shape[0] * o_shape[1], o_shape[2]).T
        values = ls_fn(X, Y)
        return values.reshape(o_shape[0], o_shape[1])

    geo_X = Property

    def _get_geo_X(self):
        '''Get global coordinates geometrical points of the intersected elements.
        '''
        return self.fe_grid.geo_grid.get_cell_point_X_arr(self.elems)

    geo_x = Property

    def _get_geo_x(self):
        return self.fe_grid.geo_grid.get_cell_point_x_arr(self.elems)

    dof_X = Property

    def _get_dof_X(self):
        '''Get global coordinates field points of the intersected elements.
        '''
        return self.fe_grid.dof_grid.get_cell_point_X_arr(self.elems)

    r_i = Property(Array(float))  # 2 points/element

    @cached_property
    def _get_r_i(self):
        '''
        Return local coordinates of the intersection points
        Assuming rectangular parametric coordinates
        Works for 1D and 2D
        '''
        i_elements = self.fe_grid.get_intersected_elems(self.ls_function)
        el_pnts = []
        for elem in i_elements:
            inter_pts = []
            # X_mtx = self.elements[elem].get_X_mtx() # skips deactivated
            # elements
            X_mtx = self.fe_grid.geo_grid.get_cell_point_X_arr(elem)
            dim = X_mtx.shape[1]  # TODO:merge 1 and 2d
            if dim == 1:
                r_coord = self._get_intersect_pt(self.ls_fn_r, (0., X_mtx))
                if r_coord != None:
                    inter_pts.append([r_coord])
            elif dim == 2:
                for c_coord in [-1., 1.]:
                    args = (c_coord, X_mtx)
                    s_coord = self._get_intersect_pt(self.ls_fn_s, args)
                    r_coord = self._get_intersect_pt(self.ls_fn_r, args)
                    if s_coord != None:
                        inter_pts.append([c_coord, s_coord])
                    if r_coord != None:
                        inter_pts.append([r_coord, c_coord])
            elif dim == 3:
                raise NotImplementedError('not available for 3D yet')
            el_pnts.append(inter_pts)
        return np.array(el_pnts)

    dofs = Property

    def _get_dofs(self):
        '''Get number of affected DOFs

        @todo - reshape dofs so that they are of dimension
        ( elem, node, dim )
        '''
        dofs = []
        for elem in self.elems:
            dof_map = self.fe_grid.dof_grid.get_cell_dofs(elem).flatten()
            dofs.append(dof_map)
        return np.array(dofs)

    i_neg_dofs = Property

    def _get_i_neg_dofs(self):
        '''Get intersected dofs with negative value of the level set at their position.
        '''
        shape = self.dofs.shape
        n_node_dofs = shape[1] / self.dof_X.shape[1]
        return self.dofs.reshape(shape[0],
                                 self.dof_X.shape[1],
                                 n_node_dofs)[self.dof_nodes_values < 0.]

    neg_dofs = Property

    def _get_neg_dofs(self):
        '''Get dofs with negative value of the level set at their position.
        '''
        dofs = []
        for elem in self.neg_elems:
            dof_map = self.fe_grid.dof_grid.get_cell_dofs(elem).flatten()
            dofs.append(dof_map)
        if dofs == []:
            return np.array(dofs)
        return np.hstack(dofs)

    def _get_intersect_pt(self, fn, args):
        try:
            return brentq(fn, -1, 1, args=args)
        except ValueError:
            return

    def ls_fn_r(self, r, s, X_mtx):  # TODO:dimensionless treatment
        X_pnt = self.fe_grid.fets_eval.map_r2X([r, s], X_mtx)
        if X_pnt.shape[0] == 1:
            Y = 0.
        else:
            Y = X_pnt[1]
        return self.ls_function(X_pnt[0], Y)

    def ls_fn_s(self, s, r, X_mtx):
        X_pnt = self.fe_grid.fets_eval.map_r2X([r, s], X_mtx)
        return self.ls_function(X_pnt[0], X_pnt[1])

    def ls_fn_X(self, X, Y):
        return self.ls_function(X, Y)


if __name__ == '__main__':
    from ibvpy.mesh.fe_grid import FEGrid
    from ibvpy.fets.fets2D.fets2D4q import FETS2D4Q
    from ibvpy.mats.mats2D.mats2D_elastic.mats2D_elastic import MATS2DElastic

    fets_eval = FETS2D4Q(mats_eval=MATS2DElastic())
    fe_grid1 = FEGrid(coord_max=(3., 2.),
                      shape=(3, 2),
                      fets_eval=fets_eval)

    def ls_fn(X, Y):
        # print "point ", X," ",Y
        #print "ls value ", X - 1.5#
        return X - 1.5  # Y-0.2*X#temp

    print('fe_grid elemes')
    print(fe_grid1.geo_grid.cell_grid.cell_idx_grid)

    print('vertex_X_grid')
    print(fe_grid1.geo_grid.cell_grid.vertex_X_grid)

    fe_slice = fe_grid1['X - 1.5']
    print('slice ')
    print(fe_slice)
    print('elems')
    print(fe_slice.elems)
    print('node values')
    print(fe_slice.dof_nodes_values)
    print('intersecting points')
    print(fe_slice.r_i)
    print('dofs')
    print(fe_slice.dofs)
    print('geo coords')
    print(fe_slice.geo_X)
    print('dof coords')
    print(fe_slice.dof_X)
    print('neg i dofs')
    print(fe_slice.i_neg_dofs)
    print('neg dofs')
    print(fe_slice.neg_dofs)
