
from ibvpy.dots.dots_eval import \
    DOTSEval
from ibvpy.dots.dots_unstructured_eval import DOTSUnstructuredEval
from ibvpy.fets.fets_eval import FETSEval
from ibvpy.mats.mats_eval import IMATSEval
from numpy import \
     array, zeros, int_, float_, ix_, dot, linspace, hstack, vstack, arange, \
     identity, unique, average, frompyfunc, abs, linalg, argmin
from scipy.linalg import \
     inv, det, norm
from scipy.optimize import brentq
from traits.api import \
     Array, Bool, Callable, Enum, Float, HasTraits, Interface, \
     Instance, Int, Trait, Str, Enum, Callable, List, TraitDict, Any, \
     on_trait_change, Tuple, WeakRef, Delegate, Property, cached_property, \
     Class, TraitError

from tvtk.api import tvtk


class FETSLSEval(FETSEval):

    x_slice = slice(0, 0)
    parent_fets = Instance(FETSEval)

    nip_disc = Int(0)  # number of integration points on the discontinuity

    def setup(self, sctx, n_ip):
        '''
        overloading the default method
        mats state array has to account for different number of ip in elements
        Perform the setup in the all integration points.
        TODO: original setup can be used after adaptation the ip_coords param
        '''
#        print 'n_ip ', n_ip
#        print 'self.m_arr_size ',self.m_arr_size
#        print 'shape ',sctx.elem_state_array.shape
        for i in range(n_ip):
            sctx.mats_state_array = sctx.elem_state_array[(i * self.m_arr_size): ((i + 1) * self.m_arr_size)]
            self.mats_eval.setup(sctx)

    n_nodes = Property  # TODO: define dependencies

    @cached_property
    def _get_n_nodes(self):
        return self.parent_fets.n_e_dofs / self.parent_fets.n_nodal_dofs

    # dots_class = DOTSUnstructuredEval
    dots_class = Class(DOTSEval)

    int_order = Int(1)

    mats_eval = Delegate('parent_fets')
    mats_eval_pos = Trait(None, Instance(IMATSEval))
    mats_eval_neg = Trait(None, Instance(IMATSEval))
    mats_eval_disc = Trait(None, Instance(IMATSEval))
    dim_slice = Delegate('parent_fets')

    dof_r = Delegate('parent_fets')
    geo_r = Delegate('parent_fets')
    n_nodal_dofs = Delegate('parent_fets')
    n_e_dofs = Delegate('parent_fets')

    get_dNr_mtx = Delegate('parent_fets')
    get_dNr_geo_mtx = Delegate('parent_fets')

    get_N_geo_mtx = Delegate('parent_fets')

    def get_B_mtx(self, r_pnt, X_mtx, node_ls_values, r_ls_value):
        B_mtx = self.parent_fets.get_B_mtx(r_pnt, X_mtx)
        return B_mtx

    def get_u(self, sctx, u):
        N_mtx = self.parent_fets.get_N_mtx(sctx.loc)
        return dot(N_mtx, u)

    def get_eps_eng(self, sctx, u):
        B_mtx = self.parent_fets.get_B_mtx(sctx.loc,
                                           sctx.X)
        return dot(B_mtx, u)

    dof_r = Delegate('parent_fets')
    geo_r = Delegate('parent_fets')

    node_ls_values = Array(float)

    tri_subdivision = Int(0)

    def get_triangulation(self, point_set):
        dim = point_set[0].shape[1]
        n_add = 3 - dim
        if dim == 1:  # sideway for 1D
            structure = [array([min(point_set[0]),
                                max(point_set[0]),
                                min(point_set[1]),
                                max(point_set[1])], dtype=float),
                        array([[0, 1],
                               [2, 3]], dtype=int)]
            return structure
        points_list = []
        triangles_list = []
        point_offset = 0
        for pts in point_set:
            if self.tri_subdivision == 1:
                new_pt = average(pts, 0)
                pts = vstack((pts, new_pt))
            if n_add > 0:
                points = hstack([pts,
                                  zeros([pts.shape[0], n_add], dtype='float_')])
            # Create a polydata with the points we just created.
            profile = tvtk.PolyData(points=points)

            # Perform a 2D Delaunay triangulation on them.
            delny = tvtk.Delaunay2D(input=profile, offset=1.e1)
            tri = delny.output
            tri.update()  # initiate triangulation
            triangles = array(tri.polys.data, dtype=int_)
            pt = tri.points.data
            tri = (triangles.reshape((triangles.shape[0] / 4), 4))[:, 1:]
            points_list += list(pt)
            triangles_list += list(tri + point_offset)
            point_offset += len(unique(tri))  # Triangulation
        points = array(points_list)
        triangles = array(triangles_list)
        return [points, triangles]

    vtk_point_ip_map = Property(Array(Int))

    def _get_vtk_point_ip_map(self):
        '''
        mapping of the visualization point to the integration points
        according to mutual proximity in the local coordinates
        '''
        vtk_pt_arr = zeros((1, 3), dtype='float_')
        ip_map = zeros(self.vtk_r.shape[0], dtype='int_')
        for i, vtk_pt in enumerate(self.vtk_r):
            vtk_pt_arr[0, self.dim_slice] = vtk_pt
            # get the nearest ip_coord
            ip_map[i] = argmin(cdist(vtk_pt_arr, self.ip_coords))
        return array(ip_map)

    def get_ip_coords(self, int_triangles, int_order):
        '''Get the array of integration points'''
        gps = []
        points, triangles = int_triangles
        if triangles.shape[1] == 1:  # 0D - points
            if int_order == 1:
                gps.append(points[0])
            else:
                raise TraitError('does not make sense')
        elif triangles.shape[1] == 2:  # 1D - lines
            if int_order == 1:
                for id in triangles:
                    gp = average(points[ix_(id)], 0)
                    gps.append(gp)
            elif int_order == 2:
                weigths = array([[0.21132486540518713, 0.78867513459481287],
                                 [0.78867513459481287, 0.21132486540518713]])
                for id in triangles :
                    gps += average(points[ix_(id)], 0, weigths[0]), \
                            average(points[ix_(id)], 0, weigths[1])
            else:
                raise NotImplementedError
        elif triangles.shape[1] == 3:  # 2D - triangles    
            if int_order == 1:
                for id in triangles:
                    gp = average(points[ix_(id)], 0)
                    # print "gp ",gp
                    gps.append(gp)
            elif int_order == 2:
                raise NotImplementedError
            elif int_order == 3:
                weigths = array([[0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]])
                for id in triangles :
                    gps += average(points[ix_(id)], 0), \
                        average(points[ix_(id)], 0, weigths[0]), \
                        average(points[ix_(id)], 0, weigths[1]), \
                        average(points[ix_(id)], 0, weigths[2])

            elif int_order == 4:
                raise NotImplementedError
            elif int_order == 5:
                weigths = array([[0.0597158717, 0.4701420641, 0.4701420641], \
                                 [0.4701420641, 0.0597158717, 0.4701420641], \
                                 [0.4701420641, 0.4701420641, 0.0597158717], \
                                 [0.7974269853, 0.1012865073, 0.1012865073], \
                                 [0.1012865073, 0.7974269853, 0.1012865073], \
                                 [0.1012865073, 0.1012865073, 0.7974269853]])
                for id in triangles:
                    weigts_sum = False  # for debug
                    gps += average(points[ix_(id)], 0), \
                         average(points[ix_(id)], 0, weigths[0], weigts_sum), \
                         average(points[ix_(id)], 0, weigths[1], weigts_sum), \
                         average(points[ix_(id)], 0, weigths[2], weigts_sum), \
                         average(points[ix_(id)], 0, weigths[3], weigts_sum), \
                         average(points[ix_(id)], 0, weigths[4], weigts_sum), \
                         average(points[ix_(id)], 0, weigths[5], weigts_sum)
            else:
                raise NotImplementedError
        elif triangles.shape[1] == 4:  # 3D - tetrahedrons
           raise NotImplementedError
        else:
            raise TraitError('unsupported geometric form with %s nodes ' % triangles.shape[1])
        return array(gps, dtype='float_')

    def get_ip_weights(self, int_triangles, int_order):
        '''Get the array of integration points'''
        gps = []
        points, triangles = int_triangles
        if triangles.shape[1] == 1:  # 0D - points
            if int_order == 1:
                gps.append(1.)
            else:
                raise TraitError('does not make sense')
        elif triangles.shape[1] == 2:  # 1D - lines
            if int_order == 1:
                for id in triangles:
                    r_pnt = points[ix_(id)]
                    J_det_ip = norm(r_pnt[1] - r_pnt[0]) * 0.5
                    gp = 2. * J_det_ip
                    gps.append(gp)
            elif int_order == 2:
                for id in triangles:
                    r_pnt = points[ix_(id)]
                    J_det_ip = norm(r_pnt[1] - r_pnt[0]) * 0.5
                    gps += J_det_ip , J_det_ip
            else:
                raise NotImplementedError
        elif triangles.shape[1] == 3:  # 2D - triangles
            if int_order == 1:
                for id in triangles:
                    r_pnt = points[ix_(id)]
                    J_det_ip = self._get_J_det_ip(r_pnt)
                    gp = 1. * J_det_ip
                    # print "gp ",gp
                    gps.append(gp)
            elif int_order == 2:
                raise NotImplementedError
            elif int_order == 3:
                for id in triangles:
                    r_pnt = points[ix_(id)]
                    J_det_ip = self._get_J_det_ip(r_pnt)
                    gps += -0.5625 * J_det_ip, \
                            0.52083333333333337 * J_det_ip, \
                            0.52083333333333337 * J_det_ip, \
                            0.52083333333333337 * J_det_ip
            elif int_order == 4:
                raise NotImplementedError
            elif int_order == 5:
                for id in triangles:
                    r_pnt = points[ix_(id)]
                    J_det_ip = self._get_J_det_ip(r_pnt)
                    gps += 0.225 * J_det_ip, 0.1323941527 * J_det_ip, \
                            0.1323941527 * J_det_ip, 0.1323941527 * J_det_ip, \
                            0.1259391805 * J_det_ip, 0.1259391805 * J_det_ip, \
                            0.1259391805 * J_det_ip
            else:
                raise NotImplementedError
        elif triangles.shape[1] == 4:  # 3D - tetrahedrons
           raise NotImplementedError
        else:
            raise TraitError('unsupported geometric form with %s nodes ' % triangles.shape[1])
        return array(gps, dtype='float_')

    def _get_J_det_ip(self, r_pnt):
        '''
        Helper function 
        just for 2D
        #todo:3D
        @param r_pnt:
        '''
        dNr_geo = self.dNr_geo_triangle
        return det(dot(dNr_geo, r_pnt[:, :2])) / 2.  # factor 2 due to triangular form

    dNr_geo_triangle = Property(Array(float))

    @cached_property
    def _get_dNr_geo_triangle(self):
        dN_geo = array([[-1., 1., 0.],
                        [-1., 0., 1.]], dtype='float_')
        return dN_geo

    def get_corr_pred(self, sctx, u, du, tn, tn1,
                      u_avg=None,
                      B_mtx_grid=None,
                      J_det_grid=None,
                      ip_coords=None,
                      ip_weights=None):
        '''
        Corrector and predictor evaluation.

        @param u current element displacement vector
        '''
        if J_det_grid == None or B_mtx_grid == None:
            X_mtx = sctx.X

        show_comparison = True
        if ip_coords == None:
            ip_coords = self.ip_coords
            show_comparison = False
        if ip_weights == None:
            ip_weights = self.ip_weights

        # ## Use for Jacobi Transformation

        n_e_dofs = self.n_e_dofs
        K = zeros((n_e_dofs, n_e_dofs))
        F = zeros(n_e_dofs)
        sctx.fets_eval = self
        ip = 0

        for r_pnt, wt in zip(ip_coords, ip_weights):
            # r_pnt = gp[0]
            sctx.r_pnt = r_pnt
# caching cannot be switched off in the moment
#            if J_det_grid == None:
#                J_det = self._get_J_det( r_pnt, X_mtx )
#            else:
#                J_det = J_det_grid[ip, ... ]
#            if B_mtx_grid == None:
#                B_mtx = self.get_B_mtx( r_pnt, X_mtx )
#            else:
#                B_mtx = B_mtx_grid[ip, ... ]
            J_det = J_det_grid[ip, ... ]
            B_mtx = B_mtx_grid[ip, ... ]

            eps_mtx = dot(B_mtx, u)
            d_eps_mtx = dot(B_mtx, du)
            sctx.mats_state_array = sctx.elem_state_array[ip * self.m_arr_size: (ip + 1) * self.m_arr_size]
            # print 'elem state ', sctx.elem_state_array
            # print 'mats state ', sctx.mats_state_array
            sctx.r_ls = sctx.ls_val[ip]
            sig_mtx, D_mtx = self.get_mtrl_corr_pred(sctx, eps_mtx, d_eps_mtx, tn, tn1)
            k = dot(B_mtx.T, dot(D_mtx, B_mtx))
            k *= (wt * J_det)
            K += k
            f = dot(B_mtx.T, sig_mtx)
            f *= (wt * J_det)
            F += f
            ip += 1

        return F, K

    def get_J_det(self, r_pnt, X_mtx, ls_nodes, ls_r):  # unified interface for caching
        return array(self._get_J_det(r_pnt, X_mtx), dtype='float_')

    def get_mtrl_corr_pred(self, sctx, eps_mtx, d_eps, tn, tn1):
        ls = sctx.r_ls
        if ls == 0. and self.mats_eval_disc:
            sig_mtx, D_mtx = self.mats_eval_disc.get_corr_pred(sctx, eps_mtx, d_eps, tn, tn1,)
        elif ls > 0. and self.mats_eval_pos:
            sig_mtx, D_mtx = self.mats_eval_pos.get_corr_pred(sctx, eps_mtx, d_eps, tn, tn1,)
        elif ls < 0. and self.mats_eval_neg:
            sig_mtx, D_mtx = self.mats_eval_neg.get_corr_pred(sctx, eps_mtx, d_eps, tn, tn1,)
        else:
            sig_mtx, D_mtx = self.mats_eval.get_corr_pred(sctx, eps_mtx, d_eps, tn, tn1,)
        return sig_mtx, D_mtx


if __name__ == '__main__':
    from ibvpy.fets.fets2D.fets2D4q import FETS2D4Q
    from ibvpy.core.scontext import SContext
    fets_ls_eval = FETSLSEval(parent_fets=FETS2D4Q())
    point_set = [array([[0, 0],
                       [10, 0],
                       [0, 10]])
                , array([[10, 0],
                       [10, 10],
                       [0, 10]])]
    print('triangulation')
    points, triangs = fets_ls_eval.get_triangulation(point_set)
    print('points ', points)
    print('triangles ', triangs)
