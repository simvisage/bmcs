
from math import \
    pow, fabs
import time

from ibvpy.api import DOTSEval
from ibvpy.core.i_tstepper_eval import \
    ITStepperEval
from ibvpy.fets.fets_eval import \
    IFETSEval, FETSEval
from numpy import \
    array, zeros, int_, float_, ix_, dot, linspace, hstack, vstack, arange, \
    intersect1d, rank
from numpy.linalg import \
    solve
from scipy.linalg import \
    norm
from traits.api import \
    Array, Bool, Callable, Enum, Float, HasTraits, Interface, \
    Int, Trait, Str, Enum, Callable, List, TraitDict, Any, \
    on_trait_change, Tuple, WeakRef, Delegate, Property, cached_property, \
    Instance, Dict


class AveragingFunction(HasTraits):
    Radius = Float(0.6)

    def set_center(self, X):
        self.center = X

    def get_value(self, X):
        raise NotImplementedError


class QuarticAF(AveragingFunction):
    #R = Float(2.)
    def get_value(self, X):
        value = 1 - (norm(self.center - X) / self.Radius) ** 2
        if value > 0:
            return value ** 2
        else:
            return 0


class LinearAF(AveragingFunction):
    #R = Float(2.)

    def get_value(self, r):
        value = (1. / self.Radius) * (1 - norm(self.center - r) / self.Radius)
        if value > 0.:
            return value
        else:
            return 0


class DOTSEvalAvg(DOTSEval):
    '''
    Domain with uniform FE-time-step-eval.
    '''

    correction = Bool('True')
    avg_function = Instance(AveragingFunction)

    def setup(self, sctx):
        self.domain = sctx.sdomain

        ndofs = self.domain.n_dofs
        self.K = zeros((ndofs, ndofs), float_)
        self.F_int = zeros(ndofs, float_)

        e_arr_size = self.e_arr_size

        # Run the setup of sub-evaluator
        #
        for elem in sctx.sdomain.elements:
            sctx.elem = elem
            id = elem.id_number
            sctx.elem_state_array = sctx.state_array[id *
                                                     e_arr_size: (id + 1) * e_arr_size]
            self.fets_eval.setup(sctx)

        #--------------
        # Averaging
        #------------------
        t1 = time.time()
        self.C = zeros((ndofs, ndofs), float_)

        #n_nodes = sctx.sdomain.n_nodes_dof
        X_nodes = sctx.sdomain.get_X_mtx_dof()
        node_list = sctx.sdomain.nodes_dof
        n_nodal_dofs = sctx.sdomain.n_nodal_dofs
        # Loop over all nodes
        for n_nod_i in node_list:
            active_nodes = []
            active_elems = []
            center = X_nodes[n_nod_i.id_number]
            self.avg_function.set_center(center)
            # Loop to find the nodes inside the radius
            for n_nod_j in node_list:
                if norm(center - X_nodes[n_nod_j.id_number]) < self.avg_function.Radius:
                    active_nodes.append(n_nod_j.id_number)
            # Loop to create the list of active elements
            for elem_i in sctx.sdomain.elements:
                node_map = elem_i.get_node_map()
                intersection = intersect1d(node_map, active_nodes)
                if len(intersection) != 0:
                    active_elems.append(elem_i)
            # Initializing the variables
            r_00 = []
            dim = self.fets_eval.n_nodal_dofs  # @tTODO
            if self.correction == True:
                # initialisation of all the values that are calculated on the
                # fly
                r_00_tot = zeros((1, 1), float_)
                r_10_tot = zeros((1, dim), float_)
                R_11_tot = zeros((dim, dim), float_)
            else:
                r_tot = 0
#
            # Loop over the elements in the active alements list
            for elem_j in active_elems:
                sctx.elem = elem_j
                # Global coordinates of the element's nodes
                sctx.X = elem_j.get_X_mtx()
                sctx.x = elem_j.get_x_mtx()
                if self.fets_eval.dim_slice:
                    X_mtx = elem_j.get_X_mtx()[:, self.fets_eval.dim_slice]
                else:
                    X_mtx = elem_j.get_X_mtx()
                # Loop over Gauss points of the element
                for gp in self.fets_eval.gp_list:
                    r_pnt = gp[0]
                    x_pnt = dot(self.fets_eval.get_N_geo_mtx(r_pnt), sctx.X)
                    # Jacobian determinant in Gauss point
                    J_det = self.fets_eval._get_J_det(r_pnt, X_mtx)
                    # value of bell function centered in node evaluated in
                    # Gauss point
                    alpha = self.avg_function.get_value(x_pnt)
                    # Add value to the memory list
                    r_00.append(gp[1] * J_det * alpha)
                    if self.correction == True:
                        #                   #Evaluate r_00, r_10 and R_11
                        r_00_tot += gp[1] * J_det * alpha
                        r_10_tot += gp[1] * J_det * alpha * (x_pnt - self.avg_function.center)[
                            :, self.fets_eval.dim_slice]
                        R_11_tot += gp[1] * J_det * alpha * \
                            dot(((x_pnt - self.avg_function.center)[:, self.fets_eval.dim_slice]).T,
                                (x_pnt - self.avg_function.center)[:, self.fets_eval.dim_slice])
                    else:
                        r_tot += gp[1] * J_det * alpha

            if self.correction == True:
                # Solve the equation to get p_0 and p_1
                A = vstack([hstack([r_00_tot, r_10_tot]),
                            hstack([r_10_tot.T, R_11_tot])])
                b = zeros((dim + 1), float_)
                b[0] = 1.
                s = solve(A, b)
                p_0 = s[0]
                p_1 = array([s[1:]])
            else:
                # if we don't want the correcting factors the weight
                # functionhas to be normed
                p_0 = 1. / r_tot
                p_1 = zeros(dim)

            # Loop over elements and over Gauss points..
            for elem_j in active_elems:
                c = 0
                sctx.X = elem_j.get_X_mtx()
                if self.fets_eval.dim_slice:
                    X_mtx = elem_j.get_X_mtx()[:, self.fets_eval.dim_slice]
                else:
                    X_mtx = elem_j.get_X_mtx()
                ix = elem_j.get_dof_map()
                # Loop over the Gauss points
                for gp in self.fets_eval.gp_list:
                    r_pnt = gp[0]
                    x_pnt = dot(self.fets_eval.get_N_geo_mtx(r_pnt), sctx.X)
                    #.. to get the N-matrix and calculate c
                    N_mtx = self.fets_eval.get_N_mtx(r_pnt)
                    c += r_00[0] * (p_0 + dot(p_1, ((x_pnt - self.avg_function.center)
                                                    [:, self.fets_eval.dim_slice]).T)) * N_mtx
                    r_00.pop(0)
                # Assembling the matrix
                for i in range(0, n_nodal_dofs):
                    self.C[n_nod_i.dofs[i], ix_(ix)] += c[i, :]
        t2 = time.time()
        diff = t2 - t1
        print("Averaging Matrix: %8.2f sec" % diff)
        # print "C ", self.C

    def get_corr_pred(self, sctx, u, du, tn, tn1):

        self.K[:, :] = 0.0
        self.F_int[:] = 0.0
        e_arr_size = self.e_arr_size
        u_avg = dot(self.C, u)
        # print "u", u
        # print "u_avg", u_avg

        for elem in sctx.sdomain.elements:
            e_id = elem.id_number
            ix = elem.get_dof_map()
            sctx.elem = elem
            sctx.elem_state_array = sctx.state_array[e_id *
                                                     e_arr_size: (e_id + 1) * e_arr_size]
            sctx.X = elem.get_X_mtx()
            f, k = self.fets_eval.get_corr_pred(
                sctx, u[ix_(ix)], du[ix_(ix)], tn, tn1, u_avg[ix_(ix)])
            self.K[ix_(ix, ix)] += k
            self.F_int[ix_(ix)] += f

        return self.F_int, self.K
