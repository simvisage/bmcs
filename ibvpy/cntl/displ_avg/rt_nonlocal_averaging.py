from math import fabs
from time import time

from ibvpy.rtrace.rt_domain import RTraceDomain
from ibvpy.rtrace.rt_domain_list import RTraceDomainList
from numpy import array, hstack, zeros, sum as np_sum, linalg, \
    arange, tile, zeros_like, float_, frompyfunc, linspace
from scipy.sparse import coo_matrix
from scipy.spatial.distance import cdist
from traits.api import \
    Bool, Float, HasTraits, \
    Instance, Int, \
    List, \
    Property, cached_property


class AveragingFunction(HasTraits):
    radius = Float(.2)
    correction = Bool(True)

    def get_value(self, dist):
        raise NotImplementedError

    def plot(self, axes, center=0):
        vfn = frompyfunc(self.get_value, 1, 1)
        xdata = linspace(-self.radius, self.radius, 40) * 1.2
        ydata = vfn(xdata)
        axes.plot(xdata + center, ydata, color='brown')


class QuarticAF(AveragingFunction):

    def get_value(self, dist):
        if fabs(dist) - self.radius >= 0:
            return 0.
        else:
            return ((1 - (dist / self.radius) ** 2) ** 2) ** 2


class LinearAF(AveragingFunction):

    def get_value(self, dist):
        if fabs(dist) - self.radius >= 0:
            return 0.
        else:
            return (1. / self.radius) * (1 - dist / self.radius)


class RTUAvg(RTraceDomain):

    avg_fn = Instance(AveragingFunction)

    n_dofs = Int(0)

    def get_C_data(self):

        a_fn = frompyfunc(self.avg_fn.get_value, 1, 1)

        dim = self.fets_eval.n_nodal_dofs  # TODO:find better way

        #@todo: hack, works just with one fe_grid,
        # make it work for arbitrary number of fe_grids
        #
        #fe_subgrid = self.sd.fe_subgrids[0]
        sd = self.sd
        fe_subgrid = sd.fe_subgrids[0]
        n_dims = fe_subgrid.geo_grid.n_dims

        X_pnt = fe_subgrid.dof_grid.cell_grid.point_X_arr

        ip_coords = self.fets_eval.ip_coords
        n_ip = len(ip_coords)
        ip_weights = self.fets_eval.ip_weights
        data, row, col = [], [], []

        # if the same integration scheme is used the N matrices can be evaluated
        # just once - shape(n_ip,dims, dofs)
        #
        e_ip_N_mtx = array([self.fets_eval.get_N_mtx(r_pnt)
                            for r_pnt in ip_coords])
        n_elem_dofs = e_ip_N_mtx.shape[2]

        # Loop over all nodes
        for i, x_pnt in enumerate(X_pnt):

            # print 'x_pnt ', x_pnt

            # Use LS to find the elements inside the radius
            # TODO: make it for all subgrids
            # TODO:check leaking
            #
            # - check the dimensionality of sd and use the corresponding dimension of level set function
            if n_dims == 1:
                level_set = fe_subgrid['(X-%(x)f)**2 - %(r)f**2'
                                       % {'x': x_pnt[0], 'r': self.avg_fn.radius}]
            elif n_dims == 2:
                level_set = fe_subgrid['(X-%(x)f)**2 + (Y-%(y)f)**2 - %(r)f**2'
                                       % {'x': x_pnt[0], 'y': x_pnt[1], 'r':self.avg_fn.radius}]
            elif n_dims == 3:
                level_set = fe_subgrid['(X-%(x)f)**2 + (Y-%(y)f)**2 + (Z-%(z)f)**2 - %(r)f**2'
                                       % {'x': x_pnt[0], 'y': x_pnt[1], 'z': x_pnt[2], 'r':self.avg_fn.radius}]

            # print '(X-%(x)f)**2 - %(r)f**2' % { 'x' : x_pnt[0], 'r' :
            # self.avg_fn.radius }

            active_elems_orig = hstack((level_set.elems, level_set.neg_elems))
            # remove the explicitly deactivated elements from the integration
            # zone
            active_elems = active_elems_orig[fe_subgrid.activation_map[active_elems_orig]]

            n_a_elems = active_elems.size
            if n_a_elems == 0:
                'WARNING - Radius too small!'

            # print 'active_elements', active_elems

            # Generate the IP coords for all active elems
            # element transformation have to be used due distorted meshes
            #
            elems_ip_coords = zeros((n_a_elems, n_ip, dim), dtype=float)
            elems_dof_map = zeros((n_a_elems, n_elem_dofs), dtype=int)
            for j, e_id in enumerate(active_elems):
                X_mtx = fe_subgrid.geo_grid.elem_X_map[e_id, :]
                elems_ip_coords[j, :] = \
                    self.fets_eval.get_vtk_r_glb_arr(X_mtx, ip_coords)
                elems_dof_map[j, :] = fe_subgrid.dof_grid.elem_dof_map[e_id, :]

            # print 'elem_ip_coords', elems_ip_coords
            # Get the distance between the current ip and the interacting ips
            #
            arm = elems_ip_coords - x_pnt[None, None, :]
            dist = cdist(x_pnt[None, :], elems_ip_coords.reshape(n_a_elems * n_ip, dim)).\
                reshape(n_a_elems, n_ip)

            # print 'distances', dist
            # Value of the weighting function
            #
            alpha = a_fn(dist)   # ( elems,n_ip )
            # ( elems, n_ip )
            J_det = fe_subgrid.dots.J_det_grid[active_elems]

            values = (ip_weights.T * J_det) * alpha  # ( elems, n_ip )

            r_00 = values.sum()

            if self.avg_fn.correction:
                r_01 = np_sum(np_sum(values[..., None] * arm, axis=1), axis=0)
                # outer product (elems,n_ip, dims, dims)
                outer_p = arm[..., None, :] * arm[..., None]

                # TODO:is summing over more dims ad once posible?
                R_11 = np_sum(
                    np_sum(values[..., None, None] * outer_p, axis=1), axis=0)
                # evaluate the correcting factors
                A = zeros((dim + 1, dim + 1))
                A[0, 0] = r_00
                A[0, 1:] = r_01
                A[1:, 0] = r_01
                A[1:, 1:] = R_11
                b = zeros((dim + 1), dtype=float)
                b[0] = 1.
                try:
                    params = linalg.solve(A, b)
                except linalg.LinAlgError:
                    print('x_pnt\n', x_pnt)
                    raise ValueError(
                        'Integration radius too small for averaging')
                # print 'check ',dot(A,params)
                p0 = params[0]
                p1 = params[1:]
                # print 'p0', p0
                # print 'p1', p1
                m_correction = np_sum(
                    p1[None, None, :] * arm, axis=2)  # (elem,n_ip)
                c_elem = np_sum((values * (p0 + m_correction))[..., None, None] *
                                e_ip_N_mtx[None, ...], axis=1)
            else:
                p0 = 1. / r_00

                # print 'p0'
                # print p0

                # print 'e_ip_N_mtx'
                # print e_ip_N_mtx[None, ...]

                c_elem = np_sum(
                    (values * p0)[..., None, None] * e_ip_N_mtx[None, ...], axis=1)

            # print 'c_elem'
            # print c_elem.flatten()

            data.append(c_elem.flatten())
            col.append(tile(elems_dof_map, dim).flatten())
            row.append(((i * dim) + zeros_like(c_elem) +
                        arange(dim, dtype=int)[None, :, None]).flatten())

        return data, col, row

    C_mtx = Property

    @cached_property
    def _get_C_mtx(self):
        n_dofs = self.n_dofs
        _data, _col, _row = self.get_C_data()
        return coo_matrix((array(hstack(_data), dtype='float_'),
                           (array(hstack(_row), dtype='float_'),
                            array(hstack(_col), dtype='float_'))),
                          shape=(n_dofs, n_dofs), dtype=float_
                          ).tocsr()

    def __call__(self, U_k):
        return [], {'eps_avg': self.C_mtx * U_k.T}


class RTNonlocalAvg(RTraceDomainList):
    '''
    response tracer for nonlocal averaging of the
    displacement field
    TODO: set the reference to the spatial domain (bind method, setup not needed?)
    '''

    avg_fn = Instance(AveragingFunction)

    def _avg_fn_default(self):
        return QuarticAF(radius=0.21)

    verbose_time = Bool(False)

    subfields = Property(List)

    @cached_property
    def _get_subfields(self):
        # construct the RTraceDomainFields
        #
        return [RTUAvg(sd=subdomain,
                       avg_fn=self.avg_fn) for subdomain in self.sd.nonempty_subdomains]

    C_mtx = Property

    @cached_property
    def _get_C_mtx(self):
        #--------------
        # Averaging
        #------------------
        n_dofs = self.sd.n_dofs

        if self.verbose_time:
            t1 = time()

        data = []
        col = []
        row = []
        for sf in self.subfields:
            _data, _col, _row = sf.get_C_data()
            data += _data
            col += _col
            row += _row

        data = hstack(data)
        row = hstack(row)
        col = hstack(col)

        if self.verbose_time:
            t2 = time()
            diff = t2 - t1
            print("Averaging Matrix: %8.2f sec" % diff)

        return coo_matrix((data, (row, col)),
                          shape=(n_dofs, n_dofs), dtype=float_).tocsr()

    def __call__(self, U_k):
        return [], {'eps_avg': self.C_mtx * U_k.T}
