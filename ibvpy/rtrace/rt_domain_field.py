from traits.api import \
    Trait, Str, Enum, \
    Property, \
    TraitError

from traitsui.api \
    import VGroup

from numpy \
    import array, zeros, \
    float_

from ibvpy.plugins.mayavi_util.pipelines \
    import MVUnstructuredGrid

from traitsui.api import \
    View, Item, HSplit, VSplit

from ibvpy.rtrace.rt_domain import RTraceDomain

class RTraceDomainField(RTraceDomain):
    '''
    Trace encompassing the whole spatial domain.
    '''

    fets_eval = Property
    def _get_fets_eval(self):
        return self.sd.fets_eval

    var_eval = Property
    def _get_var_eval(self):
        return self.sd.dots.rte_dict.get(self.var, None)

    warp_var = Str('u')
    warp_var_eval = Property
    def _get_warp_var_eval(self):
        return self.sd.dots.rte_dict.get(self.warp_var, None)

#    idx      = Int(-1, auto_set = False, enter_set = True )
#    save_on  = Enum('update','iteration')
#    warp     = Bool(True)
#    warp_f   = Float(1.)
#    sctx     = Any
    position = Enum('int_pnts', 'nodes')

    def bind(self):
        '''
        Locate the evaluators
        '''
        pass

    def setup(self):
        '''
        Setup the spatial domain of the tracer
        '''
        if self.var_eval == None:
            self.skip_domain = True

    def add_current_values(self, sctx, U_k, *args, **kw):
        if self.position == 'nodes':
            self.current_values(sctx, U_k, *args, **kw)
        elif self.position == 'int_pnts':
            self.current_values_ip(sctx, U_k, *args, **kw)

    def current_values(self, sctx, U_k, *args, **kw):
        '''
        Invoke the evaluators in the current context for the specified control vector U_k.
        '''
        if self.skip_domain:
            return

        # Get the domain points
        # TODO - make this more compact. The element list is assumed to be uniform
        # so that all element arrays have the same shape. Thus, use slices and vectorized
        # evaluation to improve the performance

        sd = self.sd

        # loc_coords = self.dots.vtk_r_arr

#        try: loc_coords = self.fets_eval.vtk_r_arr
#        except AttributeError:
#            raise 'domain %s has no variable called %s' % ( sd.shape, self.var )

        # n_loc = loc_coords.shape[0]
        field = []
        state_array = self.sd.dots.state_array

        ip_offset = self.sd.dots.ip_offset

        for e_id, e in zip(self.sd.idx_active_elems, self.sd.elements):
            loc_coords = self.dots.get_vtk_r_arr(e_id)
            n_loc = loc_coords.shape[0]

            mats_arr_size = self.fets_eval.m_arr_size

            sctx.elem_state_array = state_array[ ip_offset[e_id] * mats_arr_size\
                                                : ip_offset[(e_id + 1)] * mats_arr_size ]  # differs from the homogenous case

            # setting the spatial context should be intermediated by the fets
            sctx.X = e.get_X_mtx()
            sctx.x = e.get_x_mtx()
            self.fets_eval.adjust_spatial_context_for_point(sctx)

            sctx.elem = e
            sctx.e_id = e_id
            field_entry = []

            ip_map = self.dots.get_vtk_pnt_ip_map(e_id)

            for i in range(n_loc):
                ip_id = ip_map[i]
                m_arr_size = self.fets_eval.m_arr_size
                sctx.mats_state_array = sctx.elem_state_array\
                                            [ip_id * m_arr_size: (ip_id + 1) * m_arr_size]
                sctx.loc = loc_coords[i]
                sctx.r_pnt = loc_coords[i]
                sctx.p_id = i
                val = self.var_eval(sctx, U_k, *args, **kw)
                field_entry.append(val)
            field += field_entry

        self.field_arr = array(field)

    def current_values_ip(self, sctx, U_k, *args, **kw):
        '''
        Invoke the evaluators in the current context for the specified control vector U_k.
        '''
        if self.var_eval == None:
            return
        # Get the domain points
        # TODO - make this more compact. The element list is assumed to be uniform
        # so that all element arrays have the same shape. Thus, use slices and vectorized
        # evaluation to improve the performance
        sd = self.sd
        sctx.fets_eval = self.fets_eval
        field = []
        dim_slice = self.fets_eval.dim_slice
        e_arr_size = self.fets_eval.get_state_array_size()
        state_array = self.sd.dots.state_array

        for e_id, e in zip(self.sd.idx_active_elems, self.sd.elements):

            sctx.elem_state_array = state_array[e_id * e_arr_size :\
                                                           (e_id + 1) * e_arr_size]
            sctx.X = e.get_X_mtx()
            sctx.x = e.get_x_mtx()
            sctx.elem = e
            sctx.e_id = e_id
            field_entry = []
            for i, ip in enumerate(self.fets_eval.ip_coords):
                m_arr_size = self.fets_eval.m_arr_size
                sctx.mats_state_array = sctx.elem_state_array\
                                            [i * m_arr_size: (i + 1) * m_arr_size]
                sctx.loc = ip
                sctx.r_pnt = ip
                sctx.p_id = i  # TODO:check this
                val = self.var_eval(sctx, U_k, *args, **kw)
                field_entry.append(val)
            field += field_entry
        self.field_arr = array(field)

    def add_current_displ(self, sctx, U_k):

#        if  self.var_eval == None or \
#            self.warp_var_eval == None:
#            return

        warp_var_eval = self.warp_var_eval
#        if self.position == 'int_pnts':
#            loc_coords = self.fets_eval.ip_coords
#        elif self.position == 'nodes':
#            loc_coords = self.dots.get_vtk_r_arr()

#        n_loc = loc_coords.shape[0]
        vector_field = []
        dim_slice = self.fets_eval.dim_slice
        for e_id, e in  zip(self.sd.idx_active_elems, self.sd.elements):
            loc_coords = self.dots.get_vtk_r_arr(e_id)
            n_loc = loc_coords.shape[0]
            sctx.X = e.get_X_mtx()
            sctx.x = e.get_x_mtx()
            sctx.elem = e
            sctx.e_id = e_id
            field_entry = []
            for i in range(n_loc):
                sctx.loc = loc_coords[i]
                sctx.p_id = i
                val = warp_var_eval(sctx, U_k)
                field_entry.append(val)
            vector_field += field_entry
        self.vector_arr = array(vector_field)

    #----------------------------------------------------------------------------
    # Visualization pipelines
    #----------------------------------------------------------------------------
    mvp_mgrid_geo = Trait(MVUnstructuredGrid)

    def _mvp_mgrid_geo_default(self):
        return MVUnstructuredGrid(name='Field %s' % self.var,
                                   points=self.rt_domain.vtk_r,
                                   cell_data=self.rt_domain.vtk_cell_data,
                                   scalars=self._get_scalars,
                                   vectors=self._get_vectors,
                                   tensors=self._get_tensors
                                   )

    def _get_warp_data(self):
        if self.vector_arr.shape[0] == 0:  # TODO:unifi check if all elems are deactivated
            return
        w_field = zeros((self.vector_arr.shape[0], 3), float_)
        w_field[:, self.fets_eval.dim_slice] = self.vector_arr[:, self.fets_eval.dim_slice]
        return w_field

    def _get_field_data(self):

        if self.var_eval == None:
            return

        shape = self.field_arr.shape[1:]
        # print "shape ", self.var, " ",shape
        if shape == ():  # TODO: subdomain where all elems are dactivated
            return
        # recognize data type (1,) = scalar
        if shape == (2, 2):  # 2D tensor - transform to 3d and flatten
            ff = zeros((self.field_arr.shape[0], 3, 3), float_)
            ff[:, :2, :2] = self.field_arr
            field = ff.reshape(self.field_arr.shape[0], 9)
        elif shape == (3, 3):  # 3D tensor - flatten
            field = self.field_arr.reshape(self.field_arr.shape[0], 9)
        elif shape == (2,):  # 2D vector - transform to 3d
            field = zeros((self.field_arr.shape[0], 3), float_)
            field[:, :2] = self.field_arr
        elif shape == (1,) or shape == (3,):
            field = self.field_arr  # is scalar or 3D vector  does not need treatment
        else:
            raise TraitError('wrong field format of tracer %s: %s' % (self.var, shape))

        return field

    def timer_tick(self, e=None):
        # self.changed = True
        pass

    def clear(self):
        pass

    view = View(HSplit(VSplit (VGroup('var', 'idx'),
                                  VGroup('record_on', 'clear_on'),
                                   Item('refresh_button', show_label=False),
                                           ),
                                           ),
                                    resizable=True)

if __name__ == '__main__':

    # Define a mesh domain adaptor as a cached property to
    # be constracted on demand

#    mgrid_adaptor = MeshGridAdaptor( n_nodal_dofs = 3,
#                                 # NOTE: the following properties must be defined and
#                                 # must correspond to the used element formulation
#                                 n_e_nodes_geo = (1,1,1),
#                                 n_e_nodes_dof = (1,1,1),
#                                 node_map_geo = [0,1,3,2,4,5,7,6],
#                                 node_map_dof = [0,1,3,2,4,5,7,6])

    # Define a mesh domain adaptor as a cached property to
    # be constracted on demand
    #    mgrid_adaptor = MeshGridAdaptor( n_nodal_dofs = 2,
    #                                     # NOTE: the following properties must be defined and
    #                                     # must correspond to the used element formulation
    #                                     n_e_nodes_geo = (1,1,0),
    #                                     n_e_nodes_dof = (3,3,0),
    #                                     node_map_geo = [0,1,3,2],
    #                                     node_map_dof = [0,3,15,12, 1,2,7,11, 14,13,8,4, 5,6,9,10] )

    #
    # Define a mesh domain adaptor as a cached property to
    # be constracted on demand
    #    mgrid_adaptor = MeshGridAdaptor( n_nodal_dofs = 2,
    #                                     # NOTE: the following properties must be defined and
    #                                     # must correspond to the used element formulation
    #                                     n_e_nodes_geo = (1,1,0),
    #                                     n_e_nodes_dof = (1,1,0),
    #                                     node_map_geo = [0,1,3,2],
    #                                     node_map_dof = [0,1,3,2] )

    # Discretization
    #
    grid_domain = MGridDomain(lengths=(3., 3., 0.),
                               shape=(1, 1, 1),
                               adaptor=mgrid_adaptor)


#    grid_domain.configure_traits()
    grid_domain.elements

#    grid_domain.changed = True

    from ibvpy.plugins.ibvpy_app import IBVPyApp
    ibvpy_app = IBVPyApp(ibv_resource=grid_domain)
    ibvpy_app.main()

class RTraceSubDomainField(RTraceDomainField):
    recursion = False
