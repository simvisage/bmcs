from traits.api import \
    Array, Bool, Enum, Float, HasTraits, HasStrictTraits, \
    Instance, Int, Trait, Str, Enum, \
    Callable, List, TraitDict, Any, Range, \
    Delegate, Event, on_trait_change, Button, \
    Interface, WeakRef, Property, cached_property, Tuple, \
    Dict

from traitsui.api \
    import Item, View, HGroup, ListEditor, VGroup, \
    HSplit, Group, Handler, VSplit, TableEditor, ListEditor

from traitsui.menu \
    import NoButtons, OKButton, CancelButton, \
    Action

from traitsui.ui_editors.array_view_editor \
    import ArrayViewEditor

from traitsui.table_column \
    import ObjectColumn, ExpressionColumn

from traitsui.table_filter \
    import TableFilter, RuleTableFilter, RuleFilterTemplate, \
    MenuFilterTemplate, EvalFilterTemplate, EvalTableFilter

from numpy \
    import ix_, mgrid, array, arange, c_, newaxis, setdiff1d, zeros, \
    float_, repeat, hstack, ndarray, append

from ibvpy.plugins.mayavi_util.pipelines \
    import MVUnstructuredGrid, MVPolyData

from ibvpy.core.rtrace \
    import RTrace

# tvtk related imports
#
from traitsui.api import \
    View, Item, HSplit, VSplit
from tvtk.api import \
    tvtk
from ibvpy.core.i_sdomain import \
    ISDomain

from ibvpy.core.sdomain import \
    SDomain

import os


class RTraceDomain(HasTraits):

    '''
    Trace encompassing the whole spatial domain.
    '''
    label = Str('RTraceDomainField')
    fets_eval = Delegate('sd')
    dots = Delegate('sd')
    position = Enum('nodes', 'int_pnts')
    sd = WeakRef(ISDomain)

    point_offset = Int
    cell_offset = Int

    # Tag that can be used to skip the domain when gathering the vtk data
    #
    skip_domain = Bool(False)

    #-------------------------ts_eval-----------------------------------------
    # Visualization pipelines
    #-------------------------------------------------------------------------
    mvp_mgrid_geo = Trait(MVUnstructuredGrid)

    def _mvp_mgrid_geo_default(self):
        return MVUnstructuredGrid(name='Response tracer mesh',
                                  warp=False,
                                  warp_var=''
                                  )

    def redraw(self):
        '''
        '''
        # self.mvp_mgrid_geo.redraw()

        print('REDRAWING', self.sd.name)

        self.mvp_mgrid_geo.rebuild_pipeline(self.vtk_node_structure)

    vtk_node_structure = Property(Instance(tvtk.UnstructuredGrid),
                                  depends_on='sd.changed_structure')

    @cached_property
    def _get_vtk_node_structure(self):
        self.position = 'nodes'
        return self.vtk_structure

    vtk_ip_structure = Property(Instance(tvtk.UnstructuredGrid),
                                depends_on='sd.changed_structure')

    @cached_property
    def _get_vtk_ip_structure(self):
        self.position = 'int_pnts'
        return self.vtk_structure

    vtk_structure = Property(Instance(tvtk.UnstructuredGrid))

    def _get_vtk_structure(self):
        ug = tvtk.UnstructuredGrid()

        if self.skip_domain:
            return ug

        cell_array, cell_offsets, cell_types = self.vtk_cell_data

        n_cells = cell_types.shape[0]
        ug.points = self.vtk_X

        vtk_cell_array = tvtk.CellArray()
        vtk_cell_array.set_cells(n_cells, cell_array)
        ug.set_cells(cell_types, cell_offsets, vtk_cell_array)
        return ug


# vtk_X = Property(Array) #TODO: cleanup
#    def _get_vtk_X(self):
#        '''Get the discretization points based on the fets_eval
#        associated with the current domain.
#        '''
#        if self.position == 'int_pnts':
#            ip_arr = self.fets_eval.ip_coords
#
#        pts = []
#        dim_slice = self.fets_eval.dim_slice
#        for e in self.sd.elements:
#            X = e.get_X_mtx()
#            if dim_slice:
#                X = X[:,dim_slice]
#                if self.position == 'int_pnts':
#                    ip_arr = ip_arr[:,dim_slice]
#            if self.position == 'nodes':
#                pts += list( self.fets_eval.get_vtk_r_glb_arr( X ) )
#            elif self.position == 'int_pnts':
#                pts += list( self.fets_eval.get_vtk_r_glb_arr( X, ip_arr) )
#        pts_array = array(pts, dtype = 'float_' )
#        return pts_array

    vtk_X = Property(Array)  # TODO: cleanup

    def _get_vtk_X(self):
        return self.dots.get_vtk_X(self.position)

#    debug_cell_data = Bool(True)

# vtk_cell_data = Property(Array, depends_on = 'point_offset,cell_offset' )#TODO:check the dependencies
#    def _get_vtk_cell_data(self):
#
#        if self.position == 'nodes':
#            subcell_offsets, subcell_lengths, subcells, subcell_types = self.dots.vtk_node_cell_data
#        elif self.position == 'int_pnts':
#            subcell_offsets, subcell_lengths, subcells, subcell_types = self.fets_eval.vtk_ip_cell_data
#
#        self.debug_cell_data = True
#
#        if self.debug_cell_data:
#            print 'subcell_offsets'
#            print subcell_offsets
#            print 'subcell_lengths'
#            print subcell_lengths
#            print 'subcells'
#            print subcells
#            print 'subcell_types'
#            print subcell_types
#
#        n_subcells = subcell_types.shape[0]
#        n_cell_points = self.n_cell_points
#        subcell_size = subcells.shape[0] + n_subcells
#
#        if self.debug_cell_data:
#            print 'n_cell_points', n_cell_points
#            print 'n_cells', self.n_cells
#
#        vtk_cell_array = zeros( (self.n_cells, subcell_size), dtype = int )
#
#        idx_cell_pnts = repeat( True, subcell_size )
#
#        if self.debug_cell_data:
#            print 'idx_cell_pnts'
#            print idx_cell_pnts
#
#        idx_cell_pnts[ subcell_offsets ] = False
#
#        if self.debug_cell_data:
#            print 'idx_cell_pnts'
#            print idx_cell_pnts
#
#        idx_lengths = idx_cell_pnts == False
#
#        if self.debug_cell_data:
#            print 'idx_lengths'
#            print idx_lengths
#
#        point_offsets = arange( self.n_cells ) * n_cell_points
#        point_offsets += self.point_offset
#
#        if self.debug_cell_data:
#            print 'point_offsets'
#            print point_offsets
#
#        vtk_cell_array[:,idx_cell_pnts] = point_offsets[:,None] + subcells[None,:]
#        vtk_cell_array[:,idx_lengths] = subcell_lengths[None,:]
#
#        if self.debug_cell_data:
#            print 'vtk_cell_array'
#            print vtk_cell_array
#
#        active_cells = self.sd.idx_active_elems
#
#        if self.debug_cell_data:
#            print 'active cells'
#            print active_cells
#
#        cell_offsets = active_cells * subcell_size
#        cell_offsets += self.cell_offset
#        vtk_cell_offsets = cell_offsets[:,None] + subcell_offsets[None,:]
#
#        if self.debug_cell_data:
#            print 'vtk_cell_offsets'
#            print vtk_cell_offsets
#
#        vtk_cell_types = zeros( self.n_cells * n_subcells, dtype = int ).reshape( self.n_cells,
#                                                                                  n_subcells )
#        vtk_cell_types += subcell_types[None,:]
#
#        if self.debug_cell_data:
#            print 'vtk_cell_types'
#            print vtk_cell_types
#
# return vtk_cell_array.flatten(), vtk_cell_offsets.flatten(),
# vtk_cell_types.flatten()

    # TODO:check the dependencies
    vtk_cell_data = Property(
        Array, depends_on='point_offset,cell_offset,sd.changed_structure')

    def _get_vtk_cell_data(self):

        #        cell_array, cell_offsets, cell_types = self.dots.get_vtk_cell_data(self.position)
        #        cell_array += self.point_offset
        #        cell_offsets += self.cell_offset
        return self.dots.get_vtk_cell_data(self.position, self.point_offset, self.cell_offset)

    # number of points
    n_points = Property(Int, depends_on='vtk_X, sd.changed_structure')

    @cached_property
    def _get_n_points(self):
        if self.skip_domain:
            return 0
        return self.vtk_X.shape[0]


#    debug_cell_data = Bool( False )


#    n_cells = Property( Int )
#    def _get_n_cells(self):
#        '''Return the total number of cells'''
#        return self.sd.n_active_elems
#
#    n_cell_points = Property( Int )
#    def _get_n_cell_points(self):
#        '''Return the number of points defining one cell'''
#        return self.fets_eval.n_vtk_r

    def clear(self):
        pass

    view = View(HSplit(VSplit(VGroup(Item('refresh_button', show_label=False),
                                     ),
                              ), ),
                resizable=True)
