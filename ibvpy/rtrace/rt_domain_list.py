import os

from ibvpy.core.i_sdomain import \
    ISDomain
from ibvpy.core.rtrace import RTrace
from ibvpy.core.sdomain import \
    SDomain
from ibvpy.plugins.mayavi_util.pipelines import \
    MVUnstructuredGrid, MVPolyData
from numpy import ix_, mgrid, array, arange, c_, newaxis, setdiff1d, zeros, \
    float_, vstack, hstack, repeat
from traits.api import \
    Array, Bool, Enum, Float, HasTraits, HasStrictTraits, \
    Instance, Int, Trait, Str, Enum, \
    Callable, List, TraitDict, Any, Range, \
    Delegate, Event, on_trait_change, Button, \
    Interface, WeakRef, Property, cached_property, Tuple, \
    Dict
from traitsui.api import Item, View, HGroup, ListEditor, VGroup, \
    HSplit, Group, Handler, VSplit, TableEditor, ListEditor
from traitsui.api import View, Item, HSplit, VSplit
from traitsui.menu import NoButtons, OKButton, CancelButton, \
    Action
from tvtk.api import tvtk
from tvtk.api import tvtk

from traitsui.table_column \
    import ObjectColumn, ExpressionColumn
from traitsui.table_filter \
    import TableFilter, RuleTableFilter, RuleFilterTemplate, \
    MenuFilterTemplate, EvalFilterTemplate, EvalTableFilter
from traitsui.ui_editors.array_view_editor \
    import ArrayViewEditor


# tvtk related imports
#
class RTraceDomainList(HasTraits):

    label = Str('RTraceDomainField')
    sd = WeakRef(ISDomain)
    position = Enum('nodes', 'int_pnts')
    subfields = List

    def redraw(self):
        '''Delegate the calculation to the pipeline
        '''
        # self.mvp_mgrid_geo.redraw() # 'label_scalars')
        self.mvp_mgrid_geo.rebuild_pipeline(self.vtk_node_structure)

    vtk_node_structure = Property(Instance(tvtk.UnstructuredGrid))
    # @cached_property

    def _get_vtk_node_structure(self):
        self.position = 'nodes'
        return self.vtk_structure

    vtk_ip_structure = Property(Instance(tvtk.UnstructuredGrid))
    # @cached_property

    def _get_vtk_ip_structure(self):
        self.position = 'int_pnts'
        return self.vtk_structure

    vtk_structure = Property(Instance(tvtk.UnstructuredGrid))

    def _get_vtk_structure(self):
        ug = tvtk.UnstructuredGrid()
        cell_array, cell_offsets, cell_types = self.vtk_cell_data
        n_cells = cell_types.shape[0]
        ug.points = self.vtk_X
        vtk_cell_array = tvtk.CellArray()
        vtk_cell_array.set_cells(n_cells, cell_array)
        ug.set_cells(cell_types, cell_offsets, vtk_cell_array)
        return ug

    vtk_X = Property

    def _get_vtk_X(self):
        point_arr_list = []
        for sf in self.subfields:
            if sf.skip_domain:
                continue
            sf.position = self.position
            sf_vtk_X = sf.vtk_X
            if sf_vtk_X.shape[0] == 0:  # all elem are deactivated
                continue
            point_arr_list.append(sf_vtk_X)
        if len(point_arr_list) > 0:
            # print 'point_arr_list ', point_arr_list
            return vstack(point_arr_list)
        else:
            return zeros((0, 3), dtype='float_')

    # point offset to use when more fields are patched together within
    # RTDomainList

    point_offset = Int(0)

    # cell offset to use when more fields are patched together within
    # RTDomainList

    cell_offset = Int(0)

    vtk_cell_data = Property

    def _get_vtk_cell_data(self):
        cell_array_list = []
        cell_offset_list = []
        cell_types_list = []
        point_offset = self.point_offset
        cell_offset = self.cell_offset
        for sf in self.subfields:
            if sf.skip_domain:
                continue
            sf.position = self.position
            sf.point_offset = point_offset
            sf.cell_offset = cell_offset
            cell_array, cell_offsets, cell_types = sf.vtk_cell_data
            cell_array_list.append(cell_array)
            cell_offset_list.append(cell_offsets)
            cell_types_list.append(cell_types)
            point_offset += sf.n_points
            cell_offset += cell_array.shape[0]
        if len(cell_array_list) > 0:
            cell_array = hstack(cell_array_list)
            cell_offsets = hstack(cell_offset_list)
            cell_types = hstack(cell_types_list)
        else:
            cell_array = array([], dtype='int_')
            cell_offsets = array([], dtype='int_')
            cell_types = array([], dtype='int_')
        return (cell_array, cell_offsets, cell_types)

    #-------------------------------------------------------------------------
    # Visualization pipelines
    #-------------------------------------------------------------------------

    mvp_mgrid_geo = Trait(MVUnstructuredGrid)

#    def _mvp_mgrid_geo_default(self):
#        return MVUnstructuredGrid( name = 'Response tracer mesh',
#                                   points = self.vtk_r,
#                                   cell_data = self.vtk_cell_data,
#                                    )

    def _mvp_mgrid_geo_default(self):
        return MVUnstructuredGrid(name='Response tracer mesh',
                                  warp=False,
                                  warp_var=''
                                  )

    view = View(resizable=True)
