import os

from ibvpy.api import RTrace
from ibvpy.core.i_sdomain import \
    ISDomain
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
    Dict, Any, Directory
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

from .rt_domain_field import RTraceDomainField
from .rt_domain_list import RTraceDomainList


# tvtk related imports
#
class RTraceDomainListField(RTrace, RTraceDomainList):

    #    sd = Instance( SDomain )
    #
    #    rt_domain = Property
    #    def _get_rt_domain(self):
    #        return self.sd.rt_bg_domain

    label = Str('RTraceDomainField')
    var = Str('')
    idx = Int(-1, enter_set=True, auto_set=False)
    position = Enum('nodes', 'int_pnts')

    save_on = Enum('update', 'iteration')
    warp = Bool(False)
    warp_f = Float(1.)
    warp_var = Str('u')

    def bind(self):
        '''
        Locate the evaluators
        '''

    def setup(self):
        '''
        Setup the spatial domain of the tracer
        '''
        for sf in self.subfields:
            sf.setup()

    subfields = Property(depends_on='sd.changed_structure')

    @cached_property
    def _get_subfields(self):
        # construct the RTraceDomainFields
        #
        return [RTraceDomainField(var=self.var,
                                  warp_var=self.warp_var,
                                  idx=self.idx,
                                  position=self.position,
                                  save_on=self.save_on,
                                  warp=self.warp,
                                  warp_f=self.warp_f,
                                  sd=subdomain) for subdomain in self.sd.nonempty_subdomains]

    # TODO: should depend on the time step
    vtk_data = Property(
        Instance(tvtk.UnstructuredGrid), depends_on='write_counter')
    # @cached_property

    def _get_vtk_data(self):
        if self.position == 'nodes':
            ug = self.vtk_node_structure
            # vtk_r = self.vtk_node_points
            # vtk_cell_data = self.vtk_node_cell_data
        elif self.position == 'int_pnts':
            ug = self.vtk_ip_structure
            # vtk_r = self.custom_vtk_r
            # vtk_cell_data = self.custom_vtk_cell_data

        # ug = self.vtk_structure

        field_arr = tvtk.DoubleArray(name=self.name)
        field_arr.from_array(self._get_field_data())  # TODO:naming
        ug.point_data.add_array(field_arr)
        # add data for warping
        if self.warp:
            warp_arr = tvtk.DoubleArray(name=self.warp_var)
            warp_arr.from_array(self._get_warp_data())
            ug.point_data.add_array(warp_arr)
        return ug

    def redraw(self):
        '''Delegate the calculation to the pipeline
        '''
        # self.mvp_mgrid_geo.redraw() # 'label_scalars')

        self.mvp_mgrid_geo.rebuild_pipeline(self.vtk_data)

    # point offset to use when more fields are patched together within
    # RTDomainList

    point_offset = Int(0)

    # cell offset to use when more fields are patched together within
    # RTDomainList

    cell_offset = Int(0)

    def add_current_values(self, sctx, U_k, *args, **kw):
        for sf in self.subfields:
            if sf.skip_domain:
                continue
            sf.add_current_values(sctx, U_k, *args, **kw)

    def add_current_displ(self, sctx, U_k):
        for sf in self.subfields:
            if sf.skip_domain:
                continue
            sf.rmgr = self.rmgr
            sf.add_current_displ(sctx, U_k)

    def register_mv_pipelines(self, e):
        pass
        # for sf in self.subfields:
        #    sf.register_mv_pipelines( e )

    writer = tvtk.UnstructuredGridWriter(file_type='binary')

    write_counter = Int(0)

    def write(self):
        '''Generate the file name within the write_dir
        and submit the request for writing to the writer
        '''
        # self.writer.scalars_name = self.name
        file_base_name = self.var + '%(direction)d%(pos)s_%(step)d.vtk' \
            % {'direction': (self.idx + 1), "pos": self.position, "step": self.write_counter}
        # full path to the data file
        file_name = os.path.join(self.dir, file_base_name)

        self.writer.input = self.vtk_data
        self.writer.file_name = file_name
        self.write_counter += 1

        self.writer.write()

    def timer_tick(self, e=None):
        # self.changed = True
        pass

    def clear(self):
        for sf in self.subfields:
            sf.clear()

    def _get_warp_data(self):
        vectors_arr_list = []
        for sf in self.subfields:
            if sf.skip_domain:
                continue
            sf_warp_data = sf._get_warp_data()
            if sf_warp_data == None:  # all elem are deactivated
                continue
            vectors_arr_list.append(sf_warp_data)
        if len(vectors_arr_list) > 0:
            return vstack(vectors_arr_list)
        else:
            return zeros((0, 3), dtype='float_')

    def _get_field_data(self):
        tensors_arr_list = []
        for sf in self.subfields:
            if sf.skip_domain:
                continue
            sf_field_data = sf._get_field_data()
            if sf_field_data == None:  # all elem are deactivated
                continue
            tensors_arr_list.append(sf_field_data)
        if len(tensors_arr_list) > 0:
            return vstack(tensors_arr_list)
        else:
            return zeros((0, 3), dtype='float_')

    #-------------------------------------------------------------------------
    # Visualization pipelines
    #-------------------------------------------------------------------------

    mvp_mgrid_geo = Trait(MVUnstructuredGrid)

    def _mvp_mgrid_geo_default(self):
        return MVUnstructuredGrid(name=self.name,
                                  warp=self.warp,
                                  warp_var=self.warp_var
                                  )

    view = View(HSplit(VSplit(VGroup('var', 'idx'),
                              VGroup('record_on', 'clear_on'),
                              Item('refresh_button', show_label=False),
                              ),
                       ),
                resizable=True)
