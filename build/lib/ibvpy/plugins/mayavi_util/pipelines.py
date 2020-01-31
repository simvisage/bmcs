from traits.api import \
    Bool, Enum, HasTraits, \
    Instance, Trait, Str, \
    Callable, Tuple

from numpy import \
    array

from tvtk.api import tvtk
# Mayavi related imports
#
from mayavi.sources.api import VTKDataSource, VTKFileReader
from mayavi.modules.api import Surface
from mayavi.filters.api import ExtractTensorComponents, \
    ExtractVectorComponents

from mayavi.filters.api import WarpVector

# MayaVI engine used for the pipeline construction
#
from ibvpy.plugins.mayavi_engine import get_engine


class MVStructuredGrid(HasTraits):
    dims = Callable

    def _dims(self):
        return lambda: (1, 1, 1)

    points = Callable

    def _points_default(self):
        return lambda: array([])

    pd = Instance(tvtk.StructuredGrid)

    def _pd_default(self):
        return tvtk.StructuredGrid()

    scalars = Callable

    vectors = Callable

    tensors = Callable

    name = Str('')

    def __init__(self, **kw):

        super(MVStructuredGrid, self).__init__(**kw)
        e = get_engine()

        from mayavi.modules.api import \
            StructuredGridOutline, GridPlane

        self.src = VTKDataSource(name=self.name, data=self.pd)
        e.add_source(self.src)

        o = StructuredGridOutline()
        e.add_module(o)

        for axis in ['x', 'y', 'z']:
            g = GridPlane(name='%s - grid plane' % axis)
            g.grid_plane.axis = axis
            e.add_module(g)

        if self.scalars or self.vectors or self.tensors:
            s = Surface()
            e.add_module(s)

    def redraw(self):

        self.pd.dimensions = self.dims()
        self.pd.points = self.points()

        if self.scalars:
            self.pd.point_data.scalars = self.scalars()
        if self.vectors:
            self.pd.point_data.vectors = self.vectors()
        if self.tensors:
            self.pd.point_data.tensors = self.tensors()

        self.src.data_changed = True


class MVUnstructuredGrid(HasTraits):

    #    vtk_cell_array = Instance(tvtk.CellArray())
    #    def _vtk_cell_array_default(self):
    #        return tvtk.CellArray()

    warp = Bool(False)

    name = Str
    position = Enum('nodes', 'int_pnts')
    engine = Trait()

    def _engine_default(self):
        return get_engine()

    def rebuild_pipeline(self, pd):
        src = VTKDataSource(name=self.name, data=pd)
        self.engine.add_source(src)
        if self.warp:
            self.engine.add_filter(WarpVector())

        if self.name in src._point_tensors_list:
            src.point_tensors_name = self.name
            self.engine.add_filter(ExtractTensorComponents())
        elif self.name in src._point_vectors_list:
            src.point_vectors_name = self.name
            self.engine.add_filter(ExtractVectorComponents())
        elif self.name in src._point_scalars_list:
            src.point_scalars_name = self.name
        s = Surface()
        s.actor.property.point_size = 5.
        self.engine.add_module(s)
        src.scene.z_plus_view()


class MVVTKSource(HasTraits):

    warp = Bool(False)

    name = Str
    position = Enum('nodes', 'int_pnts')
    engine = Trait()

    def _engine_default(self):
        return get_engine()

#    def redraw(self):
#        e = get_engine()
# directory management
#        home_dir = os.environ['HOME']
#        base_name = os.path.basename(os.getcwd())
#
#        data_dir = home_dir+'/simdata'
#        if not os.path.exists(data_dir):
#            os.mkdir(data_dir)
#            print "simdata directory created"
#
#        if not os.path.exists(data_dir +'/'+ base_name):
#            os.mkdir(data_dir +'/'+ base_name)
#            print base_name," directory created"
#
#        os.chdir(data_dir +'/'+ base_name)
#
#        self.src = VTKFileReader(base_file_name = 'nodes_0.vtk')
# self.src = VTKDataSource( name = self.name, data = self.pd )
# self.e.add_source(self.src)
#        e.add_source(self.src)
#        g = Surface()
#        e.add_module(g)
# g.module_manager.scalar_lut_manager.show_scalar_bar = True # show scalar bar
#
#        self.src.scene.z_plus_view()
# self.src.data_changed = True

    def rebuild_pipeline(self, pd):
        src = VTKFileReader(base_file_name='%s_0.vtk' % self.position)
        self.engine.add_source(src)
        if self.warp:
            self.engine.add_filter(WarpVector())

        if self.name in src._point_tensors_list:
            src.point_tensors_name = self.name
            self.engine.add_filter(ExtractTensorComponents())
        elif self.name in src._point_vectors_list:
            src.point_vectors_name = self.name
            self.engine.add_filter(ExtractVectorComponents())
        elif self.name in src._point_scalars_list:
            src.point_scalars_name = self.name
        s = Surface()
        s.actor.property.point_size = 5.
        self.engine.add_module(s)
        src.scene.z_plus_view()


class MVPBase(HasTraits):

    '''
    Mayavi Pipeline Base class
    '''

    points = Callable

    def _points_default(self):
        return lambda: array([])

    lines = Callable

    def _lines_default(self):
        return lambda: array([])

    polys = Callable

    def _polys_default(self):
        return lambda: array([])

    scalars = Callable

    vectors = Callable

    tensors = Callable

    pd = Instance(tvtk.PolyData)

    def _pd_default(self):
        return tvtk.PolyData()

    def redraw(self):

        self.pd.points = self.points()
        self.pd.lines = self.lines()
        self.pd.polys = self.polys()
        if self.scalars:
            self.pd.point_data.scalars = self.scalars()
        if self.vectors:
            self.pd.point_data.vectors = self.vectors()
        if self.tensors:
            self.pd.point_data.tensors = self.tensors()
        self.src.data_changed = True


#-------------------------------------------------------------------
# MVMeshSource - mayavi mesh source
#-------------------------------------------------------------------

class MVPolyData(MVPBase):

    '''
    Provide a Mayavi source for polar visualization visualization. 

    @TODO must be revised in the context of the view and editor
    concept.  It is now a mixin-base class providing the visualization
    functionality. Still its the question whether or not it should be
    implemented taht way.
    '''

    name = Str('')

    def __init__(self, **kw):

        e = get_engine()

        super(MVPolyData, self).__init__(**kw)
        from mayavi.modules.api import Outline, Surface, Labels

        self.src = VTKDataSource(name=self.name, data=self.pd)
        e.add_source(self.src)

        o = Outline()
        e.add_module(o)
        s = Surface()
        e.add_module(s)


class MVPointLabels(MVPBase):
    color = Tuple(1., 1., 1.)

    def __init__(self, **kw):

        e = get_engine()

        super(MVPointLabels, self).__init__(**kw)
        from mayavi.modules.api import Outline, Surface, Labels

        self.src = VTKDataSource(name=self.name, data=self.pd)
        e.add_source(self.src)

        self.labels = Labels(name='Node numbers', object=self.src,
                             label_format='%g',
                             number_of_labels=100)
        self.labels.property.color = self.color
        e.add_module(self.labels)

    def redraw(self, label_mode='label_ids'):
        super(MVPointLabels, self).redraw()
        self.labels.mapper.label_mode = label_mode
