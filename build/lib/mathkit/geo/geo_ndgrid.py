
from math import floor
from mayavi.core.source import Source
from numpy import zeros, mgrid, c_, indices, transpose, array, arange, \
    asarray, ix_, ones, random
from traits.api import \
    HasTraits, Float, Int, Array, Property, cached_property, \
    Tuple, List, Str, on_trait_change, Button, Delegate, \
    Instance, Trait
from traitsui.api import View, Item, HSplit, VSplit, InstanceEditor
from traitsui.api import \
    View, Item, Group, HGroup, VGroup, VSplit, HSplit, CheckListEditor, TextEditor
from tvtk.api import tvtk
from tvtk.pyface.scene_editor import SceneEditor
from functools import reduce

# from ibvpy.plugins.mayavi_util.pipelines import \
#     MVPolyData, MVPointLabels


# tvtk related imports
#
#---------------------------------------------------------------------
# C L A S S  GridPoint
#---------------------------------------------------------------------
class GridPoint(HasTraits):
    '''
    Editable point specification to be used for orientation 
    within the grid.
    '''
    _coords = List(Float, [0., 0., 0.], enter_set=True, auto_set=False)

    x = Property(Float, enter_set=True, auto_set=False)
    y = Property(Float, enter_set=True, auto_set=False)
    z = Property(Float, enter_set=True, auto_set=False)

    def _get_x(self):
        '''get the value of x'''
        return self._coords[0]

    def _set_x(self, value):
        '''set the value of x'''
        self._coords[0] = value

    def _get_y(self):
        '''get the value of y'''
        return self._coords[1]

    def _set_y(self, value):
        '''set the value of y'''
        self._coords[1] = value

    def _get_z(self):
        '''get the value of z'''
        return self._coords[2]

    def _set_z(self, value):
        '''set the value of z'''
        self._coords[2] = value

    def __getitem__(self, idx):
        '''Delegate access to _coords'''
        return self._coords[idx]

    def __iter__(self):
        '''Delegate iteration to _coords'''
        return iter(self._coords)

    traits_view = View(Item('x'),
                       Item('y'),
                       Item('z'),
                       resizable=True)

#---------------------------------------------------------------------
# C L A S S  GeoNDGrid
#---------------------------------------------------------------------


class GeoNDGrid(Source):
    '''
    Specification and representation of an nD-grid.

    GridND
    '''
    # The name of our scalar array.
    scalar_name = Str('scalar')

    # map of coordinate labels to the indices
    _dim_map = {'x': 0, 'y': 1, 'z': 2}

    # currently active dimensions
    active_dims = List(Str, ['x', 'y'])

    # Bottom left corner
    x_mins = Instance(GridPoint, label='Corner 1')

    def _x_mins_default(self):
        '''Bottom left corner'''
        return GridPoint()
    # Upper right corner
    x_maxs = Instance(GridPoint, label='Corner 2')

    def _x_maxs_default(self):
        '''Upper right corner'''
        return GridPoint(x=1, y=1, z=1)

    # indices of the currently active dimensions
    dim_indices = Property(Array(int), depends_on='active_dims')

    @cached_property
    def _get_dim_indices(self):
        ''' Get active indices '''
        return array([self._dim_map[dim_ix]
                      for dim_ix
                      in self.active_dims], dtype='int_')

    # number of currently active dimensions
    n_dims = Property(Int, depends_on='active_dims')

    @cached_property
    def _get_n_dims(self):
        '''Number of currently active dimensions'''
        return len(self.active_dims)

    # number of elements in each direction
    # @todo: rename to n_faces
    shape = Tuple(int, int, int, label='Elements')

    def _shape_default(self):
        '''Number of elements in each direction'''
        return (1, 0, 0)

    n_act_nodes = Property(Array, depends_on='shape, active_dims')

    @cached_property
    def _get_n_act_nodes(self):
        '''Number of active nodes respecting the active_dim'''
        act_idx = ones((3, ), int)
        shape = array(list(self.shape), dtype=int)
        act_idx[self.dim_indices] += shape[self.dim_indices]
        return act_idx

    # total number of nodes of the system grid
    n_nodes = Property(Int, depends_on='shape, active_dims')

    @cached_property
    def _get_n_nodes(self):
        '''Number of nodes used for the geometry approximation'''
        return reduce(lambda x, y: x * y, self.n_act_nodes)

    enum_nodes = Property(Array, depends_on='shape,active_dims')

    @cached_property
    def _get_enum_nodes(self):
        '''
        Returns an array of element numbers respecting the grid structure
        (the nodes are numbered first in x-direction, then in y-direction and
        last in z-direction)
        '''
        return arange(self.n_nodes).reshape(tuple(self.n_act_nodes))

    grid = Property(Array,
                    depends_on='shape,active_dims,x_mins.+,x_maxs.+')

    @cached_property
    def _get_grid(self):
        '''
        slice(start,stop,step) with step of type 'complex' leads to that number of divisions 
        in that direction including 'stop' (see numpy: 'mgrid')
        '''
        slices = [slice(x_min, x_max, complex(0, n_n))
                  for x_min, x_max, n_n
                  in zip(self.x_mins, self.x_maxs, self.n_act_nodes)]
        return mgrid[tuple(slices)]

    #-------------------------------------------------------------------------
    # Visualization pipelines
    #-------------------------------------------------------------------------
#     mvp_mgrid_geo = Trait(MVPolyData)
#
#     def _mvp_mgrid_geo_default(self):
#         return MVPolyData(name='Mesh geomeetry',
#                           points=self._get_points,
#                           lines=self._get_lines,
#                           polys=self._get_faces,
#                           scalars=self._get_random_scalars
#                           )
#
#     mvp_mgrid_labels = Trait(MVPointLabels)
#
#     def _mvp_mgrid_labels_default(self):
#         return MVPointLabels(name='Mesh numbers',
#                              points=self._get_points,
#                              scalars=self._get_random_scalars,
#                              vectors=self._get_points)

    changed = Button('Draw')

    @on_trait_change('changed')
    def redraw(self):
        '''
        '''
        self.mvp_mgrid_geo.redraw()
        self.mvp_mgrid_labels.redraw('label_scalars')

    def _get_points(self):
        '''
        Reshape the grid into a column.
        '''
        return c_[tuple([self.grid[i].flatten() for i in range(3)])]

    def _get_n_lines(self):
        '''
        Get the number of lines.
        '''
        act_idx = ones((3, ), int)
        act_idx[self.dim_indices] += self.shape[self.dim_indices]
        return reduce(lambda x, y: x * y, act_idx)

    def _get_lines(self):
        '''
        Only return data if n_dims = 1
        '''
        if self.n_dims != 1:
            return array([], int)
        #
        # Get the list of all base nodes
        #
        tidx = ones((3,), dtype='int_')
        tidx[self.dim_indices] = -1
        slices = tuple([slice(0, idx) for idx in tidx])
        base_node_list = self.enum_nodes[slices].flatten()
        #
        # Get the node map within the line
        #
        ijk_arr = zeros((3, 2), dtype=int)
        ijk_arr[self.dim_indices[0]] = [0, 1]
        offsets = self.enum_nodes[ijk_arr[0], ijk_arr[1], ijk_arr[2]]
        #
        # Setup and fill the array with line connectivities
        #
        n_lines = self._get_n_lines()
        lines = zeros((n_lines, 2), dtype='int_')
        for n_idx, base_node in enumerate(base_node_list):
            lines[n_idx, :] = offsets + base_node
        return lines

    def _get_n_faces(self):
        '''Return the number of faces.

        The number is determined by putting 1 into inactive dimensions and 
        shape into the active dimensions. 
        '''
        act_idx = ones((3, ), int)
        shape = array(self.shape, dtype=int)
        act_idx[self.dim_indices] = shape[self.dim_indices]
        return reduce(lambda x, y: x * y, act_idx)

    def _get_faces(self):
        '''
        Only return data of n_dims = 2.
        '''
        if self.n_dims != 2:
            return array([], int)
        #
        # get the slices extracting all corner nodes with
        # the smallest node number within the element
        #
        tidx = ones((3,), dtype='int_')
        tidx[self.dim_indices] = -1
        slices = tuple([slice(0, idx) for idx in tidx])
        base_node_list = self.enum_nodes[slices].flatten()
        #
        # get the node map within the face
        #
        ijk_arr = zeros((3, 4), dtype=int)
        ijk_arr[self.dim_indices[0]] = [0, 0, 1, 1]
        ijk_arr[self.dim_indices[1]] = [0, 1, 1, 0]
        offsets = self.enum_nodes[ijk_arr[0], ijk_arr[1], ijk_arr[2]]
        #
        # setup and fill the array with line connectivities
        #
        n_faces = self._get_n_faces()
        faces = zeros((n_faces, 4), dtype='int_')
        for n_idx, base_node in enumerate(base_node_list):
            faces[n_idx, :] = offsets + base_node
        return faces

    def _get_volumes(self):
        '''
        Only return data if ndims = 3
        '''
        if self.n_dims != 3:
            return array([], int)

        tidx = ones((3,), dtype='int_')
        tidx[self.dim_indices] = -1
        slices = tuple([slice(0, idx) for idx in tidx])

        en = self.enum_nodes
        offsets = array([en[0, 0, 0], en[0, 1, 0], en[1, 1, 0], en[1, 0, 0],
                         en[0, 0, 1], en[0, 1, 1], en[1, 1, 1], en[1, 0, 1]], dtype='int_')
        base_node_list = self.enum_nodes[slices].flatten()

        n_faces = self._get_n_faces()
        faces = zeros((n_faces, 8), dtype='int_')
        for n in base_node_list:
            faces[n, :] = offsets + n

        return faces

    # Identifiers
    var = Str('dummy')
    idx = Int(0)

    def _get_random_scalars(self):
        return random.weibull(1, size=self.n_nodes)

    traits_view = View(HSplit(Group(Item('changed', show_label=False),
                                    Item('active_dims@',
                                         editor=CheckListEditor(values=['x', 'y', 'z'],
                                                                cols=3)),
                                    Item('x_mins@', resizable=False),
                                    Item('x_maxs@'),
                                    Item('shape@'),
                                    ),
                              ),
                       resizable=True)


if __name__ == '__main__':
    pass
#     from mayavi.scripts import mayavi2
#     from ibvpy.plugins.mayavi_engine import set_engine
#
#     @mayavi2.standalone
#     def view():
#         from etsproxy.mayavi.modules.api import Outline, Surface
#         # 'mayavi' is always defined on the interpreter.
#         set_engine(mayavi)
#         mayavi.new_scene()
#         # Make the data and add it to the pipeline.
#         mfn = GeoNDGrid(active_dims=['x', 'y'], shape=(2, 1, 5),
#                         x_maxs=GridPoint(x=5, y=5, z=15))
#
#         mfn.redraw()
#
#     view()
