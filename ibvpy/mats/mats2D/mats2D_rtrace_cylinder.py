
from numpy import \
    array, hstack, zeros, vstack, pi as Pi, mgrid, arange, \
    linspace, sin, cos, c_, vstack, zeros, append, concatenate, \
    dot

from traits.api import \
    Str, Callable, Int, List, Bool, Trait
from traitsui.api import \
    View, HSplit, VGroup, Item
from ibvpy.api import \
    RTrace
from ibvpy.plugins.mayavi_util.pipelines import \
    MVPolyData, MVPointLabels
from .mats2D_tensor import map2d_eps_eng_to_mtx


class MATS2DRTraceCylinder(RTrace):

    '''Gather the data for the polar plot

    @todo: define the resolution in the polar direction.

    Limit this model to the set of discretized domain.
    Acquire the microplane geometry from the the model.

    In order to construct the warp, a tensor rank 2 variable must
    be specified in order to make a projection onto the cylinder
    surface. By default, the wapr is called 'eps_app'.  This trait
    attribute may be modified to other derived variables, like
    eps_plastic or eps_dev, whathever the mats supports.

    For discrete models, (microplane, the resolution might be given
    in the MATS class itself. Therefore, ask first there.

    @todo: Include the strain field as warp from the material model
    According to the current resolution of the strain space - project
    the strain tensor onto the planes.

    @todo: Add vector field with arrows to show vector magnitudes at individual microplanes.
    '''

    label = Str('MATS2DRTraceCylinder')

    # Switch the warp veriable on and off
    var_warp_on = Bool(True)

    # Variable to be used for the meridian direction
    # (distance between rings)
    var_axis = Str('time')
    var_axis_eval = Callable
    idx_axis = Int(0)

    # Variable to be used for the surface direction
    var_surface = Str('')
    var_surface_eval = Callable
    idx_surface = Int(-1)

    _trace = List()
    _warp_field = List()

    def get_n_t(self):
        '''Get the resolution of the axis.
        This associates every point in the history with a time stamp
        mapped as a z-coordinate into the 3D space.
        '''
        return len(self._trace)

    def get_n_r(self):
        '''
        Get resolution of the polar axis
        '''
        # this is special for the microplane model
        # verify - more general interface to the source material model should be
        # defined
        return 2 * self.var_surface_eval.ts.n_mp

    def __init__(self, **kwargs):
        super(MATS2DRTraceCylinder, self).__init__(**kwargs)
        self._trace = []
        if self.var_warp_on:
            self._warp_field = []

    def bind(self):
        '''
        Locate the evaluators
        '''
        # get the evaluator for the variable
        self.var_surface_eval = self.rmgr.rte_dict[self.var_surface]
        # get the warp_eval callable

    def add_current_values(self, sctx, eps_eng):
        '''
        Invoke the evaluators in the current context for the specified control vector U_k.
        '''
        surface_values = self.var_surface_eval(sctx, eps_eng)
        self._trace.append(surface_values)
        if self.var_warp_on:
            n_r = self.get_n_r()
            alpha_list = array([2 * Pi / n_r * i for i in range(0, n_r)])
            N = array([[cos(alpha), sin(alpha)] for alpha in alpha_list])

            # todo: change this so that it can be applied to a general model.
            #
            # The warp values are in the U_k input given here. It must be mapped to
            # to the strain tensor and then projected onto the planes N
            #
            eps_mtx = map2d_eps_eng_to_mtx(eps_eng)
            warp_values = array([dot(eps_mtx, n) for n in N])

            self._warp_field.append(warp_values)

    #-------------------------------------------------------------------------
    # 3D visualization in MayaVI
    #-------------------------------------------------------------------------
    def xregister_mv_pipelines(self, e):
        '''Register as a source in the pipelane
        '''
        # Remarks[rch]:
        #
        # Pipeline construction
        # ---------------------
        # Maybe this should be done only on demand
        # when the visualization is requested either by the update
        # event triggered here, or by the user. For a batch-like
        # simulation no scenes would be necessary.
        #
        # In that case, the engine could be simply registered and
        # the pipeline construction can be deffered to the explicit
        # request.
        #
        # Further question concerns the multiplicity of the relations.
        # One tracer can construct several sources. The source can be
        # for example the standard array source or parametric surface.
        # There should be no link back to the tracer.
        #
        # Links between tracers and scenes
        # --------------------------------
        # On the other hand, several tracers can contribute to a single
        # scene. Then the tracer explicitly specifies the name
        # of the scene it is contributing to. This type of sharing makes
        # sence if the spatial context of the tracer is the same.
        #
        # The scene management is a separate issue, no general
        # rules can be formulated at this time.
        #
        scene = e.new_scene()
        scene.name = 'Polar domain'

        # Construct the source
        from mayavi.sources.vtk_data_source import VTKDataSource
        from tvtk.api import tvtk

        self._mv_src = VTKDataSource(name='Time-Strain Cylinder',
                                     data=tvtk.PolyData())
        e.add_source(self._mv_src)

        # Construct the warp filter
        if self.var_warp_on:
            from mayavi.filters.api import WarpVector
            e.add_filter(WarpVector())

        # Construct visualization modules
        from mayavi.modules.api import Outline, Surface
        s = Surface()
        e.add_module(Outline())
        e.add_module(s)
        s.module_manager.scalar_lut_manager.show_scalar_bar = True
        s.module_manager.scalar_lut_manager.reverse_lut = True

    #-------------------------------------------------------------------------
    # Visualization pipelines
    #-------------------------------------------------------------------------
    mvp_mgrid_geo = Trait(MVPolyData)

    def _mvp_mgrid_geo_default(self):
        return MVPolyData(name='Mesh geomeetry',
                               points=self._get_points,
                               polys=self._get_faces,
                               scalars=self._get_scalars,
                               vectors=self._get_vectors
                          )

    mvp_mgrid_labels = Trait(MVPointLabels)

    def _mvp_mgrid_labels_default(self):
        return MVPointLabels(name='Mesh numbers',
                                  points=self._get_points,
                                  scalars=self._get_scalars,
                                  vectors=self._get_vectors)

    def redraw(self):
        '''
        '''
        self.mvp_mgrid_geo.redraw()
        # self.mvp_mgrid_labels.redraw( 'label_scalars' )

    n_t = Int
    n_r = Int

    def _get_points(self):
        '''
        Return an array of points. In order to show the values constant
        on each ring segment, generate two grids so that point values
        from left and right can be stored in order to show the jumps.
        '''
        # to get a reasonable shape of the cylinder define its dimensions
        # The axis values - time-step values must be mapped into
        # the visualization range. Thus, when using axis values, the numbers
        # do not correspond to the real calculated values along the cylinder
        # axis.
        # @todo: Maybe it is possible to adjust this in the axes filter
        #
        T = 10
        R = 2

        self.n_t = n_t = self.get_n_t()
        self.n_r = n_r = self.get_n_r()

        d_r = 2 * Pi / n_r

        alpha_slice1 = slice(0, 2 * Pi - d_r, complex(0, n_r))
        alpha_slice2 = slice(d_r, 2 * Pi, complex(0, n_r))
        axis_slice = slice(0, T, complex(0, n_t))
        grid1 = mgrid[alpha_slice1, axis_slice]
        grid2 = mgrid[alpha_slice2, axis_slice]

        grid1 = array([R * cos(grid1[0]), R * sin(grid1[0]), grid1[1]])
        grid2 = array([R * cos(grid2[0]), R * sin(grid2[0]), grid2[1]])

        points1 = c_[tuple([grid1[i].flatten() for i in range(3)])]
        points2 = c_[tuple([grid2[i].flatten() for i in range(3)])]
        points = vstack([points1, points2])
        return points

    def _get_faces(self):
        '''
        Only return data of n_dims = 2.
        '''
        # Construct  an array of node numbers respecting the grid structure
        # (the nodes are numbered first in t-direction, then in r-direction
        #
        enum_nodes = arange(
            2 * self.n_r * self.n_t).reshape((2 * self.n_r, self.n_t))
        #
        # get the slices extracting all corner nodes with the smallest
        # node number within the element
        #
        enum_nodes_grid1 = arange(
            self.n_r * self.n_t).reshape((self.n_r, self.n_t))
        base_node_list = enum_nodes_grid1[:, 0:-1].flatten()
        #
        # Get the node map within the line
        #
        goffset = self.n_r
        offsets = enum_nodes[[0, 0, goffset, goffset], [0, 1, 1, 0]]
        #
        # The number is determined by putting 1 into inactive dimensions and
        # n_t into the active dimensions.
        #
        n_faces = (self.n_r) * (self.n_t - 1)
        faces = zeros((n_faces, 4), dtype='int_')
        for n_idx, base_node in enumerate(base_node_list):
            faces[n_idx, :] = offsets + base_node
        return faces

    def _get_scalars(self):
        '''Point values to be plotted by  the surface module.
        '''
        f_data = array(self._trace, dtype='float').transpose().flatten()
        f_data = hstack([f_data, f_data])
        return hstack([f_data, f_data])

    def _get_vectors(self):
        '''Vector values that can be used in the warp field.

        The values must be expanded to match the 2-mgrid layout.
        There are two grids in order to be able to represent the
        segment-wise constant values. For a general case,
        this is not necessary. Should the fields be classified
        according to their background mesh? Then,
        '''
        if self.var_warp_on:
            # Convert the list of gathered arrays into
            # a single 2-n_steps array
            #
            # glue the list of deformation vectors together
            #
            f_data = array(self._warp_field, dtype='float')

            # swap axes to have the meridian (time) direction first
            #
            f_data = f_data.swapaxes(0, 1)

            # construct the shifted grid by putting the first
            # line at the end
            #
            g_data = vstack((f_data[1:, :, :], f_data[0:1, 0:, :]))

            # Put the grids together
            #
            f_data = vstack((f_data, g_data))

            # flatten the first two axes to get each point
            # in a single row of a two-dimensional array
            #
            n_points = f_data.shape[0] * f_data.shape[1]
            f_data = f_data.reshape(n_points, 2)

            # Augment with the zero z-vector coordinate
            #
            f_data = hstack([f_data, zeros((n_points, 1), dtype='float')])
            return f_data
        else:
            return []

    def timer_tick(self, e=None):
        pass
        # @todo: unify with redraw
        # self.redraw()

    def clear(self):
        self._trace = []
        if self.var_warp_on:
            self._warp_field = []

    view = View(HSplit(VGroup(Item('var_warp_on'),
                              VGroup('var_axis', style='readonly'),
                              VGroup('var_surface', style='readonly'),
                              VGroup('record_on', 'clear_on'),
                              VGroup(Item('refresh_button', show_label=False)),),
                       ),
                resizable=True)
