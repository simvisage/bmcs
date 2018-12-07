

from ibvpy.core.tstepper import TStepper
from ibvpy.mesh.fe_grid import FEGrid
from ibvpy.mesh.fe_ls_domain import FELSDomain
from ibvpy.mesh.fe_refinement_grid import FERefinementGrid
from ibvpy.mesh.xfe_subdomain import XFESubDomain
from ibvpy.plugins.mayavi_engine import get_engine
from mathkit.matrix_la.dense_mtx import DenseMtx
from mathkit.matrix_la.sys_mtx_array import SysMtxArray
from mathkit.matrix_la.sys_mtx_assembly import SysMtxAssembly
from mayavi.filters.api import PolyDataNormals
from mayavi.filters.api import WarpScalar
from mayavi.modules.api import Surface
from mayavi.sources.api import VTKDataSource
from numpy import ndarray, max, zeros, array
from traits.api import \
    HasTraits, Instance
from tvtk.api import tvtk

from traitsui.api \
    import View, Item, \
    TreeEditor, TreeNode, Handler
from traitsui.menu \
    import Menu, Action, Separator
# from traitsui.wx.tree_editor \
#     import NewAction


draw_action = Action(
    name='Draw',
    action='handler.draw(editor,object)')

show_stiffness_action = Action(
    name='Show stiffness',
    action='handler.show_stiffness(editor,object)')

plot_stiffness_action = Action(
    name='Plot stiffness',
    action='handler.plot_stiffness(editor,object)')


class TreeHandler (Handler):

    def draw(self, editor, object):
        object.redraw()

    def show_stiffness(self, editor, object):
        K = self._get_stiffness(editor, object)
        K_dense = DenseMtx(assemb=K)
        K_dense.configure_traits()

    def plot_stiffness(self, editor, object):
        '''This method gets the input data from the current tstepper
        which is the root of the tree. Sets up the context and 
        gets the stiffness matrix.
        '''
        K = self._get_stiffness(editor, object)
        K_dense = DenseMtx(assemb=K)

        # prepare plotting of the matrix in Mayavi
        #
        z_data = K_dense.mtx.flatten()
        z_max = max(z_data)
        n_dofs = K.n_dofs

        spoints = tvtk.StructuredPoints(origin=(0, 0, 0),
                                        spacing=(1, -1, 1),
                                        dimensions=(n_dofs, n_dofs, 1))
        spoints.point_data.scalars = z_data
        spoints.point_data.scalars.name = 'Stiffness'

        e = get_engine()
        src = VTKDataSource(data=spoints)
        e.add_source(src)
        scale_factor = .1 / float(z_max) * n_dofs
        ws = WarpScalar()
        ws.filter.scale_factor = scale_factor
        e.add_filter(ws)
        e.add_filter(PolyDataNormals())
        s = Surface()
        e.add_module(s)

    def _get_stiffness(self, editor, object):
        tstepper = editor.object.tstepper

        if isinstance(object, TStepper):
            U_k = tstepper.U_k
            d_U = tstepper.d_U
            K, R = tstepper.eval('predictor', U_k, d_U, 0, 0)
            print('constraints')
            K.print_constraints()
            K.apply_constraints(R)
        else:

            dots = object.dots
            U_k = tstepper.U_k
            d_U = tstepper.d_U
            sctx = tstepper.sctx

            F_int, K_mtx = dots.get_corr_pred(sctx, U_k, d_U, 0, 0)

            # put the matrix into the system assembly
            #
            K = SysMtxAssembly()
            if isinstance(K_mtx, ndarray):
                K.add_mtx(K_mtx)
            elif isinstance(K_mtx, SysMtxArray):
                K.sys_mtx_arrays.append(K_mtx)
            elif isinstance(K_mtx, list):
                K.sys_mtx_arrays = K_mtx
            elif isinstance(K_mtx, SysMtxAssembly):
                K.sys_mtx_arrays = K_mtx.sys_mtx_arrays

            n_dofs = tstepper.sdomain.n_dofs
            K.add_mtx(
                zeros((1, 1), dtype='float_'), array([n_dofs - 1], dtype='int_'))

        return K


domain_menu = Menu(  # NewAction,
    Separator(),
    draw_action,
    Separator(),
    show_stiffness_action,
    plot_stiffness_action)

fe_domain_tree_editor = TreeEditor(
    nodes=[
        TreeNode(node_for=[TStepper],
                 auto_open=True,
                 label='=domain',
                 children='subdomains',
                 menu=domain_menu,
                 ),
        TreeNode(node_for=[TStepper],
                 auto_open=True,
                 label='=xdomain',
                 children='xdomains',
                 menu=domain_menu,
                 ),
        TreeNode(node_for=[XFESubDomain],
                 auto_open=True,
                 label='name',
                 children='',
                 menu=domain_menu,
                 ),
        TreeNode(node_for=[FELSDomain],
                 auto_open=True,
                 label='name',
                 children='',
                 menu=domain_menu,
                 ),
        TreeNode(node_for=[FERefinementGrid],
                 auto_open=True,
                 label='name',
                 children='',
                 menu=domain_menu,
                 ),
        TreeNode(node_for=[FERefinementGrid],
                 auto_open=True,
                 label='=sublevels',
                 children='children',
                 menu=domain_menu,
                 ),
        TreeNode(node_for=[FERefinementGrid],
                 auto_open=False,
                 label='=subgrids',
                 children='fe_subgrids',
                 menu=domain_menu,
                 ),
        TreeNode(node_for=[FEGrid],
                 auto_open=False,
                 label='name',
                 children='',
                 menu=domain_menu,
                 ), ],
    hide_root=False
)


class TStepperService(HasTraits):

    # Set by envisage when this is offered as a service offer.
    window = Instance('envisage.ui.workbench.workbench_window.WorkbenchWindow')

    tstepper = Instance(TStepper)

    def _tstepper_default(self):
        return TStepper()

    def _tstepper_changed(self):
        '''Rebind the dependent services'''
        rtrace_service = \
            self.window.get_service(
                'ibvpy.plugins.rtrace_service.RTraceService')
        rtrace_service.rtrace_mngr = self.tstepper.rtrace_mngr

    ###########################################################################
    # `HasTraits` interface.
    ###########################################################################
    def default_traits_view(self):
        """The default traits view of the Engine View.
        """
        view = View(  # Group(
            Item(name='tstepper',
                 id='tstepper',
                 editor=fe_domain_tree_editor,
                 resizable=True,
                 show_label=False),
            id='simvisage.ibvpy.mesh.fe_domain_tree',
            dock='horizontal',
            #drop_class = HasTraits,
            handler=TreeHandler(),
            #buttons = [ 'Undo', 'OK', 'Cancel' ],
            resizable=True,
            scrollable=True)
        return view


if __name__ == '__main__':
    tstepper_service = TStepperService()
    tstepper_service.configure_traits()
