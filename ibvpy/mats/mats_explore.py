
from bmcs.time_functions.tfun_pwl_interactive import TFunPWLInteractive
from ibvpy.bcond import BCDof
from ibvpy.core.scontext import SContext
from ibvpy.core.tloop import TLine
from ibvpy.mats.mats_eval import IMATSEval
from ibvpy.mesh.fe_domain import FEDomain
from traits.api import \
    Callable, Float, List, Property, cached_property,\
    Instance
from traitsui.api import \
    Item, View, VGroup
from util.traits.either_type import \
    EitherType
from view.plot2d import Viz2D, Vis2D
from view.window import BMCSModel, BMCSWindow

from mats1D.mats1D_explore import MATS1DExplore
from mats1D5.mats1D5_explore import MATS1D5Explore
from mats2D.mats2D_explore import MATS2DExplore
from mats3D.mats3D_explore import MATS3DExplore
from mats_tloop import TLoop
from mats_viz2d import Viz2DSigEps


class MATSExplore(BMCSModel, Vis2D):
    '''
    Simulate the loading histories of a material point in 2D space.
    '''

    node_name = 'Composite tensile test'

    tree_node_list = List([])

    def _tree_node_list_default(self):
        return [
            self.dim,
            self.bc
        ]

    def _update_node_list(self):
        self.tree_node_list = [
            self.dim,
            self.bc
        ]

    dim = EitherType(names=['1D', '1D5', '2D', '3D'],
                     klasses=[MATS1DExplore,
                              MATS1D5Explore,
                              MATS2DExplore,
                              MATS3DExplore
                              ],
                     MAT=True)

    dimx = Property(depends_on='dim')

    @cached_property
    def _get_dimx(self):
        return self.dim

    bc = Instance(BCDof)

    def _bc_default(self):
        return BCDof(var='u', dof=0, value=-0.001)

    max_load = Float(5.0, enter_set=True, auto_set=False)

    time_function = Callable(lambda t: t)

    n_steps = Float(30, enter_set=True, auto_set=False)

    tolerance = Float(1e-5, enter_set=True, auto_set=False)

    n_iterations = Float(10, enter_set=True, auto_set=False)

    def _dim_default(self):
        dim = MATS3DExplore(explorer=self)
        dim.bcond_mngr.bcond_list = [self.bc]
        return dim

    def _dim_changed(self):
        self.dim.explorer = self
        self.dim.bcond_mngr.bcond_list = [self.bc]
        self.dim._mats_eval_changed()

    tloop = Property(Instance(TLoop), depends_on='dim')

    @cached_property
    def _get_tloop(self):

        n_steps = self.n_steps

        tloop = TLoop(ts=self.dim,
                      k_max=1000,
                      tolerance=1e-7,
                      tline=TLine(min=0.0, step=1.0 / n_steps, max=1.0))

        return tloop

    def init(self):
        self.tloop.init()

    def eval(self):
        self.tloop.eval()

    traits_view = View(
        resizable=True,
        width=1.0,
        height=1.0,
        scrollable=True,
    )

    tree_view = traits_view


if __name__ == '__main__':
    from ibvpy.mats.mats3D import MATS3DMplCSDEEQ
    from ibvpy.mats.mats3D.mats3D_plastic.mats3D_desmorat import \
        MATS3DDesmorat
    explorer = MATSExplore(
        dim=MATS3DExplore(
            mats_eval=MATS3DDesmorat()
        )
    )
    viz2d_sig_eps = Viz2DSigEps(name='stress-strain',
                                vis2d=explorer)

    w = BMCSWindow(model=explorer)

    w.viz_sheet.viz2d_list.append(viz2d_sig_eps)
    w.viz_sheet.n_cols = 1
    w.viz_sheet.monitor_chunk_size = 1
    w.offline = False
    w.run()
    w.configure_traits()
