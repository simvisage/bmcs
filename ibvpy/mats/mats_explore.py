
from traits.api import \
    Callable, Float, List, Property, cached_property,\
    Instance
from traitsui.api import \
    Item, View, VGroup

from bmcs.time_functions import \
    LoadingScenario
from ibvpy.bcond import BCDof
from simulator.api import Simulator
from simulator.xdomain.xdomain_point import XDomainSinglePoint
from util.traits.either_type import \
    EitherType
from view.window import BMCSWindow

from .mats_viz2d import Viz2DSigEps


class MATSExplore(Simulator):
    '''
    Simulate the loading histories of a material point in 2D space.
    '''

    node_name = 'Composite tensile test'

    def _bc_default(self):
        return [BCDof(
            var='u', dof=0, value=-0.001,
            time_function=LoadingScenario()
        )]

    def _model_default(self):
        return MATS3DDesmorat()

    def _xdomain_default(self):
        return XDomainSinglePoint()

    traits_view = View(
        resizable=True,
        width=1.0,
        height=1.0,
        scrollable=True,
    )

    tree_view = traits_view


if __name__ == '__main__':
    from ibvpy.mats.mats3D import MATS3DMplCSDEEQ
    from ibvpy.mats.mats3D.mats3D_plastic.vmats3D_desmorat import \
        MATS3DDesmorat
    from ibvpy.rtrace.record_vars import RecordVars
    import time
    e = MATSExplore(
        record={'vars': RecordVars()}
    )
    e.tline.step = 0.01
    viz2d_sig_eps = Viz2DSigEps(name='stress-strain',
                                vis2d=e.hist['vars'])

    w = BMCSWindow(model=e)
    w.viz_sheet.viz2d_list.append(viz2d_sig_eps)
    w.viz_sheet.n_cols = 1
    w.viz_sheet.monitor_chunk_size = 10
    w.offline = False
#     w.run()
#     time.sleep(2)
    w.configure_traits()
