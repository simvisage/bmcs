'''
'''

from bmcs.api import Simulator
from ibvpy.api import BCDof
from traits.api import Instance, List,\
    Property, cached_property, Float
from traitsui.api import View, Include, VGroup, Item
from view.plot2d import Vis2D, Viz2D
from view.window import BMCSWindow
from view.window.bmcs_window import BMCSWindow

import matplotlib.patches as patches
from view.examples.demo_model.response_tracer import ResponseTracer
from view.examples.demo_model.tline import TLine

from . import splash_screen
from .shear_crack import ShearCrack
from .tloop import TimeLoop


class RCShearZoneShapeViz2D(Viz2D):

    def plot(self, ax, vot):

        self.vis2d._plot_shear_zone(ax, vot)


class RCShearZoneModel(BMCSModel, Vis2D):
    '''Demo model of the BMCS Window

    '''
    node_name = 'shear zone model'

    tree_node_list = List

    def _tree_node_list_default(self):
        return [self.sc, self.tline, self.rt]

    H = Float(0.3, auto_set=False, enter_set=True)

    L = Float(0.3, auto_set=False, enter_set=True)

    tline = Instance(TLine)
    '''Time range.
    '''

    def _tline_default(self):
        return TLine(min=0.0, step=0.1, max=0.0,
                     time_change_notifier=self.time_changed,
                     )

    # def time_changed(self, time):
        # self.ui.viz_sheet.time_changed(time)

    # def time_range_changed(self, tmax):
        #self.tline.max = tmax
        # self.ui.viz_sheet.time_range_changed(tmax)

    # def set_tmax(self, time):
        # self.time_range_changed(time)

    tloop = Property(Instance(TimeLoop))

    @cached_property
    def _get_tloop(self):
        return TimeLoop(tline=self.tline)

    def eval(self):
        self.tloop.eval()

    rt = Instance(ResponseTracer, ())

    sc = Instance(ShearCrack, ())

    def _plot_shear_zone(self, ax, vot):
        H = self.H
        L = self.L
        rect = patches.Rectangle(
            (0, 0), L, H, linewidth=1, edgecolor='black',
            facecolor='lightgray')
        ax.add_patch(rect)

    viz2d_classes = {
        'shear_zone': RCShearZoneShapeViz2D
    }

    tree_view = View(
        VGroup(
            Include('actions'),
            #            UItem('bc_dof@', height=500),
            Item('H'),
            Item('L')
        )
    )


if __name__ == '__main__':

    splash_screen.splash_screen(seconds=3)

    model = RCShearZoneModel()
    tv = BMCSWindow(model=model)
    model.rt.add_viz2d('time_profile', 'response tracer #1')
    model.add_viz2d('shear_zone', 'shear zone shape')
    model.sc.add_viz2d('shear_crack', 'shear crack geometry')

    tv.configure_traits()
