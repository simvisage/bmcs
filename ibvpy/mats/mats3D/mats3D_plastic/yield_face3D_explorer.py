from mayavi import mlab
from mayavi.core.ui.api import MlabSceneModel, SceneEditor
from traits.api import HasTraits, Instance, Button, \
    on_trait_change, Float, Property, cached_property
from traitsui.api import \
    View, UItem, Item, HSplit, Group, \
    VGroup, Spring

import numpy as np
from yield_face3D import YieldConditionAbaqus, YieldConditionWillamWarnke, \
    YieldConditionDruckerPrager


class YieldFaceViewer(HasTraits):

    min_sig = Float(-20.0)
    max_sig = Float(5.0)

    sig_i = Property(depends_on='min_sig, max_sig')

    @cached_property
    def _get_sig_i(self):
        n_sig = 100j
        sig_1, sig_2, sig_3 = np.mgrid[self.min_sig: self.max_sig: n_sig,
                                       self.min_sig: self.max_sig: n_sig,
                                       self.min_sig: self.max_sig: n_sig]
        return [sig_1, sig_2, sig_3]

    sig_ij = Property(depends_on='min_sig, max_sig')

    @cached_property
    def _get_sig_ij(self):
        sig_abcj = np.einsum('jabc->abcj', np.array(self.sig_i))
        DELTA = np.identity(3)
        sig_abcij = np.einsum('abcj,jl->abcjl', sig_abcj, DELTA)
        return sig_abcij

    yc_abaqus = Instance(YieldConditionAbaqus, arg=(), kw={})
    yc_WW = Instance(YieldConditionWillamWarnke, arg=(), kw={})
    yc_DP = Instance(YieldConditionDruckerPrager, arg=(), kw={})

    scene1 = Instance(MlabSceneModel, ())

    button1 = Button('Redraw')

    @on_trait_change('button1')
    def redraw_scene1(self):
        self.redraw_scene(self.scene1)

    def redraw_scene(self, scene):
        # Notice how each mlab call points explicitely to the figure it
        # applies to.
        mlab.clf(figure=scene.mayavi_scene)

        f = self.yc_abaqus.f(self.sig_ij)
        f_pipe = mlab.contour3d(
            self.sig_i[0], self.sig_i[1], self.sig_i[2], f,
            contours=[0.0], color=(1, 0, 0), figure=scene.mayavi_scene)

        f = self.yc_DP.f(self.sig_ij)
        f_pipe = mlab.contour3d(
            self.sig_i[0], self.sig_i[1], self.sig_i[2], f,
            contours=[0.0], color=(0, 1, 0), figure=scene.mayavi_scene)

        f = self.yc_WW.f(self.sig_ij)
        f_pipe = mlab.contour3d(
            self.sig_i[0], self.sig_i[1], self.sig_i[2], f,
            contours=[0.0], color=(0, 0, 1), figure=scene.mayavi_scene)

        mlab.axes(f_pipe)

    # The layout of the dialog created
    view = View(
        HSplit(
            Group(
                VGroup(
                    UItem('yc_abaqus@', resizable=True, width=300),
                    label='Abaqus yield face'
                ),
                VGroup(
                    UItem('yc_DP@', resizable=True),
                    label='Drucker-Prager yield face'
                ),
                VGroup(
                    UItem('yc_WW@', resizable=True),
                    label='Willam-Warnke yield face',
                ),
                Spring(),
                UItem('button1', resizable=True),
                show_labels=True,
            ),
            Group(
                Item('scene1',
                     editor=SceneEditor(), height=250,
                     width=300),
                show_labels=False,
            )
        ),
        resizable=True,
        height=0.7,
        width=0.8
    )


def run_explorer(*args, **kw):
    m = YieldFaceViewer()
    m.redraw_scene1()
    m.configure_traits(*args, **kw)


if __name__ == '__main__':
    run_explorer()