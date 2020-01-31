
from mayavi.core.ui.api import \
    MayaviScene, SceneEditor, MlabSceneModel
from traits.api import HasTraits, Instance, Button, \
    on_trait_change, Float, Property, cached_property, Bool, \
    Tuple

from traitsui.api import \
    View, UItem, Item, HSplit, Group, \
    VGroup, Spring
import numpy as np
from .yield_face3D import YieldConditionAbaqus, YieldConditionWillamWarnke, \
    YieldConditionDruckerPrager, YieldConditionVonMises, YieldConditionRankine


class YieldFaceViewer(HasTraits):

    min_sig = Float(-20.0, auto_set=False, enter_set=True, view_changed=True)
    max_sig = Float(5.0, auto_set=False, enter_set=True, view_changed=True)

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

    yc_VM = Instance(YieldConditionVonMises, arg=(), kw={})
    yc_VM_on = Bool(True, view_changed=True)
    yc_DP = Instance(YieldConditionDruckerPrager, arg=(), kw={})
    yc_DP_on = Bool(False, view_changed=True)
    yc_R = Instance(YieldConditionRankine, arg=(), kw={})
    yc_R_on = Bool(False, view_changed=True)
    yc_abaqus = Instance(YieldConditionAbaqus, arg=(), kw={})
    yc_abaqus_on = Bool(False, view_changed=True)
    yc_WW = Instance(YieldConditionWillamWarnke, arg=(), kw={})
    yc_WW_on = Bool(False, view_changed=True)
    scene = Instance(MlabSceneModel)

    def _scene_default(self):
        return MlabSceneModel()

    bgcolor = Tuple(1.0, 1.0, 1.0)
    fgcolor = Tuple(0.0, 0.0, 0.0)

    mlab = Property(depends_on='input_change')
    '''Get the mlab handle'''

    def _get_mlab(self):
        return self.scene.mlab

    fig = Property()
    '''Figure for 3D visualization.
    '''
    @cached_property
    def _get_fig(self):
        fig = self.mlab.gcf()
        bgcolor = tuple(self.bgcolor)
        fgcolor = tuple(self.fgcolor)
        self.mlab.figure(fig, fgcolor=fgcolor, bgcolor=bgcolor)
        return fig

    button1 = Button('Redraw')

    @on_trait_change('button1,+view_changed')
    def redraw_scene(self):
        # Notice how each mlab call points explicitly to the figure it
        # applies to.
        self.fig
        self.mlab.clf()
        fig = self.mlab.gcf()
        fig.scene.disable_render = True

        if self.yc_abaqus_on:
            f = self.yc_abaqus.f(self.sig_ij)
            self.mlab.contour3d(
                self.sig_i[0], self.sig_i[1], self.sig_i[2], f,
                contours=[0.0], color=(1, 0, 0)
            )

        if self.yc_DP_on:
            f = self.yc_DP.f(self.sig_ij)
            self.mlab.contour3d(
                self.sig_i[0], self.sig_i[1], self.sig_i[2], f,
                contours=[0.0], color=(0, 1, 0)
            )

        if self.yc_WW_on:
            f = self.yc_WW.f(self.sig_ij)
            self.mlab.contour3d(
                self.sig_i[0], self.sig_i[1], self.sig_i[2], f,
                contours=[0.0], color=(0, 0, 1)
            )

        if self.yc_VM_on:
            f = self.yc_VM.f(self.sig_ij)
            f_pipe = self.mlab.contour3d(
                self.sig_i[0], self.sig_i[1], self.sig_i[2], f,
                contours=[0.0], color=(0, 0, 1)
            )

        if self.yc_R_on:
            f = self.yc_R.f(self.sig_ij)
            self.mlab.contour3d(
                self.sig_i[0], self.sig_i[1], self.sig_i[2], f,
                contours=[0.0], color=(0, 0.5, .5)
            )

        fig.scene.disable_render = False

    # The layout of the dialog created
    view = View(
        HSplit(
            Group(
                VGroup(
                    UItem('yc_VM_on', label='visible', full_size=True,
                          resizable=True, width=300),
                    UItem('yc_VM@', resizable=True),
                    label='Von-Mises yield face',
                ),
                VGroup(
                    UItem('yc_DP_on', label='visible', full_size=True,
                          resizable=True, width=300),
                    UItem('yc_DP@', resizable=True),
                    label='Drucker-Prager yield face'
                ),
                VGroup(
                    UItem('yc_R_on', label='visible', full_size=True,
                          resizable=True, width=300),
                    UItem('yc_R@', resizable=True),
                    label='Rankine yield face',
                ),
                VGroup(
                    UItem('yc_abaqus_on', label='visible', full_size=True,
                          resizable=True, width=300),
                    UItem('yc_abaqus@', full_size=True,
                          resizable=True, width=300),
                    label='Abaqus yield face'
                ),
                VGroup(
                    UItem('yc_WW_on', label='visible', full_size=True,
                          resizable=True, width=300),
                    UItem('yc_WW@', resizable=True),
                    label='Willam-Warnke yield face',
                ),
                Spring(),
                VGroup(
                    Item('max_sig', label='compression', full_size=True,
                         resizable=True, width=300),
                    Item('min_sig', label='tension', resizable=True),
                    label='Display range',
                ),
                UItem('button1', resizable=True),
                show_labels=True,
            ),
            Group(
                Item('scene',
                     editor=SceneEditor(scene_class=MayaviScene), height=250,
                     width=300),
                show_labels=False,
            ),
        ),
        resizable=True,
        height=0.7,
        width=0.8
    )


def run_explorer(*args, **kw):
    m = YieldFaceViewer()
    m.redraw_scene()
    m.configure_traits(*args, **kw)


if __name__ == '__main__':
    run_explorer()
