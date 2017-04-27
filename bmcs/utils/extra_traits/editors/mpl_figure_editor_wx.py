
import wx

import matplotlib
from matplotlib.backends.backend_wx import NavigationToolbar2Wx, StatusBarWx
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure
from traits.api import Instance
from traitsui.wx.basic_editor_factory import BasicEditorFactory
from traitsui.wx.editor import Editor


# We want matplotlib to use a wxPython backend
matplotlib.use('WXAgg')


class _MPLFigureEditor(Editor):

    scrollable = True

    def init(self, parent):
        self.control = self._create_canvas(parent)
        self.object.on_trait_change(self.update_editor, 'data_changed')
        self.set_tooltip()

    def update_editor(self):
        figure = self.value
        figure.canvas.mpl_connect('key_press_event', self.key_press_callback)
        figure.canvas.draw()

    def _create_canvas(self, parent):
        """ Create the MPL canvas. """
        # The panel lets us add additional controls.
        panel = wx.Panel(parent, -1, style=wx.CLIP_CHILDREN)
        sizer = wx.BoxSizer(wx.VERTICAL)
        panel.SetSizer(sizer)

        # matplotlib commands to create a canvas
        mpl_control = FigureCanvas(panel, 1, self.value)
        statbar = StatusBarWx(panel)
        toolbar = NavigationToolbar2Wx(mpl_control)
        toolbar.set_status_bar(statbar)

        sizer.Add(toolbar, 0, wx.EXPAND)
        sizer.Add(mpl_control, 1, wx.EXPAND)

        sizer.Add(toolbar.statbar, 0, wx.EXPAND)
        self.value.canvas.SetMinSize((100, 100))
        return panel

    # TODO: generalize
    def key_press_callback(self, event):
        'whenever a key is pressed'
        figure = self.value
        if not event.inaxes:
            return
        if event.key == 'k':
            if figure.axes[0].get_xscale() == 'log':
                figure.axes[0].set_xscale('linear')
                figure.canvas.draw()
            else:
                figure.axes[0].set_xscale('log')
                figure.canvas.draw()

        if event.key == 'l':
            if figure.axes[0].get_yscale() == 'log':
                figure.axes[0].set_yscale('linear')
                figure.canvas.draw()
            else:
                figure.axes[0].set_yscale('log')
                figure.canvas.draw()


class MPLFigureEditor(BasicEditorFactory):

    klass = _MPLFigureEditor


if __name__ == "__main__":
    # Create a window to demo the editor
    from traits.api import HasTraits
    from traitsui.api import View, Item
    from numpy import sin, cos, linspace, pi

    class Test(HasTraits):

        figure = Instance(Figure, ())

        view = View(Item('figure', editor=MPLFigureEditor(),
                         show_label=False),
                    width=400,
                    height=300,
                    resizable=True)

        def __init__(self):
            super(Test, self).__init__()
            axes = self.figure.add_subplot(111)
            t = linspace(0, 2 * pi, 200)
            axes.plot(
                sin(t) * (1 + 0.5 * cos(11 * t)), cos(t) * (1 + 0.5 * cos(11 * t)))

    Test().configure_traits()
