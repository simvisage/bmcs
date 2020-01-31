from chaco.api import *
from enable.component_editor import ComponentEditor
from numpy import arange, linspace, random
from traits.api import *
from traitsui.api import *
from traitsui.key_bindings import KeyBinding, KeyBindings


class CodeHandler(Handler, HasTraits):
    info = Any

    def action1(self, info):
        print(info)


class show_trace(HasTraits):

    command = Str
    container = Instance(VPlotContainer)
    pd = Instance(ArrayPlotData)
    plot1 = Instance(Plot)
    key_bindings = KeyBindings(
        KeyBinding(binding1='Ctrl-d',  # CTRL-D works, while D doesn't.
                   description='Restitution Displacement',
                   method_name='action1'),
    )

    view = View(
        Item('container', editor=ComponentEditor()), Item('command'),
        key_bindings=key_bindings,
        handler=CodeHandler(),
        resizable=True,
        buttons=["OK"],
        width=1024, height=768,
        title='DB FrontEnd 2 - Show Trace')

    def __init__(self):
        self.pd = ArrayPlotData()
        x = linspace(0, 100, 101)
        y = random.random_integers(-100, 100, 101)
        self.pd.set_data('index', x)
        self.pd.set_data('value', y)
        p = Plot(self.pd)
        p.plot(('index', 'value'))
        self.plot1 = p
        self.container = VPlotContainer()
        self.container.add(self.plot1)


S = show_trace()
S.configure_traits()
