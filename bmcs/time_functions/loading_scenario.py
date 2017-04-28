
from bmcs.view.ui import BMCSTreeNode, BMCSLeafNode
from matplotlib.figure import Figure
from scipy.interpolate import interp1d
from traits.api import \
    Property, Instance, cached_property, Str, Button, Enum, \
    Range, on_trait_change, Array, List, Float
from traitsui.api import \
    View, Item, Group, VGroup
from util.traits.editors import MPLFigureEditor


class TimeFunctionInteractive(BMCSLeafNode):

    node_name = Str('Loading Scenario')
    number_of_cycles = Float(1.0)
    maximum_loading = Float(0.2)
    unloading_ratio = Range(0., 1., value=0.5)
    number_of_increments = Float(10)
    loading_type = Enum("Monotonic", "Cyclic")
    amplitude_type = Enum("Increased_Amplitude", "Constant_Amplitude")
    loading_range = Enum("Non_symmetric", "Symmetric")

    time = Range(0.00, 1.00, value=1.00)

    d_t = Float(0.005)
    t_max = Float(1.)
    k_max = Float(200)
    tolerance = Float(1e-4)

    d_array = Property(
        depends_on=' maximum_loading , number_of_cycles , loading_type , loading_range , amplitude_type, unloading_ratio')
