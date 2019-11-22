
''' a dictionary filled with distribution names (keys) and
    scipy.stats.distribution attributes having None
    or 1 shape parameters (values)'''

from math import sqrt
import tempfile

from matplotlib.figure import Figure
from numpy import linspace
from pyface.image_resource import ImageResource
from traits.api import HasTraits, Float, Int, Event, Array, Interface, \
    Tuple, Property, cached_property, Instance, Enum, on_trait_change, Dict
from traitsui.api import Item, View, Group, HSplit, VGroup, Tabbed
from traitsui.api import ModelView
from traitsui.menu import OKButton, CancelButton
from util.traits.editors.mpl_figure_editor import MPLFigureEditor

from .distribution import Distribution
import scipy.stats as stats


distr_dict = {}
distr_enum = []

for distr in stats.distributions.__all__:
    d = stats.distributions.__dict__[distr]
    if hasattr(d, 'numargs'):
        if d.numargs == 0:
            distr_dict[distr] = d
            distr_enum.append(distr)

        elif d.numargs == 1:
            distr_dict[distr] = d
            distr_enum.append(distr)
    beta = stats._continuous_distns.beta
    distr_dict['beta'] = beta
    distr_enum.append('beta')


class IPDistrib(Interface):

    n_segments = Int


class PDistrib(HasTraits):

    implements = IPDistrib

    # puts all chosen continuous distributions distributions defined
    # in the scipy.stats.distributions module as a list of strings
    # into the Enum trait
    distr_choice = Enum(distr_enum)
    distr_dict = Dict(distr_dict)

#    distr_choice = Enum('sin2x', 'weibull_min', 'sin_distr', 'uniform', 'norm')
#    distr_dict = {'sin2x' : sin2x,
#                  'uniform' : uniform,
#                  'norm' : norm,
#                  'weibull_min' : weibull_min,
#                  'sin_distr' : sin_distr}

    # instantiating the continuous distributions
    distr_type = Property(Instance(Distribution), depends_on='distr_choice')

    @cached_property
    def _get_distr_type(self):
        return Distribution(self.distr_dict[self.distr_choice])

    # change monitor - accumulate the changes in a single event trait
    changed = Event

    @on_trait_change('distr_choice, distr_type.changed, quantile, n_segments')
    def _set_changed(self):
        self.changed = True

    # ------------------------------------------------------------------------
    # Methods setting the statistical modments
    # ------------------------------------------------------------------------
    mean = Property

    def _get_mean(self):
        return self.distr_type.mean

    def _set_mean(self, value):
        self.distr_type.mean = value

    variance = Property

    def _get_variance(self):
        return self.distr_type.mean

    def _set_variance(self, value):
        self.distr_type.mean = value

    # ------------------------------------------------------------------------
    # Methods preparing visualization
    # ------------------------------------------------------------------------

    quantile = Float(0.00001, auto_set=False, enter_set=True)
    range = Property(Tuple(Float), depends_on='distr_type.changed, quantile')

    @cached_property
    def _get_range(self):
        return (self.distr_type.distr.ppf(self.quantile),
                self.distr_type.distr.ppf(1 - self.quantile))

    n_segments = Int(500, auto_set=False, enter_set=True)

    dx = Property(Float, depends_on='distr_type.changed, quantile, n_segments')

    @cached_property
    def _get_dx(self):
        range_length = self.range[1] - self.range[0]
        return range_length / self.n_segments

    # -------------------------------------------------------------------------
    # Discretization of the distribution domain
    # -------------------------------------------------------------------------
    x_array = Property(Array('float_'), depends_on='distr_type.changed,'
                       'quantile, n_segments')

    @cached_property
    def _get_x_array(self):
        '''Get the intrinsic discretization of the distribution
        respecting its  bounds.
        '''
        return linspace(self.range[0], self.range[1], self.n_segments + 1)

    # ===========================================================================
    # Access function to the scipy distribution
    # ===========================================================================
    def pdf(self, x):
        return self.distr_type.distr.pdf(x)

    def cdf(self, x):
        return self.distr_type.distr.cdf(x)

    def rvs(self, n):
        return self.distr_type.distr.rvs(n)

    def ppf(self, e):
        return self.distr_type.distr.ppf(e)

    # ===========================================================================
    # PDF - permanent array
    # ===========================================================================

    pdf_array = Property(Array('float_'), depends_on='distr_type.changed,'
                         'quantile, n_segments')

    @cached_property
    def _get_pdf_array(self):
        '''Get pdf values in intrinsic positions'''
        return self.distr_type.distr.pdf(self.x_array)

    def get_pdf_array(self, x_array):
        '''Get pdf values in externally specified positions'''
        return self.distr_type.distr.pdf(x_array)

    # ===========================================================================
    # CDF permanent array
    # ===========================================================================
    cdf_array = Property(Array('float_'), depends_on='distr_type.changed,'
                         'quantile, n_segments')

    @cached_property
    def _get_cdf_array(self):
        '''Get cdf values in intrinsic positions'''
        return self.distr_type.distr.cdf(self.x_array)

    def get_cdf_array(self, x_array):
        '''Get cdf values in externally specified positions'''
        return self.distr_type.distr.cdf(x_array)

    # -------------------------------------------------------------------------
    # Randomization
    # -------------------------------------------------------------------------
    def get_rvs_array(self, n_samples):
        return self.distr_type.distr.rvs(n_samples)


class PDistribView(ModelView):

    def __init__(self, **kw):
        super(PDistribView, self).__init__(**kw)
        self.on_trait_change(
            self.refresh, 'model.distr_type.changed, model.quantile, model.n_segments')
        self.refresh()

    model = Instance(PDistrib)

    figure = Instance(Figure)

    def _figure_default(self):
        figure = Figure(facecolor='white')
        return figure

    data_changed = Event

    def plot(self, fig):
        figure = fig
        figure.clear()
        axes = figure.gca()
        # plot PDF
        axes.plot(self.model.x_array, self.model.pdf_array, lw=1.0, color='blue',
                  label='PDF')
        axes2 = axes.twinx()
        # plot CDF on a separate axis (tick labels left)
        axes2.plot(self.model.x_array, self.model.cdf_array, lw=2, color='red',
                   label='CDF')
        # fill the unity area given by integrating PDF along the X-axis
        axes.fill_between(self.model.x_array, 0, self.model.pdf_array, color='lightblue',
                          alpha=0.8, linewidth=2)
        # plot mean
        mean = self.model.distr_type.distr.stats('m')
        axes.plot([mean, mean], [0.0, self.model.distr_type.distr.pdf(mean)],
                  lw=1.5, color='black', linestyle='-')
        # plot stdev
        stdev = sqrt(self.model.distr_type.distr.stats('v'))
        axes.plot([mean - stdev, mean - stdev],
                  [0.0, self.model.distr_type.distr.pdf(mean - stdev)],
                  lw=1.5, color='black', linestyle='--')
        axes.plot([mean + stdev, mean + stdev],
                  [0.0, self.model.distr_type.distr.pdf(mean + stdev)],
                  lw=1.5, color='black', linestyle='--')

        axes.legend(loc='center left')
        axes2.legend(loc='center right')
        axes.ticklabel_format(scilimits=(-3., 4.))
        axes2.ticklabel_format(scilimits=(-3., 4.))

        # plot limits on X and Y axes
        axes.set_ylim(0.0, max(self.model.pdf_array) * 1.15)
        axes2.set_ylim(0.0, 1.15)
        range = self.model.range[1] - self.model.range[0]
        axes.set_xlim(self.model.x_array[0] - 0.05 * range,
                      self.model.x_array[-1] + 0.05 * range)
        axes2.set_xlim(self.model.x_array[0] - 0.05 * range,
                       self.model.x_array[-1] + 0.05 * range)

    def refresh(self):
        self.plot(self.figure)
        self.data_changed = True

    icon = Property(Instance(
        ImageResource), depends_on='model.distr_type.changed, model.quantile, model.n_segments')

    @cached_property
    def _get_icon(self):
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(4, 4), facecolor='white')
        self.plot(fig)
        tf_handle, tf_name = tempfile.mkstemp('.png')
        fig.savefig(tf_name, dpi=35)
        return ImageResource(name=tf_name)

    traits_view = View(HSplit(VGroup(Group(Item('model.distr_choice', show_label=False),
                                           Item(
                                               '@model.distr_type', show_label=False),
                                           ),
                                     id='pdistrib.distr_type.pltctrls',
                                     label='Distribution parameters',
                                     scrollable=True,
                                     ),
                              Tabbed(Group(Item('figure',
                                                editor=MPLFigureEditor(),
                                                show_label=False,
                                                resizable=True),
                                           scrollable=True,
                                           label='Plot',
                                           ),
                                     Group(Item('model.quantile', label='quantile'),
                                           Item(
                                         'model.n_segments', label='plot points'),
                                  label='Plot parameters'
                              ),
        label='Plot',
        id='pdistrib.figure.params',
        dock='tab',
    ),
        dock='tab',
        id='pdistrib.figure.view'
    ),
        id='pdistrib.view',
        dock='tab',
        title='Statistical distribution',
        buttons=[OKButton, CancelButton],
        scrollable=True,
        resizable=True,
        width=600, height=400
    )


if __name__ == '__main__':
    pdistrib = PDistrib()
    pdistribview = PDistribView(model=pdistrib)
    pdistribview.refresh()
    pdistribview.configure_traits()
