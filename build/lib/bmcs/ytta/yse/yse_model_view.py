'''
Created on Sep 22, 2009

@author: rostislav
'''
from matplotlib.figure import Figure
from traits.api import \
    Instance, DelegatesTo, Bool, on_trait_change, Event
from traitsui.api import \
    View, Item, VGroup, HGroup, ModelView, HSplit, VSplit
from traitsui.menu import OKButton
from util.traits.editors.mpl_figure_editor import MPLFigureEditor

from .yse import YSE


class YSEModelView(ModelView):
    '''
    Size effect depending on the yarn length
    '''
    model = Instance(YSE)

    def _model_default(self):
        return YSE()

    l_plot = DelegatesTo('model')

    min_plot_length = DelegatesTo('model')

    n_points = DelegatesTo('model')

    test_2_on = Bool(True,
                     modified=True)

    test_3_on = Bool(True,
                     modified=True)

    Asymp = Bool(True,
                 modified=True)

    Gauss = Bool(True,
                 modified=True)
    stdev = Bool(False,
                 modified=True)

    mean_app = Bool(False,
                    modified=True)

    figure = Instance(Figure)

    def _figure_default(self):
        figure = Figure(facecolor='white')
        figure.add_axes([0.08, 0.13, 0.85, 0.74])
        return figure

    test_length_r = DelegatesTo('model')
    test_length_f = DelegatesTo('model')

    data_changed = Event(True)

    @on_trait_change('+modified,model.+modified')
    def _redraw(self):

        data = self.model._get_values()
        xdata = []
        ydata = []
        linecolor = []
        linewidth = []
        linestyle = []
        legend = []

        if self.Gauss is True:
            xdata.append(data[0])
            ydata.append(data[2])
            linecolor.append('blue')
            linewidth.append(2)
            linestyle.append('solid')
            legend.append('median Gauss')
        if self.Asymp is True:
            xdata.append(data[1])
            ydata.append(data[3])
            linecolor.append('black')
            linewidth.append(1)
            linestyle.append('dashed')
            legend.append('asymptotes')

        if self.test_2_on:
            xdata.append(self.test_length_r[0])
            ydata.append(self.test_length_r[1])
            linecolor.append('red')
            linewidth.append(1)
            linestyle.append('dashed')
            legend.append('test - single bundle')

        if self.test_3_on:
            xdata.append(self.test_length_f[0])
            ydata.append(self.test_length_f[1])
            linecolor.append('red')
            linewidth.append(1)
            linestyle.append('dashed')
            legend.append('test - chain of bundles')

        if self.mean_app is True:
            xdata.append(data[7])
            xdata.append(data[7])
            xdata.append(data[7])
            ydata.append(data[8])
            ydata.append(data[9])
            ydata.append(data[10])
            linecolor.append('red')
            linecolor.append('black')
            linecolor.append('green')
            linewidth.append(1)
            linewidth.append(1)
            linewidth.append(1)
            linestyle.append('dashed')
            linestyle.append('dotted')
            linestyle.append('solid')
            legend.append('mean Mirek')
            legend.append('mean Gumbel')
            legend.append('median Gumbel')
        if self.stdev and self.Gauss is True:
            xdata.append(data[4])
            xdata.append(data[4])
            ydata.append(data[5])
            ydata.append(data[6])
            linecolor.append('lightblue')
            linecolor.append('lightblue')
            linewidth.append(2)
            linewidth.append(2)
            linestyle.append('solid')
            linestyle.append('solid')
            legend.append('st. deviation')
            test_on = True
        if self.stdev is False or self.Gauss is False:
            test_on = False

        figure = self.figure
        figure.clear()
        axes = figure.gca()

        for x, y, c, w, s in zip(xdata[:], ydata[:], linecolor[:],
                                 linewidth[:], linestyle[:]):
            axes.plot(x, y, color=c, linewidth=w, linestyle=s)

        if test_on is True:
            axes.fill_between(xdata[-1], ydata[len(ydata) - 1],
                              ydata[len(xdata) - 2], color='lightblue', )

        else:
            pass
        axes.set_xlabel('length [m]', weight='semibold')
        axes.set_ylabel('stress [MPa]', weight='semibold')
        axes.set_title('strength size effect',
                       size='large', color='black',
                       weight='bold', position=(.5, 1.03))
        axes.set_axis_bgcolor(color='white')
        axes.ticklabel_format(scilimits=(-3., 4.))
        axes.grid(color='gray', linestyle='--', linewidth=0.1, alpha=0.4)
        axes.legend((legend), loc='best')
        axes.set_xscale('log')  # , subsx = [0, 1, 2, 3 ] )
        axes.set_yscale('log')  # , subsy = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] )
        # axes.set_xlim(10.e-4)

        limits = max(ydata[:][0]) - min(ydata[:][0])

        # annotations
        #
        x, y = self.min_plot_length, self.model.mu_sigma_0
        axes.annotate('%.1f' % y, xy=(x, y), xytext=(x * 1.1, y + limits * 0.08),  # fontsize=18,family='serif',
                      )

        x, y = self.model.l_b, self.model.mu_b
        axes.annotate('[%.3f,%.1f]' % (x, y), xy=(x, y),
                      # fontsize=18,family='serif',
                      xytext=(x * 1.1, y + limits * 0.08),
                      arrowprops=dict(facecolor='blue', shrink=0.01)
                      )

        axes.set_ylim(
            min(ydata[:][0]) - limits * 0.15, max(ydata[:][0]) + limits * 0.30)
        self.data_changed = True

    traits_view = View(HSplit(
        VGroup(
            Item('model@', show_label=False, resizable=True),
            label='data sheet',
            id='yse.viewmodel.model',
            dock='tab',
        ),
        VSplit(
            VGroup(
                Item('figure', editor=MPLFigureEditor(),
                     resizable=True, show_label=False),
                label='plot sheet',
                id='yse.viewmodel.figure_window',
                dock='tab',
            ),
            VGroup(
                HGroup(
                    Item('min_plot_length', label='minimum [m]',
                         springy=True),
                    Item('l_plot', label='maximum [m]',
                         springy=True),
                    Item('n_points', label='plot points',
                         springy=True),
                ),
                VGroup(
                    Item('test_2_on', label='I. test'),
                    Item('Asymp', label='asymptotes'),
                    Item('Gauss', label='Gauss'),
                    Item('test_3_on', label='II. test'),
                    Item('stdev', label='st. deviadion'),
                    Item('mean_app', label='mean approx.'),
                    columns=3,
                ),
                label='plot parameters',
                id='yse.viewmodel.view_params',
                dock='tab',
            ),
            id='yse.viewmodel.right',
        ),
        id='yse.viewmodel.splitter',
    ),
        title='Yarn Size Effect',
        id='yse.viewmodel',
        dock='tab',
        resizable=True,
        height=0.8, width=0.8,
        buttons=[OKButton])


def run():
    yse = YSEModelView(model=YSE())
    yse.configure_traits()

if __name__ == '__main__':
    run()
