'''
Created on Jun 28, 2010

@author: rostislav
'''
from matplotlib.figure import Figure
from traits.api import Instance, on_trait_change, Event
from traitsui.api import ModelView, View, Item, VSplit, HGroup
from traitsui.menu import OKButton
from util.traits.editors.mpl_figure_editor import MPLFigureEditor

from .chob import ChainOfBundlesAnalysis as CHOB


class CHOBModelView(ModelView):
    '''
    Size effect depending on the yarn length
    '''

    model = Instance(CHOB)

    def _model_default(self):
        return CHOB()

    figure = Instance(Figure)

    def _figure_default(self):
        figure = Figure(facecolor='white')
        figure.add_axes([0.08, 0.13, 0.85, 0.74])
        return figure

    data_changed = Event(True)

    @on_trait_change('model.+input')
    def refresh(self):
        distribs = self.model.distribs
        xdata = []
        ydata = []
        legend = []
        if self.model.fiber_total is True:
            xdata.append(distribs[0].x)
            ydata.append(distribs[0].pdf)
            legend.append('fiber $l_t$')
        if self.model.bundle_total is True:
            xdata.append(distribs[1].x)
            ydata.append(distribs[1].pdf)
            legend.append('bundle $l_t$')
        if self.model.fiber_bundle is True:
            xdata.append(distribs[2].x)
            ydata.append(distribs[2].pdf)
            legend.append('fiber $l_b$')
        if self.model.bundle_bundle is True:
            xdata.append(distribs[3].x)
            ydata.append(distribs[3].pdf)
            legend.append('bundle $l_b$')
        if self.model.yarn_total is True:
            xdata.append(distribs[4].x)
            ydata.append(distribs[4].pdf)
            legend.append('yarn $l_t$')

        figure = self.figure
        figure.clear()
        axes = figure.gca()

        for x, y, l in zip(xdata[:], ydata[:], legend[:]):
            axes.plot(x, y, linewidth=2, label=l)
        axes.set_xlabel('strength', weight='semibold')
        axes.set_ylabel('PDF', weight='semibold')
        axes.set_title('chain of bundles analysis',
                       size='large', color='black',
                       weight='bold')
        axes.set_axis_bgcolor(color='white')
        axes.ticklabel_format(scilimits=(-3., 4.))
        axes.grid(color='gray', linestyle='--', linewidth=0.1, alpha=0.4)
        axes.legend(loc='best')

        self.data_changed = True

    traits_view = View(VSplit(
        HGroup(
            Item('model@', show_label=False, resizable=True),
            label='data sheet',
            id='chob.viewmodel.model',
            dock='tab',
        ),
        HGroup(Item('figure', editor=MPLFigureEditor(),
                    resizable=True, show_label=False),
               label='plot sheet',
               id='chob.viewmodel.figure_window',
               dock='tab',
               ),
        id='chob.vsplit'
    ),
        title='chain of bundles',
        id='chob.viewmodel',
        dock='tab',
        resizable=True,
        buttons=[OKButton],
        height=0.8, width=0.8)


def run():
    chob = CHOBModelView(model=CHOB())
    chob.configure_traits()

if __name__ == '__main__':
    run()
