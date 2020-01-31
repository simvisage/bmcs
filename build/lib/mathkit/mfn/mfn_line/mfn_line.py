from matplotlib.figure import \
    Figure
from scipy import interpolate as ip
from traits.api import Array, Float, Event, \
    ToolbarButton, on_trait_change, \
    Property, cached_property, Enum, Instance, Bool
from traitsui.api import View, VGroup, UItem
from util.traits.editors import \
    MPLFigureEditor
from view.ui import BMCSLeafNode

import matplotlib.pyplot as plt
import numpy as np


class MFnLineArray(BMCSLeafNode):

    # Public Traits
    xdata = Array(float, value=[0.0, 1.0])

    def _xdata_default(self):
        '''
        convenience default - when xdata not defined created automatically as
        an array of integers with the same shape as ydata
        '''
        return np.arange(self.ydata.shape[0])

    ydata = Array(float, value=[0.0, 1.0])

    def __init__(self, *args, **kw):
        super(MFnLineArray, self).__init__(*args, **kw)
        self.replot()

    extrapolate = Enum('constant', 'exception', 'diff', 'zero')
    '''
    Vectorized interpolation using scipy.interpolate
    '''

    def values(self, x, k=1):
        '''
        vectorized interpolation, k is the spline order, default set to 1 (linear)
        '''
        tck = ip.splrep(self.xdata, self.ydata, s=0, k=k)

        x = np.array([x]).flatten()

        if self.extrapolate == 'diff':
            values = ip.splev(x, tck, der=0)
        elif self.extrapolate == 'exception':
            if x.all() < self.xdata[0] and x.all() > self.xdata[-1]:
                values = values = ip.splev(x, tck, der=0)
            else:
                raise ValueError('value(s) outside interpolation range')
        elif self.extrapolate == 'constant':
            values = ip.splev(x, tck, der=0)
            values[x < self.xdata[0]] = self.ydata[0]
            values[x > self.xdata[-1]] = self.ydata[-1]
        elif self.extrapolate == 'zero':
            values = ip.splev(x, tck, der=0)
            values[x < self.xdata[0]] = 0.0
            values[x > self.xdata[-1]] = 0.0
        return values

    def __call__(self, x):
        return self.values(x)

    yrange = Property
    '''Get min max values on the vertical axis
    '''

    def _get_yrange(self):
        return np.min(self.ydata), np.max(self.ydata)

    xrange = Property
    '''Get min max values on the vertical axis
    '''

    def _get_xrange(self):
        return np.min(self.xdata), np.max(self.xdata)

    data_changed = Event

    figure = Instance(Figure)

    def _figure_default(self):
        figure = Figure(facecolor='white')
        return figure

    def diff(self, x, k=1, der=1):
        '''
        vectorized interpolation, der is the nth derivative, default set to 1;
        k is the spline order of the data inetrpolation, default set to 1 (linear)
        '''
        xdata = np.sort(np.hstack((self.xdata, x)))
        idx = np.argwhere(np.diff(xdata) == 0).flatten()
        xdata = np.delete(xdata, idx)
        tck = ip.splrep(xdata, self.values(xdata, k=k), s=0, k=k)
        return ip.splev(x, tck, der=der)

    dump_button = ToolbarButton('Print data',
                                style='toolbar')

    @on_trait_change('dump_button')
    def print_data(self, event=None):
        print('x = ', repr(self.xdata))
        print('y = ', repr(self.ydata))

    integ = Property(Float(), depends_on='ydata')

    @cached_property
    def _get_integ(self):
        _xdata = self.xdata
        _ydata = self.ydata
        # integral under the stress strain curve
        return np.trapz(_ydata, _xdata)

    def clear(self):
        self.xdata = np.array([])
        self.ydata = np.array([])

    def plot(self, axes, *args, **kw):
        self.mpl_plot(axes, *args, **kw)

    def mpl_plot(self, axes, *args, **kw):
        '''plot within matplotlib window'''
        axes.plot(self.xdata, self.ydata, *args, **kw)

    def mpl_plot_diff(self, axes, *args, **kw):
        '''plot within matplotlib window'''
        ax_dx = axes.twinx()
        x = np.linspace(self.xdata[0], self.xdata[-1],
                        np.size(self.xdata) * 20.0)
        y_dx = self.diff(x, k=1, der=1)
        ax_dx.plot(x, y_dx, *args + ('-',), **kw)

    plot_diff = Bool(False)

    def replot(self):
        self.figure.clf()
        ax = self.figure.add_subplot(111)
        self.mpl_plot(ax)
        if self.plot_diff:
            self.mpl_plot_diff(ax, color='orange')
        self.data_changed = True

    def savefig(self, fname):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        self.mpl_plot(ax)
        self.mpl_plot_diff(ax, color='orange')
        fig.savefig(fname)

    tree_view = View(
        VGroup(
            VGroup(
                UItem('figure', editor=MPLFigureEditor(),
                      resizable=True,
                      springy=True),
                scrollable=True,
            ),
        )
    )

    traits_view = tree_view


if __name__ == '__main__':
    import pylab as plt

#    from matplotlib import pyplot as plt
    x = np.linspace(-2, 7, 20)
    xx = np.linspace(-4, 8, 100)
    y = np.sin(x)

    mf = MFnLineArray(xdata=x, ydata=y)

    # plots raw data
    def data():
        plt.plot(x, y, 'ro', label='data')

    # plots values with extrapolation as constant value
    def constant():
        mf.extrapolate = 'constant'
        plt.plot(xx, mf(xx), label='constant')
        plt.plot(xx, mf.diff(xx), label='constant diff')

    # plots values with extrapolation as zero
    def zero():
        mf.extrapolate = 'zero'
        plt.plot(xx, mf(xx), label='zero')
        plt.plot(xx, mf.diff(xx), label='zero diff')

    # plots values with extrapolation with constant slope
    def diff():
        mf.extrapolate = 'diff'
        plt.plot(xx, mf(xx), label='diff')
        plt.plot(xx, mf.diff(xx,), label='diff diff')

    # raises an exception if data are outside the interpolation range
    def exception():
        mf.extrapolate = 'exception'
        plt.plot(xx, mf(xx), label='exception')

    data()
    # constant()
    # zero()
    diff()
    # exception()
    plt.legend(loc='best')
    plt.show()

    mf.replot()
    mf.configure_traits()
