import numpy as np
from enthought.traits.api import Array, Float, Event, HasTraits, \
                                 ToolbarButton, on_trait_change, \
                                 Property, cached_property, Enum

from scipy import interpolate as ip

class MFnLineArray(HasTraits):

    # Public Traits
    xdata = Array(float, value = [0.0, 1.0])
    def _xdata_default(self):
        '''
        convenience default - when xdata not defined created automatically as
        an array of integers with the same shape as ydata
        '''
        return np.arange(self.ydata.shape[0])

    ydata = Array(float, value = [0.0, 1.0])

    extrapolate = Enum('constant', 'exception', 'diff', 'zero')

    # alternative vectorized interpolation using scipy.interpolate   
    def get_values(self, x, k = 1):
        '''
        vectorized interpolation, k is the spline order, default set to 1 (linear)
        '''
        tck = ip.splrep(self.xdata, self.ydata, s = 0, k = k)

        x = np.array([x]).flatten()

        if self.extrapolate == 'diff':
            values = ip.splev(x, tck, der = 0)
        elif self.extrapolate == 'exception':
            if x.all() < self.xdata[0] and x.all() > self.xdata[-1]:
                values = values = ip.splev(x, tck, der = 0)
            else:
                raise ValueError('value(s) outside interpolation range')

        elif self.extrapolate == 'constant':
            mask = x >= self.xdata[0]
            mask *= x <= self.xdata[-1]
            l_mask = x < self.xdata[0]
            r_mask = x > self.xdata[-1]
            extrapol_left = np.repeat(ip.splev(self.xdata[0], tck, der = 0), len(x)) * l_mask
            extrapol_right = np.repeat(ip.splev(self.xdata[-1], tck, der = 0), len(x)) * r_mask
            extrapol = extrapol_left + extrapol_right
            values = ip.splev(x, tck, der = 0) * mask + extrapol
        elif self.extrapolate == 'zero':
            mask = x >= self.xdata[0]
            mask *= x <= self.xdata[-1]
            mask_extrapol = mask == False
            extrapol = np.zeros(len(x)) * mask_extrapol
            values = ip.splev(x, tck, der = 0) * mask + extrapol
        return values

    def get_value(self, x):
        x2idx = self.xdata.searchsorted(x)
        if x2idx == len(self.xdata):
            x2idx -= 1
        x1idx = x2idx - 1
        x1 = self.xdata[ x1idx ]
        x2 = self.xdata[ x2idx ]
        dx = x2 - x1
        y1 = self.ydata[ x1idx ]
        y2 = self.ydata[ x2idx ]
        dy = y2 - y1
        y = y1 + dy / dx * (x - x1)
        return y

    data_changed = Event

    def get_diffs(self, x, k = 1, der = 1):
        '''
        vectorized interpolation, der is the nth derivative, default set to 1;
        k is the spline order of the data inetrpolation, default set to 1 (linear)
        '''
        xdata = np.sort(np.hstack((self.xdata, x)))
        idx = np.argwhere(np.diff(xdata) == 0).flatten()
        xdata = np.delete(xdata, idx)
        tck = ip.splrep(xdata, self.get_values(xdata, k = k), s = 0, k = k)
        return ip.splev(x, tck, der = der)

    def get_diff(self, x):
        x2idx = self.xdata.searchsorted(x)
        if x2idx == len(self.xdata):
            x2idx -= 1
        x1idx = x2idx - 1
        x1 = self.xdata[ x1idx ]
        x2 = self.xdata[ x2idx ]
        dx = x2 - x1
        y1 = self.ydata[ x1idx ]
        y2 = self.ydata[ x2idx ]
        dy = y2 - y1
        return dy / dx

    dump_button = ToolbarButton('Print data',
                                style = 'toolbar')
    @on_trait_change('dump_button')
    def print_data(self, event = None):
        print 'x = ', repr(self.xdata)
        print 'y = ', repr(self.ydata)

    integ_value = Property(Float(), depends_on = 'ydata')
    @cached_property
    def _get_integ_value(self):
        _xdata = self.xdata
        _ydata = self.ydata
        # integral under the stress strain curve
        E_t = np.trapz(_ydata, _xdata)
        # area of the stored elastic energy  
        U_t = 0.0
        if len(_xdata) != 0:
            U_t = 0.5 * _ydata[-1] * _xdata[-1]
        return E_t - U_t

    def clear(self):
        self.xdata = np.array([])
        self.ydata = np.array([])

    def plot(self, axes, *args, **kw):
        self.mpl_plot(axes, *args, **kw)

    def mpl_plot(self, axes, *args, **kw):
        '''plot within matplotlib window'''
        axes.plot(self.xdata, self.ydata, *args, **kw)

if __name__ == '__main__':
    import pylab as plt

#    from matplotlib import pyplot as plt
    x = np.linspace(-2, 7, 20)
    xx = np.linspace(-4, 8, 100)
    y = np.sin(x)

    mf = MFnLineArray(xdata = x, ydata = y)

    # plots raw data
    def data():
        plt.plot(x, y, 'ro', label = 'data')

    # plots interpolation and extrapolation using scalar methods
    def scalar():
        plt.plot(xx, [mf.get_value(xi) for xi in xx], label = 'values scalar')
        plt.plot(xx, [mf.get_diff(xi) for xi in xx], label = 'diff scalar')

    # plots values with extrapolation as constant value
    def constant():
        mf.extrapolate = 'constant'
        plt.plot(xx, mf.get_values(xx), label = 'constant')
        plt.plot(xx, mf.get_diffs(xx,), label = 'constant diff')

    # plots values with extrapolation as zero
    def zero():
        mf.extrapolate = 'zero'
        plt.plot(xx, mf.get_values(xx), label = 'zero')
        plt.plot(xx, mf.get_diffs(xx,), label = 'zero diff')

    # plots values with extrapolation with constant slope
    def diff():
        mf.extrapolate = 'diff'
        plt.plot(xx, mf.get_values(xx), label = 'diff')
        plt.plot(xx, mf.get_diffs(xx,), label = 'diff diff')

    # raises an exception if data are outside the interpolation range
    def exception():
        mf.extrapolate = 'exception'
        plt.plot(xx, mf.get_values(xx), label = 'diff')

    data()
    #scalar()
    constant()
    #zero()
    #diff()
    #exception()
    plt.legend(loc = 'best')
    plt.show()
