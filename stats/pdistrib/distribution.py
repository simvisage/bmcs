''' The Distribution class computes, for a statistical distribution function, the
    statistical parameters from a given set of statistical moments and vice versa.
    The __init__ method takes a scipy.stats distribution or in general a method
    which takes the parameters location, scale (optional shape1, shape2...) and
    returns statistical moments, pdf, cdf and ppf (percent point function - inverse
    cdf). The pdf, cdf, mean and stdev are plotted in the module pdistrib.py'''

from decimal import Decimal
import decimal

from numpy import array, arange, hstack, zeros, infty
from scipy.optimize import fsolve
import scipy.stats
from scipy.stats.distributions import rv_continuous
from traits.api import HasTraits, Float, Property, cached_property, \
    Event, Array, Instance, Bool
from traitsui.api import Item, View, Group, VGroup, Label

import numpy as np


class Distribution(HasTraits):
    ''' takes a scipy.stats distribution '''

    def __init__(self, distribution, **kw):
        super(Distribution, self).__init__(**kw)
        self.distribution = distribution
        self.changes()

    distribution = Instance(rv_continuous)

    def add_listeners(self):
        self.on_trait_change(self.changes, '+params,+moments')

    def remove_listeners(self):
        self.on_trait_change(self.changes, '+params,+moments', remove=True)

    # precision for displayed numbers = 12 numbers corresponds with the numbers
    # displayed in the UI.
    decimal.getcontext().prec = 12

    # event that triggers the replot in pdistrib.py
    changed = Event

    # freezes the location to 0.0
    loc_zero = Bool(True)

    # old values are compared with new values to recognize which value changed
    old_values = Array(Float, value=zeros(7))
    new_values = Array(Float, value=zeros(7))

    # statistical parameters
    loc = Float(0.0, auto_set=False, enter_set=True, params=True)
    scale = Float(1.0, auto_set=False, enter_set=True, params=True)
    shape = Float(1.0, auto_set=False, enter_set=True, params=True)

    # statistical moments
    mean = Float(0.0, auto_set=False, enter_set=True, moments=True)
    variance = Float(0.0, auto_set=False, enter_set=True, moments=True)
    skewness = Float(0.0, auto_set=False, enter_set=True, moments=True)
    kurtosis = Float(0.0, auto_set=False, enter_set=True, moments=True)

    stdev = Property(depends_on='variance')

    def _get_stdev(self):
        return self.variance ** (0.5)

    def get_mean(self):
        ''' Methods for evaluating the statistical moments. Decimal together with
        precision are needed in order to get the number which is actually displayed
        in the UI. Otherwise clicking in the interface or pressing enter on the
        displayed values would trigger new computation because these values are a
        representation of the computed values rounded to 12 numbers. '''
        self.mean = float(Decimal(str((self.distr.stats('m')))) / 1)

    def get_variance(self):
        self.variance = float(Decimal(str((self.distr.stats('v')))) / 1)

    def get_skewness(self):
        self.skewness = float(Decimal(str((self.distr.stats('s')))) / 1)

    def get_kurtosis(self):
        self.kurtosis = float(Decimal(str((self.distr.stats('k')))) / 1)

    def get_moments(self, specify):
        ''' specify is a string containing some of the letters 'mvsk' '''
        self.remove_listeners()

        moments = self.distr.stats(specify)

        moment_names = ['mean', 'variance', 'skewness', 'kurtosis']
        for idx, value in enumerate(moments):
            setattr(self, moment_names[idx][0], value)

        dict = {'m': self.get_mean,
                'v': self.get_variance,
                's': self.get_skewness,
                'k': self.get_kurtosis}

        # chooses the methods to calculate the three moments which didn't
        # trigger this method
        for idx in specify:
            dict[idx]()

        self.add_listeners()

    def changes(self):
        ''' coordinates the methods for computing
        parameters and moments when a change has occurred '''
        self.remove_listeners()
        self.new_values = array([self.shape, self.loc, self.scale, self.mean,
                                 self.variance, self.skewness, self.kurtosis])
        # test which parameters or moments are significant
        print((self.old_values))
        print((self.new_values))
        diff_old_new = abs(self.old_values - self.new_values)
        indexing = np.where(diff_old_new != 0)[0]
        print(('indexing', indexing))
        #indexing = arange(8)[ix]
        if len(indexing) > 0 and indexing[0] < 3:
            self.get_moments('mvsk')
        elif len(indexing) > 0 and indexing[0] > 2:
            self.param_methods[indexing[0] - 3]()
        else:
            pass
        self.old_values = array([self.shape, self.loc, self.scale, self.mean,
                                 self.variance, self.skewness, self.kurtosis])
        self.add_listeners()
        self.changed = True

    param_methods = Property(Array, depends_on='distribution')

    @cached_property
    def _get_param_methods(self):
        methods = array([self.mean_change, self.variance_change_scale,
                         self.variance_change_shape, self.skewness_change,
                         self.kurtosis_change])
        if self.distribution.shapes == None:
            return methods[0:2]
        else:
            if len(self.distribution.shapes) == 1:
                return hstack((methods[0], methods[2:5]))
            else:
                print('more than 1 shape parameters')

    def shape_scale_mean_var_residuum(self, params):
        shape = params[0]
        scale = params[1]
        res_mean = self.mean - self.distribution(shape,
                                                 loc=self.loc, scale=scale).stats('m')
        res_var = self.variance - self.distribution(shape,
                                                    loc=self.loc, scale=scale).stats('v')
        return [res_mean, res_var]

    def mean_change(self):
        if self.loc_zero == True and self.distribution.__dict__['shapes'] != None:
            self.loc = 0.0
            result = fsolve(self.shape_scale_mean_var_residuum, [1., 1.])
            self.shape = float(Decimal(str(result[0].sum())) / 1)
            self.scale = float(Decimal(str(result[1].sum())) / 1)
        else:
            self.loc += float(Decimal(str(self.mean -
                                          self.distr.stats('m'))) / 1)

    def scale_variance_residuum(self, scale):
        return self.variance - self.distribution(
            loc=self.loc, scale=scale).stats('v')

    def variance_change_scale(self):
        self.scale = float(
            Decimal(str(fsolve(self.scale_variance_residuum, 1).sum())) / 1)

    def shape_variance_residuum(self, shape):
        return self.variance - self.distribution(shape,
                                                 loc=self.loc, scale=self.scale).stats('v')

    def variance_change_shape(self):
        self.shape = float(
            Decimal(str(fsolve(self.shape_variance_residuum, 1).sum())) / 1)
        self.get_moments('msk')

    def shape_skewness_residuum(self, shape):
        return self.skewness - self.distribution(shape,
                                                 loc=self.loc, scale=self.scale).stats('s')

    def skewness_change(self):
        self.shape = float(
            Decimal(str(fsolve(self.shape_skewness_residuum, 1).sum())) / 1)
        self.get_moments('mvk')

    def shape_kurtosis_residuum(self, shape):
        return self.kurtosis - self.distribution(shape,
                                                 loc=self.loc, scale=self.scale).stats('k')

    def kurtosis_change(self):
        self.shape = float(
            Decimal(str(fsolve(self.shape_kurtosis_residuum, 1).sum())) / 1)
        self.get_moments('mvs')

    distr = Property(depends_on='+params')

    @cached_property
    def _get_distr(self):
        if self.distribution.__dict__['numargs'] == 0:
            return self.distribution(self.loc, self.scale)
        elif self.distribution.__dict__['numargs'] == 1:
            return self.distribution(self.shape, self.loc, self.scale)
        elif self.distribution.__dict__['numargs'] == 2:
            return self.distribution(self.shape, self.kurtosis,
                                     self.loc, self.scale)
        else:
            print(('Number of arguments', self.distribution.numargs))

    def default_traits_view(self):
        '''checks the number of shape parameters of the distribution and adds them to
        the view instance'''
        label = str(self.distribution.name)
        if self.distribution.shapes == None:
            params = Item()
            if self.mean == infty:
                moments = Item(label='No finite moments defined')
            else:
                moments = Item('mean', label='mean'), \
                    Item('variance', label='variance'), \
                    Item('stdev', label='st. deviation', style='readonly')

        elif len(self.distribution.shapes) == 1:
            params = Item('shape', label='shape')
            if self.mean == infty:
                moments = Item(label='No finite moments defined')
            else:
                moments = Item('mean', label='mean'), \
                    Item('variance', label='variance'), \
                    Item('stdev', label='st. deviation', style='readonly'), \
                    Item('skewness', label='skewness'), \
                    Item('kurtosis', label='kurtosis'),
        else:
            params = Item('shape', label='shape')
            moments = Item('mean', label='mean'), \
                Item('variance', label='variance'), \
                Item('stdev', label='st. deviation', style='readonly'), \
                Item('skewness', label='skewness'), \
                Item('kurtosis', label='kurtosis'),

        view = View(VGroup(Label(label, emphasized=True),
                           Group(params,
                                 Item('loc', label='location'),
                                 Item('scale', label='scale'),
                                 Item('loc_zero', label='loc = 0.0'),
                                 show_border=True,
                                 label='parameters',
                                 id='pdistrib.distribution.params'
                                 ),
                           Group(moments,
                                 id='pdistrib.distribution.moments',
                                 show_border=True,
                                 label='moments',
                                 ),
                           id='pdistrib.distribution.vgroup'
                           ),
                    kind='live',
                    resizable=True,
                    id='pdistrib.distribution.view'
                    )
        return view


if __name__ == '__main__':
    #    distr = Distribution(scipy.stats.norm)
    distr = Distribution(scipy.stats._continuous_distns.beta)
    distr.configure_traits()
