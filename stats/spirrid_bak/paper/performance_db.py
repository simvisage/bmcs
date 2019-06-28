#-------------------------------------------------------------------------------
#
# Copyright (c) 2009, IMB, RWTH Aachen.
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in simvisage/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.simvisage.com/licenses/BSD.txt
#
# Thanks for using Simvisage open source!
#
# Created on Jan 17, 2011 by: rch

from math import pi
import pickle

from matplotlib.figure import Figure
from numpy import sign, exp, sqrt, cos, max as nmax, array, power as pow, arange
from stats.spirrid_bak import RF, IRF, SPIRRID
from stats.spirrid_bak.rf_filament import Filament as RFFilament
from util.traits.editors.mpl_figure_editor import MPLFigureEditor

from etsproxy.traits.api import \
    HasTraits, Int, Float, Str, \
    Bool, Dict, Property, cached_property, \
    Instance, Array, WeakRef, List, Tuple, Event, Button, \
    on_trait_change
from etsproxy.traits.ui.api import \
    View, Item, HGroup
from matresdev.db.simdb.simdb_class import \
    SimDBClass, SimDBClassExt
import numexpr as np

config_dict = {'I' :
            (
             {'cached_dG'         : False,
              'compiled_QdG_loop'  : True,
              'compiled_eps_loop' : True },
              'bx-',
              '$\mathrm{C}_{{e},\\theta} ( q(e,\\theta) \cdot g[\\theta_1]  g[\\theta_2] \dots g[\\theta_n] ) $'
             ),
            'II' :
            (
             {'cached_dG'         : True,
              'compiled_QdG_loop'  : True,
              'compiled_eps_loop' : True },
              'go-',
              '$\mathrm{C}_{e,\\theta} ( q(e,\\theta) \cdot G[\\theta] ) $',
              ),
            'III' :
            (
             {'cached_dG'         : False,
              'compiled_QdG_loop'  : True,
              'compiled_eps_loop' : False },
             'r-2',
             '$\mathrm{P}_{e} ( \mathrm{C}_{\\theta} ( q(e,\\theta) \cdot g[\\theta_1]  g[\\theta_2] \dots g[\\theta_n] ) ) $',
             ),
            'IV' :
            (
             {'cached_dG'         : True,
              'compiled_QdG_loop'  : False,
              'compiled_eps_loop' : False },
             'y--',
             '$\mathrm{P}_{e} ( \mathrm{N}_{\\theta} ( q(e,\\theta) \cdot G[\\theta] ) ) $'
             ),
            }


class RFBase(RF):

    implements(IRF)

    title = Str('RFSumParams')

    a = Float(1, auto_set=False, enter_set=True,
                distr=['uniform'],
                loc=1.0, scale=0.1, shape=0.1)

    b = Float(1, auto_set=False, enter_set=True,
                distr=['uniform'],
                loc=1.0, scale=0.1, shape=0.1)

    c = Float(1, auto_set=False, enter_set=True,
                distr=['uniform'],
                loc=1.0, scale=0.1, shape=0.1)

    d = Float(1, auto_set=False, enter_set=True,
                distr=['uniform'],
                loc=1.0, scale=0.1, shape=0.1)

    ee = Float(1, auto_set=False, enter_set=True,
                distr=['uniform'],
                loc=1.0, scale=0.1, shape=0.1)

    f = Float(1, auto_set=False, enter_set=True,
                distr=['uniform'],
                loc=1.0, scale=0.1, shape=0.1)

    g = Float(1, auto_set=False, enter_set=True,
                distr=['uniform'],
                loc=1.0, scale=0.1, shape=0.1)

    h = Float(1, auto_set=False, enter_set=True,
                distr=['uniform'],
                loc=1.0, scale=0.1, shape=0.1)

    i = Float(1, auto_set=False, enter_set=True,
                distr=['uniform'],
                loc=1.0, scale=0.1, shape=0.1)

    k = Float(1, auto_set=False, enter_set=True,
                distr=['uniform'],
                loc=1.0, scale=0.1, shape=0.1)


class RFSumParam(RFBase):

    C_code = '''
            // Computation of the q( ... ) function
            q = ( ( a ) + ( b ) + ( c ) + ( d ) + ( ee ) + ( f ) + ( g ) + ( h ) + ( i ) + (k) ) * eps;
        '''

    def __call__(self, eps, a, b, c, d, ee, f, g, h, i, k):
        '''
        Implements the response function with arrays as variables.
        first extract the variable discretizations from the orthogonal grid.
        '''
        # return np.evaluate( '( ( a ) + ( b ) + ( c ) + ( d ) + ( ee ) + ( f ) + ( g ) + ( h ) + ( i ) + ( k ) ) * eps' )

        return ((a) + (b) + (c) + (d) + (ee) + (f) + (g) + (h) + (i) + (k)) * eps


class RFCosParam(RFBase):

    C_code = '''
            // Computation of the q( ... ) function
            q = ( cos( a ) + cos( b ) + cos( c ) + cos( d ) + cos( ee ) + cos( f ) + cos( g ) + cos( h ) + cos( i ) + cos( k )) * eps;
        '''

    def __call__(self, eps, a, b, c, d, ee, f, g, h, i, k):
        '''
        Implements the response function with arrays as variables.
        first extract the variable discretizations from the orthogonal grid.
        '''
        return (cos(a) + cos(b) + cos(c) + cos(d) + cos(ee) + cos(f) + cos(g) + cos(h) + cos(i) + cos(k)) * eps


class RFPowParam(RFBase):

    C_code = '''
            // Computation of the q( ... ) function
            q = ( a + 
                b * b + 
                c * c * c +
                d * d * d * d +
                ee * ee * ee * ee * ee +
                f * f * f * f * f * f +
                g * g * g * g * g * g * g + 
                h * h * h * h * h * h * h * h + 
                i * i * i * i * i * i * i * i * i +
                k * k * k * k * k * k * k * k * k * k ) * eps;
        '''

    def __call__(self, eps, a, b, c, d, ee, f, g, h, i, k):
        '''
        Implements the response function with arrays as variables.
        first extract the variable discretizations from the orthogonal grid.
        '''
        return (a + pow(b, 2) + pow(c, 3) + pow(d, 4) + pow(ee, 5) + pow(f, 6) \
                +pow(g, 7) + pow(h, 8) + pow(i, 9) + pow(k, 10)) * eps;


class SingleRun(HasTraits):
    '''Pairing of algorithm configuration and randomization pattern.
    '''
    rf = Property

    def _get_rf(self):
        return self.run_table.rf

    config = Tuple

    conf_idx = Property

    def _get_conf_idx(self):
        return self.config[0]

    conf_options = Property

    def _get_conf_options(self):
        return self.config[1][0]

    conf_latex_label = Property

    def _get_conf_latex_label(self):
        return self.config[1][1]

    rand_idx_arr = Array(int, rand=True)

    run_table = WeakRef

    # derived stuff
    s = Property

    def _get_s(self):
        return self.run_table.s

    n_rv = Property

    def _get_n_rv(self):
        return len(self.rand_idx_arr)

    memsize = Property

    def _get_memsize(self):
        return self.run_table.memsize

    n_int = Property

    def _get_n_int(self):
        return int(pow(self.memsize, 1 / float(self.n_rv)))

    real_memsize = Property

    def _get_real_memsize(self):
        return pow(self.n_int, self.n_rv)

    exec_time = Property

    def _get_exec_time(self):

        '''Run spirrid with the given dictionary of configurations.
        '''
        s = self.s

        # apply the randomization pattern
        s.clear_rv()
        for rv in self.rand_idx_arr:
            param_key = self.rf.param_keys[ rv ]
            s.add_rv(param_key, n_int=self.n_int)

        # setup the spirrid exec configuration
        s.set(**self.conf_options)

        return s.exec_time

    def __str__(self):
        return 'conf: %s, rand_idx_arr %s' % (self.conf_options, self.rand_idx_arr)


class RunTable(SimDBClass):
    '''Manage the combinations of exec configurations and randomization patterns.
    '''

    name = Str(simdb=True)

    memsize = Float(1e4, simdb=True)

    s = Property(Instance(SPIRRID), depends_on='rf')

    @cached_property
    def _get_s(self):
        return SPIRRID(rf=self.rf,
                     min_eps=0.00, max_eps=1.0, n_eps=20,
                     compiler_verbose=0
                     )

    rf = Instance(IRF, simdb=True)

    config_list = List(config=True)

    def _config_list_default(self):
        return ['I', 'IV']

    config_dict = Property(depends_on='config_list')

    @cached_property
    def _get_config_dict(self):
        cd = {}
        for config_idx in self.config_list:
            cd[ config_idx ] = config_dict[ config_idx ]
        return cd

    rand_list = List(rand=True)

    run_arr = Property(Array, depends_on='+rand,+config')

    @cached_property
    def _get_run_arr(self):
        # generate the runs to be performed 
        run_table = [
                     [
                      SingleRun(run_table=self, config=config,
                                 rand_idx_arr=rand_idx_arr)
                      for rand_idx_arr in self.rand_list
                      ]
                     for config in list(self.config_dict.items())
                     ]

        return array(run_table)

    exec_time_arr = Array
    n_int_arr = Array
    real_memsize_arr = Array

    calculate = Button()

    def _calculate_fired(self):
        s = self.run_arr.shape
        self.exec_time_arr = array([ run.exec_time for run in self.run_arr.flatten() ]).reshape(s)
        self.n_int_arr = array([ run.n_int for run in self.run_arr.flatten() ]).reshape(s)
        self.real_memsize_arr = array([ run.real_memsize for run in self.run_arr.flatten() ]).reshape(s)
        self.save()
        self._redraw_fired()

    clear = Button()

    def _clear_fired(self):
        figure = self.figure
        figure.clear()
        self.data_changed = True

    figure = Instance(Figure, transient=True)

    def _figure_default(self):
        figure = Figure(facecolor='white')
        # figure.add_axes( [0.08, 0.13, 0.85, 0.74] )
        return figure

    data_changed = Event(True)

    normalized_numpy = Bool(True)
    c_code = Bool(False)

    redraw = Button()

    def _redraw_fired(self):
        figure = self.figure
        axes = figure.gca()
        self.plot(axes)
        self.data_changed = True

    redraw_in_window = Button()

    def _redraw_in_window_fired(self):
        figure = plt.figure(0)
        axes = figure.gca()
        self.plot(axes)
        plt.show()

    def plot(self, ax):

        exec_time_arr = self.exec_time_arr
        n_int_arr = self.n_int_arr[0, :]
        real_memsize_arr = self.real_memsize_arr[0, :]

        rand_arr = arange(len(self.rand_list)) + 1
        width = 0.45

        if exec_time_arr.shape[0] == 1:
            shift = width / 2.0
            ax.bar(rand_arr - shift, exec_time_arr[0, :], width, color='lightgrey')

        elif self.exec_time_arr.shape[0] == 2:
            max_exec_time = nmax(exec_time_arr)

            ax.set_ylabel('$\mathrm{execution \, time \, [sec]}$', size=20)
            ax.set_xlabel('$n_{\mathrm{rnd}}  \;-\; \mathrm{number \, of \, random \, parameters}$', size=20)

            ax.bar(rand_arr - width, exec_time_arr[0, :], width,
                    hatch='/', color='white', label='C')  # , color = 'lightgrey' )
            ax.bar(rand_arr, exec_time_arr[1, :], width,
                    color='lightgrey', label='numpy')

            yscale = 1.25
            ax_xlim = rand_arr[-1] + 1
            ax_ylim = max_exec_time * yscale

            ax.set_xlim(0, ax_xlim)
            ax.set_ylim(0, ax_ylim)

            ax2 = ax.twinx()
            ydata = exec_time_arr[1, :] / exec_time_arr[0, :]
            ax2.plot(rand_arr, ydata, '-o', color='black',
                      linewidth=1, label='numpy/C')

            ax2.plot([rand_arr[0] - 1, rand_arr[-1] + 1], [1, 1], '-')
            ax2.set_ylabel('$\mathrm{time}(  \mathsf{numpy}  ) / \mathrm{ time }(\mathsf{C}) \; [-]$', size=20)
            ax2_ylim = nmax(ydata) * yscale
            ax2_xlim = rand_arr[-1] + 1
            ax2.set_ylim(0, ax2_ylim)
            ax2.set_xlim(0, ax2_xlim)

            ax.set_xticks(rand_arr)
            ax.set_xticklabels(rand_arr, size=14)
            xticks = [ '%.2g' % n_int for n_int in n_int_arr ]
            ax3 = ax.twiny()
            ax3.set_xlim(0, rand_arr[-1] + 1)
            ax3.set_xticks(rand_arr)
            ax3.set_xlabel('$n_{\mathrm{int}}$', size=20)
            ax3.set_xticklabels(xticks, rotation=30)

            'set the tick label size of the lower X axis'
            X_lower_tick = 14
            xt = ax.get_xticklabels()
            for t in xt:
                t.set_fontsize(X_lower_tick)

            'set the tick label size of the upper X axis'
            X_upper_tick = 12
            xt = ax3.get_xticklabels()
            for t in xt:
                t.set_fontsize(X_upper_tick)

            'set the tick label size of the Y axes'
            Y_tick = 14
            yt = ax2.get_yticklabels() + ax.get_yticklabels()
            for t in yt:
                t.set_fontsize(Y_tick)

            'set the legend position and font size'
            leg_fontsize = 16
            leg = ax.legend(loc=(0.02, 0.83))
            for t in leg.get_texts():
                t.set_fontsize(leg_fontsize)
            leg = ax2.legend(loc=(0.705, 0.90))
            for t in leg.get_texts():
                t.set_fontsize(leg_fontsize)

    traits_view = View(Item('name'),
                        Item('memsize'),
                        Item('rf'),
                        Item('config_dict'),
                        Item('rand_list'),
                        HGroup(Item('calculate', show_label=False),
                                Item('redraw', show_label=False),
                                Item('clear', show_label=False),
                                Item('redraw_in_window', show_label=False),
                                ),
                        Item('figure', editor=MPLFigureEditor(),
                              resizable=True, show_label=False),
                        buttons=['OK', 'Cancel' ]
                        )


RunTable.db = SimDBClassExt(
                            klass=RunTable,
                            )


def add_studies():
    ''' Run a study and save it to the file'''

    rand_list = [ arange(0, i) for i in range(1, 11) ]
    print(rand_list)

    memsize = 5e4  # 3e+7 maximum

    rf = RFCosParam()
    rt = RunTable(name='cos', rf=rf, memsize=memsize, rand_list=rand_list)
    RunTable.db['cos'] = rt
    rt.save()

    rf = RFSumParam()
    rt = RunTable(name='sum', rf=rf, memsize=memsize, rand_list=rand_list)
    RunTable.db['sum'] = rt
    rt.save()


if __name__ == '__main__':
    from matplotlib import pyplot as plt, rcParams, rc
    rc('font', family='serif',
        style='normal', variant='normal', stretch='normal')
    # rcParams['text.latex.preamble'] = '\usepackage{bm}'

#    add_studies()
#    print RunTable.db.keys()
    # del RunTable.db['']

    RunTable.db.configure_traits()

#    rand_list = [ array( [0], dtype = int ),
#                  array( [1], dtype = int ),
#                  array( [2], dtype = int ),
#                  array( [3], dtype = int ),
#                  array( [4], dtype = int ),
#                  array( [5], dtype = int ),
#                  array( [6], dtype = int ),
#                  array( [7], dtype = int ),
#                  array( [8], dtype = int ),
#                 ]
#    rf = RFPowParam()
#    rt = RunTable( name = 'pow', rf = rf, memsize = 500000, rand_list = rand_list,
#                   config_list = ['I'] ) # , 'IV'] )
#    print RunTable.db.keys()
#    RunTable.db['pow'] = rt
#    rt.save()

#    rand_list = [ array( [0, 1], dtype = int ),
#                  array( [1, 2], dtype = int ),
#                  array( [2, 3], dtype = int ),
#                  array( [3, 4], dtype = int ),
#                  array( [4, 5], dtype = int ),
#                  array( [5, 6], dtype = int ),
#                  array( [6, 7], dtype = int ),
#                  array( [7, 8], dtype = int ),
#                 ]
#    rf = RFPowParam()
#    rt = RunTable( name = 'pow-2-params', rf = rf, memsize = 500000, rand_list = rand_list,
#                   config_list = ['I'] ) # , 'IV'] )
#    print RunTable.db.keys()
#    RunTable.db['pow-2-params`'] = rt
#    rt.save()

#    rand_list = [ array( [0, 1, 2], dtype = int ),
#                  array( [1, 2, 3], dtype = int ),
#                  array( [2, 3, 4], dtype = int ),
#                  array( [3, 4, 5], dtype = int ),
#                  array( [4, 5, 6], dtype = int ),
#                  array( [5, 6, 7], dtype = int ),
#                  array( [6, 7, 8], dtype = int ),
#                  array( [7, 8, 9], dtype = int ),
#                 ]
#    rf = RFPowParam()
#    rt = RunTable( name = 'pow-3-params', rf = rf, memsize = 500000, rand_list = rand_list,
#                   config_list = ['I'] ) # , 'IV'] )
#    print RunTable.db.keys()
#    RunTable.db['pow-3-params`'] = rt
#    rt.save()

#
#    rf = RFSumParam()
#    rt = RunTable( name = 'sum-single-rv9', rf = rf, memsize = 5000, rand_list = [1],
#                   config_list = ['I'] )
#    print RunTable.db.keys()
#    RunTable.db['sin-single-rv'] = rt
#    rt.save()

    rand_list = [ array([0, 1, 2, 3], dtype=int),
                  array([1, 2, 3, 4], dtype=int),
                  array([0, 2, 3, 4], dtype=int),
                  array([0, 1, 3, 4], dtype=int),
                  array([0, 1, 2, 4], dtype=int),
                 ]
    rand_list = [ array([3, 1], dtype=int),
                  array([1, 2], dtype=int),
                  array([3, 2], dtype=int),
                 ]
    rf = RFFilament()
    rt = RunTable(name='filament-new', rf=rf, memsize=5000, rand_list=rand_list,
                   config_list=['I'])
    print((list(RunTable.db.keys())))
    RunTable.db['filament-new'] = rt
    rt.save()
