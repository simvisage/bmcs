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

from numpy import sign, exp, sqrt, cos
from stats.spirrid_bak import RF, IRF

from etsproxy.traits.api import \
    HasTraits, Float, Str, Bool


class RFSumParams(RF):

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

    eps = Float(0, ctrl_range=(0, 0.2, 10))

    C_code = '''
            // Computation of the q( ... ) function
            //q = ( cos( a ) + cos( b ) + cos( c ) + cos( d ) + cos( ee ) + cos( f ) + cos( g ) + cos( h ) + cos( i )) * eps;
             q = ( ( a ) + ( b ) + ( c ) + ( d ) + ( ee ) + ( f ) + ( g ) + ( h ) + ( i )) * eps;
        '''

    def __call__(self, eps, a, b, c, d, ee, f, g, h, i):
        '''
        Implements the response function with arrays as variables.
        first extract the variable discretizations from the orthogonal grid.
        '''
        # broadcast eps also in the xi - dimension 
        # (by multiplying with array containing ones with the same shape as xi )
        #
        # return ( cos( a ) + cos( b ) + cos( c ) + cos( d ) + cos( ee ) + cos( f ) + cos( g ) + cos( h ) + cos( i ) ) * eps
        return ((a) + (b) + (c) + (d) + (ee) + (f) + (g) + (h) + (i)) * eps;


if __name__ == '__main__':

    from stats.spirrid_bak.old.spirrid import SPIRRID
    from stats.spirrid_bak.ui.rf_model_view import RFModelView
    from stats.spirrid_bak.ui.spirrid_model_view import SPIRRIDModelView

    from matplotlib import pyplot as plt, rcParams, rc
    rc('font', family='serif',
        style='normal', variant='normal', stretch='normal', size=16)
    # rcParams['text.latex.preamble'] = '\usepackage{bm}'
    plt.figure(0)
    plt.subplots_adjust(wspace=0.0, hspace=0.0, bottom=.11)

    from numpy import arange, array

    def run_study(s, run_dict, offset=0, width=0.35):
        legend = []
        exec_times = []

        for idx, run in list(run_dict.items()):
            run_options, plot_options, legend_string = run

            s.set(**run_options)
#            s.mean_curve
#            print 'xdata', s.mean_curve.xdata
#            print 'ydata', s.mean_curve.ydata
#            s.mean_curve.plot( plt, plot_options )

            print(('---- code %d ---' % idx))
            print(('cached', s.cached_dG))
            print(('compiled dG', s.compiled_QdG_loop))
            print(('compiled eps', s.compiled_eps_loop))
#                print s.C_code

            # print 'integral of the pdf theta', s.eval_i_dG_grid()
            print(('execution time', s.exec_time))
            legend.append(legend_string)  # % s.exec_time )
            exec_times.append(s.exec_time)

        time_for_version_1 = exec_times[0]
        version_arr = arange(1, len(run_dict) + 1)
        time_arr = array(exec_times, dtype=float)
        time_arr /= time_for_version_1

        plt.figure(1)
        rects = plt.bar(version_arr + offset, time_arr, width, color='lightgrey')
        plt.xticks(version_arr + width / 2.0,
                    ('$\mathrm{I}$', '$\mathrm{II}$',
                      '$\mathrm{III}$', '$\mathrm{IV}$'),
                    size=20)

        def autolabel(rects, legend, exec_times):
            # attach some text labels
            for rect, le, exec_time in zip(rects, legend, exec_times):
    #                    plt.text( rect.get_x() + rect.get_width() / 2., 0.05, le,
    #                              color = 'black', size = 17,
    #                              ha = 'center', va = 'bottom', rotation = 90 )
                plt.text(rect.get_x() + rect.get_width() / 2., 1.02 * rect.get_height(),
                          '%4.2f $\mathrm{sec}$' % exec_time,
                          color='black', size=14,
                          ha='center', va='bottom', rotation=90)

        autolabel(rects, legend, exec_times)
        return time_arr

    run_dict = {1 :
                (
                 {'cached_dG'         : False,
                  'compiled_QdG_loop'  : True,
                  'compiled_eps_loop' : True },
                  'bx-',
                  '$\mathrm{C}_{{e},\\theta} ( q(e,\\theta) \cdot g[\\theta_1]  g[\\theta_2] \dots g[\\theta_n] ) $'
                 ),
                2 :
                (
                 {'cached_dG'         : False,
                  'compiled_QdG_loop'  : True,
                  'compiled_eps_loop' : True },
                  'bx-',
                  '$\mathrm{C}_{{e},\\theta} ( q(e,\\theta) \cdot g[\\theta_1]  g[\\theta_2] \dots g[\\theta_n] ) $'
                 ),
                3 :
                (
                 {'cached_dG'         : True,
                  'compiled_QdG_loop'  : False,
                  'compiled_eps_loop' : False },
                 'y--',
                 '$\mathrm{P}_{e} ( \mathrm{N}_{\\theta} ( q(e,\\theta) \cdot G[\\theta] ) ) $'
                 ),
                }

    memsize = 5e6

    rf = RFSumParams()

    n_params = len(rf.param_keys)

    width = 0.08
    time_plot = []

    for idx in range(n_params):
        offset = idx * width
        n_rv = idx + 1

        n_int = int(pow(memsize, 1 / float(n_rv)))
        print(('n_int', n_int))

        s = SPIRRID(rf=rf,
                     min_eps=0.00, max_eps=1.0, n_eps=20,
                     compiler_verbose=0
                     )

        for rv in range(n_rv):
            param_key = rf.param_keys[ rv ]
            s.add_rv(param_key, n_int=n_int)
        time_plot.append(run_study(s, run_dict, offset, width)[1])

    plt.figure(0)
    plt.plot(list(range(1, len(rf.param_keys) + 1)), time_plot, '-o', color='black', linewidth=1)

    plt.figure(1)
    plt.plot([1.0, 2.0 + width * n_params], [1.0, 1.0], '-o', color='black')

    plt.figure(0)
    plt.ylabel('$\mathrm{normalized \, execution \, time \, [-]}$', size=20)
    plt.xlabel('$\mathrm{number \, of \, randomized \, parameters}$', size=20)
    newYTicks = [ ('$%i$' % y) for y in plt.yticks()[0]]
    plt.yticks(plt.yticks()[0], newYTicks)
    newXTicks = [ ('$%i$' % x) for x in plt.xticks()[0]]
    plt.xticks(plt.xticks()[0], newXTicks, position=(0, -.01))

    plt.show()

