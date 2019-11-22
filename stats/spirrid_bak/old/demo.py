'''
Created on May 27, 2010

@author: rostislav
'''

from stats.spirrid_bak.old.spirrid import SPIRRID
from matplotlib import pyplot as plt
from stats.spirrid_bak.rf_filament import Filament
from math import pi

def run():
    # Quantities for the response function
    # and randomization
    # 
    E_mod = 70 * 1e+9 # Pa
    sig_u = 1.25 * 1e+9 # Pa
    D = 26 * 1.0e-6 # m
    A = ( D / 2.0 ) ** 2 * pi
    xi_u = sig_u / E_mod

    # construct a default response function for a single filament

    rf = Filament( E_mod = 70.e9, xi = 0.02, A = A, theta = 0, lambd = 0 )

    # construct the integrator and provide it with the response function.

    s = SPIRRID( rf = rf,
                 min_eps = 0.00, max_eps = 0.05, n_eps = 20 )

    # construct the random variables

    n_int = 40

    s.add_rv( 'xi', distribution = 'weibull_min', scale = 0.02, shape = 10., n_int = n_int )
    s.add_rv( 'E_mod', distribution = 'uniform', loc = 70e+9, scale = 15e+9, n_int = n_int )
    s.add_rv( 'theta', distribution = 'uniform', loc = 0.0, scale = 0.01, n_int = n_int )
    s.add_rv( 'lambd', distribution = 'uniform', loc = 0.0, scale = .2, n_int = n_int )
    s.add_rv( 'A', distribution = 'uniform', loc = A * 0.3, scale = 0.7 * A, n_int = n_int )

    # define a tables with the run configurations to start in a batch

    run_list = [
                ( 
                 {'cached_dG'         : False,
                  'compiled_QdG_loop'  : True,
                  'compiled_eps_loop' : True },
                  'bx-',
                  '$\mathrm{C}_{e,\\theta} ( q(e,\\theta) \cdot g[\\theta_1]  g[\\theta_2] \dots g[\\theta_n] ) $ - %4.2f sec'
                 ),
                ( 
                 {'cached_dG'         : False,
                  'compiled_QdG_loop'  : True,
                  'compiled_eps_loop' : False },
                 'r-2',
                 '$\mathrm{P}_{e} ( \mathrm{C}_{\\theta} ( q(e,\\theta) \cdot g[\\theta_1]  g[\\theta_2] \dots g[\\theta_n] ) ) $ - %4.2f sec',
                 ),
                ( 
                 {'cached_dG'         : True,
                  'compiled_QdG_loop'  : True,
                  'compiled_eps_loop' : True },
                  'go-',
                  '$\mathrm{C}_{e,\\theta} ( q(e,\\theta) \cdot G[\\theta] ) $ - %4.2f sec',
                  ),
                ( 
                 {'cached_dG'         : True,
                  'compiled_QdG_loop'  : False,
                  'compiled_eps_loop' : False },
                 'y--',
                 '$\mathrm{P}_{e} ( \mathrm{N}_{\\theta} ( q(e,\\theta) \cdot G[\\theta] ) ) $ - %4.2f sec'
                 ),
                ]

    legend = []

    for idx, run in enumerate( run_list ):
        run_options, plot_options, legend_string = run
        print('run', idx, end=' ')
        s.set( **run_options )
        print('xdata', s.mean_curve.xdata)
        print('ydata', s.mean_curve.ydata)
        s.mean_curve.plot( plt, plot_options )

        print('integral of the pdf theta', s.eval_i_dG_grid())
        print('execution time', s.exec_time)
        legend.append( legend_string % s.exec_time )

    plt.xlabel( 'strain [-]' )
    plt.ylabel( 'stress' )
    plt.legend( legend )

    plt.title( s.rf.title )
    plt.show()

if __name__ == '__main__':
    run()
