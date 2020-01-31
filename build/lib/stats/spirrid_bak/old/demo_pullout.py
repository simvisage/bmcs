'''
Created on Jun 16, 2010

@author: rostislav
'''
'''
Created on May 27, 2010

@author: rostislav
'''

from stats.spirrid_bak.old.spirrid import SPIRRID
from matplotlib import pyplot as plt
from quaducom.pullout.constant_friction_finite_fiber import ConstantFrictionFiniteFiber


def run():
    # Quantities for the response function
    # and randomization

    # construct a default response function for a single filament

    rf = ConstantFrictionFiniteFiber(fu = 1200e15, qf = 1200,
                                     L = 0.02, A = 0.00000002,
                                     E = 210.e9, z = 0.004,
                                     phi = 0.5, f = 0.01)

    # construct the integrator and provide it with the response function.

    s = SPIRRID(rf = rf,
                 min_eps = 0.00, max_eps = 0.0008, n_eps = 380)

    # construct the random variables

    n_int = 25

    s.add_rv('E_mod', distribution = 'uniform', loc = 170.e9, scale = 250.e9, n_int = n_int)
    s.add_rv('L', distribution = 'uniform', loc = 0.02, scale = 0.03, n_int = n_int)
    s.add_rv('phi', distribution = 'sin_distr', loc = 0., scale = 1., n_int = n_int)
    s.add_rv('z', distribution = 'uniform', loc = 0, scale = rf.L / 2., n_int = n_int)

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
                 'b--',
                 '$\mathrm{P}_{e} ( \mathrm{N}_{\\theta} ( q(e,\\theta) \cdot G[\\theta] ) ) $ - %4.2f sec'
                 ),
                ]

    for idx, run in enumerate(run_list):
        run_options, plot_options, legend_string = run
        print('run', idx, end=' ')
        s.set(**run_options)
        s.mean_curve.plot(plt, plot_options, linewidth = 2, label = legend_string % s.exec_time)
        print('execution time', s.exec_time)

#    def f():
#        print 'exec_time', s.exec_time
#
#    global f
#
#    import cProfile
#
#    cProfile.run('f()', 'spirrid.tprof')
#    import pstats
#    p = pstats.Stats('spirrid.tprof')
#    p.strip_dirs()
#    print 'cumulative'
#    p.sort_stats('cumulative').print_stats(50)
#    print 'time'
#    p.sort_stats('time').print_stats(50)

    plt.xlabel('strain [-]')
    plt.ylabel('stress')
    plt.legend(loc = 'lower right')

    plt.title(s.rf.title)
    plt.show()

if __name__ == '__main__':
    run()

