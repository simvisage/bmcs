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
# Created on Sep 8, 2011 by: rch

from etsproxy.traits.api import HasTraits, Array, Property, DelegatesTo, \
    Instance, Int, Str, List, on_trait_change, Button, Enum, Bool, Directory
from etsproxy.traits.ui.api import View, Item
from stats.spirrid_bak import ErrorEval
from itertools import combinations, chain
from matplotlib import rc
from socket import gethostname
from stats.spirrid_bak import SPIRRID
import numpy as np
import os.path
import pylab as p # import matplotlib with matlab interface
import types
import shutil
from os.path import expanduser

#===============================================================================
# Helper functions
#===============================================================================
def Heaviside(x):
    ''' Heaviside function '''
    return x >= 0

def powerset(iterable):
    '''
        Return object of all combination of iterable. 
        powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    '''
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

#===============================================================================
# convergence_study
#===============================================================================

class SPIRRIDLAB(HasTraits):
    '''Class used for elementary parametric studies of spirrid.
    '''

    s = Instance(SPIRRID)

    evars = DelegatesTo('s')

    tvars = DelegatesTo('s')

    q = DelegatesTo('s')

    exact_arr = Array('float')

    dpi = Int

    plot_mode = Enum(['subplots', 'figures'])

    fig_output_dir = Directory('fig')
    @on_trait_change('fig_output_dir')
    def _check_dir(self):
        if os.access(self.fig_output_dir, os.F_OK) == False:
            os.mkdir(self.fig_output_dir)

    e_arr = Property
    def _get_e_arr(self):
        return self.s.evar_lst[0]

    hostname = Property
    def _get_hostname(self):
        return gethostname()

    qname = Str

    def get_qname(self):
        if self.qname == '':
            if isinstance(self.q, types.FunctionType):
                qname = self.q.__name__
            else: # if isinstance(self.q, types.ClassType):
                qname = self.q.__class__.__name__
        else:
            qname = self.qname
        return qname

    show_output = False

    save_output = True

    plot_sampling_idx = Array(value = [0, 1], dtype = int)

    def _plot_sampling(self,
                       i,
                       n_col,
                      sampling_type,
                      p = p,
                      ylim = None,
                      xlim = None):
        '''Construct a spirrid object, run the calculation
        plot the mu_q / e curve and save it in the subdirectory.
        '''

        s = self.s
        s.sampling_type = sampling_type

        plot_idx = self.plot_sampling_idx

        qname = self.get_qname()

        # get n randomly selected realizations from the sampling
        theta = s.sampling.get_samples(500)
        tvar_x = s.tvar_lst[plot_idx[0]]
        tvar_y = s.tvar_lst[plot_idx[1]]
        min_x, max_x, d_x = s.sampling.get_theta_range(tvar_x)
        min_y, max_y, d_y = s.sampling.get_theta_range(tvar_y)
        # for vectorized execution add a dimension for control variable
        theta_args = [ t[:, np.newaxis] for t in theta]
        q_arr = s.q(self.e_arr[None, :], *theta_args)
        if self.plot_mode == 'figures':
            f = p.figure(figsize = (7., 6.))
            f.subplots_adjust(left = 0.15, right = 0.97, bottom = 0.15, top = 0.92)
        if self.plot_mode == 'subplots':
            if i == 0:
                f = p.figure()
            p.subplot('2%i%i' % (n_col, (i + 1)))

        p.plot(theta[plot_idx[0]], theta[plot_idx[1]], 'o', color = 'grey')
        p.xlabel('$\lambda$')
        p.ylabel('$\\xi$')

        p.xlim(min_x, max_x)
        p.ylim(min_y, max_y)
        p.title(s.sampling_type)

        if self.save_output:
            fname = os.path.join(self.fig_output_dir,
                                 qname + '_sampling_' + s.sampling_type + '.png')
            p.savefig(fname, dpi = self.dpi)

        if self.plot_mode == 'figures':
            f = p.figure(figsize = (7., 5))
            f.subplots_adjust(left = 0.15, right = 0.97, bottom = 0.18, top = 0.91)
        elif self.plot_mode == 'subplots':
            p.subplot('2%i%i' % (n_col, (i + 5)))

        p.plot(self.e_arr, q_arr.T, color = 'grey')

        if len(self.exact_arr) > 0:
            p.plot(self.e_arr, self.exact_arr, label = 'exact solution',
                    color = 'black', linestyle = '--', linewidth = 2)

        # numerically obtained result
        p.plot(self.e_arr, s.mu_q_arr, label = 'numerical integration',
               linewidth = 3, color = 'black')
        p.title(s.sampling_type)
        p.xlabel('$\\varepsilon$ [-]')
        p.ylabel(r'$q(\varepsilon;\, \lambda,\, \xi)$')
        if ylim:
            p.ylim(0.0, ylim)
        if xlim:
            p.xlim(0.0, xlim)
        p.xticks(position = (0, -.015))
        p.legend(loc = 2)

        if self.save_output:
            fname = os.path.join(self.fig_output_dir, qname + '_' + s.sampling_type + '.png')
            p.savefig(fname, dpi = self.dpi)

    sampling_structure_btn = Button(label = 'compare sampling structure')
    @on_trait_change('sampling_structure_btn')
    def sampling_structure(self, **kw):
        '''Plot the response into the file in the fig subdirectory.
        '''
        if self.plot_mode == 'subplots':
            p.rcdefaults()
        else:
            fsize = 28
            p.rcParams['font.size'] = fsize
            rc('legend', fontsize = fsize - 8)
            rc('axes', titlesize = fsize)
            rc('axes', labelsize = fsize + 6)
            rc('xtick', labelsize = fsize - 8)
            rc('ytick', labelsize = fsize - 8)
            rc('xtick.major', pad = 8)

        s_lst = ['TGrid', 'PGrid', 'MCS', 'LHS']

        for i, s in enumerate(s_lst):
            self._plot_sampling(i, len(s_lst), sampling_type = s, **kw)

        if self.show_output:
            p.show()

    n_int_range = Array()

    #===========================================================================
    # Output file names for sampling efficiency
    #===========================================================================
    fname_sampling_efficiency_time_nint = Property
    def _get_fname_sampling_efficiency_time_nint(self):
        return self.get_qname() + '_' + '%s' % self.hostname + '_time_nint' + '.png'

    fname_sampling_efficiency_error_nint = Property
    def _get_fname_sampling_efficiency_error_nint(self):
        return self.get_qname() + '_' + '%s' % self.hostname + '_error_nint' + '.png'

    fname_sampling_efficiency_error_time = Property
    def _get_fname_sampling_efficiency_error_time(self):
        return self.get_qname() + '_' + '%s' % self.hostname + '_error_time' + '.png'

    fnames_sampling_efficiency = Property
    def _get_fnames_sampling_efficiency(self):
        fnames = [self.fname_sampling_efficiency_time_nint]
        if len(self.exact_arr) > 0:
            fnames += [self.fname_sampling_efficiency_error_nint,
                       self.fname_sampling_efficiency_error_time ]
        return fnames

    #===========================================================================
    # Run sampling efficiency studies
    #===========================================================================

    sampling_types = Array(value = ['TGrid', 'PGrid', 'MCS', 'LHS'], dtype = str)

    sampling_efficiency_btn = Button(label = 'compare sampling efficiency')
    @on_trait_change('sampling_efficiency_btn')
    def sampling_efficiency(self):
        '''
        Run the code for all available sampling types.
        Plot the results.
        '''
        def run_estimation(n_int, sampling_type):
            # instantiate spirrid with samplingetization methods 
            print(('running', sampling_type, n_int))
            self.s.set(n_int = n_int, sampling_type = sampling_type)
            n_sim = self.s.sampling.n_sim
            exec_time = np.sum(self.s.exec_time)
            return self.s.mu_q_arr, exec_time, n_sim

        # vectorize the estimation to accept arrays
        run_estimation_vct = np.vectorize(run_estimation, [object, float, int])

        #===========================================================================
        # Generate the inspected domain of input parameters using broadcasting
        #===========================================================================

        run_estimation_vct([5], ['PGrid'])

        sampling_types = self.sampling_types
        sampling_colors = np.array(['grey', 'black', 'grey', 'black'], dtype = str) # 'blue', 'green', 'red', 'magenta'
        sampling_linestyle = np.array(['--', '--', '-', '-'], dtype = str)

        # run the estimation on all combinations of n_int and sampling_types
        mu_q, exec_time, n_sim_range = run_estimation_vct(self.n_int_range[:, None],
                                                          sampling_types[None, :])

        p.rcdefaults()
        f = p.figure(figsize = (12, 6))
        f.subplots_adjust(left = 0.06, right = 0.94)

        #===========================================================================
        # Plot the results
        #===========================================================================
        p.subplot(1, 2, 1)
        p.title('response for %d $n_\mathrm{sim}$' % n_sim_range[-1, -1])
        for i, (sampling, color, linestyle) in enumerate(zip(sampling_types,
                                                             sampling_colors,
                                                             sampling_linestyle)):
            p.plot(self.e_arr, mu_q[-1, i], color = color,
                   label = sampling, linestyle = linestyle)

        if len(self.exact_arr) > 0:
            p.plot(self.e_arr, self.exact_arr, color = 'black', label = 'Exact solution')

        p.legend(loc = 1)
        p.xlabel('e', fontsize = 18)
        p.ylabel('q', fontsize = 18)

        # @todo: get n_sim - x-axis
        p.subplot(1, 2, 2)
        for i, (sampling, color, linestyle) in enumerate(zip(sampling_types,
                                                             sampling_colors,
                                                             sampling_linestyle)):
            p.loglog(n_sim_range[:, i], exec_time[:, i], color = color,
                     label = sampling, linestyle = linestyle)

        p.legend(loc = 2)
        p.xlabel('$n_\mathrm{sim}$', fontsize = 18)
        p.ylabel('$t$ [s]', fontsize = 18)

        if self.save_output:
            basename = self.fname_sampling_efficiency_time_nint
            fname = os.path.join(self.fig_output_dir, basename)
            p.savefig(fname, dpi = self.dpi)

        #===========================================================================
        # Evaluate the error
        #===========================================================================

        if len(self.exact_arr) > 0:
            er = ErrorEval(exact_arr = self.exact_arr)

            def eval_error(mu_q, error_measure):
                return error_measure(mu_q)
            eval_error_vct = np.vectorize(eval_error)

            error_measures = np.array([er.eval_error_max,
                                        er.eval_error_energy,
                                        er.eval_error_rms ])
            error_table = eval_error_vct(mu_q[:, :, None],
                                          error_measures[None, None, :])

            f = p.figure(figsize = (14, 6))
            f.subplots_adjust(left = 0.07, right = 0.97, wspace = 0.26)

            p.subplot(1, 2, 1)
            p.title('max rel. lack of fit')
            for i, (sampling, color, linestyle) in enumerate(zip(sampling_types, sampling_colors, sampling_linestyle)):
                p.loglog(n_sim_range[:, i], error_table[:, i, 0], color = color, label = sampling, linestyle = linestyle)

            #p.ylim( 0, 10 )
            p.legend()
            p.xlabel('$n_\mathrm{sim}$', fontsize = 18)
            p.ylabel('$\mathrm{e}_{\max}$ [-]', fontsize = 18)

            p.subplot(1, 2, 2)
            p.title('rel. root mean square error')
            for i, (sampling, color, linestyle) in enumerate(zip(sampling_types, sampling_colors, sampling_linestyle)):
                p.loglog(n_sim_range[:, i], error_table[:, i, 2], color = color, label = sampling, linestyle = linestyle)
            p.legend()
            p.xlabel('$n_{\mathrm{sim}}$', fontsize = 18)
            p.ylabel('$\mathrm{e}_{\mathrm{rms}}$ [-]', fontsize = 18)

            if self.save_output:
                basename = self.fname_sampling_efficiency_error_nint
                fname = os.path.join(self.fig_output_dir, basename)
                p.savefig(fname, dpi = self.dpi)

            f = p.figure(figsize = (14, 6))
            f.subplots_adjust(left = 0.07, right = 0.97, wspace = 0.26)

            p.subplot(1, 2, 1)
            p.title('rel. max lack of fit')
            for i, (sampling, color, linestyle) in enumerate(zip(sampling_types, sampling_colors, sampling_linestyle)):
                p.loglog(exec_time[:, i], error_table[:, i, 0], color = color, label = sampling, linestyle = linestyle)
            p.legend()
            p.xlabel('time [s]', fontsize = 18)
            p.ylabel('$\mathrm{e}_{\max}$ [-]', fontsize = 18)

            p.subplot(1, 2, 2)
            p.title('rel. root mean square error')
            for i, (sampling, color, linestyle) in enumerate(zip(sampling_types, sampling_colors, sampling_linestyle)):
                p.loglog(exec_time[:, i], error_table[:, i, 2], color = color, label = sampling, linestyle = linestyle)
            p.legend()
            p.xlabel('time [s]', fontsize = 18)
            p.ylabel('$\mathrm{e}_{\mathrm{rms}}$ [-]', fontsize = 18)

            if self.save_output:
                basename = self.fname_sampling_efficiency_error_time
                fname = os.path.join(self.fig_output_dir, basename)
                p.savefig(fname, dpi = self.dpi)

        if self.show_output:
            p.show()

    #===========================================================================
    # Efficiency of numpy versus C code
    #===========================================================================
    run_lst_detailed_config = Property(List)
    def _get_run_lst_detailed_config(self):
        run_lst = []
        if hasattr(self.q, 'c_code'):
            run_lst += [
#                ('c',
#                 {'cached_dG'         : True,
#                  'compiled_eps_loop' : True },
#                  'go-',
#                  '$\mathsf{C}_{\\varepsilon} \{\, \mathsf{C}_{\\theta} \{\,  q(\\varepsilon,\\theta) \cdot G[\\theta] \,\}\,\} $ - %4.2f sec',
#                  ),
#                ('c',
#                 {'cached_dG'         : True,
#                  'compiled_eps_loop' : False },
#                 'r-2',
#                 '$\mathsf{Python} _{\\varepsilon} \{\, \mathsf{C}_{\\theta} \{\,  q(\\varepsilon,\\theta) \cdot G[\\theta] \,\}\,\} $ - %4.2f sec'
#                 ),
#                ('c',
#                 {'cached_dG'         : False,
#                  'compiled_eps_loop' : True },
#                 'r-2',
#                 '$\mathsf{C}_{\\varepsilon} \{\, \mathsf{C}_{\\theta} \{\, q(\\varepsilon,\\theta) \cdot g[\\theta_1] \cdot \ldots \cdot g[\\theta_m] \,\}\,\} $ - %4.2f sec'
#                 ),
                ('c',
                 {'cached_dG'         : False,
                  'compiled_eps_loop' : False },
                  'bx-',
                  '$\mathsf{Python} _{\\varepsilon} \{\, \mathsf{C}_{\\theta}  \{\, q(\\varepsilon,\\theta) \cdot g[\\theta_1] \cdot \ldots \cdot g[\\theta_m] \,\} \,\} $ - %4.2f sec',
                 )
                ]
        if hasattr(self.q, 'cython_code'):
            run_lst += [
#                ('cython',
#                 {'cached_dG'         : True,
#                  'compiled_eps_loop' : True },
#                  'go-',
#                  '$\mathsf{Cython}_{\\varepsilon} \{\, \mathsf{Cython}_{\\theta} \{\,  q(\\varepsilon,\\theta) \cdot G[\\theta] \,\}\,\} $ - %4.2f sec',
#                  ),
#                ('cython',
#                 {'cached_dG'         : True,
#                  'compiled_eps_loop' : False },
#                 'r-2',
#                 '$\mathsf{Python} _{\\varepsilon} \{\, \mathsf{Cython}_{\\theta} \{\,  q(\\varepsilon,\\theta) \cdot G[\\theta] \,\}\,\} $ - %4.2f sec'
#                 ),
#                ('cython',
#                 {'cached_dG'         : False,
#                  'compiled_eps_loop' : True },
#                 'r-2',
#                 '$\mathsf{Cython}_{\\varepsilon} \{\, \mathsf{Cython}_{\\theta} \{\, q(\\varepsilon,\\theta) \cdot g[\\theta_1] \cdot \ldots \cdot g[\\theta_m] \,\}\,\} $ - %4.2f sec'
#                 ),
#                ('cython',
#                 {'cached_dG'         : False,
#                  'compiled_eps_loop' : False },
#                  'bx-',
#                  '$\mathsf{Python} _{\\varepsilon} \{\, \mathsf{Cython}_{\\theta}  \{\, q(\\varepsilon,\\theta) \cdot g[\\theta_1] \cdot \ldots \cdot g[\\theta_m] \,\} \,\} $ - %4.2f sec',
#                 )
                ]
        if hasattr(self.q, '__call__'):
            run_lst += [
#                ('numpy',
#                 {},
#                 'y--',
#                 '$\mathsf{Python}_{\\varepsilon} \{\,  \mathsf{Numpy}_{\\theta} \{\,  q(\\varepsilon,\\theta) \cdot G[\\theta] \,\} \,\} $ - %4.2f sec'
#                 )
                ]
        return run_lst

    # number of recalculations to get new time. 
    n_recalc = Int(2)

    def codegen_efficiency(self):
        # define a tables with the run configurations to start in a batch

        basenames = []

        qname = self.get_qname()

        s = self.s

        legend = []
        legend_lst = []
        time_lst = []
        p.figure()

        for idx, run in enumerate(self.run_lst_detailed_config):
            code, run_options, plot_options, legend_string = run
            s.codegen_type = code
            s.codegen.set(**run_options)
            print(('run', idx, run_options))

            for i in range(self.n_recalc):
                s.recalc = True # automatically proagated within spirrid
                print(('execution time', s.exec_time))

            p.plot(s.evar_lst[0], s.mu_q_arr, plot_options)

            # @todo: this is not portable!!
            #legend.append(legend_string % s.exec_time)
            #legend_lst.append(legend_string[:-12])
            time_lst.append(s.exec_time)

        p.xlabel('strain [-]')
        p.ylabel('stress')
        #p.legend(legend, loc = 2)
        p.title(qname)

        if self.save_output:
            print('saving codegen_efficiency')
            basename = qname + '_' + 'codegen_efficiency' + '.png'
            basenames.append(basename)
            fname = os.path.join(self.fig_output_dir, basename)
            p.savefig(fname, dpi = self.dpi)

        self._bar_plot(legend_lst, time_lst)
        p.title('%s' % s.sampling_type)
        if self.save_output:
            basename = qname + '_' + 'codegen_efficiency_%s' % s.sampling_type + '.png'
            basenames.append(basename)
            fname = os.path.join(self.fig_output_dir, basename)
            p.savefig(fname, dpi = self.dpi)

        if self.show_output:
            p.show()

        return basenames

    #===========================================================================
    # Efficiency of numpy versus C code
    #===========================================================================
    run_lst_language_config = Property(List)
    def _get_run_lst_language_config(self):
        run_lst = []
        if hasattr(self.q, 'c_code'):
            run_lst += [
                ('c',
                 {'cached_dG'         : False,
                  'compiled_eps_loop' : False },
                  'bx-',
                  '$\mathsf{Python} _{\\varepsilon} \{\, \mathsf{C}_{\\theta}  \{\, q(\\varepsilon,\\theta) \cdot g[\\theta_1] \cdot \ldots \cdot g[\\theta_m] \,\} \,\} $ - %4.2f sec',
                 )]
        if hasattr(self.q, 'cython_code'):
            run_lst += [
                ('cython',
                 {'cached_dG'         : False,
                  'compiled_eps_loop' : False },
                  'bx-',
                  '$\mathsf{Python} _{\\varepsilon} \{\, \mathsf{Cython}_{\\theta}  \{\, q(\\varepsilon,\\theta) \cdot g[\\theta_1] \cdot \ldots \cdot g[\\theta_m] \,\} \,\} $ - %4.2f sec',
                 )]
        if hasattr(self.q, '__call__'):
            run_lst += [
                ('numpy',
                 {},
                 'y--',
                 '$\mathsf{Python}_{\\varepsilon} \{\,  \mathsf{Numpy}_{\\theta} \{\,  q(\\varepsilon,\\theta) \cdot G[\\theta] \,\} \,\} $ - %4.2f sec'
                 )]
        return run_lst

    extra_compiler_args = Bool(True)
    le_sampling_lst = List(['LHS', 'PGrid'])
    le_n_int_lst = List([440, 5000])

    #===========================================================================
    # Output file names for language efficiency
    #===========================================================================
    fnames_language_efficiency = Property
    def _get_fnames_language_efficiency(self):
        return ['%s_codegen_efficiency_%s_extra_%s.png' %
                (self.qname, self.hostname, extra)
                for extra in [self.extra_compiler_args]]

    language_efficiency_btn = Button(label = 'compare language efficiency')
    @on_trait_change('language_efficiency_btn')
    def codegen_language_efficiency(self):
        # define a tables with the run configurations to start in a batch

        home_dir = expanduser("~")

#        pyxbld_dir = os.path.join(home_dir, '.pyxbld')
#        if os.path.exists(pyxbld_dir):
#            shutil.rmtree(pyxbld_dir)

        python_compiled_dir = os.path.join(home_dir, '.python27_compiled')
        if os.path.exists(python_compiled_dir):
            shutil.rmtree(python_compiled_dir)

        for extra, fname in zip([self.extra_compiler_args], self.fnames_language_efficiency):
            print(('extra compilation args:', extra))
            legend_lst = []
            error_lst = []
            n_sim_lst = []
            exec_times_sampling = []

            meth_lst = list(zip(self.le_sampling_lst, self.le_n_int_lst))
            for item, n_int in meth_lst:
                print(('sampling method:', item))
                s = self.s
                s.exec_time # eliminate first load time delay (first column)
                s.n_int = n_int
                s.sampling_type = item
                exec_times_lang = []

                for idx, run in enumerate(self.run_lst_language_config):
                    code, run_options, plot_options, legend_string = run

                    #os.system('rm -fr ~/.python27_compiled')

                    s.codegen_type = code
                    s.codegen.set(**run_options)
                    if s.codegen_type == 'c':
                        s.codegen.set(**dict(use_extra = extra))
                    print(('run', idx, run_options))

                    exec_times_run = []
                    for i in range(self.n_recalc):
                        s.recalc = True # automatically propagated
                        exec_times_run.append(s.exec_time)
                        print(('execution time', s.exec_time))

                    legend_lst.append(legend_string[:-12])
                    if s.codegen_type == 'c':
                        # load weave.inline time from tmp file and fix values in time_arr
                        #@todo - does not work on windows
                        import tempfile
                        tdir = tempfile.gettempdir()
                        f = open(os.path.join(tdir, 'w_time'), 'r')
                        value_t = float(f.read())
                        f.close()
                        exec_times_run[0][1] = value_t
                        exec_times_run[0][2] -= value_t
                        exec_times_lang.append(exec_times_run)
                    else:
                        exec_times_lang.append(exec_times_run)

                print(('legend_lst', legend_lst))
                n_sim_lst.append(s.sampling.n_sim)
                exec_times_sampling.append(exec_times_lang)
                #===========================================================================
                # Evaluate the error
                #===========================================================================
                if len(self.exact_arr) > 0:
                    er = ErrorEval(exact_arr = self.exact_arr)
                    error_lst.append((er.eval_error_rms(s.mu_q_arr), er.eval_error_max(s.mu_q_arr)))

            times_arr = np.array(exec_times_sampling, dtype = 'd')
            self._multi_bar_plot(meth_lst, legend_lst, times_arr, error_lst, n_sim_lst)
            if self.save_output:
                fname_path = os.path.join(self.fig_output_dir, fname)
                p.savefig(fname_path, dpi = self.dpi)

        if self.show_output:
            p.show()

    def combination_efficiency(self, tvars_det, tvars_rand):
        '''
        Run the code for all available random parameter combinations.
        Plot the results.
        '''
        qname = self.get_qname()

        s = self.s
        s.set(sampling_type = 'TGrid')

        # list of all combinations of response function parameters
        rv_comb_lst = list(powerset(list(s.tvars.keys())))

        p.figure()
        exec_time_lst = []

        for id, rv_comb in enumerate(rv_comb_lst[163:219]): # [1:-1]
            s.tvars = tvars_det
            print(('Combination', rv_comb))

            for rv in rv_comb:
                s.tvars[rv] = tvars_rand[rv]

            #legend = []
            #p.figure()
            time_lst = []
            for idx, run in enumerate(self.run_lst):
                code, run_options, plot_options, legend_string = run
                print(('run', idx, run_options))
                s.codegen_type = code
                s.codegen.set(**run_options)

                #p.plot(s.evar_lst[0], s.mu_q_arr, plot_options)

                #print 'integral of the pdf theta', s.eval_i_dG_grid()
                print(('execution time', s.exec_time))
                time_lst.append(s.exec_time)
                #legend.append(legend_string % s.exec_time)
            exec_time_lst.append(time_lst)
        p.plot(np.array((1, 2, 3, 4)), np.array(exec_time_lst).T)
        p.xlabel('method')
        p.ylabel('time')

        if self.save_output:
            print('saving codegen_efficiency')
            fname = os.path.join(self.fig_output_dir, qname + '_' + 'combination_efficiency' + '.png')
            p.savefig(fname, dpi = self.dpi)

        if self.show_output:
            p.title(s.q.title)
            p.show()

    def _bar_plot(self, legend_lst, time_lst):
        rc('font', size = 15)
        #rc('font', family = 'serif', style = 'normal', variant = 'normal', stretch = 'normal', size = 15)
        fig = p.figure(figsize = (10, 5))

        n_tests = len(time_lst)
        times = np.array(time_lst)
        x_norm = times[1]
        xmax = times.max()
        rel_xmax = xmax / x_norm
        rel_times = times / x_norm
        m = int(rel_xmax % 10)

        if m < 5:
            x_max_plt = int(rel_xmax) - m + 10
        else:
            x_max_plt = int(rel_xmax) - m + 15

        ax1 = fig.add_subplot(111)
        p.subplots_adjust(left = 0.45, right = 0.88)
        #fig.canvas.set_window_title('window title')
        pos = np.arange(n_tests) + 0.5
        rects = ax1.barh(pos, rel_times, align = 'center',
                          height = 0.5, color = 'w', edgecolor = 'k')

        ax1.set_xlabel('normalized execution time [-]')
        ax1.axis([0, x_max_plt, 0, n_tests])
        ax1.set_yticks(pos)
        ax1.set_yticklabels(legend_lst)

        for rect, t in zip(rects, rel_times):
            width = rect.get_width()

            xloc = width + (0.03 * rel_xmax)
            clr = 'black'
            align = 'left'

            yloc = rect.get_y() + rect.get_height() / 2.0
            ax1.text(xloc, yloc, '%4.2f' % t, horizontalalignment = align,
                     verticalalignment = 'center', color = clr)#, weight = 'bold')

        ax2 = ax1.twinx()
        ax1.plot([1, 1], [0, n_tests], 'k--')
        ax2.set_yticks([0] + list(pos) + [n_tests])
        ax2.set_yticklabels([''] + ['%4.2f s' % s for s in list(times)] + [''])
        ax2.set_xticks([0, 1] + list(range(5 , x_max_plt + 1, 5)))
        ax2.set_xticklabels(['%i' % s for s in ([0, 1] + list(range(5 , x_max_plt + 1, 5)))])

    def _multi_bar_plot(self, title_lst, legend_lst, time_arr, error_lst, n_sim_lst):
        '''Plot the results if the code efficiency. 
        '''
        p.rcdefaults()
        fsize = 14
        fig = p.figure(figsize = (15, 3))
        rc('font', size = fsize)
        rc('legend', fontsize = fsize - 2)
        legend_lst = ['weave', 'cython', 'numpy']

        # times are stored in 3d array - dimensions are:
        n_sampling, n_lang, n_run, n_times = time_arr.shape
        print(('arr', time_arr.shape))
        times_sum = np.sum(time_arr, axis = n_times)

        p.subplots_adjust(left = 0.1, right = 0.95, wspace = 0.1,
                          bottom = 0.15, top = 0.8)

        for meth_i in range(n_sampling):

            ax1 = fig.add_subplot(1, n_sampling, meth_i + 1)
            ax1.set_xlabel('execution time [s]')
            ytick_pos = np.arange(n_lang) + 1

    #        ax1.axis([0, x_max_plt, 0, n_lang])
            # todo: **2 n_vars
            if len(self.exact_arr) > 0:
                ax1.set_title('%s: $ n_\mathrm{sim} = %s, \mathrm{e}_\mathrm{rms}=%s, \mathrm{e}_\mathrm{max}=%s$' %
                           (title_lst[meth_i][0], self._formatSciNotation('%.2e' % n_sim_lst[meth_i]),
                            self._formatSciNotation('%.2e' % error_lst[meth_i][0]), self._formatSciNotation('%.2e' % error_lst[meth_i][1])))
            else:
                ax1.set_title('%s: $ n_\mathrm{sim} = %s$' %
                              (title_lst[meth_i][0], self._formatSciNotation('%.2e' % n_sim_lst[meth_i])))
            ax1.set_yticks(ytick_pos)
            if meth_i == 0:
                ax1.set_yticklabels(legend_lst, fontsize = fsize + 2)
            else:
                ax1.set_yticklabels([])

            ax1.set_xlim(0, 1.2 * np.max(times_sum[meth_i]))

            distance = 0.2
            height = 1.0 / n_run - distance
            offset = height / 2.0

            colors = ['w', 'w', 'w', 'r', 'y', 'b', 'g', 'm' ]
            hatches = [ '/', '\\', 'x', '-', '+', '|', 'o', 'O', '.', '*' ]
            label_lst = ['sampling', 'compilation', 'integration']

            for i in range(n_run):
                pos = np.arange(n_lang) + 1 - offset + i * height
                end_bar_pos = np.zeros((n_lang,), dtype = 'd')
                for j in range(n_times):
                    if i > 0:
                        label = label_lst[j]
                    else:
                        label = None
                    bar_lengths = time_arr[meth_i, :, i, j]
                    rects = ax1.barh(pos, bar_lengths , align = 'center',
                                     height = height, left = end_bar_pos,
                                     color = colors[j], edgecolor = 'k', hatch = hatches[j], label = label)
                    end_bar_pos += bar_lengths
                for k in range(n_lang):
                    x_val = times_sum[meth_i, k, i] + 0.01 * np.max(times_sum[meth_i])
                    ax1.text(x_val, pos[k], '$%4.2f\,$s' % x_val, horizontalalignment = 'left',
                         verticalalignment = 'center', color = 'black')#, weight = 'bold')
                    if meth_i == 0:
                         ax1.text(0.02 * np.max(times_sum[0]), pos[k], '$%i.$' % (i + 1), horizontalalignment = 'left',
                         verticalalignment = 'center', color = 'black',
                         bbox = dict(pad = 0., ec = "w", fc = "w"))
            p.legend(loc = 0)

    def _formatSciNotation(self, s):
        # transform 1e+004 into 1e4, for example
        tup = s.split('e')
        try:
            significand = tup[0].rstrip('0').rstrip('.')
            sign = tup[1][0].replace('+', '')
            exponent = tup[1][1:].lstrip('0')
            if significand == '1':
                # reformat 1x10^y as 10^y
                significand = ''
            if exponent:
                exponent = '10^{%s%s}' % (sign, exponent)
            if significand and exponent:
                return r'%s{\cdot}%s' % (significand, exponent)
            else:
                return r'%s%s' % (significand, exponent)

        except IndexError as msg:
            return s


    def _bar_plot_2(self, title_lst, legend_lst, time_lst):
        legend_lst = ['weave', 'cython', 'numpy']
        rc('font', size = 15)
        #rc('font', family = 'serif', style = 'normal', variant = 'normal', stretch = 'normal', size = 15)
        fig = p.figure(figsize = (15, 3))
        idx = int(len(time_lst) / 2.)
        n_tests = len(time_lst[:idx])
        times = np.array(time_lst[:idx])
        x_norm = np.min(times)
        xmax = times.max()
        rel_xmax = xmax / x_norm
        rel_times = times / x_norm
        m = int(rel_xmax % 10)

        if m < 5:
            x_max_plt = int(rel_xmax) - m + 10
        else:
            x_max_plt = int(rel_xmax) - m + 15

        ax1 = fig.add_subplot(121)
        p.subplots_adjust(left = 0.35, right = 0.88, wspace = 0.3, bottom = 0.2)
        #fig.canvas.set_window_title('window title')
        pos = np.arange(n_tests) + 0.5
        rects = ax1.barh(pos, rel_times, align = 'center',
                          height = 0.5, color = 'w', edgecolor = 'k')

        ax1.set_xlabel('normalized execution time [-]')
        ax1.axis([0, x_max_plt, 0, n_tests])
        ax1.set_title(title_lst[0])
        ax1.set_yticks(pos)
        ax1.set_yticklabels(legend_lst[:idx])

        for rect, t in zip(rects, rel_times):

            width = rect.get_width()
            xloc = width + (0.03 * rel_xmax)
            clr = 'black'
            align = 'left'

            yloc = rect.get_y() + rect.get_height() / 2.0
            ax1.text(xloc, yloc, '%4.2f' % t, horizontalalignment = align,
                    verticalalignment = 'center', color = clr)#, weight = 'bold')

        ax2 = ax1.twinx()
        ax1.plot([1, 1], [0, n_tests], 'k--')
        ax2.set_yticks([0] + list(pos) + [n_tests])
        ax2.set_yticklabels([''] + ['%4.2f s' % s for s in list(times)] + [''])
        ax2.set_xticks([0, 1] + list(range(5 , x_max_plt + 1, 5)))
        ax2.set_xticklabels(['%i' % s for s in ([0, 1] + list(range(5 , x_max_plt + 1, 5)))])

        n_tests = len(time_lst[idx:])
        times = np.array(time_lst[idx:])
        x_norm = np.min(times)
        xmax = times.max()
        rel_xmax = xmax / x_norm
        rel_times = times / x_norm
        m = int(rel_xmax % 10)

        if m < 5:
            x_max_plt = int(rel_xmax) - m + 10
        else:
            x_max_plt = int(rel_xmax) - m + 15

        ax3 = fig.add_subplot(122)
        #fig.canvas.set_window_title('window title')
        pos = np.arange(n_tests) + 0.5
        rects = ax3.barh(pos, rel_times, align = 'center',
                          height = 0.5, color = 'w', edgecolor = 'k')

        ax3.set_xlabel('normalized execution time [-]')
        ax3.axis([0, x_max_plt, 0, n_tests])
        ax3.set_title(title_lst[1])
        ax3.set_yticks(pos)
        ax3.set_yticklabels([])

        for rect, t in zip(rects, rel_times):
            width = rect.get_width()
            xloc = width + (0.03 * rel_xmax)
            clr = 'black'
            align = 'left'

            yloc = rect.get_y() + rect.get_height() / 2.0
            ax3.text(xloc, yloc, '%4.2f' % t, horizontalalignment = align,
                     verticalalignment = 'center', color = clr)#, weight = 'bold')

        ax4 = ax3.twinx()
        ax3.plot([1, 1], [0, n_tests], 'k--')
        ax4.set_yticks([0] + list(pos) + [n_tests])
        ax4.set_yticklabels([''] + ['%4.2f s' % s for s in list(times)] + [''])
        ax4.set_xticks([0, 1] + list(range(5 , x_max_plt + 1, 5)))
        ax4.set_xticklabels(['%i' % s for s in ([0, 1] + list(range(5 , x_max_plt + 1, 5)))])

    traits_view = View(Item('sampling_structure_btn', show_label = False),
                       Item('sampling_efficiency_btn', show_label = False),
                       Item('language_efficiency_btn', show_label = False),
                       width = 0.2,
                       height = 0.2,
                       buttons = ['OK', 'Cancel'])

if __name__ == '__main__':

    from stats.spirrid_bak.rv import RV
    from scipy.special import erf
    import math

    # response function
    def fiber_tt_2p(e, la, xi):
        ''' Response function of a single fiber '''
        return la * e * Heaviside(xi - e)

    # statistical characteristics (mean, stdev)
    m_la, std_la = 10., 1.0
    m_xi, std_xi = 1.0, 0.1

    # discretize the control variable (x-axis)
    e_arr = np.linspace(0, 1.2, 40)

    # Exact solution
    def mu_q_ex(e, m_xi, std_xi, m_la):
        return e * (0.5 - 0.5 * erf(0.5 * math.sqrt(2) * (e - m_xi) / std_xi)) * m_la

    mu_q_ex_arr = mu_q_ex(e_arr, m_xi, std_xi, m_la)

    g_la = RV('norm', m_la, std_la)
    g_xi = RV('norm', m_xi, std_xi)

    s = SPIRRID(q = fiber_tt_2p,
                e_arr = e_arr,
                n_int = 10,
                tvars = dict(la = g_la, xi = g_xi),
                )

    mu_q_ex_arr = mu_q_ex(e_arr, m_xi, std_xi, m_la)

    slab = SPIRRIDLAB(s = s, save_output = False, show_output = True,
                      exact_arr = mu_q_ex(e_arr, m_xi, std_xi, m_la))

    slab.configure_traits(kind = 'live')

#    powers = np.linspace(1, math.log(200, 10), 50)
#    n_int_range = np.array(np.power(10, powers), dtype = int)
#
#    slab.sampling_efficiency(n_int_range = n_int_range)
