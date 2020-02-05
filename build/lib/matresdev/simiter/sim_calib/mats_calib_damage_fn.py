#-------------------------------------------------------------------------
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
# Created on May 29, 2009 by: rch

from copy import copy
import os
from os.path import join
import pickle

from ibvpy.core.scontext import SContext
from ibvpy.core.tloop import TLoop, TLine
from ibvpy.core.tstepper import TStepper
from ibvpy.mats.mats2D.mats2D_explore import MATS2DExplore
from ibvpy.mats.mats_explore import MATSExplore
from mathkit.mfn import MFnLineArray
from matplotlib.figure import Figure
from pyface.api import ImageResource
from scipy.optimize import brentq, newton, fsolve, brenth
from traits.api import \
    Float, Instance, Array, Int, Property, cached_property, on_trait_change, Bool, \
    HasTraits, File, Event, Str
from traitsui.api import \
    View, Item, FileEditor, HSplit, Group, VSplit, \
    Handler
from traitsui.menu import \
    Action, CloseAction, HelpAction, Menu, \
    MenuBar, NoButtons, Separator, ToolBar, CancelButton, OKButton
from util.traits.editors.mpl_figure_editor import MPLFigureEditor

from matresdev.db.exdb.ex_run import ExRun
from matresdev.db.simdb.simdb import simdb
import numpy as np
import pylab as p


data_file_editor = FileEditor(filter=['*.DAT'])


# ---------------------------------------------------
# Calibration controller
# ---------------------------------------------------

class MATSCalibDamageFnController(Handler):
    '''Handle the dynamic interaction with the calibrator.
    '''

    #-------------------------------------------------------------------------
    # Public Controller interface
    #-------------------------------------------------------------------------
    def run_calibration(self, ui_info):

        calibrator = ui_info.object

        calibrator.init()
        calibrator.fit_response()
        calibrator.store()

# ---------------------------------------------------
# Calibrator of the damage function from uniaxial test:
# ---------------------------------------------------


class MATSCalibDamageFn(MATSExplore):
    '''
    Fitting algorithm for the damage function of the
    quasi-ductile anisotropic material model.

    The algorithm uses the TLoop instance to proceed step
    by step with the computation. The value of the damage function
    for the time step t_n is identified iteratively by adjusting
    the values and evaluating the corresponding equilibrated stresses.

    The control parameters of the algorithm are:

    @param step_size: time step for fitting the damage parameter.
    @param tmax: end time for fitting, it might be also be set implicitly
    for integrity = 1 - full damage of the material.
    '''

    # store the fitted 'PhiFn' in the data base, i.e. 'CCSUniteCell'
    store_fitted_phi_fn = Bool(True)

    # default settings are overloaded with settings specified in 'ec_config'

    max_eps = Property(Float)

    def _get_max_eps(self):
        # return 0.007  # set explicit value when calibration is aborted (mean
        # value of strain)
        return self.mfn_line_array_target.xdata[-1]

    n_steps = Int(1)

    log = Bool(False)

    # TLine parameter
    #
    KMAX = Int
    tolerance = Float
    RESETMAX = Float

    step_size = Property(Float, depends_on='max_eps,n_steps')

    @cached_property
    def _get_step_size(self):
        print('step_size = ', self.max_eps / self.n_steps)
        return self.max_eps / self.n_steps

    def run_through(self):
        '''Run the computation without fitting from the start to the end
        '''
        self.tloop.tline.max = self.tmax
        self.tloop.tline.step = self.step_size
        self.tloop.eval()
        print('ending time', self.tloop.t_n1)
        # show the response

    def run_step_by_step(self):
        '''Run the computation step by step from the start to the end
        '''
        n_steps = int(self.tmax / self.step_size)
        self.tloop.tline.step = self.step_size
        current_time = 0.
        tmax = 0.
        for i in range(n_steps):
            print('STEP', i)
            self.run_one_step()

    def run_trial_step(self):
        '''Run the computation one step starting from the
        current time t_n to iterate the value for phi_new
        which gives a fit with macroscopic stress curve.
        NOTE: The trial step does not update 'U_n' or 't_n'!
        '''
        if self.log:
            print('--------- run trial step: --------- ')
        if len(self.tloop.U_n) == 0:
            current_U_n = self.tloop.tstepper.new_cntl_var()
            print('U_n = None: tloop.tstepper.new_cntl_var()', self.tloop.tstepper.new_cntl_var())
        else:
            current_U_n = self.tloop.U_n[:]
        current_time = self.tloop.t_n
        self.run_one_step()

        # reset the current time back
        self.tloop.t_n = current_time
        self.tloop.U_n[:] = current_U_n[:]
        if self.log:
            print('--------- end of trial step --------- ')

        self.tloop.tstepper.sctx.update_state_on = False

    def run_one_step(self):
        '''Run the computation one step starting from the
        current time t_n with the iterated value for phi_new
        in order to update TLoop and save the new phi value in
        the array ydata of PhiFnGeneral
        NOTE: The calculated step does update 'U_n' or 't_n'!
        '''
        self.tloop.tline.step = self.step_size
        current_time = self.tloop.t_n
        tmax = current_time + self.step_size
        self.tloop.tline.max = tmax
        self.tloop.eval()
        self.update_e_max_value_new = True

    #--------------------------------------------------
    # Data source for calibration within simdb
    #--------------------------------------------------

    ex_run = Instance(ExRun)

    composite_tensile_test = Property

    def _get_composite_tensile_test(self):
        return self.ex_run.ex_type

    composite_cross_section = Property

    def _get_composite_cross_section(self):
        return self.composite_tensile_test.ccs

    def get_target_data_exdb_tensile_test(self):
        '''Use the data from the ExDB
        '''
        ctt = self.composite_tensile_test

        # save 'sig_eps_arr' in directory "/simdb/simdata/mats_calib_damage_fn"
        simdata_dir = os.path.join(simdb.simdata_dir, 'mats_calib_damage_fn')
        if os.path.isdir(simdata_dir) == False:
            os.makedirs(simdata_dir)
        ctt_key = str(self.composite_tensile_test.key)
        filename = os.path.join(simdata_dir, 'eps-sig-arr_' + ctt_key + '.csv')

        xdata, ydata = ctt.eps_c_interpolated_smoothed[:, None], \
            ctt.sig_c_interpolated_smoothed[:, None]
        eps_sig_arr = np.hstack([xdata, ydata])
        print('eps_sig_arr')
        np.savetxt(filename, eps_sig_arr, delimiter=';')
        print('eps-sig-data saved to file %s' % (filename))

        # smoothed data without jumps with interpolated starting point in the
        # origin
        return ctt.eps_c_interpolated_smoothed, ctt.sig_c_interpolated_smoothed
#        return ctt.eps_c_interpolated, ctt.sig_c_interpolated  # original data without jumps with interpolated starting point in the origin
#        return ctt.eps_ironed, ctt.sig_c_ironed  # original data without smoothing (without jumps)
#        return ctt.eps_smooth, ctt.sig_c_smooth #smoothed data

    #--------------------------------------------------
    # interpolation function for fitting data:
    #--------------------------------------------------
    mfn_line_array_target = Property(Instance(MFnLineArray),
                                     depends_on='ex_run')

    @cached_property
    def _get_mfn_line_array_target(self):
        xdata, ydata = self.get_target_data_exdb_tensile_test()
        print('xdata[-1]', xdata[-1])
        return MFnLineArray(xdata=xdata, ydata=ydata)

    fitted_phi_fn = Instance(MFnLineArray)

    #---------------------------------------------------------------
    # PLOT OBJECT
    #-------------------------------------------------------------------
    figure = Instance(Figure)

    def _figure_default(self):
        figure = Figure(facecolor='white')
        figure.add_axes([0.12, 0.13, 0.85, 0.74])
        return figure

    data_changed = Event

    def init(self):
        #--------------------------------------------------
        # for fitting use 'General'-function for 'phi_fn':
        #--------------------------------------------------
        # The value pair for the piecewise linear definition
        # of 'phi_fn' value consists of current strain and the
        # iterated 'phi_value'. The microplanes with a lower
        # microplane strain level use an interpolated value
        # for 'phi'
        self.fitted_phi_fn = self.dim.mats_eval.phi_fn.mfn
        self.fitted_phi_fn.xdata = [0]
        self.fitted_phi_fn.ydata = [1]
        self.fitted_phi_fn.data_changed = True
        # initialize TLoop parameters:
        self.tloop.setup()
        self.tloop.tstepper.sctx.mats_state_array[:] = 0.0
        self.tloop.U_n[:] = 0.0
        self.tloop.rtrace_mngr.clear()
        self.tloop.verbose_iteration = False
        self.tloop.verbose_load_step = False
        self.tloop.verbose_time = False
        # set TLine parameters
        self.tloop.tline.KMAX = self.KMAX
        self.tloop.tline.tolerance = self.tolerance
        self.tloop.tline.RESETMAX = self.RESETMAX

    # store trial step data in the lists if trial steps are to be stored
    # for the plotting method 'plot_trail_steps'
    #
    rec_trial_steps = True
    phi_trial_list_i = []
    sig_trial_list_i = []
    phi_trial_list_n = []
    sig_trial_list_n = []

    def get_lack_of_fit(self, phi_trial):
        '''Return the difference between the macroscopic stress calculated
        based on the value of phi_trial (damage at the next step) and the
        macroscopic stress defined as target data (=fitting curve)
        '''
        if self.log:
            print('\n')
            print("#'get_lack_of_fit' for the trial value # START")
            print('    phi_trial    = ', phi_trial)

        # value of the principle macroscopic strain corresponds to control
        # variable
        current_time = self.tloop.t_n

        if self.log:
            print('    current_time = ', current_time)
            print('    step_size    = ', self.step_size)

        # ------------------------------------
        # add new pair in fitted_phi_fn
        # ------------------------------------
        # consisting of 'e_max_value_new' and 'phi_trial'
        x = np.hstack([self.fitted_phi_fn.xdata[:],
                       current_time + self.step_size])
        y = np.hstack([self.fitted_phi_fn.ydata[:], phi_trial])
        self.fitted_phi_fn.set(xdata=x, ydata=y)
        self.fitted_phi_fn.data_changed = True

        # ------------------------------------
        # get state array before trial:
        # ------------------------------------
        mats_state_array_old = copy(self.tloop.tstepper.sctx.mats_state_array)

        # ------------------------------------
        # run trial step:
        # ------------------------------------
        if self.log:
            print('    reset current_U_n   =', self.tloop.U_n)
            print('CURRENT PHI', self.dim.mats_eval.phi_fn.mfn.ydata)
        # try the next equilibrium
        self.run_trial_step()

        # ------------------------------------
        # reset mats_state_array:
        # ------------------------------------
        # Note: the material state array (i.e. the maximum microstrains) are
        # updated within the iterations of each trial step, therefore a reset
        # is necessary in order to start each trial step with the same state
        # variables
        self.tloop.tstepper.sctx.mats_state_array[:] = mats_state_array_old[:]
        if self.log:
            print('    reset state array')

        # ------------------------------------
        # remove trial value in fitted_phi_fn
        # ------------------------------------
        x = self.fitted_phi_fn.xdata[:-1]
        y = self.fitted_phi_fn.ydata[:-1]
        self.fitted_phi_fn.set(xdata=x, ydata=y)
        self.fitted_phi_fn.data_changed = True

        # ------------------------------------
        # get the lack of fit
        # ------------------------------------
        # get calculated value for 'sig_app' based on the current value of 'phi_trial':
        # and evaluate the difference between the obtained stress and the
        # measured response
        self.tloop.rtrace_mngr.rtrace_bound_list[0].redraw()
        sig_app_trial = self.tloop.rtrace_mngr.rtrace_bound_list[0].trace.ydata[-1]
        # get corresponding value from the target data:
        sig_app_target = self.mfn_line_array_target.get_value(
            current_time + self.step_size)
        # absolut error:
        lack_of_fit_absolut = sig_app_trial - sig_app_target
        # relative error:
        lack_of_fit_relative = lack_of_fit_absolut / sig_app_target

        if self.log:
            print('    sig_app_trial ', sig_app_trial)
            print('    sig_app_target', sig_app_target)
            print('    lack_of_fit_absolute  ', lack_of_fit_absolut)
            print('    lack_of_fit_relative  ', lack_of_fit_relative)
            print('# get_lack_of_fit # END ')

        if self.rec_trial_steps:
            # store all trial values of 'phi_trail' and 'sig_app_trail' within each iteration to a global list
            #
            self.phi_trial_list_i.append(phi_trial)
            self.sig_trial_list_i.append(sig_app_trial)

        return lack_of_fit_relative

    param_key = Str('')

    phi_max_factor = Float(1.0)

    xtol = Float(1e-6,
                 label='lack-of-fit tolerance')

    def fit_response(self):
        '''iterate phi_trial in each incremental step such that the
        lack of fit between the calculated stress and the target
        curve is smaller then xtol defined in function 'brentq'.
        NOTE: the method 'get_lack_of_fit' returns the relative error.
        '''

        self.tloop.reset()

        phi_old = 1.0

        # map the array dimensions to the plot axes
        #
        figure = self.figure

        axes = figure.axes[0]

        print('n_steps', self.n_steps)
        for n in range(self.n_steps):

            axes.clear()

            phi_new = phi_old

            # use scipy-functionality to get the iterated value of phi_new
            # If the trial value calculated with phi_trial = phi_old
            # is smaller then target value get_lack_of_fit has no sign change
            # for phi_trial = phi_old and phi_trial = 0. which is a requirement
            # for the function call 'brentq'. In this case the old value
            # for phi_trial is used and tloop moves on one step
            try:
                # The method brentq has optional arguments such as
                #   'xtol'    - absolut error (default value = 1.0e-12)
                #   'rtol'    - relative error (not supported at the time)
                #   'maxiter' - maximum numbers of iterations used
                #
                # Here xtol is used to specify the allowed RELATIVE error!
                # therefore the relative lack of fit is returned in
                # method 'get_lack_of_fit'
                _xtol = self.xtol
                phi_max = min(1.0, phi_old * self.phi_max_factor)
                phi_min = phi_old * 0.2  # 0.0  # phi_old * 0.3
                phi_new = brentq(self.get_lack_of_fit,
                                 phi_min, phi_max, xtol=_xtol)
                # @todo: check if 'brenth' gives better fitting results; faster?
#                phi_new = brenth( self.get_lack_of_fit, 0., phi_old )
                print('(#) n = ', n, phi_new, phi_max, phi_old, phi_old - phi_new)
            except ValueError:

                if self.log:
                    lof_0 = self.get_lack_of_fit(0.)
                    lof_phi_old = self.get_lack_of_fit(phi_old)
                    print('No sign change between get_lack_of_fit(phi_old) = ', lof_phi_old, ' and ')
                    print('get_lack_of_fit(0.) = ', lof_0)
                    print('Use old value for phi_trial. phi_old = ', phi_old)
                else:
                    print('(!) n = ', n, phi_new, phi_max, phi_old, phi_old - phi_new)
                    phi_new = phi_old

            # current time corresponds to the current strain applied
            #
            current_time = self.tloop.t_n

            # replace old 'phi_value' with iterated value:
            #
            phi_old = phi_new

            # get mats_state_array:
#            mats_state_array = copy(self.tloop.tstepper.sctx.mats_state_array)

            # update phi_data:
            #
            x = np.hstack([self.fitted_phi_fn.xdata[:],
                           current_time + self.step_size])
            y = np.hstack([self.fitted_phi_fn.ydata[:], phi_new])

            axes.plot(x, y, color='blue', linewidth=2)
            self.data_changed = True

            self.fitted_phi_fn.set(xdata=x, ydata=y)
            self.fitted_phi_fn.data_changed = True

            # run one step with the iterated value for phi in order to
            # update the state array and to move forward one step:
            if self.log:
                print('\n')
                print('### run_one_step ###')
                print('### step', n, '###')
                print('### current time:', current_time)

            if self.rec_trial_steps:
                # add entries of the iterations ('i') in the current step ('n')
                # (yields a list of lists)
                #
                self.phi_trial_list_n.append(self.phi_trial_list_i)
                self.sig_trial_list_n.append(self.sig_trial_list_i)
                # delete the entries of the iterations ('i') in the last step ('n')
                # and fill it with the iterations of the next step ('n+1')
                #
                self.phi_trial_list_i = []
                self.sig_trial_list_i = []

            self.run_one_step()
#            print '(g%)' %(n)

        self.fitted_phi_fn.changed = True

    def store(self):
        mats_key = self.dim.mats_eval.__class__.__name__
        ctt_key = str(self.composite_tensile_test.key)
        if self.store_fitted_phi_fn:
            print("stored 'fitted_phi_fn' in CCSUnitCell with material model %s and calibration test %s" % (mats_key, ctt_key))
            print('ctt_key + self.param_key', ctt_key + self.param_key)
            self.composite_cross_section.set_param(mats_key, ctt_key + self.param_key,
                                                   # self.composite_cross_section.set_param(mats_key,
                                                   # ctt_key,
                                                   copy(self.fitted_phi_fn))
            # save 'sig_eps_arr' in directory
            # "/simdb/simdata/mats_calib_damage_fn"
            simdata_dir = os.path.join(
                simdb.simdata_dir, 'mats_calib_damage_fn')
            if os.path.isdir(simdata_dir) == False:
                os.makedirs(simdata_dir)
            ctt_key = str(self.composite_tensile_test.key)
            filename = os.path.join(
                simdata_dir, 'eps-phi-arr_' + ctt_key + self.param_key + '.csv')

            xdata, ydata = self.fitted_phi_fn.xdata[:,
                                                    None], self.fitted_phi_fn.ydata[:, None]
            eps_phi_arr = np.hstack([xdata, ydata])
            np.savetxt(filename, eps_phi_arr, delimiter=';')
            print('eps-phi-data saved to file %s' % (filename))

    format_ticks = Bool(False)

    def plot_trial_steps(self):
        '''Plot target (sig-eps-curve of the tensile test) and trial curves
        and corresponding phi function together with trail steps from the iteration process.
        NOTE: the global variable 'rec_trial_steps' must be set to 'True' in order to store the iteration values
              within the global variables 'phi_trial_list_n' and 'sig_trial_list_n'
        n - index of the time steps to be considered
        i - index of the iteration steps performed in order to fit the target curve
        '''
        #-------------------------------------------------------------------
        # configure the style of the font to be used for labels and ticks
        #-------------------------------------------------------------------
        #
        from matplotlib.font_manager import FontProperties
        font = FontProperties()
#        font.serif         : Times, Palatino, New Century Schoolbook, Bookman, Computer Modern Roman
#        font.sans-serif    : Helvetica, Avant Garde, Computer Modern Sans serif
#        font.cursive       : Zapf Chancery
#        font.monospace     : Courier, Computer Modern Typewriter
        font.set_name('Script MT')
        # name = ['Times New Roman', 'Helvetica', 'Script MT'] #?
        font.set_family('serif')
        # family = ['serif', 'sans-serif', 'cursive', 'fantasy', 'monospace']
        font.set_style('normal')
        # style  = ['normal', 'italic', 'oblique']
        font.set_size('small')
        # size  = ['xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large', '11']
        font.set_variant('normal')
        # variant= ['normal', 'small-caps']
        font.set_weight('medium')
        # weight = ['light', 'normal', 'medium', 'semibold', 'bold', 'heavy', 'black']

        #-------------------------------------------------------------------

        p.figure(facecolor='white', dpi=600,
                 figsize=(8, 6))  # white background

        # time list corresponding to the specified numbers of steps and step size
        #
        step_list = [n * self.step_size for n in range(self.n_steps + 1)]

        # get list of lists containing the trial values of 'sig_app' and 'phi_trail'
        # the lists are defined as global variables of 'MATSCalibDamageFn' and are filled
        # within the iteration process when the method 'get_lack_of_fit" is called
        #
        phi_trial_list_n = [[1.]] + self.phi_trial_list_n
        sig_trial_list_n = [[0.]] + self.sig_trial_list_n

        xrange = 10.  # plotting range for strain [mm/m]
        yrange = 15.  # plotting range for stress [MPa]

        for n in range(self.n_steps):
            for i in range(len(phi_trial_list_n[n + 1])):
                x = np.array([step_list[n], step_list[n + 1]])
                eps = 1000. * x  # plot strains in permil on the x-axis
                #--------------------------------------
                # sig-eps trial
                #--------------------------------------
                # plot the numerically calculated sig-eps-curve (tensile test)
                # (with trial steps)
                #
                sig_trail = np.array(
                    [sig_trial_list_n[n][-1], sig_trial_list_n[n + 1][i]])
                p.subplot(222)
                p.plot(eps, sig_trail, color='k', linewidth=1)
                p.xlabel(r'strain $\varepsilon$ [1E-3]', fontproperties=font)
                p.ylabel('stress $\sigma$ [MPa]', fontproperties=font)
                if self.format_ticks:
                    # format ticks for plot
                    p.axis([0, xrange, 0., yrange], fontproperties=font)
                    locs, labels = p.xticks()
                    p.xticks(locs, ["%.0f" %
                                       x for x in locs], fontproperties=font)
                    locs, labels = p.yticks()
                    p.yticks(locs, ["%.0f" %
                                       x for x in locs], fontproperties=font)

                #--------------------------------------
                # phi_trail
                #--------------------------------------
                # plot the fitted phi-function
                # (with trial steps)
                #
                p.subplot(224)
                phi_trail = np.array(
                    [phi_trial_list_n[n][-1], phi_trial_list_n[n + 1][i]])
                p.plot(eps, phi_trail, color='k', linewidth=1)
                p.xlabel(r'strain $\varepsilon$ [1E-3]', fontproperties=font)
                p.ylabel('integrity $\phi$ [-]', fontproperties=font)
                if self.format_ticks:
                    # format ticks for plot
                    p.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
                    p.axis([0, xrange, 0., 1.])
                    locs, labels = p.xticks()
                    p.xticks(locs, ["%.0f" %
                                       x for x in locs], fontproperties=font)
                    locs, labels = p.yticks()
                    p.yticks(locs, ["%.1f" %
                                       x for x in locs], fontproperties=font)

        #--------------------------------------
        # sig-eps target
        #--------------------------------------
        # plot the sig-eps-target curve (tensile test)
        #
        p.subplot(221)
        eps = 1000. * self.mfn_line_array_target.xdata[:-1]
        sig_target = self.mfn_line_array_target.ydata[:-1]
        p.plot(eps, sig_target, color='black', linewidth=1)
        p.xlabel(r'strain $\varepsilon$ [1E-3]', fontproperties=font)
        p.ylabel('stress $\sigma$ [MPa]', fontproperties=font)
        if self.format_ticks:
            # format ticks for plot
            p.axis([0, xrange, 0., yrange])
            locs, labels = p.xticks()
            p.xticks(locs, ["%.0f" %
                               x for x in locs], fontproperties=font)
            locs, labels = p.yticks()
            p.yticks(locs, ["%.0f" %
                               x for x in locs], fontproperties=font)

        #--------------------------------------
        # phi_trail (final)
        #--------------------------------------
        # plot the corresponding fitted phi-function
        # (without trial steps)
        #
        p.subplot(223)
        eps = 1000. * self.fitted_phi_fn.xdata[:-1]
        phi_fn = self.fitted_phi_fn.ydata[:-1]
        p.plot(eps, phi_fn, color='black', linewidth=1)
        p.xlabel(r'strain $\varepsilon$ [1E-3]', fontproperties=font)
        p.ylabel('integrity $\phi$ [-]', fontproperties=font)
        if self.format_ticks:
            # format ticks for plot
            p.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            p.axis([0, xrange, 0., 1.])
            locs, labels = p.xticks()
            p.xticks(locs, ["%.0f" %
                               x for x in locs], fontproperties=font)
            locs, labels = p.yticks()
            p.yticks(locs, ["%.1f" %
                               x for x in locs], fontproperties=font)

        # save figure with calibration process in directory
        # "/simdb/simdata/lcc_table/output_images/save_fig_to_file.png"
        simdata_dir = os.path.join(simdb.simdata_dir, 'mats_calib_damage_fn')
        if os.path.isdir(simdata_dir) == False:
            os.makedirs(simdata_dir)

        ctt_key = self.test_key
        filename = os.path.join(simdata_dir, ctt_key + self.param_key + '.pdf')
        p.savefig(filename)
        print('plot_trail_steps.png saved to file %s' % (filename))
        filename = os.path.join(simdata_dir, ctt_key + self.param_key + '.png')
        p.savefig(filename, dpi=600)
        print('plot_trail_steps.png saved to file %s' % (filename))

        p.show()

    test_key = Property

    def _get_test_key(self):
        return str(self.composite_tensile_test.key)

    #-------------------------------------------------------------------------
    # User interaction
    #-------------------------------------------------------------------------
    toolbar = ToolBar(
        Action(name="Run Calibration",
               tooltip='Run damage function calibration for the current parameters',
               image=ImageResource('kt-start'),
               action="run_calibration"),
        image_size=(22, 22),
        show_tool_names=False,
        show_divider=True,
        name='calibration_toolbar')

    traits_view = View(HSplit(
        Item('ex_run@',
             show_label=False),
        VSplit(
            Item('dim@',
                 id='mats_calib_damage_fn.run.split',
                 dock='tab',
                 resizable=True,
                 label='experiment run',
                 show_label=False),
            id='mats_calib_damage_fn.mode_plot_data.vsplit',
            dock='tab',
        ),
        VSplit(
            Group(
                Item('figure', editor=MPLFigureEditor(),
                     resizable=True, show_label=False),
                id='mats_calib_damage_fn.plot_sheet',
                label='fitted damage function',
                dock='tab',
            ),
            id='mats_calib_damage_fn.plot.vsplit',
            dock='tab',
        ),
        id='mats_calib_damage_fn.hsplit',
        dock='tab',
    ),
        #                        menubar = self.default_menubar(),
        resizable=True,
        toolbar=toolbar,
        handler=MATSCalibDamageFnController(),
        title='Simvisage: damage function calibration',
        id='mats_calib_damage_fn',
        dock='tab',
        buttons=[OKButton, CancelButton],
        height=0.8,
        width=0.8)


def run():
    #-------------------------------------------------------------------------
    # Example using the mats2d_explore
    #-------------------------------------------------------------------------
    from ibvpy.mats.mats2D.mats2D_explore import MATS2DExplore
    from ibvpy.mats.mats2D.mats2D_rtrace_cylinder import MATS2DRTraceCylinder

    from ibvpy.mats.mats2D.mats2D_cmdm.mats2D_cmdm_rtrace_Gf_mic import \
        MATS2DMicroplaneDamageTraceGfmic, \
        MATS2DMicroplaneDamageTraceEtmic, MATS2DMicroplaneDamageTraceUtmic

    from ibvpy.mats.mats2D.mats2D_cmdm.mats2D_cmdm_rtrace_Gf_mac import \
        MATS2DMicroplaneDamageTraceGfmac, \
        MATS2DMicroplaneDamageTraceEtmac, MATS2DMicroplaneDamageTraceUtmac

    from ibvpy.mats.mats2D.mats2D_cmdm.mats2D_cmdm import \
        MATS2DMicroplaneDamage, MATS1DMicroplaneDamage

    from ibvpy.mats.matsXD.matsXD_cmdm import \
        PhiFnGeneral, PhiFnStrainHardening

    from ibvpy.api import RTraceGraph, RTraceArraySnapshot

    from mathkit.mfn import MFnLineArray
    from numpy import array, hstack

    from ibvpy.mats.mats2D.mats2D_explorer_bcond import BCDofProportional
    from os.path import join

    ec = {
        # overload the default configuration
        'bcond_list': [BCDofProportional(max_strain=1.0, alpha_rad=0.0)],
        'rtrace_list': [
            RTraceGraph(name='stress - strain',
                        var_x='eps_app', idx_x=0,
                        var_y='sig_app', idx_y=0,
                        record_on='iteration'),
        ],
    }

    mats_eval = MATS2DMicroplaneDamage(
        n_mp=15,
        # mats_eval = MATS1DMicroplaneDamage(
        elastic_debug=False,
        stress_state='plane_stress',
        symmetrization='sum-type',
        model_version='compliance',
        phi_fn=PhiFnGeneral,
    )

#    print 'normals', mats_eval._MPN
#    print 'weights', mats_eval._MPW

    fitter = MATSCalibDamageFn(KMAX=300,
                               tolerance=5e-4,  # 0.01,
                               RESETMAX=0,
                               dim=MATS2DExplore(
                                   mats_eval=mats_eval,
                                   explorer_config=ec,
                               ),
                               store_fitted_phi_fn=True,
                               log=False
                               )

    #-------------------------------------------
    # run fitter for entire available test data:
    #-------------------------------------------

    calibrate_all = False

    if calibrate_all:
        from matresdev.db.exdb.ex_run_table import \
            ExRunClassExt
#         from matresdev.db.exdb.ex_composite_tensile_test import \
#             ExCompositeTensileTest
#         ex = ExRunClassExt(klass=ExCompositeTensileTest)
#         for ex_run in ex.ex_run_list:
#             if ex_run.ready_for_calibration:
#                 print 'FITTING', ex_run.ex_type.key
#                 # 'E_c' of each test is different, therefore 'mats_eval'
#                 # needs to be defined for each test separately.
#                 #
#                 E_c = ex_run.ex_type.E_c
#                 nu = ex_run.ex_type.ccs.concrete_mixture_ref.nu
#
#                 # run calibration
#                 #
#                 fitter.ex_run = ex_run
#                 fitter.dim.mats_eval.E = E_c
#                 fitter.dim.mats_eval.nu = nu
#                 fitter.init()
#                 fitter.fit_response()

    else:

        test_file = join(simdb.exdata_dir,
                         'tensile_tests',
                         'dog_bone',
                         #                              'buttstrap_clamping',

                         '2010-02-09_TT-10g-3cm-a-TR_TRC11',
                         #                              'TT11-10a-average.DAT' )
                         'TT-10g-3cm-a-TR-average.DAT')

        #-----------------------------------
        # tests for 'BT-3PT-12c-6cm-TU_ZiE'
        #-----------------------------------
        # 'ZiE-S1': test series no. 1 (age = 11d)
        #
#                                 '2011-05-23_TT-12c-6cm-0-TU_ZiE',
#                                 'TT-12c-6cm-0-TU-V2.DAT')

        # 'ZiE-S2': test series no. 2 (age = 9d)
        #
#                                 '2011-06-10_TT-12c-6cm-0-TU_ZiE',
#                                 'TT-12c-6cm-0-TU-V2.DAT')

        #-----------------------------------
        # tests for 'BT-4PT-12c-6cm-TU_SH4'
        # tests for 'ST-12c-6cm-TU' (fresh)
        #-----------------------------------
        # @todo: add missing front strain information from Aramis3d testing
        #
#                               '2012-04-12_TT-12c-6cm-0-TU_SH4-Aramis3d',
#                               'TT-12c-6cm-0-TU-SH4-V2.DAT')

#                                '2012-02-14_TT-12c-6cm-0-TU_SH2',
#                                'TT-12c-6cm-0-TU-SH2-V2.DAT')

#                                '2012-02-14_TT-12c-6cm-0-TU_SH2',
#                                'TT-12c-6cm-0-TU-SH2F-V3.DAT')

        # used for suco(!)
#                                '2012-02-14_TT-12c-6cm-0-TU_SH2',
#                                'TT-12c-6cm-0-TU-SH2-V1.DAT')

        #-----------------------------------
        # tests for 'BT-3PT-6c-2cm-TU_bs'
        #-----------------------------------
        # barrelshell
        #
#                                 # TT-bs1
#                                 '2013-05-17_TT-6c-2cm-0-TU_bs1',
#                                 'TT-6c-2cm-0-TU-V3_bs1.DAT')
#                                 # TT-bs2
#                                 '2013-05-21-TT-6c-2cm-0-TU_bs2',
#                                 'TT-6c-2cm-0-TU-V1_bs2.DAT')
#                                 # TT-bs3
#                                 '2013-06-12_TT-6c-2cm-0-TU_bs3',
#                                 'TT-6c-2cm-0-TU-V1_bs3.DAT')
#                                 # TTb-bs4-Aramis3d
#                                  '2013-07-09_TTb-6c-2cm-0-TU_bs4-Aramis3d',
#                                  'TTb-6c-2cm-0-TU-V2_bs4.DAT')

        #-----------------------------------
        # tests for 'TT-6c-2cm-90-TU'
        #-----------------------------------
        #
#                                '2013-05-22_TTb-6c-2cm-90-TU-V3_bs1',
#                                'TTb-6c-2cm-90-TU-V3_bs1.DAT')

#                               '2013-05-17_TT-6c-2cm-0-TU_bs1',
#                               'TT-6c-2cm-90-TU-V3_bs1.DAT')

#        test_file = join(simdb.exdata_dir,
#                               'tensile_tests',
#                               'buttstrap_clamping',
#                               '2013-07-18_TTb-6c-2cm-0-TU_bs5',
#                               'TTb-6c-2cm-0-TU-V1_bs5.DAT')
# #                               'TTb-6c-2cm-0-TU-V3_bs5.DAT')
#
#        test_file = join(simdb.exdata_dir,
#                               'tensile_tests',
#                               'buttstrap_clamping',
#                               '2013-07-09_TTb-6c-2cm-0-TU_bs4-Aramis3d',
#                               'TTb-6c-2cm-0-TU-V2_bs4.DAT')

        #-----------------------------------
        # tests for 'TT-6g-2cm-0-TU' (ARG-1200-TU)
        #-----------------------------------
        #
        # test series no.1
        #
#        test_file = join(simdb.exdata_dir,
#                               'tensile_tests',
#                               'dog_bone',
#                               '2012-12-10_TT-6g-2cm-0-TU_bs',
#                               'TT-6g-2cm-0-V2.DAT')
        # test series no.3
        #
#        test_file = join(simdb.exdata_dir,
#                               'tensile_tests',
#                               'buttstrap_clamping',
#                               '2013-07-09_TTb-6g-2cm-0-TU_bs4-Aramis3d',
#                               'TTb-6g-2cm-0-TU-V1_bs4.DAT')

        # test series NxM_1
        #
#        test_file = join(simdb.exdata_dir,
#                          'tensile_tests',
#                          'buttstrap_clamping',
#                          '2014-04-30_TTb-6c-2cm-0-TU_NxM1',
#                          'TTb-6c-2cm-0-TU-V16_NxM1.DAT')

        #------------------------------------------------------------------
        # set 'ex_run' of 'fitter' to selected calibration test
        #------------------------------------------------------------------
        #
        ex_run = ExRun(data_file=test_file)
        fitter.ex_run = ex_run

        #------------------------------------------------------------------
        # specify the parameters used within the calibration
        #------------------------------------------------------------------
        #
        # get the composite E-modulus and Poisson's ratio as stored
        # in the experiment data base for the specified age of the tensile test
        #
        E_c = ex_run.ex_type.E_c
        print('E_c', E_c)

#        # use the value as graphically determined from the tensile test (= initial stiffness for tension)
#        E_c = 28000.

        # age, Em(age), and nu of the slab test or bending test determines the
        # calibration parameters. Those are used for calibration and are store in the 'param_key'
        # appendet to the calibration-test-key
        #
        age = 28

        # E-modulus of the concrete matrix at the age of testing
        # NOTE: value is more relevant as compression behavior is determined by it in the bending tests and slab tests;
        # behavior in the tensile zone is defined by calibrated 'phi_fn' with the predefined 'E_m'
#        E_m = ex_run.ex_type.ccs.get_E_m_time(age)
        E_c = ex_run.ex_type.ccs.get_E_c_time(age)

        # use average E-modul from 0- and 90-degree direction for fitter in both directions
        # this yields the correct tensile behavior and returns the best average compressive behavior
        #
#        E_c = 22313.4

        # alternatively use maximum E-modul from 90-direction also for 0-degree direction for fitting
        # this yields the correct tensile behavior also in the linear elastic regime for both directions corresponding to the
        # tensile test behavior (note that the compressive E-Modulus in this case is overestimated in 0-degree direction; minor influence
        # assumed as behavior is governed by inelastic tensile behavior and anisotropic redistrirbution;
        #
#        E_c = 29940.2
#        E_c = 29100.
#        E_c = 22390.4
#        E_c = 18709.5
        E_c = 28700.

        # smallest value for matrix E-modulus obtained from cylinder tests (d=150mm)
#        E_m = 18709.5

        # set 'nu'
        # @todo: check values stored in 'mat_db'
        #
        nu = 0.20
        ex_run.ex_type.ccs.concrete_mixture_ref.nu = nu

        n_steps = 200
        fitter.n_steps = n_steps

        fitter.format_ticks = True

        fitter.ex_run.ex_type.age = age
        print('age = %g used for calibration' % age)
        fitter.ex_run = ex_run

#        print 'E_m(age) = %g used for calibration' % E_m
#        fitter.dim.mats_eval.E = E_m

        print('E_c(age) = %g used for calibration' % E_c)
        fitter.dim.mats_eval.E = E_c

        print('nu = %g used for calibration' % nu)
        fitter.dim.mats_eval.nu = nu

        print('n_steps = %g used for calibration' % n_steps)

        max_eps = fitter.max_eps
        print('max_eps = %g used for calibration' % max_eps)

        #------------------------------------------------------------------
        # set 'param_key' of 'fitter' to store calibration params in the name
        #------------------------------------------------------------------
        #
#        param_key = '_age%g_Em%g_nu%g_nsteps%g' % (age, E_m, nu, n_steps)
#        param_key = '_age%g_Ec%g_nu%g_nsteps%g__smoothed' % (age, E_c, nu, n_steps, max_eps)
        param_key = '_age%g_Ec%g_nu%g_nsteps%g_smoothed' % (
            age, E_c, nu, n_steps)

        fitter.param_key = param_key
        print('param_key = %s used in calibration name' % param_key)

        #------------------------------------------------------------------
        # run fitting procedure
        #------------------------------------------------------------------
        #
        import pylab as p
        ax = p.subplot(111)
        fitter.mfn_line_array_target.mpl_plot(ax)
        p.show()

        fitter.init()
        fitter.fit_response()
        fitter.store()
        fitter.plot_trial_steps()

    return

    #---------------------------
    # basic testing of fitter methods:
    #---------------------------

    # set to True for basic testing of the methods:
    basic_tests = False

    if basic_tests:
        fitter.run_through()
        #    fitter.tloop.rtrace_mngr.rtrace_bound_list[0].configure_traits()
        fitter.tloop.rtrace_mngr.rtrace_bound_list[0].redraw()
        last_strain_run_through = fitter.tloop.rtrace_mngr.rtrace_bound_list[0].trace.xdata[:]
        last_stress_run_through = fitter.tloop.rtrace_mngr.rtrace_bound_list[0].trace.ydata[:]
        print('last strain (run-through) value', last_strain_run_through)
        print('last stress (run-through) value', last_stress_run_through)

        fitter.tloop.reset()
        fitter.run_step_by_step()
        # fitter.tloop.rtrace_mngr.rtrace_bound_list[0].configure_traits()
        fitter.tloop.rtrace_mngr.rtrace_bound_list[0].redraw()
        last_strain_step_by_step = fitter.tloop.rtrace_mngr.rtrace_bound_list[0].trace.xdata[:]
        last_stress_step_by_step = fitter.tloop.rtrace_mngr.rtrace_bound_list[0].trace.ydata[:]
        print('last stress (step-by-step) value', last_stress_step_by_step)

        fitter.run_trial_step()
        fitter.run_trial_step()
        fitter.tloop.rtrace_mngr.rtrace_bound_list[0].redraw()
        strain_after_trial_steps = fitter.tloop.rtrace_mngr.rtrace_bound_list[0].trace.xdata[:]
        stress_after_trial_steps = fitter.tloop.rtrace_mngr.rtrace_bound_list[0].trace.ydata[:]
        print('stress after trial', stress_after_trial_steps)

        fitter.init()
        # fitter.mats2D_eval.configure_traits()
        lof = fitter.get_lack_of_fit(1.0)
        print('1', lof)
        lof = fitter.get_lack_of_fit(0.9)
        print('2', lof)

        # fitter.tloop.rtrace_mngr.configure_traits()
        fitter.run_trial_step()

    else:
        from ibvpy.plugins.ibvpy_app import IBVPyApp
        ibvpy_app = IBVPyApp(ibv_resource=fitter)
        ibvpy_app.main()


if __name__ == '__main__':
    run()
