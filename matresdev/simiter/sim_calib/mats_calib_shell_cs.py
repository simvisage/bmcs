
from os.path import join

from traits.api import \
    Float, File, Bool, Property, cached_property,\
    List, HasStrictTraits, Dict, Instance, Str, Directory

from ibvpy.api import RTDofGraph
from ibvpy.mats.mats2D.mats2D_cmdm.mats2D_cmdm import \
    MATS2DMicroplaneDamage
from ibvpy.mats.mats2D.mats2D_explore import MATS2DExplore
from ibvpy.mats.mats2D.mats2D_explorer_bcond import BCDofProportional
from ibvpy.mats.matsXD.matsXD_cmdm import \
    PhiFnGeneral
from ibvpy.mats.mats_explore import MATSExplore
from mathkit.array.smoothing import smooth as smooth_array
from mathkit.mfn import MFnLineArray
from matresdev.db.simdb.simdb import simdb
import numpy as np
import pylab as p

from .mats_calib_damage_fn import MATSCalibDamageFn


def get_test_data_dir():
    return join(simdb.exdata_dir,
                'tensile_tests',
                'buttstrap_clamping',
                '2017-06-22-TTb-sig-eps-dresden-girder')


class MATSCalibDamageFnSigEps(MATSCalibDamageFn):

    file_name = Str(input=True)

    test_dir = Directory

    def _test_dir_default(self):
        return get_test_data_dir()

    test_file = Property(File, depends_on='file_name')

    @cached_property
    def _get_test_file(self):
        return join(self.test_dir, self.file_name + '.txt')

    phi_file = Property(File, depends_on='file_name')

    @cached_property
    def _get_phi_file(self):
        return join(self.test_dir, self.file_name + '_concr_phi_data.txt')

    E_c_file = Property(File, depends_on='file_name')

    @cached_property
    def _get_E_c_file(self):
        return join(self.test_dir, self.file_name + '_E_c.txt')

    sig_f_eps_file = Property(File, depends_on='file_name')

    @cached_property
    def _get_sig_f_eps_file(self):
        return join(self.test_dir, self.file_name + '_sig_f_eps.txt')

    smooth = Bool(True)

    experimental_data = Property(depends_on='+input')

    def _get_experimental_data(self):
        data = np.loadtxt(self.test_file)
        return data.T

    sig_c_eps = Property(depends_on='+input')
    '''Composite stress versus strain.
    '''
    @cached_property
    def _get_sig_c_eps(self):
        xdata, ydata = self.experimental_data

        xdata *= 0.001
        if self.smooth:

            xdata = xdata[:np.argmax(ydata)]
            ydata = ydata[:np.argmax(ydata)]
            eps = smooth_array(xdata, 60, 'flat')
            sig_c = smooth_array(ydata, 60, 'flat')
        else:
            eps = xdata
            sig_c = ydata
        return sig_c, eps

    def get_E_c(self):

        data = np.loadtxt(self.test_file)
        xdata, ydata = data.T
        xdata *= 0.001
        xdata = xdata[:np.argmax(ydata)]
        ydata = ydata[:np.argmax(ydata)]
        del_idx = np.arange(60)
        xdata = np.delete(xdata, del_idx)
        ydata = np.delete(ydata, del_idx)
        xdata = np.append(0.0, xdata)
        ydata = np.append(0.0, ydata)
        E_c = np.average((ydata[2:4] - ydata[0]) / (xdata[2:4] - xdata[0]))
        E_c = 20000.0
        return E_c

    width = Float(0.1, unit='m', CS=True)

    thickness = Float(0.01, unit='m', CS=True)

    A_c = Property
    '''Composite area.
    '''

    def _get_A_c(self):
        return self.thickness * self.width

    a_f = Float(unit='m2', CS=True)
    '''Reinforcement area.
    '''

    A_f = Property(Float)
    '''Reinforcement area.
    '''

    def _get_A_f(self):
        print('self.a_f', self.a_f)
        return self.a_f * self.width

    rho = Property(depends_on='+CS')
    r'''Reinforcement ratio.
    '''
    @cached_property
    def _get_rho(self):
        return self.A_f / self.A_c

    E_cf = Property(depends_on='+input')

    @cached_property
    def _get_E_cf(self):
        sig_c, eps = self.sig_c_eps
        eps_max = eps[-1]
        eps2 = eps[int(len(eps) / 2)]
        sig_c_max = sig_c[-1]
        sig2_c = sig_c[int(len(eps) / 2)]
        return (sig_c_max - sig2_c) / (eps_max - eps2)

    fn_sig_c_eps = Property(depends_on='+input')

    @cached_property
    def _get_fn_sig_c_eps(self):
        sig_c, eps = self.sig_c_eps
        return MFnLineArray(xdata=eps, ydata=sig_c)

    fn_sig_cf_eps = Property(depends_on='+input')

    @cached_property
    def _get_fn_sig_cf_eps(self):
        sig_c, eps = self.sig_c_eps
        eps_f_max = eps[-1]
        sig_cf_max = self.E_cf * eps_f_max
        delta_sig_c = sig_c[-1] - sig_cf_max

        if delta_sig_c < 0:

            sig_cf_max += delta_sig_c

            E_c_secant = np.fabs(sig_c / eps)
            argmin_E_c_secant = np.argmin(E_c_secant)

            sig_c_lambda = sig_c[argmin_E_c_secant]
            eps_lambda = eps[argmin_E_c_secant]

            E_cf0 = sig_c_lambda / eps_lambda

            eps_cf_eta = -delta_sig_c / (self.E_cf - E_cf0)
            sig_cf_eta = E_cf0 * eps_cf_eta

            fn_sig_cf_eps = MFnLineArray(xdata=[0, eps_cf_eta, eps_f_max],
                                         ydata=[0, sig_cf_eta, sig_cf_max])

        else:
            fn_sig_cf_eps = MFnLineArray(xdata=[0, eps_f_max],
                                         ydata=[0, sig_cf_max])

        return fn_sig_cf_eps

    sig_f_eps = Property

    def _get_sig_f_eps(self):
        return self.fn_sig_f_eps.ydata, self.fn_sig_f_eps.xdata

    fn_sig_f_eps = Property(depends_on='+input,+CS')

    @cached_property
    def _get_fn_sig_f_eps(self):
        sig_cf = self.fn_sig_cf_eps.ydata
        eps = self.fn_sig_cf_eps.xdata
        return MFnLineArray(xdata=eps,
                            ydata=sig_cf / self.rho)

    fn_sig_cm_eps = Property(depends_on='+input')

    @cached_property
    def _get_fn_sig_cm_eps(self):
        fn_sig_c_eps = self.fn_sig_c_eps
        eps_c = fn_sig_c_eps.xdata
        sig_c = fn_sig_c_eps.ydata
        fn_sig_cf_eps = self.fn_sig_cf_eps
        sig_cf = fn_sig_cf_eps(eps_c)
        eps_cm = eps_c
        sig_cm = sig_c - sig_cf

        smooth_from = np.argmax(sig_cm)
        s_r_asc = int(smooth_from * 0.1)
        s_r_desc = int((len(sig_cm) - smooth_from) * 0.3)
        eps_cm_asc = smooth_array(eps_cm[:smooth_from], s_r_asc, 'flat')
        sig_cm_asc = smooth_array(sig_cm[:smooth_from], s_r_asc, 'flat')
        eps_cm_desc = smooth_array(eps_cm[smooth_from:], s_r_desc, 'flat')
        sig_cm_desc = smooth_array(sig_cm[smooth_from:], s_r_desc, 'flat')

        eps_cm = np.hstack([eps_cm_asc, eps_cm_desc])
        sig_cm = np.hstack([sig_cm_asc, sig_cm_desc])
        return MFnLineArray(xdata=eps_cm, ydata=sig_cm)

    fn_sig_m_eps = Property(depends_on='+input,+CS')

    @cached_property
    def _get_fn_sig_m_eps(self):
        sig_cm = self.fn_sig_cm_eps.ydata
        eps = self.fn_sig_cm_eps.xdata
        return MFnLineArray(xdata=eps,
                            ydata=sig_cm / (1 - self.rho))

    def _get_mfn_line_array_target(self):
        return self.fn_sig_m_eps

    def _get_test_key(self):
        return 'girder_dresden'

    def show_test_file(self, ax, **plot_kw):

        data = np.loadtxt(self.test_file)
        xdata, ydata = data.T
        xdata *= 0.001

        if self.smooth:
            xdata = xdata[:np.argmax(ydata)]
            ydata = ydata[:np.argmax(ydata)]
            del_idx = np.arange(60)
            xdata = np.delete(xdata, del_idx)
            ydata = np.delete(ydata, del_idx)
            xdata = np.append(0.0, xdata)
            ydata = np.append(0.0, ydata)
            eps = smooth_array(xdata, 60, 'flat')
            sig = smooth_array(ydata, 60, 'flat')
        else:
            eps = xdata
            sig = ydata

        E_c = np.average((ydata[2:4] - ydata[0]) / (xdata[2:4] - xdata[0]))
        E_c = 20000.0
        print('e_mod', E_c)

    #     p.plot([0, 0.001], [0, E_c * 0.001], color='blue')

        p.plot(xdata, ydata)
        self.fn_sig_c_eps.mpl_plot(ax, **plot_kw)
        self.fn_sig_cf_eps.mpl_plot(ax, **plot_kw)
        self.fn_sig_cm_eps.mpl_plot(ax, **plot_kw)

    def calibrate_damage_function(self, ax):

        #----------------------------------------------------------------------
        # Example using the mats2d_explore
        #----------------------------------------------------------------------
        rt = RTraceGraph(name='stress - strain',
                         var_x='eps_app', idx_x=0,
                         var_y='sig_app', idx_y=0,
                         record_on='update')

        ec = {
            # overload the default configuration
            'bcond_list': [BCDofProportional(max_strain=1.0, alpha_rad=0.0,
                                             )],
            'rtrace_list': [
                rt
            ],
        }

        mats_eval = MATS2DMicroplaneDamage(
            n_mp=15,
            elastic_debug=False,
            stress_state='plane_stress',
            symmetrization='sum-type',
            model_version='compliance',
            phi_fn=PhiFnGeneral,
        )

        calib_params = dict(KMAX=300,
                            n_steps=100,
                            phi_max_factor=1.0,
                            tolerance=5e-4,  # 0.01,
                            RESETMAX=0,
                            dim=MATS2DExplore(
                                mats_eval=mats_eval,
                                explorer_config=ec,
                            ),
                            store_fitted_phi_fn=True,
                            log=False)
        self.trait_set(**calib_params)

        nu = 0.20

        self.format_ticks = True
        E_c = self.get_E_c()
        self.dim.mats_eval.E = E_c
        self.dim.mats_eval.nu = nu

        print('n_steps = %g used for calibration' % self.n_steps)
        print('max_eps = %g used for calibration' % self.max_eps)

        #------------------------------------------------------------------
        # set 'param_key' of 'fitter' to store calibration params in the name
        #------------------------------------------------------------------
        #
        age = 28
        param_key = '_age%g_Ec%g_nu%g_nsteps%g_smoothed' % (
            age, E_c, nu, self.n_steps)

        self.param_key = param_key
        print('param_key = %s used in calibration name' % param_key)

        #------------------------------------------------------------------
        # run fitting procedure
        #------------------------------------------------------------------
        #
        self.init()
        self.fit_response()
        self.fitted_phi_fn
        # fitter.fitted_phi_fn.mpl_plot(ax)
        rt.redraw()
        ax.plot(rt.trace.xdata,
                rt.trace.ydata)
        self.mfn_line_array_target.mpl_plot(ax)

    def save(self):
        E_c = self.get_E_c()
        xdata = self.fitted_phi_fn.xdata
        ydata = self.fitted_phi_fn.ydata

        n = len(xdata)

        xdata = smooth_array(xdata, int(0.05 * n), 'flat')
        ydata = smooth_array(ydata, int(0.05 * n), 'flat')

        results = np.c_[xdata, ydata]
        np.savetxt(self.phi_file, results)
        with open(self.E_c_file, 'w') as f:
            f.write(r'''E_c = %g''' % E_c)
        sig_f_eps_data = np.c_[self.fn_sig_f_eps.xdata,
                               self.fn_sig_f_eps.ydata]
        np.savetxt(self.sig_f_eps_file, sig_f_eps_data)

    def verify_sig_m_eps(self, ax, **kw):

        e_phi_data = np.loadtxt(self.phi_file)
        e_data, phi_data = e_phi_data.T
        E_c = self.get_E_c()

        rt = RTDofGraph(name='stress - strain',
                        var_x='eps_app', idx_x=0,
                        var_y='sig_app', idx_y=0,
                        record_on='update')
        ec = {
            # overload the default configuration
            'bcond_list': [BCDofProportional(max_strain=np.max(e_data),
                                             alpha_rad=0.0)],
            'rtrace_list': [
                rt
            ],
        }

        mats_eval = MATS2DMicroplaneDamage(
            n_mp=30,
            E=E_c,
            nu=0.2,
            elastic_debug=False,
            stress_state='plane_stress',
            symmetrization='sum-type',
            model_version='compliance',
            phi_fn=PhiFnGeneral(mfn=MFnLineArray(xdata=e_data,
                                                 ydata=phi_data)),
        )

        me = MATSExplore(KMAX=300,
                         tolerance=5e-4,  # 0.01,
                         RESETMAX=0,
                         dim=MATS2DExplore(
                             mats_eval=mats_eval,
                             explorer_config=ec,
                         ),
                         store_fitted_phi_fn=True,
                         log=False
                         )

        #------------------------------------------------------------------
        # specify the parameters used within the calibration
        #------------------------------------------------------------------
        #

        me.n_steps = 200
        me.tloop.tline.step = 0.01
        me.format_ticks = True

        me.tloop.eval()

        rt.redraw()

        eps_m = rt.trace.xdata
        sig_m = rt.trace.ydata
        ax.plot(eps_m,
                sig_m, **kw)

    def show_damage_function(self, ax):
        results = np.loadtxt(self.phi_file)
        xdata, ydata = results.T
        with open(self.E_c_file, 'r') as f:
            E_c_str = f.read()
        ax.plot(xdata, ydata, label='%s: %s' % (self.file_name, E_c_str))
        ax.set_ylim(ymin=0.0)


class CalibTestsSeries(HasStrictTraits):

    name = Str('<unnamed>')
    file_names = List([], input=True)
    test_params = Dict({}, input=True)

    avg_calibrator = Property(Instance(MATSCalibDamageFnSigEps),
                              depends_on='+input')

    @cached_property
    def _get_avg_calibrator(self):
        eps, sig_c = self.get_average_sig_c_eps()
        test_file = join(
            get_test_data_dir(), self.name + '.txt')
        np.savetxt(test_file, np.c_[1000.0 * eps, sig_c])
        cf = MATSCalibDamageFnSigEps(file_name=self.name,
                                     smooth=False)
        cf.trait_set(**self.test_params)
        return cf

    calibrators = Property(depends_on='file_nemes_items, test_params_items')

    @cached_property
    def _get_calibrators(self):
        cf_list = []
        for file_name in self.file_names:
            cf = MATSCalibDamageFnSigEps(file_name=file_name,
                                         smooth=True)
            cf.trait_set(**self.test_params)
            cf_list.append(cf)
        return cf_list

    def show_test_files(self, ax):
        for cf in self.calibrators:
            cf.show_test_file(ax)
        self.avg_calibrator.show_test_file(ax, lw=4)

    def calibrate_damage_functions(self, ax):
        for cf in self.calibrators:
            cf.calibrate_damage_function(ax)
            cf.save()

    def verify_sig_m_eps(self, ax):
        for cf in self.calibrators:
            cf.verify_sig_m_eps(ax)

    def show_damage_functions(self, ax):
        for cf in self.calibrators:
            cf.show_damage_function(ax)
        p.legend(loc=1)

    def get_average_sig_c_eps(self, n_steps=20000):
        mfn_list = []
        for cf in self.calibrators:
            mfn_list.append(cf.fn_sig_c_eps)

        if len(mfn_list) == 0:
            return None, None
        min_max_e = np.min([mfn.xdata[-1] for mfn in mfn_list])
        e_avg = np.linspace(0, min_max_e, n_steps)
        sig_c_list = []
        for mfn in mfn_list:
            sig_c_list.append(mfn(e_avg))
        sig_c_arr = np.array(sig_c_list)
        sig_c_avg = np.average(sig_c_arr, axis=0)
        return e_avg, sig_c_avg

    def calibrate_average_damage_function(self, ax):
        if len(self.file_names) == 0:
            return
        self.avg_calibrator.calibrate_damage_function(ax)
        self.avg_calibrator.save()

    def verify_average_sig_m_eps(self, ax):
        if len(self.file_names) == 0:
            return
        self.avg_calibrator.verify_sig_m_eps(ax)

    def show_average_damage_function(self, ax):
        if len(self.file_names) == 0:
            return
        self.avg_calibrator.show_damage_function(ax)


calib_series_800 = CalibTestsSeries(
    name='tt-dk-800tex',
    file_names=[
        #        'tt-dk1-800tex',
        'tt-dk2-800tex',
        'tt-dk3-800tex',
        'tt-dk4-800tex'
    ],
    test_params={
        'a_f': 6.160e-5,
        'phi_max_factor': 1.0
    }
)

calib_series_3300 = CalibTestsSeries(
    name='tt-dk-3300tex',
    file_names=[
        # 'tt-dk1-3300tex',
        #    'tt-dk2-3300tex',
        'tt-dk3-3300tex',
        'tt-dk4-3300tex'
    ],
    test_params={
        'a_f': 1.713e-4,
        'phi_max_factor': 1.5
    }
)


def calibrate_damage_fn_for_all_tests():
    p.figure(figsize=(9, 6))
    ax = p.subplot(241)
    calib_series_800.show_test_files(ax)
    ax = p.subplot(242)
    calib_series_800.calibrate_average_damage_function(ax)
    ax = p.subplot(243)
    calib_series_800.show_average_damage_function(ax)
    ax = p.subplot(244)
    calib_series_800.verify_average_sig_m_eps(ax)

    ax = p.subplot(245)
    calib_series_3300.show_test_files(ax)
    ax = p.subplot(246)
    calib_series_3300.calibrate_average_damage_function(ax)
    ax = p.subplot(247)
    calib_series_3300.show_average_damage_function(ax)
    ax = p.subplot(248)
    calib_series_3300.verify_average_sig_m_eps(ax)

    p.tight_layout()
    p.show()


def show_all_tests(x=1, y=2, i=0):
    p.figure(figsize=(9, 6))
    ax = p.subplot(x, y, i + 1)
    calib_series_800.show_test_files(ax)
    ax = p.subplot(x, y, i + 2)
    calib_series_3300.show_test_files(ax)

    p.tight_layout()
    p.show()


def show_sig_eps_f_3300_800():
    a_f_800 = calib_series_800.test_params['a_f']
    a_f_3300 = calib_series_3300.test_params['a_f']
    a_f = a_f_800 + a_f_3300
    print('a_f_800', a_f_800)
    print('a_f_3300', a_f_3300)
    print('a_f', a_f)

    thickness = 0.01
    rho_800 = a_f_800 / thickness
    rho_3300 = a_f_3300 / thickness
    rho_3300_800 = a_f / thickness
    print('rho_3300', rho_3300)
    print('rho_3300_800', rho_3300_800)

    eta_800 = a_f_800 / a_f
    print('eta_800', eta_800)
    sig_f_800, eps_f_800 = calib_series_800.avg_calibrator.sig_f_eps
    sig_f_3300, eps_f_3300 = calib_series_3300.avg_calibrator.sig_f_eps
    E_f_800 = sig_f_800[-1] / eps_f_800[-1]
    print('E_f_800', E_f_800)
    E_f_3300 = ((sig_f_3300[-1] - sig_f_3300[-2]) /
                (eps_f_3300[-1] - eps_f_3300[-2]))
    print('E_f_3300', E_f_3300)
    eps_f_3300_800 = eps_f_3300
    sig_f_3300_800 = np.array([0,
                               (1.0 - eta_800) * sig_f_3300[1] +
                               eta_800 * E_f_800 * eps_f_3300[1],
                               (1.0 - eta_800) * sig_f_3300[2] +
                               eta_800 * E_f_800 * eps_f_3300[2]
                               ])
    print('eps', eps_f_3300_800)
    print('sig', sig_f_3300_800)

    test_file = join(
        get_test_data_dir(), 'tt-dk-3300+800tex_sig_f_eps' + '.txt')
    np.savetxt(test_file, np.c_[eps_f_3300_800, sig_f_3300_800])

    E_f_3300_800 = ((sig_f_3300_800[-1] - sig_f_3300_800[-2]) /
                    (eps_f_3300_800[-1] - eps_f_3300_800[-2]))

    print('E_f', E_f_3300_800)
    print('E_c_3300', rho_3300 * E_f_3300)
    print('E_c_3300_800', rho_3300_800 * E_f_3300_800)

    ax = p.subplot(131)
    ax.plot(eps_f_800, sig_f_800, color='red')
    ax.plot(eps_f_3300, sig_f_3300, color='blue')
    ax.plot(eps_f_3300_800, sig_f_3300_800, label='a_f=%g' % a_f,
            color='green')
    ax.legend()

    ax = p.subplot(132)
    ax.plot(eps_f_800, rho_800 * np.array(sig_f_800),
            color='red')
    ax.plot(eps_f_3300, rho_3300 * np.array(sig_f_3300),
            color='blue')
    ax.plot(eps_f_3300_800, rho_3300_800 * sig_f_3300_800,
            color='green')
    ax.plot(eps_f_3300, rho_3300_800 * np.array(sig_f_3300),
            color='orange')
    ax.legend()

    ax = p.subplot(133)
    fn_sig_m_eps_3300 = calib_series_3300.avg_calibrator.fn_sig_m_eps
    eps_c = np.linspace(0, np.max(eps_f_3300_800))
    sig_m_3300 = fn_sig_m_eps_3300(eps_c)
    fn_f_3300_800 = MFnLineArray(xdata=eps_f_3300_800,
                                 ydata=sig_f_3300_800)
    sig_f_3300_800 = fn_f_3300_800(eps_c)
    sig_c_3300_800 = ((1.0 - rho_3300_800) * sig_m_3300 +
                      rho_3300_800 * sig_f_3300_800)

    p.plot(eps_c, sig_c_3300_800, lw=5, color='green')
    calib_series_800.avg_calibrator.fn_sig_c_eps.mpl_plot(ax, color='red')
    calib_series_3300.avg_calibrator.fn_sig_c_eps.mpl_plot(ax, color='blue')

    p.show()


def correct_sig_eps_f_3300_800():
    a_f_800 = calib_series_800.test_params['a_f']
    a_f_3300 = calib_series_3300.test_params['a_f']
    a_f = a_f_800 + a_f_3300
    print('a_f_800', a_f_800)
    print('a_f_3300', a_f_3300)
    print('a_f', a_f)

    thickness = 0.01
    rho_800 = a_f_800 / thickness
    rho_3300 = a_f_3300 / thickness
    rho_3300_800 = a_f / thickness
    print('rho_3300', rho_3300)
    print('rho_3300_800', rho_3300_800)

    eta_800 = a_f_800 / a_f
    eta_3300 = a_f_3300 / a_f
    print('eta_800', eta_800)
    sig_f_800, eps_f_800 = calib_series_800.avg_calibrator.sig_f_eps
    sig_f_3300, eps_f_3300 = calib_series_3300.avg_calibrator.sig_f_eps
    eps_fbar = np.sort(np.unique(np.hstack([eps_f_800, eps_f_3300])))
    sig_fbar_x800 = np.interp(eps_fbar, eps_f_800, sig_f_800)
    sig_fbar_x3300 = np.interp(eps_fbar, eps_f_3300, sig_f_3300)
    print('eps_f_800', eps_f_800)
    print('sig_f_3300', sig_f_800)
    print('eps_f_800', eps_f_3300)
    print('sig_f_3300', sig_f_3300)
    print('eps_f', eps_fbar)
    print('sig_f_800', sig_fbar_x800)
    print('sig_f_3300', sig_fbar_x3300)

    sig_fbar = eta_800 * sig_fbar_x800 + eta_3300 * sig_fbar_x3300

    E_f_800 = sig_f_800[-1] / eps_f_800[-1]
    print('E_f_800', E_f_800)
    E_f_3300 = ((sig_f_3300[-1] - sig_f_3300[-2]) /
                (eps_f_3300[-1] - eps_f_3300[-2]))
    print('E_f_3300', E_f_3300)
    eps_f_3300_800 = eps_f_3300
    sig_f_3300_800 = np.array([0,
                               eta_3300 * sig_f_3300[1] +
                               eta_800 * E_f_800 * eps_f_3300[1],
                               eta_3300 * sig_f_3300[2] +
                               eta_800 * E_f_800 * eps_f_3300[2]
                               ])
    print('eps', eps_f_3300_800)
    print('sig', sig_f_3300_800)

    test_file = join(
        get_test_data_dir(), 'tt-dk-3300+800tex_sig_f_eps' + '.txt')
    np.savetxt(test_file, np.c_[eps_f_3300_800, sig_f_3300_800])

    E_f_3300_800 = ((sig_f_3300_800[-1] - sig_f_3300_800[-2]) /
                    (eps_f_3300_800[-1] - eps_f_3300_800[-2]))

    print('E_f', E_f_3300_800)
    print('E_c_800', rho_800 * E_f_800)
    print('E_c_3300', rho_3300 * E_f_3300)
    print('E_c_3300_800', rho_3300_800 * E_f_3300_800)

    ax = p.subplot(131)
    ax.plot(eps_f_800, sig_f_800, color='red')
    ax.plot(eps_f_3300, sig_f_3300, color='blue')
    ax.plot(eps_fbar, sig_fbar, label='a_f=%g' % a_f,
            color='orange')
    ax.plot(eps_f_3300_800, sig_f_3300_800, label='a_f=%g' % a_f,
            color='green')
    ax.legend()

    ax = p.subplot(132)
    ax.plot(eps_f_800, rho_800 * np.array(sig_f_800),
            color='red')
    ax.plot(eps_f_3300, rho_3300 * np.array(sig_f_3300),
            color='blue')
    ax.plot(eps_f_3300_800, rho_3300_800 * sig_f_3300_800,
            color='green')
    ax.plot(eps_f_3300, rho_3300_800 * np.array(sig_f_3300),
            color='orange')
    ax.legend()

    ax = p.subplot(133)
    fn_sig_m_eps_3300 = calib_series_3300.avg_calibrator.fn_sig_m_eps
    eps_c = np.linspace(0, np.max(eps_f_3300_800))
    fn_sig_m_eps_800 = calib_series_800.avg_calibrator.fn_sig_m_eps
    sig_m_3300 = fn_sig_m_eps_3300(eps_c)
    sig_m_800 = fn_sig_m_eps_800(eps_c)
    sig_m_3300_800 = (sig_m_3300 + sig_m_800) / 2.0
    fn_f_3300_800 = MFnLineArray(xdata=eps_f_3300_800,
                                 ydata=sig_f_3300_800)
    sig_f_3300_800 = fn_f_3300_800(eps_c)
    sig_c_3300_800 = ((1.0 - rho_3300_800) * sig_m_3300_800 +
                      rho_3300_800 * sig_f_3300_800)

    p.plot(eps_c, sig_c_3300_800, lw=5, color='green')
    ax.plot(eps_f_800, rho_800 * np.array(sig_f_800),
            color='red')
    calib_series_800.avg_calibrator.fn_sig_c_eps.mpl_plot(ax, color='red')
    ax.plot(eps_f_3300, rho_3300 * np.array(sig_f_3300),
            '.', color='blue')
    calib_series_3300.avg_calibrator.fn_sig_c_eps.mpl_plot(ax, color='blue')

    p.show()


if __name__ == '__main__':
    # show_all_tests()
    # calibrate_damage_fn_for_all_tests()
    correct_sig_eps_f_3300_800()
    # show_sig_eps_f_3300_800()
