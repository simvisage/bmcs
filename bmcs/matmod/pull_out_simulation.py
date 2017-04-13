'''
Created on 12.01.2016
@author: Yingxiong, ABaktheer, RChudoba
'''

from bmcs.matmod.tloop import TLoop
from bmcs.matmod.tstepper import TStepper
from ibvpy.core.bcond_mngr import BCondMngr
from matplotlib.figure import Figure
from scipy.interpolate import interp1d
from traits.api import \
    Property, Instance, cached_property, Str, Button, Enum, \
    Range, on_trait_change, Array, List, Float
from traitsui.api import \
    View, Item, Group, VGroup
from util.traits.editors import MPLFigureEditor

from fets1d52ulrhfatigue import FETS1D52ULRHFatigue
from mats_bondslip import MATSEvalFatigue
import numpy as np
from view.ui import BMCSTreeNode, BMCSLeafNode


class Material(BMCSLeafNode):

    node_name = Str('material parameters')
    E_b = Float(100,
                label="E_b ",
                desc="Bond Stiffness",
                enter_set=True,
                auto_set=False)

    gamma = Float(10,
                  label="Gamma ",
                  desc="Kinematic hardening modulus",
                  enter_set=True,
                  auto_set=False)

    K = Float(10,
              label="K ",
              desc="Isotropic harening",
              enter_set=True,
              auto_set=False)

    S = Float(10,
              label="S ",
              desc="Damage cumulation parameter",
              enter_set=True,
              auto_set=False)

    r = Float(1,
              label="r ",
              desc="Damage cumulation parameter",
              enter_set=True,
              auto_set=False)

    c = Float(1.2,
              label="c ",
              desc="Damage cumulation parameter",
              enter_set=True,
              auto_set=False)

    tau_pi_bar = Float(5,
                       label="Tau_pi_bar ",
                       desc="Reversibility limit",
                       enter_set=True,
                       auto_set=False)

    pressure = Float(-5,
                     label="Pressure",
                     desc="Lateral pressure",
                     enter_set=True,
                     auto_set=False)

    a = Float(1.7,
              label="a",
              desc="Lateral pressure coefficient",
              enter_set=True,
              auto_set=False)

    view = View(VGroup(Group(Item('E_b'),
                             Item('tau_pi_bar'), show_border=True, label='Bond Stiffness and reversibility limit'),
                       Group(Item('gamma'),
                             Item('K'), show_border=True, label='Hardening parameters'),
                       Group(Item('S'),
                             Item('r'), Item('c'), show_border=True, label='Damage cumulation parameters'),
                       Group(Item('pressure'),
                             Item('a'), show_border=True, label='Lateral Pressure')))


class Geometry(BMCSLeafNode):

    node_name = Str('Geometry')
    L_x = Range(1, 700, value=300)
    A_m = Float(100 * 8 - 9 * 1.85, desc='matrix area [mm2]')
    A_f = Float(9 * 1.85, desc='reinforcement area [mm2]')
    P_b = Float(10., desc='perimeter of the bond interface [mm]')


class LoadingScenario(BMCSLeafNode):

    node_name = Str('Loading Scenario')
    number_of_cycles = Float(1.0)
    maximum_loading = Float(0.5)
    unloading_ratio = Range(0., 1., value=0.5)
    number_of_increments = Float(10)
    loading_type = Enum("Monotonic", "Cyclic")
    amplitude_type = Enum("Increased_Amplitude", "Constant_Amplitude")
    loading_range = Enum("Non_symmetric", "Symmetric")

    time = Range(0.00, 1.00, value=1.00)

    d_t = Float(0.005)
    t_max = Float(1.)
    k_max = Float(200)
    tolerance = Float(1e-4)

    d_array = Property(
        depends_on=' maximum_loading , number_of_cycles , loading_type , loading_range , amplitude_type, unloading_ratio')

    @cached_property
    def _get_d_array(self):

        if self.loading_type == "Monotonic":
            self.number_of_cycles = 1
            d_levels = np.linspace(
                0, self.maximum_loading, self.number_of_cycles * 2)
            d_levels[0] = 0
            d_levels.reshape(-1, 2)[:, 0] *= 0
            d_history = d_levels.flatten()
            d_arr = np.hstack([np.linspace(d_history[i], d_history[i + 1], self.number_of_increments)
                               for i in range(len(d_levels) - 1)])

            return d_arr

        if self.amplitude_type == "Increased_Amplitude" and self.loading_range == "Symmetric":
            d_levels = np.linspace(
                0, self.maximum_loading, self.number_of_cycles * 2)
            d_levels.reshape(-1, 2)[:, 0] *= -1
            d_history = d_levels.flatten()
            d_arr = np.hstack([np.linspace(d_history[i], d_history[i + 1], self.number_of_increments)
                               for i in range(len(d_levels) - 1)])

            return d_arr

        if self.amplitude_type == "Increased_Amplitude" and self.loading_range == "Non_symmetric":
            d_levels = np.linspace(
                0, self.maximum_loading, self.number_of_cycles * 2)
            d_levels.reshape(-1, 2)[:, 0] *= 0
            d_history = d_levels.flatten()
            d_arr = np.hstack([np.linspace(d_history[i], d_history[i + 1], self.number_of_increments)
                               for i in range(len(d_levels) - 1)])

            return d_arr

        if self.amplitude_type == "Constant_Amplitude" and self.loading_range == "Symmetric":
            d_levels = np.linspace(
                0, self.maximum_loading, self.number_of_cycles * 2)
            d_levels.reshape(-1, 2)[:, 0] = -self.maximum_loading
            d_levels[0] = 0
            d_levels.reshape(-1, 2)[:, 1] = self.maximum_loading
            d_history = d_levels.flatten()
            d_arr = np.hstack([np.linspace(d_history[i], d_history[i + 1], self.number_of_increments)
                               for i in range(len(d_levels) - 1)])

            return d_arr

        if self.amplitude_type == "Constant_Amplitude" and self.loading_range == "Non_symmetric":
            # d_1 = np.zeros(self.number_of_cycles*2 + 1)
            d_1 = np.zeros(1)
            d_2 = np.linspace(
                0, self.maximum_loading, self.number_of_cycles * 2)
            d_2.reshape(-1, 2)[:, 0] = self.maximum_loading
            d_2.reshape(-1, 2)[:, 1] = self.maximum_loading * \
                self.unloading_ratio
            d_history = d_2.flatten()
            d_arr = np.hstack((d_1, d_history))

            return d_arr

    time_func = Property(depends_on='maximum_loading, t_max , d_array')

    @cached_property
    def _get_time_func(self):
        t_arr = np.linspace(0, self.t_max, len(self.d_array))
        return interp1d(t_arr, self.d_array)

    figure = Instance(Figure)

    def _figure_default(self):
        figure = Figure()
        return figure

    update = Button()

    def _update_fired(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        x = np.arange(0, self.t_max, self.d_t)
        ax.plot(x, self.time_func(x))
        ax.set_xlabel('time')
        ax.set_ylabel('displacement')
        self.figure.canvas.draw()

    view = View(VGroup(Group(Item('loading_type')),
                       Group(Item('maximum_loading')),
                       Group(Item('number_of_cycles'),
                             Item('amplitude_type'),
                             Item('loading_range'), Item('unloading_ratio'), show_border=True, label='Cyclic load inputs'),
                       Group(Item('d_t'),
                             Item('t_max'),
                             Item('k_max'), show_border=True, label='Solver Settings')),
                Group(Item('update', label='Plot Loading scenario')),
                Item('figure', editor=MPLFigureEditor(),
                     dock='horizontal',
                     show_label=False),
                Item('time', label='t/T_max')
                )


class PullOutSimulation(BMCSTreeNode):

    node_name = Str('pull out simulation')

    tree_node_list = List([])

    def _tree_node_list_default(self):

        return [self.material, self.geometry, self.loading_scenario, self.bcond_mngr]

    material = Instance(Material)

    def _material_default(self):
        return Material()

    loading_scenario = Instance(LoadingScenario)

    def _loading_scenario_default(self):
        return LoadingScenario()

    geometry = Instance(Geometry)

    def _geometry_default(self):
        return Geometry()

    mats_eval = Property(Instance(MATSEvalFatigue))
    '''Material model'''
    @cached_property
    def _get_mats_eval(self):
        return MATSEvalFatigue()

    fets_eval = Property(Instance(FETS1D52ULRHFatigue))
    '''Finite element time stepper implementing the corrector
    predictor operators at the element level'''
    @cached_property
    def _get_fets_eval(self):
        return FETS1D52ULRHFatigue()

    time_stepper = Property(Instance(TStepper))
    '''Objects representing the state of the model providing
    the predictor and corrector functionality needed for time-stepping
    algorithm.
    '''
    @cached_property
    def _get_time_stepper(self):
        return TStepper()

    time_loop = Property(Instance(TLoop))
    '''Algorithm controlling the time stepping.
    '''
    @cached_property
    def _get_time_loop(self):
        return TLoop(ts=self.time_stepper)

    bcond_mngr = Property(Instance(BCondMngr))

    @cached_property
    def _get_bcond_mngr(self):
        return self.time_stepper.bcond_mngr

    t_record = Array
    U_record = Array
    F_record = Array
    sf_record = Array
    eps_record = List
    sig_record = List
    D_record = List
    w_record = List

    figure = Instance(Figure)

    def _figure_default(self):
        figure = Figure()
        return figure

    # plot = Button()

    def plot(self, figure, color='blue', linestyle='-',
             linewidth=1, label='<unnamed>'):
        # assign the material parameters
        self.mats_eval.E_b = self.material.E_b
        self.mats_eval.gamma = self.material.gamma
        self.mats_eval.S = self.material.S
        self.mats_eval.tau_pi_bar = self.material.tau_pi_bar
        self.mats_eval.r = self.material.r
        self.mats_eval.K = self.material.K
        self.mats_eval.c = self.material.c
        self.mats_eval.a = self.material.a
        self.mats_eval.pressure = self.material.pressure

        # assign the geometry parameters
        self.fets_eval.A_m = self.geometry.A_m
        self.fets_eval.P_b = self.geometry.P_b
        self.fets_eval.A_f = self.geometry.A_f
        self.time_stepper.L_x = self.geometry.L_x

        # assign the parameters for solver and loading_scenario
        self.time_loop.t_max = self.loading_scenario.t_max
        self.time_loop.d_t = self.loading_scenario.d_t
        self.time_loop.k_max = self.loading_scenario.k_max
        self.time_loop.tolerance = self.loading_scenario.tolerance

        # assign the bc
        self.time_stepper.bcond_list[1].value = 1
        self.time_stepper.bcond_list[
            1].time_function = self.loading_scenario.time_func

        # self.time = 1.00

        s_arr = self.loading_scenario._get_d_array()
        tau_arr, w_arr, xs_pi_arr, xs_pi_cum = self.mats_eval.get_bond_slip(
            s_arr)

        ax1 = figure.add_subplot(231)
        ax1.cla()
        ax1.plot(s_arr, tau_arr)
        ax1.set_title('Bond_slip curve')
        ax1.set_xlabel('Slip')
        ax1.set_ylabel('Stress')

        self.U_record, self.F_record, self.sf_record, self.t_record, self.eps_record, \
            self.sig_record, self.w_record, self.D_record = self.time_loop.eval()
        n_dof = 2 * self.time_stepper.domain.n_active_elems + 1

        ax2 = figure.add_subplot(232)
        ax2.cla()
        ax2.plot(self.U_record[:, n_dof], self.F_record[:, n_dof] / 1000, lw=linewidth, color=color,
                 ls=linestyle, label=label)
        ax2.plot(
            self.U_record[-1, n_dof], self.F_record[-1, n_dof] / 1000, 'ro')
        ax2.set_title('pull-out force-displacement curve')

        ax3 = figure.add_subplot(233)
        ax3.cla()
        X = np.linspace(0, self.time_stepper.L_x, self.time_stepper.n_e_x + 1)
        X_ip = np.repeat(X, 2)[1:-1]
        ax3.plot(X_ip, self.w_record[-1].flatten())
        ax3.set_ylim(0, 1)
        ax3.set_title('Damage distribution')

        ax4 = figure.add_subplot(234)
        ax4.cla()
        X = np.linspace(0, self.time_stepper.L_x, self.time_stepper.n_e_x + 1)
        X_ip = np.repeat(X, 2)[1:-1]
        ax4.plot(X_ip, self.sf_record[-1, :])
        ax4.set_ylim(np.amin(self.sf_record), np.amax(self.sf_record))
        ax4.set_title('shear flow in the bond interface')

        ax5 = figure.add_subplot(235)
        ax5.cla()
        ax5.plot(X_ip, self.eps_record[-1][:, :, 0].flatten())
        ax5.plot(X_ip, self.eps_record[-1][:, :, 2].flatten())
        ax5.set_title('strain')

        ax6 = figure.add_subplot(236)
        ax6.cla()
        ax6.plot(X_ip, self.sig_record[-1][:, :, 0].flatten())
        ax6.plot(X_ip, self.sig_record[-1][:, :, 2].flatten())
        ax6.set_title('stress')

        # ax6 = figure.add_subplot(236)
        # ax6.cla()
        # ax6.plot(self.U_record[-1, n_dof], self.D_record[:][-1, :, 1 , 1].flatten())
        # ax6.plot(X_ip, self.D_record[-1][:, :, 1,1].flatten())
        # ax6.set_title('stiffness')
        # ax6.set_ylim(np.amin(self.sig_record), np.amax(self.sig_record))

    def _time(self):
        return self.loading_scenario.time

    # self.time = self.loading_scenario.time

    @on_trait_change('time')
    def plot_t(self, figure):

        self.time = self.loading_scenario.time
        # assign the material parameters
        self.mats_eval.E_b = self.material.E_b
        self.mats_eval.gamma = self.material.gamma
        self.mats_eval.S = self.material.S
        self.mats_eval.tau_pi_bar = self.material.tau_pi_bar
        self.mats_eval.r = self.material.r
        self.mats_eval.K = self.material.K
        self.mats_eval.c = self.material.c
        self.mats_eval.a = self.material.a
        self.mats_eval.pressure = self.material.pressure

        # assign the geometry parameters
        self.fets_eval.A_m = self.geometry.A_m
        self.fets_eval.P_b = self.geometry.P_b
        self.fets_eval.A_f = self.geometry.A_f
        self.time_stepper.L_x = self.geometry.L_x

        # assign the parameters for solver and loading_scenario
        self.time_loop.t_max = self.loading_scenario.t_max
        self.time_loop.d_t = self.loading_scenario.d_t
        self.time_loop.k_max = self.loading_scenario.k_max
        self.time_loop.tolerance = self.loading_scenario.tolerance

        # assign the bc
        self.time_stepper.bcond_list[1].value = 1
        self.time_stepper.bcond_list[
            1].time_function = self.loading_scenario.time_func

        # self.time = 1.00

        s_arr = self.loading_scenario._get_d_array()
        tau_arr, w_arr, xs_pi_arr, xs_pi_cum = self.mats_eval.get_bond_slip(
            s_arr)

        ax1 = figure.add_subplot(231)
        ax1.cla()
        ax1.plot(s_arr, tau_arr)
        ax1.set_title('Bond_slip curve')
        ax1.set_xlabel('Slip')
        ax1.set_ylabel('Stress')

        #self.U_record, self.F_record, self.sf_record, self.t_record, \
        #self.eps_record, self.sig_record, self.w_record = self.time_loop.eval()
        # n_dof = 2 * self.time_stepper.domain.n_active_elems + 1

        idx = (np.abs(self.time * max(self.t_record) - self.t_record)).argmin()
        n_dof = 2 * self.time_stepper.domain.n_active_elems + 1

        ax2 = figure.add_subplot(232)
        ax2.cla()
        ax2.plot(self.U_record[:, n_dof], self.F_record[:, n_dof])
        ax2.plot(self.U_record[idx, n_dof], self.F_record[idx, n_dof], 'ro')
        ax2.set_title('pull-out force-displacement curve')

        ax3 = figure.add_subplot(233)
        ax3.cla()
        X = np.linspace(0, self.time_stepper.L_x, self.time_stepper.n_e_x + 1)
        X_ip = np.repeat(X, 2)[1:-1]
        ax3.plot(X_ip, self.w_record[idx].flatten())
        ax3.set_ylim(0, 1)
        ax3.set_title('Damage')

        ax4 = figure.add_subplot(234)
        ax4.cla()
        X = np.linspace(0, self.time_stepper.L_x, self.time_stepper.n_e_x + 1)
        X_ip = np.repeat(X, 2)[1:-1]
        ax4.plot(X_ip, self.sf_record[idx, :])
        ax4.set_ylim(np.amin(self.sf_record), np.amax(self.sf_record))
        ax4.set_title('shear flow in the bond interface')

        ax5 = figure.add_subplot(235)
        ax5.cla()
        ax5.plot(X_ip, self.eps_record[idx][:, :, 0].flatten())
        ax5.plot(X_ip, self.eps_record[idx][:, :, 2].flatten())
        ax5.set_title('strain')

        ax6 = figure.add_subplot(236)
        ax6.cla()
        ax6.plot(X_ip, self.sig_record[idx][:, :, 0].flatten())
        ax6.plot(X_ip, self.sig_record[idx][:, :, 2].flatten())
        ax6.set_ylim(np.amin(self.sig_record), np.amax(self.sig_record))
        ax6.set_title('stress')

        # figure.canvas.draw()

    def plot_custom(self, ax1, ax2, ax3, color='blue', linestyle='-',
                    linewidth=1, label='<unnamed>'):
        # assign the material parameters
        self.mats_eval.E_b = self.material.E_b
        self.mats_eval.gamma = self.material.gamma
        self.mats_eval.S = self.material.S
        self.mats_eval.tau_pi_bar = self.material.tau_pi_bar
        self.mats_eval.r = self.material.r
        self.mats_eval.K = self.material.K
        self.mats_eval.c = self.material.c
        self.mats_eval.a = self.material.a
        self.mats_eval.pressure = self.material.pressure

        # assign the geometry parameters
        self.fets_eval.A_m = self.geometry.A_m
        self.fets_eval.P_b = self.geometry.P_b
        self.fets_eval.A_f = self.geometry.A_f
        self.time_stepper.L_x = self.geometry.L_x

        # assign the parameters for solver and loading_scenario
        self.time_loop.t_max = self.loading_scenario.t_max
        self.time_loop.d_t = self.loading_scenario.d_t
        self.time_loop.k_max = self.loading_scenario.k_max
        self.time_loop.tolerance = self.loading_scenario.tolerance

        # assign the bc
        self.time_stepper.bcond_list[1].value = 1
        self.time_stepper.bcond_list[
            1].time_function = self.loading_scenario.time_func

        self.U_record, self.F_record, self.sf_record, self.t_record, self.eps_record, \
            self.sig_record, self.w_record, self.D_record = self.time_loop.eval()
        n_dof = 2 * self.time_stepper.domain.n_active_elems + 1

        if self.loading_scenario.loading_type == "Monotonic" or "Cyclic":
            ax1.plot(self.U_record[:, n_dof], self.F_record[:, n_dof] / 1000, lw=linewidth, color=color,
                     ls=linestyle, label=label)
            # ax1.plot(self.U_record[-1, n_dof], self.F_record[-1, n_dof], 'ro')
            ax1.set_title('pull-out force-displacement curve')
            ax1.set_xlabel('Slip(mm)')
            ax1.set_ylabel('Force (KN)')
            ax1.legend(loc=4)

        X = np.linspace(0, self.time_stepper.L_x, self.time_stepper.n_e_x + 1)
        X_ip = np.repeat(X, 2)[1:-1]
        ax2.plot(X_ip, self.w_record[-1].flatten(), lw=linewidth, color=color,
                 ls=linestyle, label=label)
        ax2.set_ylim(0, 1)
        ax2.set_title('Damage distribution')
        ax2.set_xlabel('Bond Interface length')
        ax2.set_ylabel('Damage')
        ax2.legend(loc=4)

        # plotting the max slip for each cycle (S_N curve)
        n = (len(self.loading_scenario.d_array) - 1) / 2
        u_max_1 = np.zeros(1)
        u_max_2 = np.zeros(1)
        u_min = np.zeros(1)
        #E_ed = np.zeros(n)
        t = np.zeros(1)

        for i in range(0, n, 1):
            idx = (2 * i + 1) * (self.loading_scenario.t_max) / \
                (2 * n * self.loading_scenario.d_t)
            if idx >= len(self.t_record):
                break
            else:
                # max slip of the loaded end
                u_max_1 = np.vstack((u_max_1, self.U_record[idx, n_dof]))
                # max slip of the unloaded end
                u_max_2 = np.vstack((u_max_2, self.U_record[idx, n_dof]))
                t = np.vstack((t, self.t_record[idx]))

        for i in range(1, n + 1, 1):
            idx = (2 * i) * (self.loading_scenario.t_max) / \
                (2 * n * self.loading_scenario.d_t)
            if idx >= len(self.t_record):
                break
            else:
                u_min = np.vstack((u_min, self.U_record[idx, n_dof]))
                #t = np.vstack((t,self.t_record[idx]))

        if self.loading_scenario.loading_type == "Cyclic":
            ax3.plot(t[1:-1] * (self.loading_scenario.number_of_cycles / self.loading_scenario.t_max), u_max_1[1:-1],
                     lw=linewidth, color=color, ls=linestyle, label=label)
            ax3.plot(t[1:-1] * (self.loading_scenario.number_of_cycles / self.loading_scenario.t_max), u_max_2[1:-1],
                     lw=linewidth, color=color, ls=linestyle, label=label)

            #ax3.set_xlim(0, 1)
            ax3.set_title('Max slip vs. number of cycles')
            ax3.set_xlabel('N')
            ax3.set_ylabel('Max Slip')
            ax3.legend(loc=4)

        # plotting the stiffness vs. number of cycles
        '''           
        if   self.loading_scenario.loading_type == "Cyclic":
            ax3.plot(t[1:-1] *(self.loading_scenario.number_of_cycles / self.loading_scenario.t_max),
                      (self.loading_scenario.maximum_loading - self.loading_scenario.maximum_loading * 
                       self.loading_scenario.unloading_ratio) / 
                                (u_max[1:-1]), 
                                lw=linewidth, color=color,
                                ls=linestyle, label=label)
            ax3.set_xlim(0, 1)
            ax3.set_title('Stiffness vs.  number of cycles')
            ax3.set_xlabel('N'); ax3.set_ylabel('Stiffness')
            ax3.legend(loc=4)
        '''

        # Extra plots - plot only specific cycles

        '''
        if   self.loading_scenario.loading_type == "Cyclic":
            
            r = 3
            m = (2 * r + 1) * ((self.loading_scenario.t_max) / (2 * n * self.loading_scenario.d_t)) + r
            U_sp = np.zeros(m)
            F_sp = np.zeros(m)
        
            idx_0 = (self.loading_scenario.t_max) / (2 * n * self.loading_scenario.d_t) + 1
            U_sp[:idx_0 ] = self.U_record[:idx_0  , n_dof]
            F_sp[:idx_0 ] = self.F_record[:idx_0  , n_dof]
        
            idx_1 = (2 * 1 + 1 )*(self.loading_scenario.t_max) / (2 * n * self.loading_scenario.d_t) +1
            U_sp[idx_0  :idx_1 ] = self.U_record[idx_0  :idx_1  , n_dof]
            F_sp[idx_0  :idx_1 ] = self.F_record[idx_0  :idx_1  , n_dof]
        
            idx_2 = (2 * 200 + 1 )*(self.loading_scenario.t_max) / (2 * n * self.loading_scenario.d_t) 
            idx_22 = (2 * 201 + 1 )*(self.loading_scenario.t_max) / (2 * n * self.loading_scenario.d_t)  +1
            U_sp[idx_1  :idx_1 + 2*idx_0 -1] = self.U_record[idx_2  :idx_22 , n_dof]
            F_sp[idx_1  :idx_1 + 2*idx_0 -1] = self.F_record[idx_2  :idx_22 , n_dof]
            
            idx_3 = (2 * 598 + 1 )*(self.loading_scenario.t_max) / (2 * n * self.loading_scenario.d_t)
            idx_33 = (2 * 599 + 1 )*(self.loading_scenario.t_max) / (2 * n * self.loading_scenario.d_t)  +1
            
            
            U_sp[idx_1 + 2*idx_0 -1  : idx_1 + 4*idx_0 -2 ] = self.U_record[idx_3  : idx_33 , n_dof]
            F_sp[idx_1 + 2*idx_0 -1 : idx_1 + 4*idx_0 -2 ] = self.F_record[idx_3 : idx_33  , n_dof]
                      
            ax1.plot(U_sp , F_sp /1000 , lw=linewidth, color=color,ls=linestyle, label=label)
            ax1.set_title('pull-out force-displacement curve')
            ax1.set_xlabel('Slip(mm)'); ax1.set_ylabel('Force (KN)')
            ax1.legend(loc=4)       
        '''

    trait_view = View(Item('fets_eval'),
                      )

if __name__ == '__main__':
    print 'xxxx#'
    ps1 = PullOutSimulation()
    ps1.configure_traits()
