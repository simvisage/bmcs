'''
Created on 05.12.2016

@author: abaktheer
'''

from os.path import join

from ibvpy.api import MATSEval
from mathkit.mfn.mfn_line.mfn_line import MFnLineArray
from simulator.api import \
    Model, TLoopImplicit, TStepBC
from traits.api import  \
    Constant, Float, Tuple, \
    Instance, Str, Button, Property
from traitsui.api import View, VGroup, Item, UItem
from view.ui import BMCSTreeNode

import numpy as np


class MATSBondSlipMultiLinear(Model, MATSEval, BMCSTreeNode):

    node_name = "multilinear bond law"

    # To use the model directly in the simulator specify the
    # time stepping classes
    tloop_type = TLoopImplicit
    tstep_type = TStepBC

    def __init__(self, *args, **kw):
        super(MATSBondSlipMultiLinear, self).__init__(*args, **kw)
        self.bs_law.replot()

    state_arr_shape = Tuple((0,))

    E_m = Float(28000.0, tooltip='Stiffness of the matrix [MPa]',
                MAT=True, unit='MPa', symbol='E_\mathrm{m}',
                desc='E-modulus of the matrix',
                auto_set=True, enter_set=True)

    E_f = Float(170000.0, tooltip='Stiffness of the fiber [MPa]',
                MAT=True, unit='MPa', symbol='E_\mathrm{f}',
                desc='E-modulus of the reinforcement',
                auto_set=False, enter_set=True)

    s_data = Str('', tooltip='Comma-separated list of strain values',
                 MAT=True, unit='mm', symbol='s',
                 desc='slip values',
                 auto_set=True, enter_set=False)

    tau_data = Str('', tooltip='Comma-separated list of stress values',
                   MAT=True, unit='MPa', symbol=r'\tau',
                   desc='shear stress values',
                   auto_set=True, enter_set=False)

    s_tau_table = Property

    def _set_s_tau_table(self, data):
        s_data, tau_data = data
        if len(s_data) != len(tau_data):
            raise ValueError('s array and tau array must have the same size')
        self.bs_law.set(xdata=s_data,
                        ydata=tau_data)

    update_bs_law = Button(label='update bond-slip law')

    def _update_bs_law_fired(self):
        s_data = np.fromstring(self.s_data, dtype=np.float_, sep=',')
        tau_data = np.fromstring(self.tau_data, dtype=np.float_, sep=',')
        if len(s_data) != len(tau_data):
            raise ValueError('s array and tau array must have the same size')
        self.bs_law.set(xdata=s_data,
                        ydata=tau_data)
        self.bs_law.replot()

    bs_law = Instance(MFnLineArray)

    def _bs_law_default(self):
        return MFnLineArray(
            xdata=[0.0, 1.0],
            ydata=[0.0, 1.0],
            plot_diff=False
        )

    #=========================================================================
    # Configurational parameters
    #=========================================================================
    U_var_shape = (1,)
    '''Shape of the primary variable required by the TStepState.
    '''

    state_var_shapes = {}
    r'''
    Shapes of the state variables
    to be stored in the global array at the level 
    of the domain.
    '''

    node_name = 'multiply_linear bond'

    def get_corr_pred(self, s, t_n1):

        n_e, n_ip, _ = s.shape
        D = np.zeros((n_e, n_ip, 3, 3))
        D[..., 0, 0] = self.E_m
        D[..., 2, 2] = self.E_f

        tau = np.einsum('...st,...t->...s', D, s)
        s = s[..., 1]
        shape = s.shape
        signs = np.sign(s.flatten())
        s_pos = np.fabs(s.flatten())
        tau[..., 1] = (signs * self.bs_law(s_pos)).reshape(*shape)
        D_tau = self.bs_law.diff(s_pos).reshape(*shape)
        D[..., 1, 1] = D_tau

        return tau, D

    def write_figure(self, f, rdir, rel_path):
        fname = 'fig_' + self.node_name.replace(' ', '_') + '.pdf'
        f.write(r'''
\multicolumn{3}{r}{\includegraphics[width=5cm]{%s}}\\
''' % join(rel_path, fname))
        self.bs_law.replot()
        self.bs_law.savefig(join(rdir, fname))

    def plot(self, ax, **kw):
        ax.plot(self.bs_law.xdata, self.bs_law.xdata, **kw)

    tree_view = View(
        VGroup(
            VGroup(
                Item('E_m', full_size=True, resizable=True),
                Item('E_f'),
                Item('s_data'),
                Item('tau_data'),
                UItem('update_bs_law')
            ),
            UItem('bs_law@')
        )
    )


if __name__ == '__main__':
    m = MATSBondSlipMultiLinear()
    #m = MATSBondSlipDP()
    #m = MATSBondSlipFRPDamage()
    # m.configure_traits()
    import matplotlib.pyplot as p
    m.plot(p.axes())
    p.show()
