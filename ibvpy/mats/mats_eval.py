
from numpy import zeros, linalg, tensordot, dot, min, max, argmax, array, pi, \
    append
from scipy.linalg import eigh
from traits.api import \
    Array, Bool, Callable, Enum, Float, HasStrictTraits, Interface, implements, \
    Instance, Int, Trait, Str, Enum, Callable, List, TraitDict, Any, \
    on_trait_change, Tuple, WeakRef, Delegate, Property, cached_property, \
    Dict, Constant

from ibvpy.core.i_tstepper_eval import \
    ITStepperEval
from ibvpy.core.tstepper_eval import \
    TStepperEval


#-------------------------------------------------------------------
# IMATSEval - interface for fe-numerical quadrature
#-------------------------------------------------------------------
class IMATSEval(ITStepperEval):

    def get_state_array_size(self):
        pass

    def setup(self, sctx):
        pass

    explorer_rtrace_list = Property

    explorer_config = Property

#-------------------------------------------------------------------
# MATSEval - general implementation of the fe-numerical quadrature
#-------------------------------------------------------------------


class MATSEval(HasStrictTraits, TStepperEval):

    implements(IMATSEval)

    # Callable specifying spatial profile of an initial strain field
    # the parameter is X - global coordinates of the material point
    #
    initial_strain = Callable

    # Callable specifying spatial profile of an initial stress field
    # the parameter is X - global coordinates of the material point
    #
    initial_stress = Callable

    # number of spatial dimensions of an integration cell for the material model
    #
    n_dims = Constant(Float)

    id_number = Int

    #-------------------------------------------------------------------------
    # Dimensionally dependent mappings between tensors
    #-------------------------------------------------------------------------
    # These are the handbook methods to be specialized in subclasses.

    # Mappings between tensorial and engineering variables
    #
    map_eps_eng_to_mtx = Callable(transient=True)
    map_sig_eng_to_mtx = Callable(transient=True)
    map_eps_mtx_to_eng = Callable(transient=True)
    map_sig_mtx_to_eng = Callable(transient=True)
    compliance_mapping = Callable(transient=True)
    map_tns4_to_tns2 = Callable(transient=True)

    def get_state_array_size(self):
        return 0

    def setup(self, sctx):
        pass

    explorer_rtrace_list = Property

    @cached_property
    def _get_explorer_rtrace_list(self):
        return []

    #-------------------------------------------------------------------------
    # Response trace evaluators
    #-------------------------------------------------------------------------
    def get_msig_pos(self, sctx, eps_app_eng, *args, **kw):
        '''
        get biggest positive principle stress
        @param sctx:
        @param eps_app_eng:
        '''
        sig_eng, D_mtx = self.get_corr_pred(sctx, eps_app_eng, 0, 0, 0)
        ms_vct = zeros(3)
        shape = sig_eng.shape[0]
        if shape == 3:
            s_mtx = self.map_sig_eng_to_mtx(sig_eng)
            m_sig, m_vct = linalg.eigh(s_mtx)

            # @todo: - this must be written in a more readable way
            #
            if m_sig[-1] > 0:
                # multiply biggest positive stress with its vector
                ms_vct[:2] = m_sig[-1] * m_vct[-1]
        elif shape == 6:
            s_mtx = self.map_sig_eng_to_mtx(sig_eng)
            m_sig = linalg.eigh(s_mtx)
            if m_sig[0][-1] > 0:
                # multiply biggest positive stress with its vector
                ms_vct = m_sig[0][-1] * m_sig[1][-1]
        return ms_vct

    def get_msig_pm(self, sctx, eps_app_eng, *args, **kw):
        sig_eng, D_mtx = self.get_corr_pred(sctx, eps_app_eng, 0, 0, 0)
        t_field = zeros(9)
        shape = sig_eng.shape[0]
        if shape == 3:
            s_mtx = self.map_sig_eng_to_mtx(sig_eng)
            m_sig = linalg.eigh(s_mtx)
            if m_sig[0][-1] > 0:
                t_field[0] = m_sig[0][-1]  # biggest positive stress
        elif shape == 6:
            s_mtx = self.map_sig_eng_to_mtx(sig_eng)
            m_sig = linalg.eigh(s_mtx)
            if m_sig[0][-1] > 0:
                t_field[0] = m_sig[0][-1]
        return t_field

    def get_max_principle_sig(self, sctx, eps_app_eng, *args, **kw):
        '''
        get biggest positive principle stress
        @param sctx:
        @param eps_app_eng:
        '''
        sig_eng, D_mtx = self.get_corr_pred(sctx, eps_app_eng, 0, 0, 0)
        s_mtx = self.map_sig_eng_to_mtx(sig_eng)
        m_sig, m_vct = linalg.eigh(s_mtx)
        max_principle_sig = max(m_sig[:])
        # return max_principle_sig
        return array([max_principle_sig], dtype='float')

    def get_sig_app(self, sctx, eps_app_eng, *args, **kw):
        sig_eng, D_mtx = self.get_corr_pred(
            sctx, eps_app_eng, 0, 0, 0, *args, **kw)
        s_tensor = zeros((3, 3))
        s_tensor[:self.n_dims, :self.n_dims] = self.map_sig_eng_to_mtx(sig_eng)
        return s_tensor

    def get_eps_app(self, sctx, eps_app_eng, *args, **kw):
        e_tensor = zeros((3, 3))
        e_tensor[:self.n_dims, :self.n_dims] = self.map_eps_eng_to_mtx(
            eps_app_eng)
        return e_tensor

    # This is only relevant for strain softening models
    #
    def get_regularizing_length(self, sctx, eps_app_eng, *args, **kw):

        X_mtx = sctx.X_reg

        # first principle strain unit vector
        #
        eigval, eigvec = eigh(self.map_eps_eng_to_mtx(eps_app_eng))

        # Get the eigenvector associated with maximum aigenvalue
        # it is located in the last column of the matrix
        # of eigenvectors eigvec
        #
        eps_one = eigvec[:, -1]
        print X_mtx
        # Project the coordinate vectors into the determined direction
        #
        proj = dot(X_mtx, eps_one)

        # Find the maximum distance between the projected coordinates
        #
        h = max(proj) - min(proj)
        return h

    def get_strain_energy(self, sctx, eps_app_eng, *args, **kw):
        sig_app = self.get_sig_app(sctx, eps_app_eng)
        eps_app = self.get_eps_app(sctx, eps_app_eng)
        energy = tensordot(sig_app, eps_app) * 0.5
        return energy

    # Declare and fill-in the explorer config
    # Each material model can define the default configuration
    # to present itself in the explorer.
    #
    explorer_config = Property(Dict)

    @cached_property
    def _get_explorer_config(self):
        from ibvpy.api import BCDof, TLine, RTDofGraph
        return {'bcond_list': [BCDof(var='u',
                                     dof=0, value=0.01,
                                     time_function=lambda t: t)],
                'rtrace_list': [RTDofGraph(name='strain - stress',
                                            var_x='eps_app', idx_x=0,
                                            var_y='sig_app', idx_y=0,
                                            record_on='update'),
                                RTDofGraph(name='strain - strain',
                                            var_x='eps_app', idx_x=0,
                                            var_y='eps_app', idx_y=1,
                                            record_on='update'),
                                RTDofGraph(name='stress - stress',
                                            var_x='sig_app', idx_x=0,
                                            var_y='sig_app', idx_y=1,
                                            record_on='update'),
                                RTDofGraph(name='Stress - Strain',
                                            var_x='F_int', idx_x=0,
                                            var_y='U_k', idx_y=0,
                                            record_on='update'),
                                RTDofGraph(name='Strain - Strain',
                                            var_x='U_k', idx_x=0,
                                            var_y='U_k', idx_y=1,
                                            record_on='update'),
                                RTDofGraph(name='Stress - Stress',
                                            var_x='F_int', idx_x=0,
                                            var_y='F_int', idx_y=1,
                                            record_on='update'),
                                RTDofGraph(name='sig1 - eps1',
                                            var_x='F_int', idx_x=0,
                                            var_y='U_k', idx_y=0,
                                            record_on='update'),
                                RTDofGraph(name='sig2 - sig3',
                                            var_x='F_int', idx_x=1,
                                            var_y='F_int', idx_y=2,
                                            record_on='update'),
                                RTDofGraph(name='eps2 - eps3',
                                            var_x='U_k', idx_x=1,
                                            var_y='U_k', idx_y=2,
                                            record_on='update')
                                ],
                'tline': TLine(step=0.1, max=1.0)
                }

        def _set_explorer_config(self, value):
            self._explorer_config = value
