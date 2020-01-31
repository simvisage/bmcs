
from numpy import array, ones, trapz, frompyfunc, dot

from traits.api import List, Callable
from ibvpy.api import RTDofGraph
from mathkit.mfn import MFnLineArray


class MATS2DMicroplaneDamageTraceGfmac(RTDofGraph):
    '''Evaluate the energy contributions spread over 
    several fields of stress-strain responses. 
    '''

    _trace = List()
    var_time_eval = Callable

    def __init__(self, **kwrds):
        super(MATS2DMicroplaneDamageTraceGfmac, self).__init__(**kwrds)
        self._trace = []

    def bind(self):
        '''
        Locate the evaluators
        '''
        super(MATS2DMicroplaneDamageTraceGfmac, self).bind()
        # get the evaluator for the variable
        self.var_time_eval = self.rmgr.rte_dict['time']

    def add_current_values(self, sctx, eps_eng):
        '''
        Invoke the evaluators in the current context for the specified control vector U_k.
        '''
        super(MATS2DMicroplaneDamageTraceGfmac, self).add_current_values(
            sctx, eps_eng)
        time_value = self.var_time_eval(sctx, eps_eng)
        self._trace.append(time_value[0])

    def redraw(self):
        if self.idx_x < 0 or self.idx_y < 0 or \
                self._xdata == [] or self._ydata == []:
            return

        xarray = array(self._xdata)[:, self.idx_x]
        yarray = array(self._ydata)[:, self.idx_y]
        print('in Gf_mac: ydata', yarray)

        # macroscopic total energy:
        E_t = self._get_E_t(xarray, yarray)
        print('E_t_mac = %.10f' % (E_t))

        # macroscopic elastic energy:
        U_t = self._get_U_t(xarray, yarray)
        print('U_t_mac = %.10f' % (U_t))

        G_f = E_t - U_t
        print('G_f_mac = %.10f' % (G_f))

        self.trace.xdata = array(self._trace, dtype=float)
        self.trace.ydata = G_f * ones(self.trace.xdata.shape, dtype=float)

    def _get_E_t(self, xdata, ydata):
        # integral under the stress strain curve
        E_t = trapz(ydata, xdata)
        return E_t

    def _get_U_t(self, xdata, ydata):
        # area of the stored elastic energy
        U_t = 0.0
        if len(xdata) != 0:
            U_t = 0.5 * ydata[-1] * xdata[-1]
        return U_t

    def clear(self):
        super(MATS2DMicroplaneDamageTraceGfmac, self).clear()
        self._trace = []


class MATS2DMicroplaneDamageTraceEtmac(MATS2DMicroplaneDamageTraceGfmac):
    '''Evaluate the energy contributions spread over 
    several fields of stress-strain responses. 
    '''

    def redraw(self):
        if self.idx_x < 0 or self.idx_y < 0 or \
                self._xdata == [] or self._ydata == []:
            return
        #
        xarray = array(self._xdata)[:, self.idx_x]
        yarray = array(self._ydata)[:, self.idx_y]
        # macroscopic total energy:
        E_t = self._get_E_t(xarray, yarray)
        #
        self.trace.xdata = array(self._trace, dtype=float)
        self.trace.ydata = E_t * ones(self.trace.xdata.shape, dtype=float)


class MATS2DMicroplaneDamageTraceUtmac(MATS2DMicroplaneDamageTraceGfmac):
    '''Evaluate the energy contributions spread over 
    several fields of stress-strain responses. 
    '''

    def redraw(self):
        if self.idx_x < 0 or self.idx_y < 0 or \
                self._xdata == [] or self._ydata == []:
            return
        #
        xarray = array(self._xdata)[:, self.idx_x]
        yarray = array(self._ydata)[:, self.idx_y]
        # macroscopic total energy:
        U_t = self._get_U_t(xarray, yarray)
        #
        self.trace.xdata = array(self._trace, dtype=float)
        self.trace.ydata = U_t * ones(self.trace.xdata.shape, dtype=float)
