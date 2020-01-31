
from traits.api import List
from ibvpy.api import RTDofGraph, RTraceDomainListField
from numpy import array, ones, trapz, frompyfunc, dot, zeros, fabs

from mathkit.mfn import MFnLineArray


class MATS2DMicroplaneDamageDomainTraceGfmic( RTraceDomainListField ):
    '''Evaluate the energy contributions spread over 
    several fields of stress-strain responses. 
    '''
#    E_t = zeros((1,))
#    U_t = zeros((1,))
#    G_f = zeros((1,))

    def __init__( self, **kwrds ):
        super( MATS2DMicroplaneDamageDomainTraceGfmic, self ).__init__( **kwrds )
        self._trace_time = []
#        self._trace_Gf   = []

    def bind( self ):
        '''
        Locate the evaluators
        '''
        super( MATS2DMicroplaneDamageDomainTraceGfmic, self ).bind()
        # get the evaluator for the variable
        self.var_time_eval = self.rmgr.rte_dict['time']


    def get_MPW( self ):
        '''
        Get the microplanes weighting coefficients used for numerical integration
        '''
        # this is special for the microplane model
        return self.var_x_eval.ts._MPW


    def add_current_values( self, sctx, eps_eng ):
        '''
        Invoke the evaluators in the current context for the specified control vector U_k.
        '''
        super( MATS2DMicroplaneDamageDomainTraceGfmic, self ).add_current_values( sctx, eps_eng )
        time_value = self.var_time_eval( sctx, eps_eng )
#        Gf_value   = self.G_f
        self._trace_time.append( time_value[0] )
#        self._trace_Gf.append(Gf_value[0])    

    def redraw( self ):

        if self.idx_x < 0 or self.idx_y < 0 or \
            self._xdata == [] or self._ydata == []:
            return

        xarray = array( self._xdata )
        yarray = array( self._ydata )

        # get the microplane weighting coefficients:
        MPW = self.get_MPW()

        # derive the number of specified microplanes base on the shape of the trace 
        n_mp = xarray.shape[1]

        # @todo: find an alternativ (faster) numpy way to calculate the value for G_f 
        #trapz.reduce( ( yarray[0,:], xarray[0,:] )


        # total energy contribution of the microplanes:
        E_t_arr = array( [ self._get_E_t( xarray[:, i_mp], yarray[:, i_mp] ) for i_mp in range( n_mp )] )

        # elastic energy contribution of the microplanes:
        U_t_arr = array( [ self._get_U_t( xarray[:, i_mp], yarray[:, i_mp] ) for i_mp in range( n_mp )] )

        G_f_arr = E_t_arr - U_t_arr

        # total "total energy" = weighted sum over the microplanes:
        E_t = array( [ dot( E_t_arr, MPW ) ] )
        print('E_t_mic = %.10f' % ( E_t ))

        # total "elastic energy" = weighted sum over the microplanes:
        U_t = array( [ dot( U_t_arr, MPW ) ] )
        print('U_t_mic = %.10f' % ( U_t ))

        # total "fracture energy" = weighted sum over the microplanes:
        G_f = array( [ dot( G_f_arr, MPW ) ] )
        print('G_f_mic = %.10f' % ( G_f ))
        print('\n')

        self.trace.xdata = array( self._trace_time, dtype = float )
#        self.trace.ydata = array( self._trace_Gf, dtype = float )
        self.trace.ydata = G_f * ones( self.trace.xdata.shape, dtype = float )


    def _get_E_t( self, xdata, ydata ):
        # integral under the stress strain curve
        E_t = trapz( ydata, xdata )
        return E_t


    def _get_U_t( self, xdata, ydata ):
        # area of the stored elastic energy  
        U_t = 0.0
        if len( xdata ) != 0:
            U_t = 0.5 * ydata[-1] * xdata[-1]
        return U_t


    def clear( self ):
        super( MATS2DMicroplaneDamageTraceGfmic, self ).clear()
        self._trace_time = []



class MATS2DMicroplaneDamageDomainTraceEtmic( MATS2DMicroplaneDamageDomainTraceGfmic ):
    '''Evaluate the energy contributions spread over 
    several fields of stress-strain responses. 
    '''
    def redraw( self ):
        if self.idx_x < 0 or self.idx_y < 0 or \
            self._xdata == [] or self._ydata == []:
            return
        xarray = array( self._xdata )
        yarray = array( self._ydata )
        # get the microplane weighting coefficients:
        MPW = self.get_MPW()
        # derive the number of specified microplanes base on the shape of the trace 
        n_mp = xarray.shape[1]
        # total energy contribution of the microplanes:
        E_t_arr = array( [ self._get_E_t( xarray[:, i_mp], yarray[:, i_mp] ) for i_mp in range( n_mp )] )
        # total "total energy" = weighted sum over the microplanes:
        E_t = array( [ dot( E_t_arr, MPW ) ] )
        print('E_t_mic = %.10f' % ( E_t ))

        self.trace.xdata = array( self._trace_time, dtype = float )
        self.trace.ydata = E_t * ones( self.trace.xdata.shape, dtype = float )



class MATS2DMicroplaneDamageDomainTraceUtmic( MATS2DMicroplaneDamageDomainTraceGfmic ):
    '''Evaluate the energy contributions spread over 
    several fields of stress-strain responses. 
    '''
    def redraw( self ):
        if self.idx_x < 0 or self.idx_y < 0 or \
            self._xdata == [] or self._ydata == []:
            return

        xarray = array( self._xdata )
        yarray = array( self._ydata )

        # get the microplane weighting coefficients:
        MPW = self.get_MPW()

        # derive the number of specified microplanes base on the shape of the trace 
        n_mp = xarray.shape[1]

        # elastic energy contribution of the microplanes:
        U_t_arr = array( [ self._get_U_t( xarray[:, i_mp], yarray[:, i_mp] ) for i_mp in range( n_mp )] )
        # total "elastic energy" = weighted sum over the microplanes:
        U_t = array( [ dot( U_t_arr, MPW ) ] )

        self.trace.xdata = array( self._trace_time, dtype = float )
        self.trace.ydata = U_t * ones( self.trace.xdata.shape, dtype = float )
