from enthought.traits.api import Array, Bool, Enum, Float, HasStrictTraits, \
                                 Instance, Int, Trait, Str, Enum, \
                                 Callable, List, TraitDict, Any, Range, \
                                 Delegate, Event, on_trait_change, Button, \
                                 Interface, implements, Property, cached_property
from enthought.traits.ui.api import Item, View, HGroup, ListEditor, VGroup, \
     HSplit, Group, Handler, VSplit
from enthought.traits.ui.menu import NoButtons, OKButton, CancelButton, \
     Action
from enthought.traits.ui.api \
    import View, Item, VSplit, TableEditor, ListEditor
from enthought.traits.ui.table_column \
    import ObjectColumn, ExpressionColumn
from numpy import \
    ix_, array, int_, dot, newaxis, float_, copy, repeat
from ibvpy.core.i_bcond import \
    IBCond

class BCDof(HasStrictTraits):
    '''
    Implements the IBC functionality for a constrained dof.
    '''
    implements(IBCond)

    var = Enum('u', 'f', 'eps', 'sig',)
    dof = Int
    value = Float

    # List of dofs that determine the value of the current dof
    #
    # If this list is empty, then the current dof is
    # prescribed. Otherwise, the dof value is given by the 
    # linear combination of DOFs in the list (see the example below)
    #
    link_dofs = List(Int)

    # Coefficients of the linear combination of DOFs specified in the
    # above list.
    #
    link_coeffs = List(Float)

    # Example of a complex constraint:
    #
    # For example the specification
    #
    # BCDof( var = 'f',
    #        value = 0.,
    #        dof = 2,
    #        link_dofs = [3,4],
    #        link_coeffs = [0.5,0.5] )
    #
    # means that
    #
    # U[2] = 0.5*U[3] + 0.5*U[4] Note that U[2] is non-zero
    #
    # and is regarded as a natural boundary condition.
    #
    # On the ther hand, the specification
    #
    # cos(alpha) * U[2] + sin(slpha) * U[3] = 0.4
    #
    # can be expressed as for U[2] as
    # U[2] = - sin(alpha) / cos(alpha) * U[3] + 0.4 / cos(alpha)
    # so that the corresponding BCDof specification has the form
    #
    # BCDof( var = 'u',
    #        value = 0.4 / cos(alpha),
    #        dof = 2,
    #        link_dofs = [3],
    #        link_coeffs = [-sin(alpha)/cos(alpha) ] )
    #
    time_function = Callable

    def _time_function_default(self):
        return lambda t: t

    def is_essential(self):
        return self.var == 'u'

    def is_linked(self):
        return self.link_dofs != []

    def is_constrained(self):
        '''
        Return true if a DOF is either explicitly prescribed or it depends on other DOFS.
        '''
        return self.is_essential() or self.is_linked()

    def is_natural(self):
        return self.var == 'f' or self.var == 'eps' or self.var == 'sig'

    def get_dofs(self):
        return [self.dof]

    def setup(self, sctx):
        '''
        Locate the spatial context.
        '''
        return

    _constraint = Any

    def apply_essential(self, K):
        '''Register the boundary conditions in the equation system.
        '''
        a = self.dof   # affected dof
        alpha = array(self.link_coeffs, float_)

        # Prepare the indexes and index arrays
        #
        n = self.link_dofs  # constraining dofs
        a_ix = ix_([a])   # constrained dof as array
        n_ix = ix_(n)   # constraining dofs as array
        n_ix_arr = array(list(self.link_dofs), dtype = int)

        #------------------------------------
        # Handle essential boundary condition
        #------------------------------------
        if self.is_essential():
            self._constraint = K.register_constraint(a = a, u_a = self.value,
                                                     alpha = alpha, ix_a = n_ix_arr)

    def apply(self, step_flag, sctx, K, R, t_n, t_n1):
        '''
        According to the kind specification add the 
        '''

        a = self.dof   # affected dof
        alpha = array(self.link_coeffs, float_)

        # Prepare the indexes and index arrays
        #
        n = self.link_dofs  # constraining dofs
        a_ix = ix_([a])   # constrained dof as array
        n_ix = ix_(n)   # constraining dofs as array
        n_ix_arr = array(list(self.link_dofs), dtype = int)

        #------------------------------------
        # Handle essential boundary condition
        #------------------------------------
        if self.is_essential():

            # The displacement is applied only in the first iteration step!.
            #
            if step_flag == 'predictor':
                ua_n = self.value * float(self.time_function(t_n))
                ua_n1 = self.value * float(self.time_function(t_n1))
                u_a = ua_n1 - ua_n
            elif step_flag == 'corrector':
                u_a = 0

            self._constraint.u_a = u_a

        elif self.is_natural():

            R_a = self.value * float(self.time_function(t_n1))

            # Add the value to the proper equation.
            #
            # if a is involved in another essential constraint, redistribute 
            # it according to the link coefficients infolved in that constraint!
            # 
            R[self.dof] += R_a

            if self.is_linked():

                # Distribute the load contribution to the proportionally loaded dofs
                #
                R[n_ix] += alpha.transpose() * R_a


if __name__ == '__main__':

    from ibvpy.mesh.fe_grid import FEGrid
    from ibvpy.mesh.fe_domain_list import FEDomainList
    from ibvpy.fets.fets1D.fets1D2l import FETS1D2L
    from ibvpy.api import \
    TStepper as TS, RTraceGraph, RTraceDomainField, TLoop, \
    TLine, IBVPSolve as IS, DOTSEval
    from ibvpy.mats.mats1D.mats1D_elastic.mats1D_elastic import MATS1DElastic


    fets_eval = FETS1D2L(mats_eval = MATS1DElastic(E = 10., A = 1.))

    # Discretization
    fe_domain1 = FEGrid(coord_max = (10., 0., 0.),
                               shape = (10,),
                               fets_eval = fets_eval)

    fe_domain2 = FEGrid(coord_min = (10., 0., 0.),
                                       coord_max = (20., 0., 0.),
                                       shape = (10,),
                                       fets_eval = fets_eval)

    fe_domain = FEDomainList(subdomains = [ fe_domain1, fe_domain2 ])
    ts = TS(dof_resultants = True,
             sdomain = fe_domain,
             bcond_list = [ BCDof(var = 'u', dof = 0, value = 0.),
                                   BCDof(var = 'u', dof = 5, link_dofs = [16], link_coeffs = [1.], value = 0.),
                                   BCDof(var = 'f', dof = 21, value = 10) ],
             rtrace_list = [ RTraceGraph(name = 'Fi,right over u_right (iteration)' ,
                                           var_y = 'F_int', idx_y = 0,
                                           var_x = 'U_k', idx_x = 1),
                                           ]
                        )

    # Add the time-loop control
    tloop = TLoop(tstepper = ts, tline = TLine(min = 0.0, step = 1, max = 1.0))


    ts.set(sdomain = FEDomainList(subdomains = [ fe_domain1, fe_domain2 ]))

    ts.set(bcond_list = [BCDof(var = 'u', dof = 0, value = 0.),
                          BCDof(var = 'u', dof = 5, link_dofs = [16], link_coeffs = [1.], value = 0.),
                          BCDof(var = 'f', dof = 21, value = 10) ])
    print tloop.eval()
