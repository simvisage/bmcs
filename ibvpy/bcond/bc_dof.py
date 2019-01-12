from traits.api import Float, \
    Int,  Enum, Instance, \
    List,  Any, provides
from traitsui.api import \
    View, Item, UItem, VGroup, VSplit

from ibvpy.core.i_bcond import \
    IBCond
from mathkit.mfn import MFnLineArray
import numpy as np
from view.plot2d import Vis2D, Viz2DTimeFunction
from view.ui import BMCSTreeNode


@provides(IBCond)
class BCDof(BMCSTreeNode):
    '''
    Implements the IBC functionality for a constrained dof.

    Example of a complex constraint:

    For example the specification

    BCDof( var = 'f',
           value = 0.,
           dof = 2,
           link_dofs = [3,4],
           link_coeffs = [0.5,0.5] )

    means that

    U[2] = 0.5*U[3] + 0.5*U[4] Note that U[2] is non-zero

    and is regarded as a natural boundary condition.

    On the ther hand, the specification

    cos(alpha) * U[2] + sin(slpha) * U[3] = 0.4

    can be expressed as for U[2] as
    U[2] = - sin(alpha) / cos(alpha) * U[3] + 0.4 / cos(alpha)
    so that the corresponding BCDof specification has the form

    BCDof( var = 'u',
           value = 0.4 / cos(alpha),
           dof = 2,
           link_dofs = [3],
           link_coeffs = [-sin(alpha)/cos(alpha) ] )

    '''
    node_name = 'boundary condition'
    tree_node_list = List()

    def _tree_node_list_default(self):
        return [self.time_function]

    var = Enum('u', 'f', 'eps', 'sig',
               label='Variable',
               BC=True
               )
    dof = Int(label='Degree of freedom',
              BC=True,
              )
    value = Float(label='Value',
                  BC=True,
                  )
    link_dofs = List(Int,
                     BC=True,
                     label='Linear dependencies',
                     tooltip='Degrees of freedom linked\n'
                     'with the current by link coefficients')
    '''
    List of dofs that determine the value of the current dof
    
    If this list is empty, then the current dof is
    prescribed. Otherwise, the dof value is given by the
    linear combination of DOFs in the list (see the example below)
    '''
    link_coeffs = List(Float,
                       BC=True,
                       label='Link coefficients',
                       tooltip='Multipliers for linear combination\n'
                       'equation')
    '''
    Coefficients of the linear combination of DOFs specified in the
    above list.
    '''
    time_function = Instance(MFnLineArray,
                             BC=True)
    '''
    Time function prescribing the evolution of the boundary condition.
    '''

    def _time_function_default(self):
        return MFnLineArray(xdata=[0, 1], ydata=[0, 1], extrapolate='diff')

    def get_viz2d_data(self):
        return self.time_function.xdata, self.time_function.ydata

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
        if self.is_essential():
            a = self.dof   # affected dof
            alpha = np.array(self.link_coeffs, np.float_)
            n_ix_arr = np.array(list(self.link_dofs), dtype=int)
            self._constraint = K.register_constraint(a=a, u_a=self.value,
                                                     alpha=alpha, ix_a=n_ix_arr)

    def apply(self, step_flag, sctx, K, R, t_n, t_n1):
        '''
        According to the kind specification add the
        '''
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

                # Prepare the indexes and index arrays
                #
                n = self.link_dofs  # constraining dofs
                n_ix = np.ix_(n)   # constraining dofs as array

                # Distribute the load contribution to the proportionally loaded dofs
                #
                alpha = np.array(self.link_coeffs, np.float_)
                R[n_ix] += alpha.transpose() * R_a

    tree_view = View(
        VGroup(
            VSplit(
                VGroup(
                    Item('var', full_size=True, resizable=True,
                         tooltip='Type of variable: u - essential, f- natural'),
                    Item('dof',
                         tooltip='Number of the degree of freedom'),
                    Item('value',
                         tooltip='Value of the boundary condition to\n'
                         'be multiplied with the time function'),
                ),
                UItem('time_function@', full_size=True, springy=True,
                      resizable=True)
            ),
        )
    )

    traits_view = tree_view


if __name__ == '__main__':

    from ibvpy.mesh.fe_grid import FEGrid
    from ibvpy.mesh.fe_domain import FEDomain
    from ibvpy.fets.fets1D.fets1D2l import FETS1D2L
    from ibvpy.api import \
        TStepper as TS, RTDofGraph, TLoop, \
        TLine
    from ibvpy.mats.mats1D.mats1D_elastic.mats1D_elastic import MATS1DElastic

    fets_eval = FETS1D2L(mats_eval=MATS1DElastic(E=10., A=1.))

    # Discretization
    fe_domain1 = FEGrid(coord_max=(10., 0., 0.),
                        shape=(10,),
                        fets_eval=fets_eval)

    fe_domain2 = FEGrid(coord_min=(10., 0., 0.),
                        coord_max=(20., 0., 0.),
                        shape=(10,),
                        fets_eval=fets_eval)

    fe_domain = FEDomain(subdomains=[fe_domain1, fe_domain2])
    ts = TS(dof_resultants=True,
            sdomain=fe_domain,
            bcond_list=[BCDof(var='u', dof=0, value=0.),
                        BCDof(
                            var='u', dof=5, link_dofs=[16], link_coeffs=[1.], value=0.),
                        BCDof(var='f', dof=21, value=10)],
            rtrace_list=[RTDofGraph(name='Fi,right over u_right (iteration)',
                                    var_y='F_int', idx_y=0,
                                    var_x='U_k', idx_x=1),
                         ]
            )

    # Add the time-loop control
    tloop = TLoop(tstepper=ts, tline=TLine(min=0.0, step=1, max=1.0))

    ts.set(sdomain=FEDomain(subdomains=[fe_domain1, fe_domain2]))

    ts.set(bcond_list=[BCDof(var='u', dof=0, value=0.),
                       BCDof(
                           var='u', dof=5, link_dofs=[16], link_coeffs=[1.], value=0.),
                       BCDof(var='f', dof=21, value=10)])
    print(tloop.eval())
