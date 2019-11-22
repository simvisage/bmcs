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
# Created on Oct 26, 2010 by: rch

from traits.api import \
    Array, provides, Instance, Property, \
    cached_property, Int, Float, List
from ibvpy.core.i_tstepper_eval import \
    ITStepperEval
from ibvpy.core.tstepper_eval import \
    TStepperEval
from ibvpy.rtrace.rt_domain import RTraceDomain
from mathkit.matrix_la.sys_mtx_assembly import \
    SysMtxArray
import numpy as np
from .fe_subdomain import FESubDomain
from .i_fe_subdomain import IFESubDomain
from .i_fe_uniform_domain import IFEUniformDomain


#-----------------------------------------------------------------------------
# Integrator for a general regular domain.
#-----------------------------------------------------------------------------
@provides(ITStepperEval)
class SpringDOTSEval(TStepperEval):
    '''
    Domain with uniform FE-time-step-eval.
    '''

    sdomain = Instance(IFEUniformDomain)

    ##### check the three following operators ###
    # they might be deleted
    def new_cntl_var(self):
        return np.zeros(self.sdomain.n_dofs, np.float_)

    def new_resp_var(self):
        return np.zeros(self.sdomain.n_dofs, np.float_)

    def new_tangent_operator(self):
        '''
        Return the tangent operator used for the time stepping
        '''
        return SysMtxArray()

    # cached zeroed array for element stiffnesses
    k_arr = Property(Array, depends_on='sdomain.+changed_structure')

    @cached_property
    def _get_k_arr(self):
        return np.zeros((self.sdomain.n_elems, 2, 2), dtype='float_')

    F_int = Property(Array, depends_on='sdomain.+changed_structure')

    @cached_property
    def _get_F_int(self):
        return np.zeros(self.sdomain.n_dofs, np.float_)

    k_mtx = Property(Array, depends_on='sdomain.+changed_structure')

    @cached_property
    def _get_k_mtx(self):
        k = self.sdomain.k_value
        return np.array([[k, -k],
                         [-k, k]], dtype='float_')

    def get_corr_pred(self, sctx, U, dU, tn, tn1, F_int, *args, **kw):

        # in order to avoid allocation of the array in every time step
        # of the computation
        k_arr = self.k_arr
        k_arr[...] = self.k_mtx[None, :, :]

        tstepper = self.sdomain.tstepper
        U = tstepper.U_k

        # @todo - avoid the loop - use numpy array operator
        for i in range(self.sdomain.n_elems):
            ix = self.sdomain.elem_dof_map[i]
            u = U[ix]
            f = np.dot(self.k_mtx, u)
            F_int[ix] += f

        return SysMtxArray(mtx_arr=k_arr, dof_map_arr=self.sdomain.elem_dof_map)


@provides(IFESubDomain, IFEUniformDomain)
class FESpringArray(FESubDomain):

    dofs_1 = Array(int)

    dofs_2 = Array(int)

    k_value = Float

    dots = Property

    @cached_property
    def _get_dots(self):
        '''Construct and return a new instance of domain
        time stepper.
        '''
        return SpringDOTSEval(sdomain=self)

    #-----------------------------------------------------------------
    # Feature: response tracer background mesh
    #-----------------------------------------------------------------

    rt_bg_domain = Property(depends_on='+changed_structure,+changed_geometry')

    @cached_property
    def _get_rt_bg_domain(self):
        return RTraceDomain(sd=self)

    def redraw(self):
        self.rt_bg_domain.redraw()

    # get the number of dofs in the subgrids
    #  - consider caching
    n_dofs = Property(Int)

    def _get_n_dofs(self):
        '''Total number of dofs'''
        return 0

    # @todo - remove these - they come from the uniform domain that uses the
    # standard DOTSEval but they are not necessary here.
    #
    # Refine the dependency between DOTSEval and FEGridLevel
    elements = List

    shape = Property

    n_elems = Property()

    def _get_n_elems(self):
        return len(self.dofs_1)

    n_active_elems = Property

    def _get_n_active_elems(self):
        return self.n_elems

    elem_dof_map = Property

    def _get_elem_dof_map(self):
        return np.hstack([self.dofs_1[:, None], self.dofs_2[:, None]])

    elem_X_map = Array

    elem_x_map = Array

    def apply_on_ip_grid(self, fn, ip_mask):
        '''
        Apply the function fn over the first dimension of the array.
        @param fn: function to apply for each ip from ip_mask and each element. 
        @param ip_mask: specifies the local coordinates within the element.     
        '''

    def get_spring_forces(self, U):

        k_arr = self.dots.k_arr
        k_arr[...] = self.dots.k_mtx[None, :, :]

        tstepper = self.tstepper
        U = tstepper.U_k

        F_arr = np.zeros((self.n_elems, ), dtype=float)

        u = U[self.elem_dof_map]
        delta_u = u[:, 0] - u[:, 1]

        for i in range(self.n_elems):
            ix = self.elem_dof_map[i]
            u = U[ix]
            f = np.dot(self.dots.k_mtx, u)
            F_arr[i] = f[1]

        return F_arr

    def plot_spring_forces(self, U, axes, *args, **kw):
        #
        F_arr = self.get_spring_forces(U)

        xdata = np.arange(self.n_elems)
        ydata = F_arr

        axes.bar(xdata, ydata, *args, **kw)
