
from numpy import \
    zeros, array, dot
from scipy.optimize import fsolve
from traits.api import \
    Bool, provides, \
    Instance, WeakRef, Delegate
from ibvpy.core.i_tstepper_eval import \
    ITStepperEval
from ibvpy.core.tstepper_eval import \
    TStepperEval


@provides(ITStepperEval)
class SubDOTSEval(TStepperEval):
    '''
    Domain with uniform FE-time-step-eval.
    '''
    dots_integ = Instance(ITStepperEval)

    new_cntl_var = Delegate('dots_integ')
    new_resp_var = Delegate('dots_integ')
    new_tangent_operator = Delegate('dots_integ')

    # The following operators should be run on each subdomain separately
    # the state array should be merged together from the several grids.
    state_array_size = Delegate('dots_integ')
    state_array = Delegate('dots_integ')
    ip_offset = Delegate('dots_integ')
    setup = Delegate('dots_integ')
    get_corr_pred = Delegate('dots_integ')
    map_u = Delegate('dots_integ')
    rte_dict = Delegate('dots_integ')
    get_vtk_cell_data = Delegate('dots_integ')
    get_vtk_X = Delegate('dots_integ')
    get_vtk_r_arr = Delegate('dots_integ')
    get_current_values = Delegate('dots_integ')
    get_vtk_pnt_ip_map = Delegate('dots_integ')

    sdomain = WeakRef

    debug = Bool(False)

    def apply_constraints(self, K):

        # Take care for kinematic compatibility between the subdomains and domain
        #
        # = Purpose =
        #
        # At this stage, the spatial domain has been refined - it contains the
        # list of registered refinements. These refinements have been added
        # during the problem setup or by the adaptive strategy.
        #
        # The manipulation of the domains is done using the DOTSList interface.
        # Within this interface, new refinement levels can be added with the backward
        # reference to the original level. The refinement levels provide the skeleton
        # for spatial refinement steps that is done incrementally by specifying the cells
        # of the coarse level to be moved/refined into the finer levels. The process may
        # run recursively.
        #
        # In this setup - run the loop over the refinement steps and impose
        # kinematic constraints between the parent and child domain levels.
        #
        parent_domain = self.sdomain.parent

        if parent_domain == None:
            return

        parent_fets_eval = parent_domain.fets_eval

        for p, fe_domain in self.sdomain.subgrids():

            dof_grid = fe_domain.dof_grid
            geo_grid = fe_domain.geo_grid
            if self.debug:
                print('parent')
                print(p)

            # Get the X coordinates from the parent !!!
            #
            # @todo: must define the dof_grid on the FEPatchedGrid
            parent_dofs = parent_domain.fe_subgrids[0][p].dofs[0].flatten()
            parent_points = parent_domain.fe_subgrids[0][p].dof_X[0]

            if self.debug:
                print('parent_dofs')
                print(parent_dofs)
                print('parent_points')
                print(parent_points)

            # Get the geometry approximation used in the super domain
            #
            N_geo_mtx = parent_fets_eval.get_N_geo_mtx

            # start vector for the search of local coordinate
            #
            lcenter = zeros(parent_points.shape[1], dtype='float_')

            # @todo - remove the [0] here - the N_geo_ntx should return an 1d array
            # to deliver simple coordinates instead of array of a single coordinate
            #
            def geo_approx(gpos, lpos): return dot(
                N_geo_mtx(lpos)[0], parent_points) - gpos

            # @todo use get_dNr_geo_mtx as fprime parameter
            #
            # geo_dapprox = ...

            # For each element in the grid evaluate the links.
            # Get the boundary dofs of the refinement.
            # Put the value into the ...
            #
            dofs, coords = dof_grid.get_boundary_dofs()
            for dofs, gpos in zip(dofs, coords):
                # find the pos within the parent domain at the position p
                #
                #
                # lpos = self.super_domain[pos].get_local_pos( pos )
                # N_mtx = self.super_domain.fets_eval.get_N_mtx( lpos )
                # K.register_constraint( a = dof, alpha = N, ix_a = super_dofs
                solution = fsolve(lambda lpos: geo_approx(gpos, lpos), lcenter)
                if isinstance(solution, float):
                    lpos = array([solution], dtype='float_')
                else:
                    lpos = solution

                if self.debug:
                    print('\tp', p, '\tdofs', dofs,
                          '\tgpos', gpos, '\tlpos', lpos)

                N_mtx = parent_fets_eval.get_N_mtx(lpos)

                if self.debug:
                    print('N_mtx')
                    print(N_mtx)

                for i, dof in enumerate(dofs):
                    K.register_constraint(
                        a=dof, alpha=N_mtx[i], ix_a=parent_dofs)
