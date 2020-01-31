'''
Created on Sep 3, 2009

@author: jakub
'''

from math import pi, sqrt
from scipy.linalg import \
    inv
from traits.api import \
    Instance, Property, \
    DelegatesTo

from ibvpy.fets.fets_eval import \
    FETSEval
import numpy as np


#-------------------------------------------------------------------------
# FETS2D4Q - 4 nodes iso-parametric quadrilateral element (2D, linear, Lagrange family)
#-------------------------------------------------------------------------
# class FETS2Drotsym( FETSEvalPrototyped ):
class FETS2Drotsym(FETSEval):

    debug_on = True

    prototype_fets = Instance(FETSEval)

    # Dimensional mapping
    dim_slice = slice(0, 2)
    dof_r = DelegatesTo('prototype_fets')
    geo_r = DelegatesTo('prototype_fets')
    n_nodal_dofs = DelegatesTo('prototype_fets')
    n_e_dofs = DelegatesTo('prototype_fets')

    get_dNr_mtx = DelegatesTo('prototype_fets')
    get_dNr_geo_mtx = DelegatesTo('prototype_fets')

    get_N_geo_mtx = DelegatesTo('prototype_fets')
    get_N_mtx = DelegatesTo('prototype_fets')

    ngp_r = Property

    def _get_ngp_r(self):
        return self.prototype_fets.ngp_r

    ngp_s = Property

    def _get_ngp_s(self):
        return self.prototype_fets.ngp_s

    vtk_cell_types = Property

    def _get_vtk_cell_types(self):
        return self.prototype_fets.vtk_cell_types

    vtk_cells = Property

    def _get_vtk_cells(self):
        return self.prototype_fets.vtk_cells

    vtk_r = Property

    def _get_vtk_r(self):
        return self.prototype_fets.vtk_r

    def _set_vtk_r(self, value):
        # @todo - WHAT's this!!!! Setter should put the value in and not return anything!
        return value

    def get_B_mtx(self, r_pnt, X_mtx):
        '''Mapping matrix from displacements to strains
        '''
        J_mtx = self.get_J_mtx(r_pnt, X_mtx)
        dNr_mtx = self.get_dNr_mtx(r_pnt)
        N_mtx = self.get_N_mtx(r_pnt)
        radius = np.dot(self.get_N_geo_mtx(r_pnt), X_mtx[:, 1])

        dNx_mtx = np.dot(inv(J_mtx), dNr_mtx)

        Bx_mtx = np.zeros((6, self.n_e_dofs), dtype='float_')
        for i in range(0, self. n_dof_r):
            Bx_mtx[0, i * 2] = dNx_mtx[0, i]  # eps_z
            Bx_mtx[1, i * 2 + 1] = dNx_mtx[1, i]  # eps_r
            Bx_mtx[2, i * 2 + 1] = N_mtx[1, i * 2 + 1] / radius  # eps_theta
            Bx_mtx[5, i * 2] = dNx_mtx[1, i]  # gamma_rz
            Bx_mtx[5, i * 2 + 1] = dNx_mtx[0, i]  # gamma_rz
        return Bx_mtx

    def get_J_det(self, r_pnt, X_mtx):
        '''
        Has to be overloaded for rotational symetry case
        @param r_pnt:
        @param X_mtx:
        '''
        #radius = 1.
        circle = 2. * pi * np.dot(self.get_N_geo_mtx(r_pnt), X_mtx[:, 1])
        return np.array(circle * self._get_J_det(r_pnt, X_mtx), dtype='float_')

    def adjust_spatial_context_for_point(self, sctx):
        '''Overloaded call to mats_eval.

        X_mtx is extended, the depth component is the circle node 
        draws around the axis of symmetry

        The extension is required in order to regularize the process zone
        within the element.
        '''
        if sctx.X.shape[1] == 2:
            X_mtx_theta = sctx.X[:, 1] * 2 * pi
            sctx.X_reg = np.append(
                np.copy(sctx.X), X_mtx_theta[:, None], axis=1)

#----------------------- example --------------------


def example_with_new_domain():
    from ibvpy.api import \
        TStepper as TS, RTraceDomainListField, TLoop, \
        TLine, BCSlice
    from ibvpy.fets.fets2D.fets2D4q8u import FETS2D4Q8U
    from ibvpy.mats.mats3D.mats3D_cmdm import \
        MATS3DMicroplaneDamage
    from ibvpy.mats.matsXD.matsXD_cmdm import PhiFnStrainSoftening

#    mats =  MATS2DElastic(E=2,nu= .2,
#                          stress_state= 'rotational_symetry')
    mats = MATS3DMicroplaneDamage(model_version='stiffness',
                                  E=34e3,
                                  nu=0.2,
                                  phi_fn=PhiFnStrainSoftening(G_f=0.001117,
                                                              f_t=2.8968))

    fets_eval = FETS2Drotsym(prototype_fets=FETS2D4Q8U(),
                             mats_eval=mats)

    fets_eval.vtk_r *= 0.9
    from ibvpy.mesh.fe_grid import FEGrid

    radius = sqrt(1. / pi)
#    f_i = (radius/2.)*2*pi
#    f_o = (radius)*2*pi
#    print 'f ',f_i,' ', f_o
    # Discretization
    fe_grid = FEGrid(  # coord_min = (0.,radius/2.,0.),
        coord_max=(1., radius, 0.),
        shape=(20, 20),
        fets_eval=fets_eval)

    tstepper = TS(sdomain=fe_grid,
                  bcond_list=[
                      BCSlice(var='u', value=0., dims=[0],
                              slice=fe_grid[0, :, 0, :]),
                      BCSlice(var='u', value=0., dims=[1],
                              slice=fe_grid[0, 0, 0, 0]),
                      BCSlice(var='u', value=1.e-3, dims=[0],
                              slice=fe_grid[-1, :, -1, :]),
                  ],

                  rtrace_list=[
                      RTraceDomainListField(name='Stress',
                                            var='sig_app', idx=0, warp=True,
                                            record_on='update'),
                      RTraceDomainListField(name='fracture_energy',
                                            var='fracture_energy', idx=0, warp=True,
                                            record_on='update'),
                      RTraceDomainListField(name='Displacement',
                                            var='u', idx=0,
                                            record_on='update',
                                            warp=True),
                      #                    RTraceDomainListField(name = 'N0' ,
                      #                                      var = 'N_mtx', idx = 0,
                      # record_on = 'update')
                  ]
                  )

    # Add the time-loop control
    #global tloop
    tloop = TLoop(tstepper=tstepper, KMAX=300, tolerance=1e-4,
                  tline=TLine(min=0.0,  step=1.0, max=1.0))

    #import cProfile
    #cProfile.run('tloop.eval()', 'tloop_prof' )
    print(tloop.eval())
    #import pstats
    #p = pstats.Stats('tloop_prof')
    # p.strip_dirs()
    # print 'cumulative'
    # p.sort_stats('cumulative').print_stats(20)
    # print 'time'
    # p.sort_stats('time').print_stats(20)

    # Put the whole thing into the simulation-framework to map the
    # individual pieces of definition into the user interface.
    #
    from ibvpy.plugins.ibvpy_app import IBVPyApp
    app = IBVPyApp(ibv_resource=tloop)
    app.main()

if __name__ == '__main__':
    example_with_new_domain()
