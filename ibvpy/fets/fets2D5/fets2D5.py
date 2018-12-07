#-------------------------------------------------------------------------------
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
# Created on Feb 22, 2010 by: rch

from traits.api import \
    HasTraits, Instance, Int, Property, Array, cached_property, Instance

from numpy import \
     zeros, dot, hstack, vstack, identity, cross, transpose, tensordot, outer

from scipy.linalg import \
     inv, norm

from ibvpy.fets.fets_eval import FETSEval
from ibvpy.fets.fets3D.fets3D8h import FETS3D8H

# import the 3D elastic matrix in order  
from ibvpy.mats.mats3D.mats3D_elastic.mats3D_elastic import \
    MATS3DElastic

from ibvpy.mats.mats2D.mats2D_tensor import \
    map2d_eps_mtx_to_eng, map2d_sig_eng_to_mtx

from ibvpy.mats.mats3D.mats3D_tensor import \
    map3d_eps_eng_to_mtx, map3d_sig_eng_to_mtx, map3d_sig_mtx_to_eng, map3d_eps_mtx_to_eng, map3d_tns2_to_tns4, map3d_tns4_to_tns2, map3d_eps_mtx_to_eng

#from i_fets_eval import IFETSEval

#-----------------------------------------------------------------------------------
# FETS2D58H -  degenerated 8 nodes isoparametric volume element (2D, linear, Lagrange family)    
#              using a 2D material model
#-----------------------------------------------------------------------------------


class FETS2D5( HasTraits ):

    # @todo: the calculation of the direction cosines 
    #        should be cached (similar to J_mtx and B_mtx)


    elastic_D_mtx3D = Property( depends_on = 'mats_eval' )
    @cached_property
    def _get_elastic_D_mtx3D( self ):
        elastic_mats_eval = MATS3DElastic( E = self.mats_eval.E, nu = self.mats_eval.nu )
        zero_3D = zeros( ( 6, ) )
        sig_eng3D, D_mtx3D = elastic_mats_eval.get_corr_pred( None, zero_3D, zero_3D, 0, 0 )
        return D_mtx3D

    def get_dircos_mtx( self, sctx ):
        '''returns the direction cosine at the integration point.
        The three column vectors v1, v2, v3 of the direction cosine
        matrix build an orthonormal system and can be used to rotate
        the 3D strains into the local coordinate system. 
        The vector v3 stands perpendicular to the plane defined by the 
        curvilinear coordinates r and s. Note that for the orientation
        of the local in-plane coordinates different choices exist.'''

        # get the Jacobian matrix
        r_pnt = sctx.r_pnt
        X_mtx = sctx.X
        J_mtx = self.get_J_mtx( r_pnt, X_mtx )

        # the first two rows of the Jacobi matrix are tangential
        # to the mid-surface at the integration point.  
        # (cf. Zienkiewicz Vol.2, Eq.(15.19))
        j1 = J_mtx[0, :]
        j2 = J_mtx[1, :]

        # v3 stands perdendicular to the surface
        v3 = cross( j1, j2 )

        # One possible choice for the direction of the local x-axis 
        # is to define it such that it coincides with the orientation
        # of the first curvilinear coordinate r (cf. FE-Skript)
        v1 = j1

        # The local y-axis is obtained as the cross product of the 
        # local x and z direction
        v2 = cross( v3, v1 )

        # normalize the vectors to obtain an ortho-normal basis 
        v1_ = v1 / norm( v1 )
        v2_ = v2 / norm( v2 )
        v3_ = v3 / norm( v3 )

        return transpose( vstack( ( v1_, v2_, v3_ ) ) )


    def get_Trans_strain_to_loc_mtx( self, dircos_mtx ):
        '''Transformation matrix in 3d shape = (6,6).
        multiply the strain vector (engineering notation) with 
        the returned matrix in order to obtain the strain vector
        (engineering notation) in the local coordinate system.
        The ordering of the strain vector in engineering notation
        corresponds to the ordering 
        [eps11,eps22,eps33,eps23,eps13,eps12]. 
        The parameter 'dircos' is the matrix of the direction
        cosine describing the relation between the global and 
        local coordinate system.
        (cf. Zienkiewicz Eq.(6.24))
        '''
        N = dircos_mtx
        T = zeros( ( 6, 6 ) )
        T[0, 0] = N[0, 0] * N[0, 0]
        T[0, 1] = N[1, 0] * N[1, 0]
        T[0, 2] = N[2, 0] * N[2, 0]
        T[0, 3] = N[1, 0] * N[2, 0]
        T[0, 4] = N[0, 0] * N[2, 0]
        T[0, 5] = N[0, 0] * N[1, 0]
        T[1, 0] = N[0, 1] * N[0, 1]
        T[1, 1] = N[1, 1] * N[1, 1]
        T[1, 2] = N[2, 1] * N[2, 1]
        T[1, 3] = N[1, 1] * N[2, 1]
        T[1, 4] = N[0, 1] * N[2, 1]
        T[1, 5] = N[0, 1] * N[1, 1]
        T[2, 0] = N[0, 2] * N[0, 2]
        T[2, 1] = N[1, 2] * N[1, 2]
        T[2, 2] = N[2, 2] * N[2, 2]
        T[2, 3] = N[1, 2] * N[2, 2]
        T[2, 4] = N[0, 2] * N[2, 2]
        T[2, 5] = N[0, 2] * N[1, 2]
        T[3, 0] = 2.0 * N[0, 2] * N[0, 1]
        T[3, 1] = 2.0 * N[1, 2] * N[1, 1]
        T[3, 2] = 2.0 * N[2, 2] * N[2, 1]
        T[3, 3] = N[1, 2] * N[2, 1] + N[2, 2] * N[1, 1]
        T[3, 4] = N[2, 2] * N[0, 1] + N[0, 2] * N[2, 1]
        T[3, 5] = N[0, 2] * N[1, 1] + N[1, 2] * N[0, 1]
        T[4, 0] = 2.0 * N[0, 2] * N[0, 0]
        T[4, 1] = 2.0 * N[1, 2] * N[1, 0]
        T[4, 2] = 2.0 * N[2, 2] * N[2, 0]
        T[4, 3] = N[1, 2] * N[2, 0] + N[2, 2] * N[1, 0]
        T[4, 4] = N[2, 2] * N[0, 0] + N[0, 2] * N[2, 0]
        T[4, 5] = N[0, 2] * N[1, 0] + N[1, 2] * N[0, 0]
        T[5, 0] = 2.0 * N[0, 1] * N[0, 0]
        T[5, 1] = 2.0 * N[1, 1] * N[1, 0]
        T[5, 2] = 2.0 * N[2, 1] * N[2, 0]
        T[5, 3] = N[1, 1] * N[2, 0] + N[2, 1] * N[1, 0]
        T[5, 4] = N[2, 1] * N[0, 0] + N[0, 1] * N[2, 0]
        T[5, 5] = N[0, 1] * N[1, 0] + N[1, 1] * N[0, 0]
        Trans_strain_to_loc_mtx = T
        return Trans_strain_to_loc_mtx


    def get_Trans_stress_from_loc_mtx( self, dircos_mtx ):
        '''Transformation matrix in 3d shape = (6,6)  
        multiply the local stress vector (engineering notation) with 
        the returned matrix in order to obtain the stress vector
        (engineering notation) in the global coordinate system.
        The ordering of the stress vector in engineering notation
        corresponds to the ordering [sig11,sig22,sig33,sig23,sig13,sig12]. 
        The parameter 'dircos' is the matrix of the direction
        cosine describing the relation between the global and 
        local coordinate system.
        (cf. Zienkiewicz Eq.(6.23))
        '''
        N = dircos_mtx
        T = zeros( ( 6, 6 ) )
        T[0, 0] = N[0, 0] * N[0, 0]
        T[0, 1] = N[0, 1] * N[0, 1]
        T[0, 2] = N[0, 2] * N[0, 2]
        T[0, 3] = 2.0 * N[0, 1] * N[0, 2]
        T[0, 4] = 2.0 * N[0, 0] * N[0, 2]
        T[0, 5] = 2.0 * N[0, 0] * N[0, 1]
        T[1, 0] = N[1, 0] * N[1, 0]
        T[1, 1] = N[1, 1] * N[1, 1]
        T[1, 2] = N[1, 2] * N[1, 2]
        T[1, 3] = 2.0 * N[1, 1] * N[1, 2]
        T[1, 4] = 2.0 * N[1, 0] * N[1, 2]
        T[1, 5] = 2.0 * N[1, 0] * N[1, 1]
        T[2, 0] = N[2, 0] * N[2, 0]
        T[2, 1] = N[2, 1] * N[2, 1]
        T[2, 2] = N[2, 2] * N[2, 2]
        T[2, 3] = 2.0 * N[2, 1] * N[2, 2]
        T[2, 4] = 2.0 * N[2, 0] * N[2, 2]
        T[2, 5] = 2.0 * N[2, 0] * N[2, 1]
        T[3, 0] = N[2, 0] * N[1, 0]
        T[3, 1] = N[2, 1] * N[1, 1]
        T[3, 2] = N[2, 2] * N[1, 2]
        T[3, 3] = N[2, 1] * N[1, 2] + N[2, 2] * N[1, 1]
        T[3, 4] = N[2, 2] * N[1, 0] + N[2, 0] * N[1, 2]
        T[3, 5] = N[2, 0] * N[1, 1] + N[2, 1] * N[1, 0]
        T[4, 0] = N[2, 0] * N[0, 0]
        T[4, 1] = N[2, 1] * N[0, 1]
        T[4, 2] = N[2, 2] * N[0, 2]
        T[4, 3] = N[2, 1] * N[0, 2] + N[2, 2] * N[0, 1]
        T[4, 4] = N[2, 2] * N[0, 0] + N[2, 0] * N[0, 2]
        T[4, 5] = N[2, 0] * N[0, 1] + N[2, 1] * N[0, 0]
        T[5, 0] = N[1, 0] * N[0, 0]
        T[5, 1] = N[1, 1] * N[0, 1]
        T[5, 2] = N[1, 2] * N[0, 2]
        T[5, 3] = N[1, 1] * N[0, 2] + N[1, 2] * N[0, 1]
        T[5, 4] = N[1, 2] * N[0, 0] + N[1, 0] * N[0, 2]
        T[5, 5] = N[1, 0] * N[0, 1] + N[1, 1] * N[0, 0]
        Trans_stress_from_loc_mtx = T
        return Trans_stress_from_loc_mtx

    def get_eps_eng( self, sctx, u ):

        eps_eng = super( FETS2D58H, self ).get_eps_eng( sctx, u )
        # get the direction cosine at the ip:
        dircos_mtx = self.get_dircos_mtx( sctx )
        dircos_mtx_T = transpose( dircos_mtx )

        # switch from engineering notation to matrix notation:
        eps_mtx3D = map3d_eps_eng_to_mtx( eps_eng )

        # transform strain from global to local coordinates (i.e. map strain into the 2D plane) 
        eps_mtx3D_ = dot( dot( dircos_mtx_T, eps_mtx3D ), dircos_mtx )

        # reduce 3D mtx to 2D mtx (neglecting the strains in local z_ direction!)
        eps_mtx2D_ = eps_mtx3D_[:2, :2]

        # switch from matrix notation to engineering notation for the strains:
        eps_eng2D_ = map2d_eps_mtx_to_eng( eps_mtx2D_ )
        return eps_eng2D_

    def get_mtrl_corr_pred( self, sctx, eps_eng, d_eps_eng, tn, tn1, eps_avg = None ):
        '''Overload the 3D material-corrector-predictor and 
        use the 2D material model instead. This requires to
        map the 3D strains into a 2D plane. This is performed
        employing the direction cosines at each integration
        point (ip). The direction cosines establish the link
        between the global and local coordinate system (the
        reference to the local coordinate system is indicated 
        by the underline character at the end of a variable
        name). 
        After employing the 2D material model the returned 
        stresses and stiffness are augmented with the z_-components
        and transformed into global coordinates. It is assumed
        that in global z-direction the material behaves linear-elastic.
        @todo: check for consistency: 
        
        var A: degenerated volume (assume plane strain eps_z_ == 0)
               for the evaluation of the 2D material model and the 
               damage in the in-plane elasticity components. 
               Then correct the stresses by adding the eps_z_ != 0 
               part. For the linear elastic regime this ends up to
               the standard 3D formulation.
               @todo: check for consistency:
               the damage functions have been obtained based on the 
               assumtion for plane stress (which corresponds to the 
               situation in the experiment). Can the 2D case be 
               reproduced with the choosen implementation? 
        
        '''

        # get the direction cosine at the ip:
        dircos_mtx = self.get_dircos_mtx( sctx )
        dircos_mtx_T = transpose( dircos_mtx )

        # get the Transformation matrices for switching between the 
        # local and global coordinate system in 3D case:
        Trans_stress_from_loc_mtx = self.get_Trans_stress_from_loc_mtx( dircos_mtx )
        Trans_strain_to_loc_mtx = self.get_Trans_strain_to_loc_mtx( dircos_mtx )

        # switch from engineering notation to matrix notation:
        eps_mtx3D = map3d_eps_eng_to_mtx( eps_eng )
        d_eps_mtx3D = map3d_eps_eng_to_mtx( d_eps_eng )

        # transform strain from global to local coordinates (i.e. map strain into the 2D plane) 
        eps_mtx3D_ = dot( dot( dircos_mtx_T, eps_mtx3D ), dircos_mtx )
        d_eps_mtx3D_ = dot( dot( dircos_mtx_T, d_eps_mtx3D ), dircos_mtx )

        # reduce 3D mtx to 2D mtx (neglecting the strains in local z_ direction!)
        eps_mtx2D_ = eps_mtx3D_[:2, :2]
        d_eps_mtx2D_ = d_eps_mtx3D_[:2, :2]
#        print 'eps_mtx2D_: ', eps_mtx2D_

        # switch from matrix notation to engineering notation for the strains:
        eps_eng2D_ = map2d_eps_mtx_to_eng( eps_mtx2D_ )
        d_eps_eng2D_ = map2d_eps_mtx_to_eng( d_eps_mtx2D_ )
#        print 'eps_eng2D_: ', eps_eng2D_

        ###
#       # @todo: verify implementation with alternative variant using
#                    Transformation matrix and engineering notation:
#        eps_eng3D_   = dot( Trans_strain_to_loc_mtx,  eps_eng  )
#        d_eps_eng3D_ = dot( Trans_strain_to_loc_mtx, d_eps_eng )
        ###

        # evaluate the 2D mtrl-model
        sig_eng2D_, D_mtx2D_ = self.mats_eval.get_corr_pred( sctx, eps_eng2D_, d_eps_eng2D_, tn, tn1 )

        #------------------------------------------------------------------------
        # get predictor (D_mtx)        
        #------------------------------------------------------------------------

        # Take the initial elastic isotropic matrix as an initial value
        # the submatrix returned by the 2D damage model shall be written
        # into the slice [0:2,0:2] and [5,5]
        #
        D_mtx3D_ = self.elastic_D_mtx3D
        D_mtx3D_[:2, :2] = D_mtx2D_[:2, :2]
        D_mtx3D_[5, 5] = D_mtx2D_[2, 2]
#        print 'D_mtx3D_', D_mtx3D_

        # transform D_mtx3D from local to global coordinates 
        D_mtx3D = dot( Trans_stress_from_loc_mtx, dot( D_mtx3D_, Trans_strain_to_loc_mtx ) )
#        print 'D_mtx3D', D_mtx3D

        # Alternative: use numpy functionality and 4th-order tensors
        # @todo: varify the numpy-alternative with the existing implementation
#        dircos_tns4 = outer( dircos_mtx, dircos_mtx ).reshape(3,3,3,3) 
#        print 'dircos_tns4', dircos_tns4.shape
#        D_tns4_3D_ = map3d_tns2_to_tns4 ( D_mtx3D_ )
#        D_tns4_3D  = tensordot( dircos_tns4, tensordot( D_tns4_3D_, dircos_tns4, [[2,3],[2,3]] ), [[2,3],[0,1]] )
#        D_mtx3D    = map3d_tns4_to_tns2( D_tns4_3D )        


        #------------------------------------------------------------------------
        # get corrector (sig_eng)        
        #------------------------------------------------------------------------

        # augment 2D matrices to the 3D-case: 
        # the material behaves linear-elastic in out-of plane directions (xz, yz);

        # linear-elastic response (used for eps_zz_, eps_yz_, eps_xz_):
        eps_eng3D_ = map3d_eps_mtx_to_eng( eps_mtx3D_ )
        sig_eng3D_ = dot( D_mtx3D_, eps_eng3D_ )

        # replace the values for xx_, yy_, and xy_ based on the evaluation of mtrl2d:
        # and correct/superpose these values by the elastic response caused by eps_zz_: 
        sig_eng3D_[0] = sig_eng2D_[0] + D_mtx3D_[0, 2] * eps_eng3D_[2]
        sig_eng3D_[1] = sig_eng2D_[1] + D_mtx3D_[1, 2] * eps_eng3D_[2]
        sig_eng3D_[5] = sig_eng2D_[2]

        # transform sig_mtx3D from local to global coordinates
        sig_mtx3D_ = map3d_sig_eng_to_mtx( sig_eng3D_ )
        sig_mtx3D = dot( dot( dircos_mtx, sig_mtx3D_ ), dircos_mtx_T )

        # switch from matrix notations to engineering notations for the stresses:
        sig_eng3D = map3d_sig_mtx_to_eng( sig_mtx3D )

        return sig_eng3D, D_mtx3D

#----------------------- example --------------------

if __name__ == '__main__':
    from ibvpy.api import \
        TStepper as TS, RTDofGraph, RTraceDomainListField, TLoop, \
        TLine, BCDofGroup, IBVPSolve as IS, DOTSEval

#    from ibvpy.mats.mats3D.mats3D_elastic.mats3D_elastic import MATS3DElastic
    from ibvpy.mats.mats2D.mats2D_elastic.mats2D_elastic import MATS2DElastic

#    fets_eval = FETS3D8H(mats_eval = MATS3DElastic(nu = 0.25))
    fets_eval = FETS2D58H( mats_eval = MATS2DElastic( nu = 0.25, stress_state = "plane_strain" ) )

    # set vtk points for shrinked mode visualization:             
    fets_eval.vtk_r = fets_eval.vtk_r * 1.0

    from ibvpy.mesh.fe_grid import FEGrid

    # Discretization
    domain = FEGrid( coord_max = ( 3., 3., 3. ),
                     shape = ( 1, 1, 1 ),
                     fets_eval = fets_eval )

    ts = TS( 
            sdomain = domain,
             bcond_list = [BCDofGroup( var = 'u', value = 0., dims = [0],
                                  get_dof_method = domain.get_left_dofs ),
                        BCDofGroup( var = 'u', value = 0., dims = [1, 2],
                                  get_dof_method = domain.get_bottom_left_dofs ),
                        BCDofGroup( var = 'u', value = 0.002, dims = [0],
                                  get_dof_method = domain.get_right_dofs ) ],
             rtrace_list = [
#                        RTDofGraph(name = 'Fi,right over u_right (iteration)' ,
#                                  var_y = 'F_int', idx_y = right_dof,
#                                  var_x = 'U_k', idx_x = right_dof,
#                                  record_on = 'update'),
#                        RTraceDomainListField(name = 'Deformation' ,
#                                       var = 'eps', idx = 0,
#                                       record_on = 'update'),
                         RTraceDomainListField( name = 'Displacement' ,
                                        var = 'u', idx = 0, warp = True ),
#                         RTraceDomainListField(name = 'Stress' ,
#                                        var = 'sig', idx = 0,
#                                        record_on = 'update'),
#                        RTraceDomainListField(name = 'N0' ,
#                                       var = 'N_mtx', idx = 0,
#                                       record_on = 'update')
                        ]
            )

    # Add the time-loop control
    tloop = TLoop( tstepper = ts, KMAX = 4, RESETMAX = 0, tolerance = 1e-3,
         tline = TLine( min = 0.0, step = 1.0, max = 1.0 ) )

    print('u', tloop.eval())

    # Put the whole stuff into the simulation-framework to map the
    # individual pieces of definition into the user interface.
    #
    from ibvpy.plugins.ibvpy_app import IBVPyApp
    app = IBVPyApp( ibv_resource = tloop )
    app.main()
