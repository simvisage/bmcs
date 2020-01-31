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
# Created on Aug 18, 2009 by: rch

from ibvpy.fets.fets_eval import FETSEval

from numpy import \
     zeros, dot, hstack, identity

from scipy.linalg import \
     inv     

#-----------------------------------------------------------------------------------
# FETS3D - Base class for 3D elements 
#-----------------------------------------------------------------------------------

class FETS3D(FETSEval):
    '''Base class for 3D elements.
    '''
    # Dimensional mapping
    dim_slice = slice(0, 3)
    
    def get_B_mtx( self, r_pnt, X_mtx ):
        '''
        Return the kinematic matrix
        @param r_pnt:local coordinates
        @param X_mtx:global coordinates
        '''
        J_mtx = self.get_J_mtx(r_pnt,X_mtx)
        dNr_mtx = self.get_dNr_mtx( r_pnt )
        dNx_mtx = dot( inv( J_mtx ), dNr_mtx  )
        n_nodes = len( self.dof_r )
        Bx_mtx = zeros( (6, n_nodes * 3 ), dtype = 'float_' )
        for i in range(0, n_nodes ):
            Bx_mtx[0, i*3]   = dNx_mtx[0, i]
            Bx_mtx[1, i*3+1] = dNx_mtx[1, i]
            Bx_mtx[2, i*3+2] = dNx_mtx[2, i]
            Bx_mtx[3, i*3+1] = dNx_mtx[2, i]
            Bx_mtx[3, i*3+2] = dNx_mtx[1, i]   
            Bx_mtx[4, i*3+0] = dNx_mtx[2, i]
            Bx_mtx[4, i*3+2] = dNx_mtx[0, i]
            Bx_mtx[5, i*3+0] = dNx_mtx[1, i]
            Bx_mtx[5, i*3+1] = dNx_mtx[0, i]
        return Bx_mtx
    