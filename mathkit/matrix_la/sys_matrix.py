

from traits.api import HasTraits, Array, Property, cached_property
from numpy import allclose, arange, eye, linalg, ones, ix_, array, zeros, \
                hstack,meshgrid, vstack, dot, newaxis, c_, r_, copy, where,\
                ones
from scipy import sparse,linalg
from scipy.sparse.linalg.dsolve import linsolve
from time import time

class SysDenseMtx( HasTraits ):
    
    def __init__(self, n_dofs, el_dof_map ):
        self.el_dof_map = el_dof_map
        self.mtx = zeros( [n_dofs, n_dofs], dtype = float )

    def set_data(self, data_l ):
        for el_dof_map, el_mtx in zip( self.el_dof_map, data_l ):
            self.mtx[ ix_( el_dof_map, el_dof_map ) ] += el_mtx
        
    def set_nond_zero(self, dof):
        K_aa = self.mtx[dof,dof]
        self.mtx[:,dof] = 0.
        self.mtx[dof,:] = 0.
        self.mtx[dof, dof]   = - K_aa
    
    def add_column(self, dof, F, factor):   
        F += factor * self.mtx[:, dof]
    
    def get_diag_elem(self, dof):
        return self.mtx[dof,dof]
    
    def adapt_linked_mtx(self, dof, n, alpha, K_aa):
        Kna = self._const_dof_diag(dof, alpha)
        self.set_nond_zero(dof)
        self._adapt_linked_stiff(dof, n, alpha, Kna, K_aa)
    
    def _const_dof_diag(self, dof, alpha):
        #print "slice ",self.mtx[:,[a]]
        #print "alpha ",alpha[newaxis] 
        return dot( self.mtx[:,[dof]], alpha[newaxis] )
    
    def _adapt_linked_stiff(self, dof, n, alpha, Kna, K_aa ):
        a_n_ix = ix_( [dof], n )  # array index for submatrix [a,n]
        n_a_ix = ix_( n, [dof] )  # array index for submatrix [n,a]
        
        self.mtx[:,n] += Kna
        self.mtx[n,:] += Kna.transpose()

        # Build in the kinematic constraint 
        #
        #print "comparison ",self.mtx[n_a_ix]," ",alpha.transpose() * K_aa
        self.mtx[n_a_ix] = alpha[newaxis] .transpose() * K_aa
        self.mtx[a_n_ix] = alpha[newaxis] * K_aa

    def solve(self, rhs):
        #tf_solve_s = time()
        u_vct = linalg.solve( self.mtx, rhs )
        #tf_solve_e = time()
        #dif_solve = tf_solve_e - tf_solve_s
        #print "Full Solve: %8.2f sec" %dif_solve
        return u_vct
    
class SysSparseMtx( HasTraits ):
    
    el_dof_map = Array( int )
    
    def __init__(self, n_dofs, el_dof_map ):
        '''Construct the symmetric sparse matrix with el_dof_map
        defining the sparsity map. The assembly is done from a list
        of element matrices that follow the same order 
        as the one in el_dof_map.''' 
        #self.mtx = sparse.sparse.coo_matrix((data,ij))
        self.n_dofs = n_dofs
        self.el_dof_map = el_dof_map
        self.Kna_data = []

    def set_data(self, mtx_arr ):
        '''Set the current array of matrices to be assembled in 
        the sparce matrices. The order of the matrices follows that in 
        el_dof_map'''
        self.mtx_arr = mtx_arr

    ij_map = Property( depends_on = 'el_dof_map' )
    @cached_property
    def _get_ij_map( self ):
        '''
        Derive the row and column indices of individual values 
        in every element matrix.
        '''
        el_dof_map = self.el_dof_map
        ij_dof_map = zeros( (el_dof_map.shape[0],
                             2,
                             el_dof_map.shape[1]**2,
                             ), dtype = 'int_' )
        for el, dof_map in enumerate( el_dof_map ):
            row_dof_map, col_dof_map = meshgrid(dof_map,dof_map)
            ij_dof_map[el,...] = vstack( [row_dof_map.flatten(), col_dof_map.flatten()] )
        return ij_dof_map 

    x_l = Property( depends_on = 'el_dof_map' )
    @cached_property
    def _get_x_l(self):
        '''Helper property to get an array of all row indices'''
        return self.ij_map[:,0,:].flatten()
    
    y_l = Property( depends_on = 'el_dof_map' )
    @cached_property
    def _get_y_l(self):
        '''Helper property to get an array of all column indices'''
        return self.ij_map[:,1,:].flatten()
                        
    def set_nond_zero(self, dof):
        '''Set the off-diagonal values associated with the row and column a
        to zero. Remark, set the diagonal terms negative. 
        
        @TODO this can be improved using the where method of numpy
        '''

        el_arr, row_arr = where( self.el_dof_map == dof )
        for el, i_dof in zip( el_arr, row_arr ):        
            k_diag = self.mtx_arr[el,i_dof,i_dof]
            self.mtx_arr[el,i_dof,:] = 0.0
            self.mtx_arr[el,:,i_dof] = 0.0
            self.mtx_arr[el,i_dof,i_dof] = -k_diag
                
    def add_column(self, dof, F, factor):   
        '''Get the slice of the a-th column.
        (used for the implementation of the essential boundary conditions)
        @TODO use where method of numpy 
        '''             
        el_arr, row_arr = where( self.el_dof_map == dof )
        for el, i_dof in zip( el_arr, row_arr ):
            rows = self.el_dof_map[el]
            F[ rows ] += factor * self.mtx_arr[ el, :, i_dof ]
        
    def get_diag_elem(self, dof):
        '''Get the value of diagonal element at a-ths dof. 
        '''
        K_aa = 0.
        el_arr, row_arr = where( self.el_dof_map == dof )
        for el, i_dof in zip( el_arr, row_arr ):
            K_aa += self.mtx_arr[el, i_dof, i_dof ]
        return K_aa
    
    def adapt_linked_mtx(self, dof, n, alpha, K_aa):
        '''Get the submatrix of the constrained degrees of freedom.
        '''
        Kna_idx = []
        self.Kna_data = []
        el_a_arr, row_a_arr = where( self.el_dof_map == dof )
        for el, i_dof in zip( el_a_arr, row_a_arr ):
            rows = self.el_dof_map[el]
            sl = self.mtx_arr[ el, :, [i_dof] ]
            #Kna[ rows ] += dot(sl.T,alpha)
            self.Kna_data.extend(dot(sl.T,alpha))
            Kna_idx.extend(rows)
            k_diag = self.mtx_arr[el,i_dof,i_dof]
            self.mtx_arr[el,i_dof,:] = 0.0
            self.mtx_arr[el,:,i_dof] = 0.0
            self.mtx_arr[el,i_dof,i_dof] = -k_diag
        
        Kna_n_idx = ones(len(Kna_idx)) * n
        self.Kna_ij = vstack((hstack((Kna_n_idx,Kna_idx)),\
                              hstack((Kna_idx,Kna_n_idx))))
        
        self.Kna_data = hstack((self.Kna_data,self.Kna_data))
            
    def solve(self, rhs):
        '''Construct the matrix and use the solver to get 
        the solution for the supplied rhs. 
        '''
        self.data_l = self.mtx_arr.ravel()        
        if self.Kna_data != []:
            ij = hstack((vstack((self.x_l,self.y_l)),self.Kna_ij))
            self.data_l = hstack((self.data_l,self.Kna_data))
        else:
            ij = vstack((self.x_l,self.y_l))

        # Assemble the system matrix from the flattened data and 
        # sparsity map containing two rows - first one are the row
        # indices and second one are the column indices.
        mtx = sparse.coo_matrix( ( self.data_l, ij ) )

        u_vct = linsolve.spsolve( mtx, rhs )
        return u_vct  

        
if __name__ == '__main__':

    # todo - run the solver for both bypes of matrices
    # and compare the solution to get a unittest.
    # 
    n_dofs = 10
    submtx = array([[1.,-1.],[-1.,1.]])
    
    el_dof_map = array( [arange( n_dofs - 1),
                         arange( n_dofs - 1) + 1 ], dtype = int ).transpose()

    rhs = zeros(n_dofs)
    rhs[-1]= 1.
    
#    tf_start = time()
#    mtx = SysDenseMtx( n_dofs )
#    for ix in ix_list:    
#        mtx[ ix_(ix,ix)  ] += submtx
#    
#   
#
#    mtx[0,:] = 0.
#    mtx[:,0] = 0.
#    mtx[0,0] = 1.
#    #print "mtx ",mtx
#    U_f=mtx.solve(rhs)
#    print "U_f ",U_f
#    tf_end = time()
#    diff = tf_end - tf_start
#    print "Full Matrix: %8.2f sec" %diff
    
    ts_start = time()

    
    x_l = []
    y_l = []
    data_l = []
    for ix in range(n_dofs-1):
        data_l.append(submtx)
    
    #print "X,Y ", x_a,y_a
    #print "data ", data_a
    #print "X,Y ",x_a,y_a
    #x_a = array([x_l]).flatten()
    #y_a = array([y_l]).flatten()
    #ix_arr = vstack((x_a,y_a))
    #print "ix_arr ", ix_arr
    #data = array(data_l).flatten()
    #print data.shape[0]
    sp_mtx = SysSparseMtx(n_dofs, el_dof_map )
     
    print('ij_map',sp_mtx.ij_map)
    
    sp_mtx.set_data( array( data_l ) )
    #sp_mtx.set_nond_row_zero(0)
    #sp_mtx.set_nond_col_zero(0)
    sp_mtx.set_nond_zero(0)
    print("class ", sp_mtx.__class__)
    #K_aa =  sp_mtx[0,:]       
    #sp_mtx[0] = 0.
    #sp_mtx[:,0] = 0.
    #sp_mtx[0,0] = 1.
    print("K_aa ",sp_mtx.get_diag_elem(2))
    U_s = sp_mtx.solve(rhs)
    print("U_s ", U_s)
    ts_end = time()
    difs = ts_end - ts_start
    print("Sparse Matrix: %8.2f sec" %difs)

#    for i in range(5,0,-1):
#        print "i ",i