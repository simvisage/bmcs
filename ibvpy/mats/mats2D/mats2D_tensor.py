'''
Module providing subsidary functions for tensor mapping in 2D
'''

from numpy import \
     array, zeros, ix_, dot, hstack, vstack

from scipy.linalg import \
    inv


#---------------------------------------------------------------------------------------------
# Switch from engineering to tensorial notation 
#---------------------------------------------------------------------------------------------

def map2d_eps_eng_to_mtx( eps_eng ):
    '''
    Switch from engineering notation to tensor notation for strains in 2D
    '''        
    eps_mtx = array( [[eps_eng[0]     , eps_eng[2] / 2.],
                      [eps_eng[2] / 2., eps_eng[1]     ]] )
    return eps_mtx


def map2d_sig_eng_to_mtx( sig_eng ):
    '''
    Switch from engineering notation to tensor notation for stresses in 2D
    '''        
    sig_mtx = array( [[sig_eng[0] , sig_eng[2] ],
                      [sig_eng[2] , sig_eng[1] ]] )
    return sig_mtx


#---------------------------------------------------------------------------------------------
# Switch from tensorial to engineering notation  
#---------------------------------------------------------------------------------------------

def map2d_eps_mtx_to_eng( eps_mtx ):
    '''
    Switch from tensor notation to engineering notation strains in 2D
    '''       
    eps_eng = array([ eps_mtx[0,0], eps_mtx[1,1], 2.0 * eps_mtx[0,1] ])
    return eps_eng


def map2d_sig_mtx_to_eng( sig_mtx ):
    '''
    Switch from tensor notation to engineering notation for stresses in 2D
    '''        
    sig_eng = array([ sig_mtx[0,0], sig_mtx[1,1], sig_mtx[0,1] ])
    return sig_eng


#---------------------------------------------------------------------------------------------
# Subsidiary index mapping functions for rank-four to rank-two tensors
#---------------------------------------------------------------------------------------------

def map2d_ijkl2mn( i,j,k,l ):
    '''
    Map the four-rank indexes to the two-rank matrix using the major
    and minor symmetry.
    '''
    # 2D-case:
    # first two indices (ij)
    if i==0 and j==0:
        m = 0
    elif i==1 and j==1:
        m = 1
    elif (i==0 and j==1) or (i==1 and j==0):
        m = 2
        
    # second two indices (kl)
    if k==0 and l==0:
        n = 0
    elif k==1 and l==1:
        n = 1
    elif (k==0 and l==1) or (k==1 and l==0):
        n = 2
        
    return m,n


#---------------------------------------------------------------------------------------------
# Subsidiary mapping functions for rank-two to rank-four tensor 
#---------------------------------------------------------------------------------------------

def map2d_tns2_to_tns4( tns2 ):
    '''
    Map a matrix to a fourth order tensor assuming minor and major symmetry,
    e.g. D_mtx (3x3) in engineering notation to D_tns(2,2,2,2)).
    '''
    n_dim = 2
    tns4 = zeros([n_dim,n_dim,n_dim,n_dim])
    for i in range(0,n_dim):
        for j in range(0,n_dim):
            for k in range(0,n_dim):
                for l in range(0,n_dim):
                    tns4[i,j,k,l] = tns2[map2d_ijkl2mn(i,j,k,l)]
    return tns4

               
#---------------------------------------------------------------------------------------------
# Subsidiary mapping functions for rank-four to rank-two tensor 
#---------------------------------------------------------------------------------------------

def map2d_tns4_to_tns2( tns4 ):
    '''
    Map a fourth order tensor to a matrix assuming minor and major symmetry,
    e.g. D_tns(2,2,2,2) to D_mtx (3x3) in engineering notation.
    (Note: Explicit assignment of components used for speedup.)
    '''
    n_eng = 3
    tns2 = zeros([n_eng,n_eng])
    
    tns2[0,0] =             tns4[0,0,0,0]
    tns2[0,1] = tns2[1,0] = tns4[0,0,1,1]
    tns2[0,2] = tns2[2,0] = tns4[0,0,0,1]
    
    tns2[1,1] =             tns4[1,1,1,1]
    tns2[1,2] = tns2[2,1] = tns4[1,1,0,1]
    
    tns2[2,2] =             tns4[0,1,0,1]
    
    return tns2

#----------------------------------------------------------------------------------------------
# Compliance mapping 2D (used for inversion of the damage effect tensor in engineering notation 
#----------------------------------------------------------------------------------------------

def compliance_mapping2d( C_mtx_2d ):
    '''
    The components of the compliance matrix are multiplied with factor 1,2 or 4 depending on their 
    position in the matrix (due to symmetry and switching of tensorial to engineering notation
    (Note: gamma_xy = 2*epsilon_xy, etc.). Necessary for evaluation of D=inv(C). 
    '''
    idx1 = [0,1]
    idx2 = [2]
    C11 = C_mtx_2d[ix_(idx1,idx1)]
    C12 = C_mtx_2d[ix_(idx1,idx2)]
    C21 = C_mtx_2d[ix_(idx2,idx1)]
    C22 = C_mtx_2d[ix_(idx2,idx2)]
    return vstack( [ hstack( [C11,   2*C12] ),
                     hstack( [2*C21, 4*C22] ) ] )

#----------------------------------------------------------------------------------------------
# Reduce the 3D-elasticity/compliance matrix to the 2D-cases plane strain or plane stress 
#----------------------------------------------------------------------------------------------

def get_D_plane_stress( D_mtx_3d ):
    '''
    Reduce the 6x6-elasticity matrix for the 3D-case to a 
    3x3 matrix for the 2D case assuming plane stress (sig_yz=0, sig_zz=0, sig_xz=0)
    '''
    idx2 = [0,1,5]
    idx3 = [2,3,4]
    D22 = D_mtx_3d[ix_(idx2,idx2)]
    D23 = D_mtx_3d[ix_(idx2,idx3)]
    D32 = D_mtx_3d[ix_(idx3,idx2)]
    D33 = D_mtx_3d[ix_(idx3,idx3)]
    D_pstress_term = dot( dot( D23, inv( D33 ) ), D32 )
    D_mtx_2d = D22 - D_pstress_term
    return D_mtx_2d


def get_D_plane_strain( D_mtx_3d ):
    '''
    Reduce the 6x6-elasticity matrix for the 3D-case to a 
    3x3 matrix for the 2D case assuming plane strain (eps_yz=0, eps_zz=0, eps_xz=0)
    '''
    idx2 = [0,1,5]
    D_mtx_2d = D_mtx_3d[ix_(idx2,idx2)]
    return D_mtx_2d


def get_C_plane_stress( C_mtx_3d ):
    '''
    Reduce the 6x6-compliance matrix for the 3D-case to a 
    3x3 matrix for the 2D case assuming plane stress (sig_yz=0, sig_zz=0, sig_xz=0)
    '''
    idx2 = [0,1,5]
    C_mtx_2d = C_mtx_3d[ix_(idx2,idx2)]
    return C_mtx_2d


def get_C_plane_strain( C_mtx_3d ):
    '''
    Reduce the 6x6-compliance matrix for the 3D-case to a 
    3x3 matrix for the 2D case assuming plane strain (eps_yz=0, eps_zz=0, eps_xz=0)
    '''
    idx2 = [0,1,5]
    idx3 = [2,3,4]
    C22 = C_mtx_3d[ix_(idx2,idx2)]
    C23 = C_mtx_3d[ix_(idx2,idx3)]
    C32 = C_mtx_3d[ix_(idx3,idx2)]
    C33 = C_mtx_3d[ix_(idx3,idx3)]
    C_pstrain_term = dot( dot( C23, inv( C33 ) ), C32 )
    C_mtx_2d = C22 - C_pstrain_term
    return C_mtx_2d

if __name__ == '__main__':
    from numpy import array
    tn2 = array([[1,4,5],
                 [4,2,6],
                 [5,6,3]])
    print('tn2')
    print(tn2)

    tn4 = map2d_tns2_to_tns4(tn2)
    
    print('tns2_to_tns4')
    print(tn4)
    
    tn2a = map2d_tns4_to_tns2(tn4)
    
    print('back')
    print('tns4_to_tns2')
    print(tn2a)
    