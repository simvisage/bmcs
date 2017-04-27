from numpy import \
     zeros, dot, hstack, vstack, identity, cross, transpose, tensordot, outer

from scipy.linalg import \
     inv, norm

from ibvpy.mats.mats2D.mats2D_tensor import \
    map2d_eps_mtx_to_eng, map2d_sig_eng_to_mtx

from ibvpy.mats.mats3D.mats3D_tensor import \
    map3d_eps_eng_to_mtx, map3d_sig_eng_to_mtx, map3d_sig_mtx_to_eng, map3d_eps_mtx_to_eng, map3d_tns2_to_tns4, map3d_tns4_to_tns2, map3d_eps_mtx_to_eng

def get_Trans_strain_from_glb_to_loc(dircos_mtx):
    '''Transformation matrix in 3d shape = (6,6).
    (eps_mtx_loc = T * eps_mtx_glb * T^t)
    Multiply the global strain vector 'eps_vct_glb' (in engineering notation) with the returned
    transformation matrix 'Trans_strain_from_glb_to_loc_mtx' in order to obtain the loacal strain vector
    (in engineering notation) in the local coordinate system (x',y',z'), e.g. in-plan shell directions.
    The ordering of the strain vector in engineering notation corresponds to the VOIGT-notation:
    eps_glb = [eps_xx, eps_yy, eps_zz, eps_yz, eps_zx, eps_xz].
    The parameter 'dircos_mtx' is the matrix of the direction cosines describing the relation between
    the global and local coordinate system.
    (the returned transformation matrix corresponds to Zienkiewicz Eq.(6.24) but used VOIGT-notation instead)
    '''
    t = dircos_mtx  # shape=(3,3)
    T = zeros((6, 6))  # shape=(6,6)
    T[0, 0] = t[0, 0] * t[0, 0]
    T[0, 1] = t[0, 1] * t[0, 1]
    T[0, 2] = t[0, 2] * t[0, 2]
    T[0, 3] = t[0, 1] * t[0, 2]
    T[0, 4] = t[0, 0] * t[0, 2]
    T[0, 5] = t[0, 0] * t[0, 1]
    T[1, 0] = t[1, 0] * t[1, 0]
    T[1, 1] = t[1, 1] * t[1, 1]
    T[1, 2] = t[1, 2] * t[1, 2]
    T[1, 3] = t[1, 1] * t[1, 2]
    T[1, 4] = t[1, 0] * t[1, 2]
    T[1, 5] = t[1, 0] * t[1, 1]
    T[2, 0] = t[2, 0] * t[2, 0]
    T[2, 1] = t[2, 1] * t[2, 1]
    T[2, 2] = t[2, 2] * t[2, 2]
    T[2, 3] = t[2, 1] * t[2, 2]
    T[2, 4] = t[2, 0] * t[2, 2]
    T[2, 5] = t[2, 0] * t[2, 1]
    T[3, 0] = 2.0 * t[2, 0] * t[1, 0]
    T[3, 1] = 2.0 * t[2, 1] * t[1, 1]
    T[3, 2] = 2.0 * t[2, 2] * t[1, 2]
    T[3, 3] = t[2, 1] * t[1, 2] + t[2, 2] * t[1, 1]
    T[3, 4] = t[2, 2] * t[1, 0] + t[2, 0] * t[1, 2]
    T[3, 5] = t[2, 0] * t[1, 1] + t[2, 1] * t[1, 0]
    T[4, 0] = 2.0 * t[2, 0] * t[0, 0]
    T[4, 1] = 2.0 * t[2, 1] * t[0, 1]
    T[4, 2] = 2.0 * t[2, 2] * t[0, 2]
    T[4, 3] = t[2, 1] * t[0, 2] + t[2, 2] * t[0, 1]
    T[4, 4] = t[2, 2] * t[0, 0] + t[2, 0] * t[0, 2]
    T[4, 5] = t[2, 0] * t[0, 1] + t[2, 1] * t[0, 0]
    T[5, 0] = 2.0 * t[1, 0] * t[0, 0]
    T[5, 1] = 2.0 * t[1, 1] * t[0, 1]
    T[5, 2] = 2.0 * t[1, 2] * t[0, 2]
    T[5, 3] = t[1, 1] * t[0, 2] + t[1, 2] * t[0, 1]
    T[5, 4] = t[1, 2] * t[0, 0] + t[1, 0] * t[0, 2]
    T[5, 5] = t[1, 0] * t[0, 1] + t[1, 1] * t[0, 0]
    Trans_strain_to_loc_mtx = T
    return Trans_strain_to_loc_mtx

def get_Trans_stress_from_loc_to_glb(dircos_mtx):
    '''Transformation matrix in 3d shape = (6,6)
    (sig_mtx_glb = T^t * sig_mtx_loc * T)
    Multiply the local stress vector 'sig_vct_loc' (in engineering notation in the local coordinate
    system (x',y', z') with the returned transformation matrix 'Trans_stress_from_loc_to_glb_mtx' in
    order to obtain the global stress vector 'sig_vct_glb' (in engineering notation) in the global
    coordinate system (x,y,z). (cf. Zienkiewicz Eq.(6.23) multiplied with T_t*(...)*T, for T being an orthonormal basis)
    The ordering of the stress vector in engineering notation corresponds to the VOIGT-notation:
    sig_vct_glb = [sig_xx, sig_yy, sig_zz, sig_yz, sig_zx, sig_xz].
    The parameter 'dircos_mtx' is the matrix of the direction cosines describing the relation between
    the global and the local coordinate system.
    '''
    t = dircos_mtx  # shape=(3,3)
    T = zeros((6, 6))  # shape=(6,6)
    T[0, 0] = t[0, 0] * t[0, 0]
    T[0, 1] = t[1, 0] * t[1, 0]
    T[0, 2] = t[2, 0] * t[2, 0]
    T[0, 3] = 2.0 * t[1, 0] * t[2, 0]
    T[0, 4] = 2.0 * t[0, 0] * t[2, 0]
    T[0, 5] = 2.0 * t[0, 0] * t[1, 0]
    T[1, 0] = t[0, 1] * t[0, 1]
    T[1, 1] = t[1, 1] * t[1, 1]
    T[1, 2] = t[2, 1] * t[2, 1]
    T[1, 3] = 2.0 * t[1, 1] * t[2, 1]
    T[1, 4] = 2.0 * t[0, 1] * t[2, 1]
    T[1, 5] = 2.0 * t[0, 1] * t[1, 1]
    T[2, 0] = t[0, 2] * t[0, 2]
    T[2, 1] = t[1, 2] * t[1, 2]
    T[2, 2] = t[2, 2] * t[2, 2]
    T[2, 3] = 2.0 * t[1, 2] * t[2, 2]
    T[2, 4] = 2.0 * t[0, 2] * t[2, 2]
    T[2, 5] = 2.0 * t[0, 2] * t[1, 2]
    T[3, 0] = t[0, 2] * t[0, 1]
    T[3, 1] = t[1, 2] * t[1, 1]
    T[3, 2] = t[2, 2] * t[2, 1]
    T[3, 3] = t[1, 2] * t[2, 1] + t[2, 2] * t[1, 1]
    T[3, 4] = t[2, 2] * t[0, 1] + t[0, 2] * t[2, 1]
    T[3, 5] = t[0, 2] * t[1, 1] + t[1, 2] * t[0, 1]
    T[4, 0] = t[0, 2] * t[0, 0]
    T[4, 1] = t[1, 2] * t[1, 0]
    T[4, 2] = t[2, 2] * t[2, 0]
    T[4, 3] = t[1, 2] * t[2, 0] + t[2, 2] * t[1, 0]
    T[4, 4] = t[2, 2] * t[0, 0] + t[0, 2] * t[2, 0]
    T[4, 5] = t[0, 2] * t[1, 0] + t[1, 2] * t[0, 0]
    T[5, 0] = t[0, 1] * t[0, 0]
    T[5, 1] = t[1, 1] * t[1, 0]
    T[5, 2] = t[2, 1] * t[2, 0]
    T[5, 3] = t[1, 1] * t[2, 0] + t[2, 1] * t[1, 0]
    T[5, 4] = t[2, 1] * t[0, 0] + t[0, 1] * t[2, 0]
    T[5, 5] = t[0, 1] * t[1, 0] + t[1, 1] * t[0, 0]
    Trans_stress_from_loc_mtx = T
    return Trans_stress_from_loc_mtx
