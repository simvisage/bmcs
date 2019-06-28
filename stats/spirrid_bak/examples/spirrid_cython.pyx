import numpy as np
cimport numpy as np
ctypedef np.double_t DTYPE_t
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def mu_q(double e_arr,np.ndarray[DTYPE_t, ndim=1] E_mod_flat,np.ndarray[DTYPE_t, ndim=1] theta_flat,np.ndarray[DTYPE_t, ndim=1] lambd_flat,np.ndarray[DTYPE_t, ndim=1] xi_flat,np.ndarray[DTYPE_t, ndim=1] A_flat):
    cdef double mu_q
    cdef double lambd, xi, E_mod, theta, A, eps = e_arr, dG, q
    cdef int i_lambd, i_xi, i_E_mod, i_theta, i_A
    mu_q = 0
    dG = 4.11523e-08
    for i_lambd from 0 <= i_lambd <30:
        lambd = lambd_flat[ i_lambd ]
        for i_xi from 0 <= i_xi <30:
            xi = xi_flat[ i_xi ]
            for i_E_mod from 0 <= i_E_mod <30:
                E_mod = E_mod_flat[ i_E_mod ]
                for i_theta from 0 <= i_theta <30:
                    theta = theta_flat[ i_theta ]
                    for i_A from 0 <= i_A <30:
                        A = A_flat[ i_A ]
                        
                        eps_ = ( eps - theta * ( 1 + lambd ) ) / ( ( 1 + theta ) * ( 1 + lambd ) )
                        # Computation of the q( ... ) function
                        if eps_ < 0 or eps_ > xi:
                            q = 0.0
                        else:
                            q = E_mod * A * eps_
                        
                            mu_q += q * dG





    return mu_q