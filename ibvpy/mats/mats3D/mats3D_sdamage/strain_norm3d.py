from traits.api import HasTraits, Float, Bool
from traitsui.api import View, Item

from numpy import where, zeros, dot, diag, linalg, sum
#from numpy.dual import *
from math import sqrt
from ibvpy.mats.mats3D.mats3D_tensor import map3d_sig_eng_to_mtx

class IStrainNorm3D(HasTraits):
    
    def get_f_trial(self, epsilon, D_el, E, nu, kappa): 
        raise NotImplementedError
    
    def get_dede(self,epsilon, D_el, E, nu):
        raise NotImplementedError

#from the maple shit damage_model_2D_euclidean_local_Code
class Euclidean(IStrainNorm3D):
    P_I = diag([1.,1.,1.,0.5,0.5,0.5])
       
    def get_f_trial(self, epsilon, D_el, E, nu, kappa): 
        '''
        Returns equivalent strain - kappa
        @param epsilon:
        @param D_el:
        @param E:
        @param kappa:
        '''
        return sqrt(dot(dot(epsilon,self.P_I), epsilon.T))-kappa

#    def get_dede(self, epsilon, D_el, E, nu):
#        dede = zeros(6)
#        t1 = pow(epsilon[0], 2.)
#        t3 = pow(epsilon[1], 2.)
#        t5 = pow(epsilon[2], 2.)
#        t7 = pow(epsilon[3], 2.)
#        t9 = pow(epsilon[4], 2.)
#        t11 = pow(epsilon[5], 2.)
#        t14 = sqrt(4. * t1 + 4. * t3 + 4. * t5 + 2. * t7 + 2. * t9 + 2. * t11)
#        t15 = 1. / t14
#        dede[0] = 2. * t15 * epsilon[0]
#        dede[1] = 2. * t15 * epsilon[1]
#        dede[2] = 2. * t15 * epsilon[2]
#        dede[3] = t15 * epsilon[3]
#        dede[4] = t15 * epsilon[4]
#        dede[5] = t15 * epsilon[5]
#        return dede



#from the maple shit damage_model_2D_euclidean_local_Code
class Energy(IStrainNorm3D):
        
    def get_f_trial(self, epsilon, D_el, E, nu, kappa):
        #print "time %8.2f sec"%diff
        return sqrt(1./E* dot(dot(epsilon,D_el), epsilon.T))-kappa
                     
#    def get_dede(self, epsilon, D_el, E, nu):
#        dede = zeros(3)
#        t1 = 1. / E
#        t2 = t1 * epsilon[0]
#        t3 = t2 * D_el[0][0]
#        t4 = t1 * epsilon[1]
#        t5 = t4 * D_el[1][0]
#        t8 = t2 * D_el[0][1]
#        t9 = t4 * D_el[1][1]
#        t12 = pow(epsilon[2], 2.)
#        t16 = sqrt(epsilon[0] * (t3 + t5) + epsilon[1] * (t8 + t9) + t12 * t1 * D_el[2][2])
#        t17 = 1. / t16
#        dede[0] = t17 * (2. * t3 + t5 + t4 * D_el[0][1]) / 2.
#        dede[1] = t17 * (t2 * D_el[1][0] + t8 + 2. * t9) / 2.
#        dede[2] = t17 * t1 * epsilon[2] * D_el[2][2]
#        return dede


#from the maple shit damage_model_2D_von_Mises_local_Code
class Mises(IStrainNorm3D):
    k = Float( 10.,
               label = "k",
               desc = "Shape Parameter")
   
    def get_f_trial(self, epsilon, D_el, E, nu, kappa):
        t1 = self.k - 1
        t2 = (epsilon[0] + epsilon[1] + epsilon[2])
        t4 = 1 / self.k
        t6 = 1 - 2 * nu
        t10 = t1 **2
        t11 = t6 **2
        t14 = t2 **2
        t16 = epsilon[0]**2
        t18 = epsilon[1]**2
        t20 = epsilon[2]**2
        t22 = epsilon[3]**2
        t24 = epsilon[4]**2
        t26 = epsilon[5]**2
        t32 = (1 + nu)**2
        t37 = sqrt((t10 / t11 * t14) + 12. *  self.k * (t16 / 2. + t18 / 2. + t20 / 2. + t22 / 4. + t24 / 4. + t26 / 4. - t14 / 6.) / t32)
        t40 = (t1 * t2 * t4 / t6) / 2. + t4 * t37 / 2.
        return t40 - kappa
    
    def get_dede(self, epsilon, D_el, E, nu):
        dede = zeros(6)
        t1 = self.k - 1
        t2 = 1 / self.k
        t5 = 1 - 2 * nu
        t8 = (t1 * t2 / t5) / 2.
        t9 = t1 * t1
        t10 = t5 * t5
        t12 = t9 / t10
        t13 = epsilon[0] + epsilon[1] + epsilon[2]
        t14 = t13 * t13
        t16 = epsilon[0]**2
        t18 = epsilon[1]**2
        t20 = epsilon[2]**2
        t22 = epsilon[3]**2
        t24 = epsilon[4]**2
        t26 = epsilon[5]**2
        t32 = (1 + nu)**2
        t33 = 1 / t32
        t37 = sqrt((t12 * t14) + 0.12e2 *  self.k * (t16 / 2. + t18 / 2. + t20 / 2. + t22 / 4. + t24 / 4. + t26 / 4. - t14 / 6.) *  t33)
        t38 = 1. / t37
        t39 = t2 * t38
        t41 = 2 * t12 * t13
        t43 = epsilon[1] / 3.
        t44 = epsilon[2] / 3.
        t54 = epsilon[0] / 3.
        dede[0] = t8 + t39 * (t41 + 12. * self.k * (2. / 3. * epsilon[0] - t43 - t44) * t33) / 4.
        dede[1] = t8 + t39 * (t41 + 12. * self.k * (2. / 3. * epsilon[1] - t54 - t44) * t33) / 4.
        dede[2] = t8 + t39 * (t41 + 12. * self.k * (2. / 3. * epsilon[2] - t54 - t43) * t33) / 4.
        dede[3] = 3. / 2. * t38 * epsilon[3] * t33
        dede[4] = 3. / 2. * t38 * epsilon[4] * t33
        dede[5] = 3. / 2. * t38 * epsilon[5] * t33
        return dede

        
      
class Rankine(IStrainNorm3D):
    '''
    computes main stresses and makes a norm of their positive part
    '''
        
    def get_f_trial(self, epsilon, D_el, E, nu, kappa):
        sigma_I = linalg.eigh(map3d_sig_eng_to_mtx(dot(D_el,epsilon)))[0]#main stresses
        eps_eqv = linalg.norm(where(sigma_I >= 0., sigma_I, zeros(3)))/E#positive part and norm
        return eps_eqv  - kappa
    
#    def get_dede(self, epsilon, D_el, E, nu):
#        dede = zeros(3)
#        t1 = D_el[0][0] / 2.
#        t2 = D_el[1][0] / 2.
#        t3 = pow(D_el[0][0], 2.)
#        t4 = pow(epsilon[0], 2.)
#        t6 = D_el[0][0] * epsilon[0]
#        t7 = D_el[0][1] * epsilon[1]
#        t13 = D_el[1][1] * epsilon[1]
#        t16 = pow(D_el[0][1], 2.)
#        t17 = pow(epsilon[1], 2.)
#        t19 = D_el[1][0] * epsilon[0]
#        t25 = pow(D_el[1][0], 2.)
#        t29 = pow(D_el[1][1], 2.)
#        t31 = pow(D_el[2][2], 2.)
#        t32 = pow(epsilon[2], 2.)
#        t35 = t3 * t4 + 2 * t6 * t7 - 2. * D_el[0][0] * t4 * D_el[1][0] - 2. * t6 * t13 + t16 * t17 - 2. * t7 * t19 - 2. * D_el[0][1] * t17 * D_el[1][1] + t25 * t4 + 2. * t19 * t13 + t29 * t17 + 4. * t31 * t32
#        t36 = sqrt(t35)
#        t37 = 1. / t36
#        t57 = t37 * (2. * t3 * epsilon[0] + 2. * D_el[0][0] * D_el[0][1] * epsilon[1] - 4. * t6 * D_el[1][0] - 2. * D_el[0][0] * D_el[1][1] * epsilon[1] - 2. * t7 * D_el[1][0] + 2. * t25 * epsilon[0] + 2. * D_el[1][0] * D_el[1][1] * epsilon[1]) / 4.
#        t59 = fabs(t6 / 2. + t7 / 2. + t19 / 2. + t13 / 2. + t36 / 2.) / (t6 / 2. + t7 / 2. + t19 / 2. + t13 / 2. + t36 / 2.)
#        t63 = 1. / E
#        t66 = D_el[0][1] / 2.
#        t67 = D_el[1][1] / 2.
#        t85 = t37 * (2. * t6 * D_el[0][1] - 2. * t6 * D_el[1][1] + 2. * t16 * epsilon[1] - 2. * D_el[0][1] * D_el[1][0] * epsilon[0] - 0.4e1 * t7 * D_el[1][1] + 2. * t19 * D_el[1][1] + 2. * t29 * epsilon[1]) / 4.
#        dede[0] = (t1 + t2 + t57 + t59 * (t1 + t2 + t57)) * t63 / 2.
#        dede[1] = (t66 + t67 + t85 + t59 * (t66 + t67 + t85)) * t63 / 2.
#        dede[2] = (t37 * t31 * epsilon[2] + t59 * t37 * t31 * epsilon[2]) * t63
#        return dede


class Mazars(IStrainNorm3D):
           
    def get_f_trial (self, epsilon, D_el, E, nu, kappa):
        epsilon_pp = where(epsilon >= 0., epsilon, zeros(6))#positive part
        return sqrt(1./E* dot(dot(epsilon_pp,D_el), epsilon_pp.T))-kappa
    



