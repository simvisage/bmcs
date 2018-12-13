
q = fiber_tt_5p(eps,lambd,xi,E_mod,theta,A)
** evars:
eps = [0, ..., 0.04] (1000)
** tvars[n_int = 15]:
lambd = norm( loc = 0.1, scale = 0.02, shape = 1)[n_int = None]
xi = norm( loc = 0.019027, scale = 0.0022891, shape = 1)[n_int = None]
E_mod = norm( loc = 7e+10, scale = 1.5e+10, shape = 1)[n_int = None]
theta = 0.005
A = norm( loc = 5.3e-10, scale = 1e-11, shape = 1)[n_int = None]
** sampling: T-grid
** codegen: numpy
 numpy
var_eval: False
