from scipy.optimize.zeros import brentq

from bmcs.time_functions import \
    LoadingScenario
from ibvpy.bcond import BCDof
from ibvpy.mats.mats3D.mats3D_plastic.vmats3D_desmorat import \
    MATS3DDesmorat
from simulator.api import Simulator, TLoop

from .interaction_scripts import run_rerun_test


class BrentqTLoop(TLoop):

    def eval(self):
        t_n1 = self.tline.val
        t_max = self.tline.max
        dt = self.tline.step
        while t_n1 <= t_max:
            print('\ttime: %g' % t_n1, end='')
            k = 0
            self.tstep.t_n1 = t_n1
            # run the iteration loop
            brentq
            while (k < self.k_max) and not self.user_wants_abort:
                if self.tstep.R_norm < self.acc:
                    print('\titer: %g' % k)
                    break
                self.tstep.make_iter()
                k += 1
            else:  # handle unfinished iteration loop
                if k >= self.k_max:  # add step size reduction
                    # no success abort the simulation
                    self.restart = True
                    print('Warning: '
                          'convergence not reached in %g iterations' % k)
                return
            # accept the time step and record the state in history
            self.tstep.make_incr()
            # update the line - launches notifiers to subscribers
            self.tline.val = t_n1
            # set a new target time
            t_n1 += dt
        return


class MATS3DDesmoratGrad(MATS3DDesmorat):
    tloop_type = BrentqTLoop


s = Simulator(
    model=MATS3DDesmorat()
)
bc = BCDof(
    var='u', dof=0, value=-0.001,
    time_function=LoadingScenario()
)
s.tstep.bcond_mngr.bcond_list = [bc]
run_rerun_test(s)
