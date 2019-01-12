'''
@author: rch
'''

from traits.api import \
    Int, Float, Bool
from bmcs.simulator import \
    TLoop


class TLoopImplicit(TLoop):
    '''Time loop with implicit time stepping controlling the newton
    based algorithms.
    '''

    k_max = Int(30, enter_set=True, auto_set=False)

    acc = Float(1e-4, enter_set=True, auto_set=False)

    def eval(self):
        t_n1 = self.tline.val
        t_max = self.tline.max
        dt = self.tline.step
        while t_n1 <= t_max:
            print('\ttime: %g' % t_n1, end='')
            k = 0
            self.tstep.t_n1 = t_n1
            # run the iteration loop
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
