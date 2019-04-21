'''
@author: rch
'''

from traits.api import \
    Int, Float, Type, Bool
from .tloop import \
    TLoop
from .tstep_bc import TStepBC


class TLoopImplicit(TLoop):
    '''Time loop with implicit time stepping controlling the newton
    based algorithms.
    '''

    tstep_type = Type(TStepBC)

    k_max = Int(100, enter_set=True, auto_set=False)

    acc = Float(1e-4, enter_set=True, auto_set=False)

    verbose = Bool(False, enter_set=True, auto_set=False)

    def eval(self):
        t_n1 = self.tline.val
        t_max = self.tline.max
        dt = self.tline.step

        if self.verbose:
            print('t:', end='')

        while t_n1 <= (t_max + 1e-8):
            if self.verbose:
                print('\t%5.2f' % t_n1, end='')

            k = 0
            self.tstep.t_n1 = t_n1
            # run the iteration loop
            while (k < self.k_max) and not self.user_wants_abort:
                if self.tstep.R_norm < self.acc:
                    if self.verbose:
                        print('(%g), ' % k, end='\n')
                    break
                self.tstep.make_iter()
                k += 1
            else:  # handle unfinished iteration loop
                if k >= self.k_max:  # add step size reduction
                    # no success abort the simulation
                    self.restart = True
                    print('')
                    raise StopIteration('Warning: '
                                        'convergence not reached in %g iterations' % k)
                return
            # accept the time step and record the state in history
            self.tstep.make_incr()
            # update the line - launches notifiers to subscribers
            self.tline.val = min(t_n1, self.tline.max)
            # set a new target time
            t_n1 += dt

        if self.verbose:
            print('')

        return
