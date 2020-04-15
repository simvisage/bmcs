r'''

Simulator implementation
========================

'''
from threading import Thread

from traits.api import \
    Instance, on_trait_change, Str, \
    Property, cached_property, Bool, List, provides
from view import BMCSTreeNode, itags_str, IBMCSModel

import traits.api as tr

from .i_tloop import ITLoop
from .i_tstep import ITStep
from .tline_mixin import TLineMixIn


class RunTimeLoopThread(Thread):
    r'''Thread launcher class used to issue a calculation.
    in an independent thread.
    '''

    def __init__(self, simulator, *args, **kw):
        super(RunTimeLoopThread, self).__init__(*args, **kw)
        self.daemon = True
        self.sim = simulator

    def run(self):
        self.sim.run()
        return


@provides(IBMCSModel)
class Simulator(BMCSTreeNode, TLineMixIn):
    r'''Base class for simulators included in the BMCS Tool Suite.
    It implements the state dependencies within the simulation tree.
    It handles also the communication between the simulation and
    the user interface in several modes of interaction.
    '''
    tree_node_list = List([])

    def _tree_node_list_default(self):
        return [
            self.tline,
        ]

    def _update_node_list(self):
        self.tree_node_list = [
            self.tline,
        ]

    title = Str

    desc = Str

    @on_trait_change(itags_str)
    def _model_structure_changed(self):
        self.tloop.restart = True

    #=========================================================================
    # TIME LOOP
    #=========================================================================

    tloop = Property(Instance(ITLoop), depends_on=itags_str)
    r'''Time loop constructed based on the current model.
    '''
    @cached_property
    def _get_tloop(self):
        return self.tstep.tloop_type(tstep=self.tstep,
                                     tline=self.tline)

    tstep = tr.WeakRef(ITStep)

    hist = tr.Property

    def _get_hist(self):
        return self.tstep.hist

    def pause(self):
        self.tloop.paused = True
        self.join_thread()

    def stop(self):
        self.tloop.restart = True
        self.join_thread()

    #=========================================================================
    # COMPUTATION THREAD
    #=========================================================================
    _run_thread = Instance(RunTimeLoopThread)
    _running = Bool(False)

    def run(self):
        r'''Run a thread if it does not exist - do nothing otherwise
        '''
        self._running = True

        if self.ui:
            # inform ui that the simulation is running in a thread
            self.ui.start_event = True
            self.ui.running = True

        try:
            # start the calculation
            self.tloop()
        except Exception as e:
            self._running = False
            if self.ui:
                self.ui.running = False
            raise e  # re-raise exception

        self._running = False

        if self.ui:
            # cleanup ui and send the finish event
            self.ui.running = False
            self.ui.finish_event = True

    def run_thread(self):
        r'''Run a thread if it does not exist - do nothing otherwise
        '''
        if self._running:
            return

        self._run_thread = RunTimeLoopThread(self)
        self._run_thread.start()

    def join_thread(self):
        r'''Wait until the thread finishes
        '''
        if self._run_thread == None:
            self._running = False
            return
        self._run_thread.join()

    @on_trait_change(itags_str)
    def signal_reset(self):
        '''Upon the change of the model parameters,
        signal the user interface that further calculation
        does not make sense.
        '''
        if self.ui:
            self.ui.stop()
