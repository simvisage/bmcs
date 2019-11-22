r'''

Simulator implementation
========================

'''
from threading import Thread

from traits.api import \
    Instance, on_trait_change, Str, \
    Property, cached_property, Bool, List, Dict, provides
from view.ui.bmcs_tree_node import BMCSRootNode, itags_str

from .hist import Hist
from .i_hist import IHist
from .i_simulator import ISimulator
from .i_tloop import ITLoop
from .i_tstep import ITStep
from .tline import TLine
from .tloop_implicit import TLoopImplicit
from .tstep_bc import TStepBC


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


@provides(ISimulator)
class Simulator(BMCSRootNode):
    r'''Base class for simulators included in the BMCS Tool Suite.
    It implements the state dependencies within the simulation tree.
    It handles also the communication between the simulation and
    the user interface in several modes of interaction.
    '''
    tree_node_list = List([])

    def _tree_node_list_default(self):
        return [
            self.tline,
            # self.domains,
            #            self.bc
        ]

    def _update_node_list(self):
        self.tree_node_list = [
            self.tline,
            # self.domains,
            #            self.bc
        ]

    title = Str

    desc = Str

    @on_trait_change(itags_str)
    def _model_structure_changed(self):
        self.tloop.restart = True

    #=========================================================================
    # Spatial domain
    #=========================================================================
    domains = List([])
    r'''Spatial domain represented by a finite element discretization.
    providing the kinematic mapping between the linear algebra (vector and
    matrix) and field representation of the primary variables.
    '''

    #=========================================================================
    # TIME LINE
    #=========================================================================
    tline = Instance(TLine)
    r'''Time line defining the time range, discretization and state,  
    '''

    def _tline_default(self):
        return TLine(
            time_change_notifier=self.time_changed,
            time_range_change_notifier=self.time_range_changed
        )

    def time_changed(self, time):
        if not(self.ui is None):
            self.ui.viz_sheet.time_changed(time)

    def time_range_changed(self, tmax):
        self.tline.max = tmax
        if self.ui != None:
            self.ui.viz_sheet.time_range_changed(tmax)

    def set_tmax(self, time):
        self.time_range_changed(time)

    #=========================================================================
    # TIME LOOP
    #=========================================================================
    tloop = Property(Instance(ITLoop))
    r'''Time loop constructed based on the current model.
    '''
    @cached_property
    def _get_tloop(self):
        return TLoopImplicit(sim=self,
                             tstep=self.tstep,
                             hist=self.hist,
                             tline=self.tline)

    bc = List
    r'''Boundary conditions
    '''
    record = Dict
    r'''Recorded variables
    '''

    tstep = Property(Instance(ITStep))
    r'''Class representing the time step and state
    '''
    @cached_property
    def _get_tstep(self):
        return TStepBC(sim=self)

    def pause(self):
        self.tloop.paused = True
        self.join_thread()

    def stop(self):
        self.tloop.restart = True
        self.join_thread()

    #=========================================================================
    # HISTORY
    #=========================================================================
    hist = Property(Instance(IHist))
    r'''History representation of the model response.
    '''
    @cached_property
    def _get_hist(self):
        return Hist(sim=self)

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
