r'''

Simulator implementation
========================

'''
from threading import Thread

from simulator.hist import Hist
from traits.api import \
    Instance, on_trait_change, Str, \
    Property, cached_property, Bool, List, Dict, provides
from view.ui.bmcs_tree_node import BMCSRootNode

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
        self.simulator = simulator

    def run(self):
        try:
            # start the calculation
            self.simulator.tloop()
        except Exception as e:
            self.simulator.running = False
            raise e  # re-raise exception
        self.simulator.running = False


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
        print('updating MATS explore', self.dim)
        self.tree_node_list = [
            self.tline,
            # self.domains,
            #            self.bc
        ]

    title = Str

    desc = Str

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
        return TLoopImplicit(tstep=self.tstep,
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
        self.join()

    def stop(self):
        self.tloop.restart = True
        self.join()

    #=========================================================================
    # HISTORY
    #=========================================================================
    hist = Instance(IHist)
    r'''History representation of the model response.
    '''

    def _hist_default(self):
        return Hist(sim=self)

    #=========================================================================
    # COMPUTATION THREAD
    #=========================================================================
    run_thread = Instance(RunTimeLoopThread)
    running = Bool(False)

    def run(self):
        r'''Run a thread if it does not exist - do nothing otherwise
        '''
        if self.running:
            return
        self.running = True

        print('RUN', self.ui)
        if self.ui:
            # inform ui that the simulation is running in a thread
            self.ui.start_event = True
            self.ui.running = True

        self.run_thread = RunTimeLoopThread(self)

        try:
            # start the calculation process
            self.run_thread.start()
        except Exception as e:
            if self.ui:
                self.ui.running = False
            raise e  # re-raise exception

        if self.ui:
            # cleanup ui and send the finish event
            self.ui.running = False
            self.ui.finish_event = True

    def join(self):
        r'''Wait until the thread finishes
        '''
        self.run_thread.join()

    @on_trait_change('MAT,ALG,CS,GEO,BC,+BC')
    def signal_reset(self):
        '''Upon the change of the model parameters,
        signal the user interface that further calculation
        does not make sense.
        '''
        if self.ui:
            self.ui.stop()
