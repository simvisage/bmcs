'''

@author: rch
'''

from threading import Thread

from reporter import Reporter
from reporter.reporter import ReportStudy
from simulator.api import ISimulator
import traits.api as tr

from .bmcs_viz_sheet import BMCSVizSheet


class XRunTimeLoopThread(Thread):
    '''Time loop thread responsible.
    '''

    def __init__(self, study, *args, **kw):
        super(XRunTimeLoopThread, self).__init__(*args, **kw)
        self.daemon = True
        self.study = study

    def run(self):
        self.study.sim.run()

    def pause(self):
        self.study.sim.paused = True

    def stop(self):
        self.study.sim.restart = True


class BMCSStudy(ReportStudy):
    '''Combine the simulater with specification of outputs
    '''

    sim = tr.Instance(ISimulator)
    '''Model of the studied phoenomenon.
    '''

    viz_sheet = tr.Instance(BMCSVizSheet, ())
    '''Sheet for 2d visualization.
    '''

    input = tr.Property

    def _get_input(self):
        return self.sim

    output = tr.Property

    def _get_output(self):
        return self.viz_sheet

    offline = tr.DelegatesTo('viz_sheet')
    n_cols = tr.DelegatesTo('viz_sheet')

    def _sim_changed(self):
        self.sim.set_ui_recursively(self)
        tline = self.sim.tline
        self.viz_sheet.time_range_changed(tline.max)
        self.viz_sheet.time_changed(tline.val)

    running = tr.Bool(False)
    enable_run = tr.Bool(True)
    enable_pause = tr.Bool(False)
    enable_stop = tr.Bool(False)

    def _running_changed(self):
        '''If the simulation is running disable the run botton,
        enable the pause button and disable changes in all 
        input parameters.
        '''
        self.enable_run = not self.running
        self.enable_pause = self.running
        self.sim.set_traits_with_metadata(self.enable_run,
                                          disable_on_run=True)

    start_event = tr.Event
    '''Event announcing the start of the calculation
    '''

    def _start_event_fired(self):
        self.viz_sheet.run_started()

    finish_event = tr.Event
    '''Event announcing the start of the calculation
    '''

    def _finish_event_fired(self):
        self.viz_sheet.run_finished()

    def run(self):
        self.sim.run_thread()
        self.enable_stop = True

    def join(self):
        self.sim.join_thread()

    def pause(self):
        self.sim.pause()

    def stop(self):
        self.sim.stop()
        self.enable_stop = False

    def report_tex(self):
        r = Reporter(report_name=self.sim.name,
                     input=self.sim,
                     output=self.viz_sheet)
        r.write()
        r.show_tex()

    def report_pdf(self):
        r = Reporter(studies=[self])
        r.write()
        r.show_tex()
        r.run_pdflatex()
        r.show_pdf()

    def add_viz2d(self, clname, name, **kw):
        self.sim.add_viz2d(clname, name, **kw)
