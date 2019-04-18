'''

@author: rch
'''

from threading import Thread
from reporter import Reporter
from reporter.reporter import ReportStudy
from bmcs.model.model import BMCSModel
from bmcs.simulator import BMCSVizSheet
import traits.api as tr


class RunTimeLoopThread(Thread):
    '''Time loop thread responsible.
    '''

    def __init__(self, study, *args, **kw):
        super(RunTimeLoopThread, self).__init__(*args, **kw)
        self.daemon = True
        self.study = study

    def run(self):
        self.study.model.run()

    def pause(self):
        self.study.model.paused = True

    def stop(self):
        self.study.model.restart = True


class XBMCSStudy(ReportStudy):
    '''Combine the model with specification of outputs
    '''

    model = tr.Instance(BMCSModel)
    '''Model of the studied phoenomenon.
    '''

    viz_sheet = tr.Instance(BMCSVizSheet, ())
    '''Sheet for 2d visualization.
    '''

    input = tr.Property

    def _get_input(self):
        return self.model

    output = tr.Property

    def _get_output(self):
        return self.viz_sheet

    offline = tr.DelegatesTo('viz_sheet')
    n_cols = tr.DelegatesTo('viz_sheet')

    def _model_changed(self):
        self.model.set_ui_recursively(self)
        tline = self.model.tline
        self.viz_sheet.time_range_changed(tline.max)
        self.viz_sheet.time_changed(tline.val)

    run_thread = tr.Instance(RunTimeLoopThread)

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
        self.model.set_traits_with_metadata(self.enable_run,
                                            disable_on_run=True)

    start_event = tr.Event
    '''Event announcing the start of the calculation
    '''

    def _start_event_fired(self):
        print('START EVENT FIRED')
        self.viz_sheet.run_started()

    finish_event = tr.Event
    '''Event announcing the start of the calculation
    '''

    def _finish_event_fired(self):
        print('FINISH EVENT FIRED')
        self.viz_sheet.run_finished()

    def run(self):
        if self.running:
            return
        self.enable_stop = True
        self.run_thread = RunTimeLoopThread(self)
        self.run_thread.start()

    def join(self):
        '''Wait until the thread finishes
        '''
        self.run_thread.join()

    def pause(self):
        self.model.pause()

    def stop(self):
        self.model.stop()
        self.enable_stop = False

    def report_tex(self):
        r = Reporter(report_name=self.model.name,
                     input=self.model,
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
        self.model.add_viz2d(clname, name, **kw)
