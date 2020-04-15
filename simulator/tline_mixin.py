'''
Created on Mar 18, 2020

@author: rch
'''
import traits.api as tr

from .tline import TLine


class TLineMixIn(tr.HasTraits):
    #=========================================================================
    # TIME LINE
    #=========================================================================
    tline = tr.Instance(TLine)
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
