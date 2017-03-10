'''
Created on Mar 7, 2017

@author: rch
'''

from viz2d import Viz2D


class TimeFunctionViz2D(Viz2D):
    '''Visualization adaptor for time function of a boundary condition'''

    def plot(self, ax, vot=0, *args, **kw):
        print 'IN Timefunction view'
        if self.vis2d.time_function:
            print 'ISSUING PLOT'
            self.vis2d.time_function.plot(ax)
            y_min, y_max = self.vis2d.time_function.yrange
            ax.plot([vot, vot], [y_min, y_max])
