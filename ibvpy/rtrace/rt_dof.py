
import os
import pickle
from traits.api import \
    Array, List, Callable, \
    Instance, Int, Str, \
    ToolbarButton, on_trait_change
from traitsui.api import \
    Item, View, HGroup, VGroup, VSplit, HSplit, Spring
from traitsui.menu import \
    OKButton, CancelButton

from ibvpy.api import RTrace
from mathkit.mfn import MFnLineArray
from mathkit.mfn.mfn_line.mfn_matplotlib_editor import MFnMatplotlibEditor
from mathkit.mfn.mfn_line.mfn_plot_adapter import MFnPlotAdapter
import numpy as np


class RTraceGraph(RTrace):

    '''
    Collects two response evaluators to make a response graph.

    The supplied strings for var_x and var_y are used to locate the rte in
    the current response manager. The bind method is used to navigate to
    the rte and is stored in here as var_x_eval and var_y_val as Callable
    object.

    The request for new response evaluation is launched by the time loop
    and directed futher by the response manager. This method is used solely
    for collecting the data, not for their visualization in the viewer.

    The timer_tick method is invoked when the visualization of the Graph
    should be synchronized with the actual contents.
    '''

    label = Str('RTraceGraph')
    var_x = Str('')
    var_x_eval = Callable(trantient=True)
    idx_x_arr = Array
    idx_x = Int(-1, enter_set=True, auto_set=False)
    var_y = Str('')
    var_y_eval = Callable(trantient=True)
    idx_y_arr = Array
    idx_y = Int(-1, enter_set=True, auto_set=False)
    transform_x = Str(enter_set=True, auto_set=False)
    transform_y = Str(enter_set=True, auto_set=False)

    trace = Instance(MFnLineArray)

    def _trace_default(self):
        return MFnLineArray()

    print_button = ToolbarButton('Print Values',
                                 style='toolbar', trantient=True)

    @on_trait_change('print_button')
    def print_values(self, event=None):
        print 'x:\t', self.trace.xdata, '\ny:\t', self.trace.ydata

    view = View(VSplit(VGroup(HGroup(VGroup(HGroup(Spring(), Item('var_x', style='readonly'),
                                                   Item('idx_x', show_label=False)),
                                            Item('transform_x')),
                                     VGroup(HGroup(Spring(), Item('var_y', style='readonly'),
                                                   Item('idx_y', show_label=False)),
                                            Item('transform_y')),
                                     VGroup('record_on', 'clear_on')),
                              HGroup(Item('refresh_button', show_label=False),
                                     Item('print_button', show_label=False)),
                              ),
                       Item('trace@',
                            editor=MFnMatplotlibEditor(
                                adapter=MFnPlotAdapter(var_x='var_x',
                                                       var_y='var_y',
                                                       min_size=(100, 100))),
                            show_label=False, resizable=True),
                       ),
                buttons=[OKButton, CancelButton],
                resizable=True,
                scrollable=True,
                height=0.5, width=0.5)

    _xdata = List(Array(float))
    _ydata = List(Array(float))

    def bind(self):
        '''
        Locate the evaluators
        '''
        self.var_x_eval = self.rmgr.rte_dict.get(self.var_x, None)
        if self.var_x_eval == None:
            raise KeyError, 'Variable %s not present in the dictionary:\n%s' % \
                            (self.var_x, self.rmgr.rte_dict.keys())

        self.var_y_eval = self.rmgr.rte_dict.get(self.var_y, None)
        if self.var_y_eval == None:
            raise KeyError, 'Variable %s not present in the dictionary:\n%s' % \
                            (self.var_y, self.rmgr.rte_dict.keys())

    def setup(self):
        self.clear()

    def close(self):
        self.write()

    def write(self):
        '''Generate the file name within the write_dir
        and submit the request for writing to the writer
        '''
        # self.writer.scalars_name = self.name
        file_base_name = 'rtrace_diagramm_%s (%s,%s).dat' % \
            (self.label, self.var_x, self.var_y)
        # full path to the data file
        file_name = os.path.join(self.dir, file_base_name)
        # file_rtrace = open( file_name, 'w' )
        self.refresh()
        np.savetxt(
            file_name, np.vstack([self.trace.xdata, self.trace.ydata]).T)
        # pickle.dump( self, file_rtrace )
        # file.close()

    def add_current_values(self, sctx, U_k, *args, **kw):
        '''
        Invoke the evaluators in the current context for the specified control vector U_k.
        '''
        x = self.var_x_eval(sctx, U_k, *args, **kw)
        y = self.var_y_eval(sctx, U_k, *args, **kw)

        self.add_pair(x.flatten(), y.flatten())

    def add_pair(self, x, y):
        self._xdata.append(np.copy(x))
        self._ydata.append(np.copy(y))

    @on_trait_change('idx_x,idx_y')
    def redraw(self, e=None):
        if ((self.idx_x < 0 and len(self.idx_x_arr) == 0) or
                (self.idx_y < 0 and len(self.idx_y_arr) == 0) or
                self._xdata == [] or
                self._ydata == []):
            return
        #
        if len(self.idx_x_arr) > 0:
            print 'x: summation for', self.idx_x_arr
            xarray = np.array(self._xdata)[:, self.idx_x_arr].sum(1)
        else:
            xarray = np.array(self._xdata)[:, self.idx_x]

        if len(self.idx_y_arr) > 0:
            print 'y: summation for', self.idx_y_arr
            yarray = np.array(self._ydata)[:, self.idx_y_arr].sum(1)

#            print 'yarray', yarray
#            yarray_arr = array( self._ydata )[:, self.idx_y_arr]
#            sym_weigth_arr = 2. * ones_like( yarray_arr[1] )
#            sym_weigth_arr[0] = 4.
#            print 'yarray_arr', yarray_arr
#            print 'sym_weigth_arr', sym_weigth_arr
#            yarray = dot( yarray_arr, sym_weigth_arr )
#            print 'yarray', yarray

        else:
            yarray = np.array(self._ydata)[:, self.idx_y]

        if self.transform_x:
            def transform_x_fn(x):
                '''makes a callable function out of the Str-attribute
                "transform_x". The vectorised version of this function is
                then used to transform the values in "xarray". Note that
                the function defined in "transform_x" must be defined in
                terms of a lower case variable "x".
                '''
                return eval(self.transform_x)
            xarray = np.frompyfunc(transform_x_fn, 1, 1)(xarray)

        if self.transform_y:
            def transform_y_fn(y):
                '''makes a callable function out of the Str-attribute
                "transform_y". The vectorised version of this function is
                then used to transform the values in "yarray". Note that
                the function defined in "transform_y" must be defined in
                terms of a lower case variable "y".
                '''
                return eval(self.transform_y)
            yarray = np.frompyfunc(transform_y_fn, 1, 1)(yarray)

        self.trace.xdata = np.array(xarray)
        self.trace.ydata = np.array(yarray)
        self.trace.data_changed = True

    def timer_tick(self, e=None):
        # @todo: unify with redraw
        pass

    def clear(self):
        self._xdata = []
        self._ydata = []
        self.trace.clear()
        self.redraw()


class RTraceArraySnapshot(RTrace):

    '''
    Plot the current value of the array along the x_axis

    Used currently for plotting the integrity factor over microplanes
    '''
    var = Str('')
    var_eval = Callable
    idx = Int(-1)

    trace = Instance(MFnLineArray)

    x = Array(float)
    y = Array(float)

    def _trace_default(self):
        return MFnLineArray()

    view = View(HSplit(VGroup(VGroup('var'),
                              VGroup('record_on', 'clear_on')),
                       Item('trace@', style='custom',
                            editor=MFnMatplotlibEditor(
                                adapter=MFnPlotAdapter(var_y='var')),
                            show_label=False, resizable=True),
                       ),
                buttons=[OKButton, CancelButton],
                resizable=True)

    def bind(self):
        '''
        Locate the evaluators
        '''
        self.var_eval = self.rmgr.rte_dict[self.var]

    def setup(self):
        pass

    def add_current_values(self, sctx, U_k, *args, **kw):
        '''
        Invoke the evaluators in the current context for the specified control vector U_k.
        '''
        self.y = np.array(self.var_eval(sctx, U_k, *args, **kw), dtype='float')
        self.x = np.arange(0, len(self.y), dtype='float')
        self.redraw()

    def timer_tick(self, e=None):
        self.redraw()

    def redraw(self):
        self.trace.xdata = self.x
        self.trace.ydata = self.y

    def clear(self):
        # @todo:
        self.x = np.zeros((0,), dtype='float_')
        self.y = np.zeros((0,), dtype='float_')

if __name__ == '__main__':

    rm1 = RTraceGraph(name='rte 1',
                      idx_x=0,
                      idx_y=0,
                      transform_x='-x')
#                       transform_x = lambda x: -x )

#    print rm1.dir
    rm1.add_pair(np.array([0.0, 0.0]), np.array([0.0, 0.0]))
    rm1.add_pair(np.array([1.0, 1.0]), np.array([1.0, 1.5]))
    rm1.add_pair(np.array([2.0, 1.5]), np.array([2.0, 2.0]))
    rm1.redraw()
    rm1.configure_traits()

    filename = '/tmp/testx'
    f = open(filename, 'w')
    pickle.dump(rm1, f)
    f.close()
