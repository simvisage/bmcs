#
# YTTB - Yarn Tensile Test Bowser
#
# - tasks - browsing the directory and reading
# specification of column types
# [rch] - now it is a list of tuples specifying the trait types and column name
# selection and filtering - what should be plotted
# [rch] - filters included - next step would be to gather the levels of
#         individual factors and providing a selection lists to construct
#         the filter
# pickup of the characteristic values in the curve
# [rch] - which action model to choose. and the persistence model
# - maximum
# - ascending branch
# - polynomial fit
# - derivative
# - integral up to maximum
# - total integral (rate effects)
# - derivative - stiffness
# - maximum stiffness
# - strain at peak stress
# - strain at peak stiffness

# potential plugins - resp-surface plotter

import csv
import os
from os.path import exists
from string import replace

from numpy import array
from numpy import loadtxt, argmax, polyfit, poly1d, frompyfunc

from enthought.traits.api import \
    HasTraits, Directory, List, Int, Float, Any, \
    on_trait_change, File, Constant, Instance, Trait, \
    Array, Str, Property, cached_property, WeakRef, \
    Dict, Button
from enthought.traits.ui.api import \
    View, Item, DirectoryEditor, TabularEditor, HSplit, VGroup, \
    TableEditor
from enthought.traits.ui.file_dialog  \
    import open_file, FileInfo, TextInfo, ImageInfo
from enthought.traits.ui.menu import \
    OKButton, CancelButton
from enthought.traits.ui.table_column import \
    ObjectColumn
from enthought.traits.ui.table_filter \
    import EvalFilterTemplate, MenuFilterTemplate, RuleFilterTemplate, \
    EvalTableFilter
from enthought.traits.ui.tabular_adapter \
    import TabularAdapter
from enthought.util.home_directory import \
    get_home_directory


#-- Tabular Adapter Definition -------------------------------------------
def fcomma2fdot(x): return float(replace(x, ',', '.'))


class ExDesignSpec(HasTraits):
    ''' Specification of the experiment design.

    Defines the parameters varied in the experiment design.
    The parameters can be read from the file included in the design.
    '''

    factors = []
    data_converters = {0: fcomma2fdot,
                       1: fcomma2fdot}

    design_file = Str('exdesign_num.csv')
    data_dir = Str('data')
    data_file_name = " '%s %dnm %d.TRA' % (self.embedding, self.torque, self.rep_number ) "


class ExRun(HasTraits):
    '''
    Represent a single test specifying the design parameters.
    and access to the measured data.
    '''
    data_dir = Str
    exdesign_reader = WeakRef

    def __init__(self, exdesign_reader, row, **kw):
        '''Retrieve the traits from the exdesign reader
        '''
        self.exdesign_reader = exdesign_reader
        factors = self.exdesign_reader.exdesign_spec.factors
        for idx, ps in enumerate(factors):
            cmd = '%s( %s("%s") )' % (ps[0], ps[1], row[idx])
            self.add_trait(ps[2], eval(cmd))
        super(ExRun, self).__init__(**kw)

    data_file = File

    def _data_file_default(self):
        return os.path.join(self.data_dir, self._get_file_name())

    @on_trait_change('data_file')
    def _reset_data_file(self):
        self.data_file = os.path.join(self.data_dir, self._get_file_name())

    def _get_file_name(self):
        return str(self.std_num) + '.txt'

    def _get_file_name(self):
        fname = eval(self.exdesign_reader.exdesign_spec.data_file_name)
        print('fname', fname)
        return fname

    _arr = Property(Array(float), depends_on='data_file')

    def _get__arr(self):
        return loadtxt(self.data_file, skiprows=2,
                       delimiter=self.exdesign_reader.exdesign_spec.data_delimiter,
                       converters=self.exdesign_reader.exdesign_spec.data_converters)

    xdata = Property(Array(float), depends_on='data_file')

    @cached_property
    def _get_xdata(self):
        return self._arr[:, 0]

    ydata = Property(Array(float), depends_on='data_file')

    @cached_property
    def _get_ydata(self):
        return self._arr[:, 1]

    max_stress_idx = Property(Int)

    def _get_max_stress_idx(self):
        return argmax(self._get_ydata())

    max_stress = Property(Float)

    def _get_max_stress(self):
        return self.ydata[self.max_stress_idx]

    strain_at_max_stress = Property(Float)

    def _get_strain_at_max_stress(self):
        return self.xdata[self.max_stress_idx]

    # get the ascending branch of the response curve
    xdata_asc = Property(Array(float))

    def _get_xdata_asc(self):
        return self.xdata[:self.max_stress_idx + 1]

    ydata_asc = Property(Array(float))

    def _get_ydata_asc(self):
        return self.ydata[:self.max_stress_idx + 1]

    # interplate the polynomial
    polyfit = Property(Array(float))

    def _get_polyfit(self):
        #
        # define x array to evaluate the derivatives properly
        #
        #xdarr = arange(0, self.xdata_asc[-1], 0.01)
        #
        # get the fit with 10-th-order polynomial
        #
        p = polyfit(self.xdata_asc, self.ydata_asc, 5)
        #
        # define the polynomial function
        #
        pf = poly1d(p)
        #
        # define universal function for the value
        # (used just for visualization)
        #
        pfun = frompyfunc(pf, 1, 1)

        approx_dat = array(pfun(self.xdata_asc), dtype=float)
        return approx_dat

    traits_view = View(Item('data_dir', style='readonly'),
                       Item('max_stress_idx', style='readonly'),
                       Item('max_stress', style='readonly'),
                       Item('strain_at_max_stress', style='readonly'),
                       )


#-------------------------------------------------------------------------
# ExDesignReader
#-------------------------------------------------------------------------
exrun_table_editor = TableEditor(
    columns_name='exdesign_table_columns',
    selection_mode='rows',
    selected='object.selected_exruns',
    #selected_indices  = 'object.selected_exruns',
    auto_add=False,
    configurable=True,
    sortable=True,
    sort_model=True,
    auto_size=False,
    filters=[EvalFilterTemplate, MenuFilterTemplate, RuleFilterTemplate],
    search=EvalTableFilter())


class EXDesignReader(HasTraits):
    '''Read the data from the directory

    The design is described in semicolon-separated
    csv file providing the information about
    design parameters.

    Each file has the name n.txt
    '''

    #--------------------------------------------------------------------
    # Specification of the design - factor list, relative paths, etc
    #--------------------------------------------------------------------
    open_exdesign = Button()

    def _open_exdesign_fired(self):
        file_name = open_file(filter=['*.eds'],
                              extensions=[FileInfo(), TextInfo()])
        if file_name != '':
            self.exdesign_spec_file = file_name

    exdesign_spec_file = File

    def _exdesign_spec_file_changed(self):
        print('changed file')
        f = file(self.exdesign_spec_file)
        str = f.read()
        self.exdesign_spec = eval('ExDesignSpec( %s )' % str)

    exdesign_spec = Instance(ExDesignSpec)

    def _exdesign_spec_default(self):
        return ExDesignSpec()

    @on_trait_change('exdesign_spec')
    def _reset_design_file(self):
        dir = os.path.dirname(self. exdesign_spec_file)
        exdesign_file = self.exdesign_spec.design_file
        self.design_file = os.path.join(dir, exdesign_file)

    #--------------------------------------------------------------------
    # file containing the association between the factor combinations
    # and data files having the data
    #--------------------------------------------------------------------
    design_file = File

    def _design_file_changed(self):
        self.exdesign = self._read_exdesign()

    exdesign_table_columns = Property(List, depends_on='exdesign_spec+')

    @cached_property
    def _get_exdesign_table_columns(self):
        return [ObjectColumn(name=ps[2],
                             editable=False,
                             width=0.15) for ps in self.exdesign_spec.factors]

    exdesign = List(Any)

    def _exdesign_default(self):
        return self._read_exdesign()

    def _read_exdesign(self):
        ''' Read the experiment design. 
        '''
        if exists(self.design_file):
            reader = csv.reader(open(self.design_file, 'r'), delimiter=';')

            data_dir = os.path.join(os.path.dirname(self.design_file),
                                    self.exdesign_spec.data_dir)

            return [ExRun(self, row, data_dir=data_dir) for row in reader]
        else:
            return []

    selected_exrun = Instance(ExRun)

    def _selected_exrun_default(self):
        if len(self.exdesign) > 0:
            return self.exdesign[0]
        else:
            return None

    last_exrun = Instance(ExRun)

    selected_exruns = List(ExRun)

    #------------------------------------------------------------------
    # Array plotting
    #-------------------------------------------------------------------
    # List of arrays to be plotted
    data = Instance(AbstractPlotData)

    def _data_default(self):
        return ArrayPlotData(x=array([]), y=array([]))

    @on_trait_change('selected_exruns')
    def _rest_last_exrun(self):
        if len(self.selected_exruns) > 0:
            self.last_exrun = self.selected_exruns[-1]

    @on_trait_change('selected_exruns')
    def _reset_data(self):
        '''
        '''
        runs, xlabels, ylabels, ylabels_fitted = self._generate_data_labels()
        for name in list(self.plot.plots.keys()):
            self.plot.delplot(name)

        for idx, exrun in enumerate(self.selected_exruns):
            if xlabels[idx] not in self.plot.datasources:
                self.plot.datasources[xlabels[idx]] = ArrayDataSource(exrun.xdata,
                                                                      sort_order='none')
            if ylabels[idx] not in self.plot.datasources:
                self.plot.datasources[ylabels[idx]] = ArrayDataSource(exrun.ydata,
                                                                      sort_order='none')

            if ylabels_fitted[idx] not in self.plot.datasources:
                self.plot.datasources[ylabels_fitted[idx]] = ArrayDataSource(exrun.polyfit,
                                                                             sort_order='none')

        for run, xlabel, ylabel, ylabel_fitted in zip(runs, xlabels, ylabels, ylabels_fitted):
            self.plot.plot((xlabel, ylabel), color='brown')
            self.plot.plot((xlabel, ylabel_fitted), color='blue')

    def _generate_data_labels(self):
        ''' Generate the labels consisting of the axis and run-number.
        '''
        return ([e.std_num for e in self.selected_exruns],
                ['x-%d' % e.std_num for e in self.selected_exruns],
                ['y-%d' % e.std_num for e in self.selected_exruns],
                ['y-%d-fitted' % e.std_num for e in self.selected_exruns])

    plot = Instance(Plot)

    def _plot_default(self):
        p = Plot()
        p.tools.append(PanTool(p))
        p.overlays.append(ZoomTool(p))
        return p

    view_traits = View(HSplit(VGroup(Item('open_exdesign',
                                          style='simple'),
                                     Item('exdesign',
                                          editor=exrun_table_editor,
                                          show_label=False, style='custom')
                                     ),
                              VGroup(Item('last_exrun@',
                                          show_label=False),
                                     Item('plot',
                                          editor=ComponentEditor(),
                                          show_label=False,
                                          resizable=True
                                          ),
                                     ),
                              ),
                       #                        handler = EXDesignReaderHandler(),
                       resizable=True,
                       buttons=[OKButton, CancelButton],
                       height=1.,
                       width=1.)


doe_reader = EXDesignReader()
doe_reader.configure_traits(view='view_traits')
