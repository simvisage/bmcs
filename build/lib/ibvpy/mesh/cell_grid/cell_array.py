
#-- Imports --------------------------------------------------------------

from numpy \
    import sqrt
from numpy.random \
    import random
from traits.api \
    import HasTraits, Property, Array, Any, Event, \
    on_trait_change, Instance, WeakRef, Int, Str, Bool, Trait, \
    Interface, provides
from traitsui.api \
    import View, Item, TabularEditor, HSplit, Group
from traitsui.menu \
    import CancelButton
from traitsui.tabular_adapter \
    import TabularAdapter


#------------------------------------------------------------------------
# Source of the data for array view
#------------------------------------------------------------------------
class ICellArraySource(Interface):

    '''Object representing a structured 1,2,3-dimensional cell grid.
    '''
    pass

#------------------------------------------------------------------------
# Cell view interface
#------------------------------------------------------------------------


class ICellView(Interface):

    '''Interface of the general cell view.
    '''

    def set_cell_traits(self):
        '''Adapt the view to the newly set cell_idx.
        '''
        raise NotImplementedError

    def redraw(self):
        '''Redraw the graphical representation of the cell 
        in the mayavi pipeline.
        '''
        raise NotImplementedError

#------------------------------------------------------------------------
# Default implementation of the cell view
#------------------------------------------------------------------------


@provides(ICellView)
class CellView(HasTraits):

    '''Get the element numbers.
    '''

    cell_idx = Int(-1)

    cell_grid = WeakRef(ICellArraySource)

    def set_cell(self, cell_idx):
        '''Method to be overloaded by subclasses. The subclass 
        can fetch the data required from the cell_grid 
        '''
        self.cell_idx = cell_idx
        self.set_cell_traits()

    def set_cell_traits(self):
        '''Specialize this function to fetch the cell data from 
        the array source.
        '''
        pass

    def redraw(self):
        '''No plotting defined by default'''
        pass

    view = View(Item('cell_idx'),
                resizable=True)

#-- Tabular Adapter Definition -------------------------------------------


class ElemTabularAdapter (TabularAdapter):

    columns = Property

    def _get_columns(self):
        data = getattr(self.object, self.name)
        if len(data.shape) != 2:
            raise ValueError('element node array must be two-dimensional')
        n_columns = getattr(self.object, self.name).shape[1]

        cols = [(str(i), i) for i in range(n_columns)]
        return [('element', 'index')] + cols

    font = 'Courier 10'
    alignment = 'right'
    format = '%d'
    index_text = Property
#    index_image = Property

    def _get_index_text(self):
        return str(self.row)

    def x_get_index_image(self):
        x, y, z = self.item
        if sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2) <= 0.25:
            return 'red_flag'
        return None

#-- Tabular Editor Definition --------------------------------------------


elem_tabular_editor = TabularEditor(
    adapter=ElemTabularAdapter(),
    selected_row='current_row',
)

#-- CellArrayView Class Definition ---------------------------------------


class CellArray(HasTraits):

    data = Array

    cell_view = Instance(ICellView)

    def _cell_view_default(self):
        return CellView()

    current_row = Int(-1)

    @on_trait_change('current_row')
    def redraw(self):
        self.cell_view.redraw()

    @on_trait_change('current_row')
    def _display_current_row(self):
        if self.current_row != -1:
            self.cell_view.set_cell(self.current_row)

    view = View(
        HSplit(
            Item('data', editor=elem_tabular_editor,
                 show_label=False, style='readonly'),
            Group(Item('cell_view@', show_label=False),
                  )),
        title='Cell Array View',
        width=0.6,
        height=0.4,
        resizable=True,
        buttons=[CancelButton]
    )


# Run the demo (if invoked from the command line):
if __name__ == '__main__':

    from traits.api import Button

    class Container(HasTraits):
        show_array = Button

        def _show_array_fired(self):
            # Create the demo:
            demo = CellArray(data=random((100000, 3)))
            demo.configure_traits()
        view = View('show_array')

    c = Container()
    c.configure_traits()
