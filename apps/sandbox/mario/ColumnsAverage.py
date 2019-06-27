'''
Created on 21.06.2019

@author: hspartali
'''
import traits.api as tr
import traitsui.api as ui


class ColumnsAverage(tr.HasStrictTraits):

    column_names = tr.List(['force', 'disp 1', 'disp 2'])

    traits_view = ui.View(ui.Item(
        'column_names', editor=ui.SetEditor(
            values=['x', 'y'],
            name='column_names',
            can_move_all=False,
            left_column_title='Remaining columns',
            right_column_title='Average of..'), show_label=False)
    )


if False:
    columns_average_selector = tr.List(editor=ui.SetEditor(
        values='',
        can_move_all=False,
        left_column_title='Remaining columns',
        right_column_title='Average of..'), show_label=False)

    def set_columns_headers_list(self, columns_headers_list):

        self.columns_average_selector = tr.List(editor=ui.SetEditor(
            values=columns_headers_list,
            can_move_all=False,
            left_column_title='Remaining columns',
            right_column_title='Average of..'), show_label=False)

c = ColumnsAverage()
c.configure_traits()
