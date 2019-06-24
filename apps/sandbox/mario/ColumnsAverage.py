'''
Created on 21.06.2019

@author: hspartali
'''
import traits.api as tr
import traitsui.api as ui


class ColumnsAverage(tr.HasStrictTraits):
            
    columns_names = ['force', 'disp 1', 'disp 2']
          
    columns_average_selector = tr.List(editor=ui.SetEditor(
        values=columns_names,
        can_move_all=False,
        left_column_title='Remaining columns',
        right_column_title='Average of..'), show_label=False)
      
    def set_columns_headers_list(self, columns_headers_list):
        
        self.columns_average_selector = tr.List(editor=ui.SetEditor(
            values=columns_headers_list,
            can_move_all=False,
            left_column_title='Remaining columns',
            right_column_title='Average of..'), show_label=False)
