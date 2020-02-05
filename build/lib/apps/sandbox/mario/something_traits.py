'''
Created on Jun 7, 2019

@author: rch
'''
import traits.api as tr
import traitsui.api as ui


class Something(tr.HasTraits):

    txt_file_name = tr.File

    openTxt = tr.Button('Open...')

    a = tr.Int(20, auto_set=False, enter_set=True,
               input=True)

    b = tr.Float(20, auto_set=False, enter_set=True,
                 input=True)

    @tr.on_trait_change('+input')
    def _handle_input_change(self):
        print('some input parameter changed')
        self.input_event = True

    input_event = tr.Event

    def _some_event_changed(self):
        print('input happend')

    def _openTxt_fired(self):
        print('do something')
        print(self.txt_file_name)

    traits_view = ui.View(
        ui.VGroup(
            ui.HGroup(
                ui.Item('openTxt', show_label=False),
                ui.Item('txt_file_name', width=200),
                ui.Item('a')
            ),
        )
    )


if __name__ == '__main__':
    s = Something()

    s.a = 10

    s.configure_traits()
