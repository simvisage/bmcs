'''
Created on 08.03.2017

@author: cthoennessen
'''
from traits.has_dynamic_views import HasDynamicViews, DynamicView
from traits.api import HasTraits, Instance, Button, Int
from traitsui.api import View, Group, UItem


class MyDynamicView(HasDynamicViews):

    current_views = 1

    tab_1 = Int(1)
    ui_1 = Group('tab_1',
                 label='View 1',
                 _mdv_order=1)
    tab_2 = Int(2)
    ui_2 = Group('tab_2',
                 label='View 2')
    tab_3 = Int(3)
    ui_3 = Group('tab_3',
                 label='View 3')
    tab_4 = Int(4)
    ui_4 = Group('tab_4',
                 label='View 4')
    tab_5 = Int(5)
    ui_5 = Group('tab_5',
                 label='View 5')

    my_uis = [ui_1, ui_2, ui_3, ui_4, ui_5]

    def create_dynamic_view(self):
        dynam_view = DynamicView(
            name='mdv',
            # id='traitsui.demos.dynamic_views',
            keywords={
                'dock':       'tab',
                'height':     0.4,
                'width':      0.4,
                'resizable':  True,
                'scrollable': True,
            },
            use_as_default=True,
        )
        self.declare_dynamic_view(dynam_view)

    def add_tab(self):
        self.current_views += 1
        ui_next = self.my_uis[self.current_views - 1]
        ui_next._mdv_order = self.current_views
        
    def remove_tab(self):
        ui_cur = self.my_uis[self.current_views - 1]
        del ui_cur._mdv_order
        self.current_views -= 1


class MyDynamicViewsExample(HasTraits):

    mdv = Instance(MyDynamicView, ())
    b_initial = Button('Initial')
    b_add = Button('Add Tab')
    b_remove = Button('Remove Tab')
    
    initial = True

    view = View(
        UItem('b_initial',
              enabled_when = 'initial'),
        UItem('b_add',
              enabled_when = 'mdv.current_views < 5'),
        UItem('b_remove',
              enabled_when = 'mdv.current_views > 0')
    )

    def _b_initial_fired(self):
            self.mdv.create_dynamic_view()
            self.mdv.configure_traits()
            self.initial = False
    
    def _b_add_fired(self):        
            self.mdv.add_tab()
            self.mdv.configure_traits()
            
    def _b_remove_fired(self):        
            self.mdv.remove_tab()
            self.mdv.configure_traits()


if __name__ == '__main__':
    mdve = MyDynamicViewsExample()
    mdve.configure_traits()
