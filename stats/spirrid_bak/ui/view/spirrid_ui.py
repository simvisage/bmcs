from etsproxy.traits.api import HasTraits, Str, List, Instance, Property
from etsproxy.traits.ui.ui_traits import Image
from etsproxy.traits.ui.api import TreeEditor, TreeNode, View, Item, Group

#from stats.spirrid_bak.spirrid_tree_structure.model.spirrid import SPIRRID
from stats.spirrid_bak.old.spirrid import SPIRRID
from stats.spirrid_bak import RFModelView
from stats.spirrid_bak import RVModelView
from stats.spirrid_bak import ResultView
from stats.spirrid_bak import SPIRRIDModelView

from quaducom.resp_func.brittle_filament import Filament

# ---------------------------------------
#    TREE STRUCTURE FOR SPIRRID VIEW
# ---------------------------------------


if __name__ == '__main__':

    # CREATE INSTANCES #
    default_rf = Filament()
    spirrid = SPIRRID( rf = default_rf, implicit_var_eval = True )

    spirrid_view = SPIRRIDModelView( model = spirrid )

    result_view = ResultView( spirrid_view = spirrid_view )

    # defining classes for tree nodes at first and second level

    class SPIRRIDUI ( HasTraits ):
        title = Str( 'simvisage SPIRRID' )
        image = Image( 'pics/spirrid.png' )
        comp_parts = List()

    class Preprocessor( HasTraits ):
        title = Str( 'preprocessor' )
        prep_components = Property( List )
        def _get_prep_components( self ):
            rf_view = RFModelView( model = default_rf,
                                   child = spirrid )
            rf_view._redraw()
            rv_view = RVModelView( model = spirrid )
            return [ rf_view, rv_view ]

    class Solver( HasTraits ):
        title = Str( 'solver' )
        solver_components = List()

    class Postprocessor( HasTraits ):
        title = Str( 'postprocessor' )
        post_components = List()

    # View for objects that aren't edited
    no_view = View()

    # Tree editor 
    tree_editor = TreeEditor( 
        nodes = [

            TreeNode( node_for = [ SPIRRIDUI ],
                      auto_open = True,
                      children = 'comp_parts',
                      label = 'title',
                      view = View( Item( 'image',
                                  show_label = False ) ) ),

                TreeNode( node_for = [ Preprocessor ],
                          auto_open = True,
                          children = 'prep_components',
                          label = 'title',
                          view = no_view ),

                    TreeNode( node_for = [ RFModelView ],
                              auto_open = False,
                              label = 'title'
                              ),
                    TreeNode( node_for = [ RVModelView ],
                              auto_open = False,
                              label = 'title',
                              ),

                TreeNode( node_for = [ Solver ],
                          auto_open = True,
                          children = 'solver_components',
                          label = 'title',
                          view = no_view ),

                    TreeNode( node_for = [ SPIRRIDModelView ],
                              auto_open = True,
                              label = 'title',
                              ),

                TreeNode( node_for = [ Postprocessor ],
                          auto_open = True,
                          children = 'post_components',
                          label = 'title',
                          view = no_view ),
                    TreeNode( node_for = [ ResultView ],
                              auto_open = True,
                              label = 'title',
                              ),
                    ]
                        )

    class STree ( HasTraits ):
        title = Str
        company = Instance( SPIRRIDUI )

        # The main view
        view = View( 
                   Group( 
                       Item( 
                            name = 'company',
                            id = 'company',
                            editor = tree_editor,
                            resizable = True,
                            show_label = False ),
                        orientation = 'vertical',
                        show_labels = True,
                        show_left = True, ),
                    title = 'SPIRRID',
                    id = \
                     'tree_editor_spirrid',
                    dock = 'horizontal',
                    drop_class = HasTraits,
                    buttons = [ 'Undo', 'OK', 'Cancel' ],
                    resizable = True,
                    width = .3,
                    height = .3 )


    # instance of the tree structure class including
    # instances and lists of instances

    tree = STree( 
        name = 'simvisage',
        company = SPIRRIDUI( 
            comp_parts = [ Preprocessor(),
                            Solver( 
                            solver_components = [spirrid_view]
                                    ),
                            Postprocessor( 
                            post_components = [result_view] )
                            ],
                            )
                )

    tree.configure_traits()

