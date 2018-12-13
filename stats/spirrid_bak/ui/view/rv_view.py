from etsproxy.traits.api import HasTraits, Str, List, Instance, Float, \
    WeakRef, Int, Bool, String, Trait, Property, cached_property, on_trait_change
from etsproxy.traits.ui.api import View, Item, VSplit, \
    HGroup, TableEditor, Spring, ModelView
from etsproxy.traits.ui.extras.checkbox_column import CheckboxColumn
from etsproxy.traits.ui.table_column import ObjectColumn

from stats.pdistrib.pdistrib import PDistrib, IPDistrib

from quaducom.resp_func.brittle_filament import Filament

from stats.spirrid_bak.old.spirrid import SPIRRID

from stats.spirrid_bak.old.spirrid import RV

# ---------------------------------
#    EDITOR FOR RANDOM VARIABLES
# ---------------------------------

class PDColumn( ObjectColumn ):
    name = 'value'

    def get_image( self, object ):
        if object.pd:
            return object.pd.icon
        else:
            return self.image


rv_list_editor = TableEditor( 
                    columns = [ ObjectColumn( name = 'varname', label = 'Name',
                                                editable = False,
                                                horizontal_alignment = 'center' ),
                                  CheckboxColumn( name = 'random', label = 'Random',
                                                editable = True,
                                                horizontal_alignment = 'center' ),
                                  PDColumn( label = 'Value',
                                                editable = False,
                                                horizontal_alignment = 'center' ),
                                  ObjectColumn( name = 'n_int', label = 'NIP',
                                                editable = True,
                                                format = '%d',
                                                horizontal_alignment = 'center' ),
 ],
                    selection_mode = 'row',
                    selected = 'object.selected_var',
                    deletable = False,
                    editable = False,
                    show_toolbar = True,
                    auto_add = False,
                    configurable = False,
                    sortable = False,
                    reorderable = False,
                    sort_model = False,
                    orientation = 'vertical',
                    auto_size = True,
        )


# -------------
#    RV VIEW
# -------------

class RIDVariable( HasTraits ):
    """
    Association between a random variable and distribution.
    """

    title = Str( 'RIDvarible' )

    s = WeakRef

    rf = WeakRef

    n_int = Int( 20, enter_set = True, auto_set = False,
                 desc = 'Number of integration points' )
    def _n_int_changed( self ):
        if self.pd:
            self.pd.n_segments = self.n_int

    # should this variable be randomized

    random = Bool( False, randomization_changed = True )
    def _random_changed( self ):
        # get the default distribution
        if self.random:
            self.s.rv_dict[ self.varname ] = RV( pd = self.pd, name = self.varname, n_int = self.n_int )
        else:
            del self.s.rv_dict[ self.varname ]

    # name of the random variable (within the response function)
    #
    varname = String

    source_trait = Trait

    trait_value = Float

    pd = Property( Instance( IPDistrib ), depends_on = 'random' )
    @cached_property
    def _get_pd( self ):
        if self.random:
            tr = self.rf.trait( self.varname )
            pd = PDistrib( distr_choice = tr.distr[0], n_segments = self.n_int )
            trait = self.rf.trait( self.varname )

            # get the distribution parameters from the metadata
            #
            distr_params = {'scale' : trait.scale, 'loc' : trait.loc, 'shape' : trait.shape }
            dparams = {}
            for key, val in list(distr_params.items()):
                if val:
                    dparams[key] = val

            pd.distr_type.set( **dparams )
            return pd
        else:
            return None

    value = Property
    def _get_value( self ):
        if self.random:
            return ''
        else:
            return '%g' % self.trait_value

    # --------------------------------------------

    # default view specification
    def default_traits_view( self ):
        return View( HGroup( Item( 'n_int', visible_when = 'random', label = 'NIP',
                                        ),
                                 Spring(),
                                 show_border = True,
                                 label = 'Variable name: %s' % self.varname
                                 ),
                    Item( 'pd@', show_label = False ),
                    resizable = True,
                    id = 'rid_variable',
                    height = 800 )


class RVModelView( ModelView ):

    '''
    ModelView class for displaying the table of parameters and
    set the distribution parameters of random variables
    '''

    title = Str( 'randomization setup' )

    model = Instance( SPIRRID )

    rv_list = List( RIDVariable )
    @on_trait_change( 'model.rf' )
    def get_rv_list( self ):
        self.rv_list = [ RIDVariable( s = self.model, rf = self.model.rf,
                              varname = nm, trait_value = st )
                 for nm, st in zip( self.model.rf.param_keys, self.model.rf.param_values ) ]

    selected_var = Instance( RIDVariable )
    def _selected_var_default( self ):
        return self.rv_list[0]

    title = Str( 'random variable editor' )

    selected_var = Instance( RIDVariable )

    traits_view = View( VSplit( 

                                HGroup( 
                                    Item( 'rv_list', editor = rv_list_editor, show_label = False ),
                                    id = 'rid.tview.randomization.rv',
                                    label = 'Model variables',
                                ),
                                HGroup( 
                                    Item( 'selected_var@', show_label = False, resizable = True ),
                                    id = 'rid.tview.randomization.distr',
                                    label = 'Distribution',
                                ),

                                scrollable = True,
                                id = 'rid.tview.tabs',
                                dock = 'tab',
                        ),

                        title = 'RANDOM VARIABLES',
                        id = 'rid.ridview',
                        dock = 'tab',
                        resizable = True,
                        height = 1.0, width = 1.0
                        )

if __name__ == '__main__':
    s = SPIRRID( rf = Filament() )
    r = RVModelView( model = s )
    r.configure_traits()
