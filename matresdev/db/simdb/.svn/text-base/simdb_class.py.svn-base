#-------------------------------------------------------------------------------
#
# Copyright (c) 2009, IMB, RWTH Aachen.
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in simvisage/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.simvisage.com/licenses/BSD.txt
#
# Thanks for using Simvisage open source!
#
# Created on Apr 15, 2010 by: rch

from enthought.traits.api import \
    HasTraits, Dict, Str, Enum, Instance, Int, Class, Type, \
    Directory, List, Property, Float, cached_property

from enthought.traits.ui.api import \
    View, Item, Tabbed, VGroup, HGroup, ModelView, HSplit, VSplit, \
    CheckListEditor, EnumEditor, TableEditor, TabularEditor, Handler

from enthought.traits.ui.menu import Action, CloseAction, HelpAction, Menu, \
                                     MenuBar, NoButtons, Separator, ToolBar

from enthought.traits.ui.tabular_adapter \
    import TabularAdapter

from enthought.traits.ui.table_column import \
    ObjectColumn

from enthought.traits.ui.tabular_adapter \
    import TabularAdapter, AnITabularAdapter

from promod.simdb import \
    SimDB

import os
import pickle
import string

# Access to the toplevel directory of the database
#
simdb = SimDB()


class SimDBTableAdapter ( TabularAdapter ):

    columns = Property

    #---------------------------------------------------------------
    # EXTRACT FACTOR NAMES AS COLUMNS FOR TABLE EDITOR
    #-------------------------------------------------------------------
    def _get_columns( self ):
        cols = [ ( 'index', 'index' ) ] # , ('key', 'dbkey') ]
        obj = self.object
        for field_name in obj.field_names:
            cols.append( ( field_name, field_name ) )
        return cols

    selection_mode = 'rows',

    font = 'Courier 10'
    alignment = 'right'
    odd_bg_color = 'lightblue'

    index_width = Float( 50 )

    index_text = Property
    def _get_index_text ( self ):
        return str( self.row )

    key_width = Float( 120 )
    key_text = Property
    def _get_key_text( self ):
        factor_idx = self.column - 1
        value = self.object.instances[ self.row, factor_idx ]
        return str( value )

simdb_table_editor = TabularEditor( adapter = SimDBTableAdapter(),
                                    selected = 'selected_instance'
                                     )

class SimDBClass( HasTraits ):
    '''Base class for instances storable as in SimDBClassExt'''

    key = Str()

    def save( self ):
        '''Let the object save itself
        '''
        self.db[ self.key ] = self

#------------------------------------------------------------------------------------------
# Class Extension - global persistent container of class instances
#------------------------------------------------------------------------------------------
class SimDBClassExt( HasTraits ):

    category = Enum( 'matdata', 'exdata' )
    def _category_default( self ):
        return 'matdata'

    path = List( [] )

    # dictionary of predefined instances - used for 
    # debugging and early stages of class developmemnt.
    #
    klass = Type

    classname = Property( depends_on = 'klass' )
    @cached_property
    def _get_classname( self ):
        return self.klass.__name__

    field_names = Property
    def _get_field_names( self ):
        '''
        Get the dictionary of factors provided by the simulation model.
        
        The factors are identified by the factor_levels metadata in the trait
        definition. For example
        
        my_param = Float( 20, factor_levels = (0, 10, 6) )
        
        specifies a float factor with  the levels [0,2,4,6,8,10]
        '''
        names = self.klass.class_trait_names( simdb = lambda x: x != None )
        traits = self.klass.class_traits( simdb = lambda x: x != None )
        return traits.keys()

    constants = Dict( {} )

    keyed_constants = Property( List )
    def _get_keyed_constants( self ):
        for key, c in self.constants.items():
            c.key = key
        return self.constants

    dirname = Str
    def _dirname_default( self ):
        '''Name of the directory for the data of the class
        '''
        klass_dir = self.klass.__name__
        full_path = ( simdb.simdb_dir, self.category ) \
                      + tuple( self.path ) + ( klass_dir, )
        path = os.path.join( *full_path )
        return path

    dir = Directory()
    def _dir_default( self ):
        '''Directory for the data of the class
        '''
        # foolproof creation of the directory
        try:
            os.makedirs( self.dirname )
        except OSError:
            if os.path.exists( self.dirname ):
                # We are nearly safe
                pass
            else:
                # There was an error on creation, so make sure we know about it
                raise
        return self.dirname

    instances = Dict
    def _instances_default( self ):
        '''Read the content of the directory
        '''
        instances = {}
        for obj_file_name in os.listdir( self.dir ):
            # check to see whether the file is pickle or not
            path = os.path.join( self.dir, obj_file_name )
            if not os.path.isfile( path ):
                continue
            obj_file = open( path, 'r' )
            key_list = string.split( obj_file_name, '.' )[:-1]
            key = reduce( lambda x, y: x + '.' + y, key_list )
            instances[ key ] = pickle.load( obj_file )
            # let the object know its key
            instances[ key ].key = key
            obj_file.close()
        return instances

    inst_list = Property
    def _get_inst_list( self ):
        return self.keyed_constants.values() + self.instances.values()

    selected_instance = Instance( SimDBClass )
    def _selected_instance_default( self ):
        if len( self.inst_list ) > 0:
            return self.inst_list[0]
        else:
            return None

    def keys( self ):
        return self.keyed_constants.keys() + self.instances.keys()

    def get( self, name, Missing ):
        it = self.keyed_constants.get( name, Missing )
        if it == Missing:
            it = self.instances.get( name, Missing )
        return it

    def __setitem__( self, key, value ):
        ''' Save the instance with the specified key.
        '''
        # check if the key corresponds to a constant
        # if yes, report an error
        it = self.keyed_constants.get( key, None )
        if it:
            raise ValueError, 'attempting to change a constant %s' % key
        else:
            for x in string.whitespace:
                key = key.replace( x, "_" )
            # write to the database
            value.key = key
            obj_file_name = os.path.join( self.dir, key + '.pickle' )
            obj_file = open( obj_file_name, 'w' )
            pickle.dump( value, obj_file, protocol = 0 ) # slow text mode
            obj_file.close()
            # register the object in the memory as well
            self.instances[ key ] = value

    def __getitem__( self, key ):
        ''' Return the instance with the specified key.
        '''
        it = self.keyed_constants.get( key, None )
        if it == None:
            it = self.instances.get( key, None )
            if it == None:
                raise ValueError, 'No database object with the key %s for class %s' % ( key, self.classname )
        return it

    def __delitem__( self, key ):
        # check if the key corresponds to a constant
        # if yes, report an error
        it = self.keyed_constants.get( key, None )
        if it:
            raise ValueError, 'attempting to delete a constant %s' % key
        else:
            for x in string.whitespace:
                key = key.replace( x, "_" )
            # write to the database
            obj_file_name = os.path.join( self.dir, key + '.pickle' )
            if os.path.exists( obj_file_name ):
                os.remove( obj_file_name )
            del self.instances[ key ]

    #---------------------------------------------------------------------------------
    # VIEW
    #---------------------------------------------------------------------------------

    traits_view = View( 
                    HSplit( 
                           VSplit( 
                                   VGroup( 
                                         HGroup( Item( 'classname',
                                                     style = 'readonly',
                                                     label = 'database extension class' )
                                                ),
                                         Item( 'inst_list',
                                               editor = simdb_table_editor,
                                               show_label = False,
                                               style = 'custom' ),
                                         label = 'database table',
                                         id = 'simbd.table.instances',
                                         dock = 'tab',
                                         ),
                                   id = 'simdb.table.left',
                                 ),
                           VGroup( 
                               VGroup( 
                                     Item( 'selected_instance@',
                                           resizable = True,
                                           show_label = False ),
                                     label = 'instance',
                                     id = 'simdb.table.instance',
                                     dock = 'tab',
                                     scrollable = True,
                                     ),
                                id = 'simdb.table.right',
                                layout = 'split',
                                label = 'selected instance',
                                dock = 'tab',
                                ),
                            id = 'simdb.table.splitter',
                        ),
                        id = 'simdb.table',
                        dock = 'tab',
                        resizable = True,
                        buttons = ['OK', 'Cancel'],
                        height = 0.8, width = 0.8,
                        )

if __name__ == '__main__':

    class MyDB( SimDBClass ):
        v1 = Int( 10, simdb = True )
        v2 = Int( 20, simdb = True )

    MyDB.db = SimDBClassExt( klass = MyDB )

    class MyDB2( SimDBClass ):
        my_key = Str( simdb = True )
        my_ref = Property
        def _get_my_ref( self ):
            return MyDB.db[ self.my_key ]

    MyDB2.db = SimDBClassExt( klass = MyDB2 )

    MyDB.db.constants = { 'object_1' : MyDB( v1 = 10, v2 = 40 ),
                        'object_2' : MyDB( v1 = 10, v2 = 50 ),
                        'object_3' : MyDB( v1 = 10, v2 = 60 ) }

    print MyDB.db['object_1'].v2
    print MyDB.db['object_2'].v2

    #MyDB.db.configure_traits()

    MyDB.db['object 4'] = MyDB( v1 = 100, v2 = 200 )
    print MyDB.db['object_4'].v2
    print MyDB.db.instances

    print 'read from the database'
    #print MyDB2.db['MDBO1'].my_ref.key

    #MyDB2.db['MDBO1'] = MyDB2( my_key = 'object_4' )
    #print MyDB2.db.instances

#    print 'v2'
#    print MyDB2.db['MDBO1'].my_ref.v2
#    
#    print MyDB2.db['MDBO1'].my_ref.key
#    
#    MyDB2.db['MDBO1'].my_key = 'object_2'
#    
#    print MyDB2.db['MDBO1'].my_ref.key
#    
#    MyDB2.db['MDBO1'].save()
#    

    MyDB2.db.configure_traits()
