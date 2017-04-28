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
# Created on Apr 20, 2010 by: rch

import sys

#--------------------------------------------------------------------------
# Helper methods specification
#--------------------------------------------------------------------------
def _find_class ( klass ):
    '''Helper method to import the specified class
    '''
    module = ''
    col    = klass.rfind( '.' )
    if col >= 0:
        module = klass[ : col ]
        klass = klass[ col + 1: ]

    mod = sys.modules.get( module )
    theClass = getattr( mod, klass, None )
    if (theClass is None) and (col >= 0):
        try:
            mod = __import__( module )
            for component in module.split( '.' )[1:]:
                mod = getattr( mod, component )

            theClass = getattr( mod, klass, None )
        except:
            pass

    return theClass
    
