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
# Created on Feb 16, 2010 by: rch

from traits.api import HasTraits, Interface

class IExType( Interface ):
    '''Read the data from the directory
    '''
    # specify inputs
    #
    
    # specify derived outputs
    #
    
    # define processing
    #
    def process_source_data( self ):
        pass

