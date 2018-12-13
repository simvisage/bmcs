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
# Created on Jun 2, 2010 by: rch

from etsproxy.traits.api import Interface, Str

#-----------------------------------------------------------------------------------
#                             RESPONSE FUNCTIONS                                  #
#-----------------------------------------------------------------------------------
class IRF( Interface ):
    """
    Abstract class representing the single response.
    As a realization any function class with the get_value member
    may be included.
    """

    # @todo: define the capability of the RF so that spirrid can set the
    # configuration of the execution accordingly.
    #
    # There is not a strict requirement that RF must implemented
    # both C and python version - but it must be able to tell the client which one is there
    #

    # C-evaluation of the response function
    C_code = Str

    def __call__( self ):
        '''Python evaluation of the response function
        '''
        pass
