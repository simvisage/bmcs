'''
Created on Feb 11, 2018

@author: rch
'''

import os

from mayavi.sources.vtk_xml_file_reader import VTKXMLFileReader
from tvtk.api import write_data

from ibvpy.mats.viz2d_field import Vis2DField, Viz2DField
import numpy as np
import traits.api as tr
