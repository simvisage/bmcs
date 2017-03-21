'''
Created on 21.03.2017

@author: cthoennessen
'''
import time
from threading import Thread
from traits.api import Instance
from view.ui.bmcs_tree_node import BMCSTreeNode
from pyface.progress_dialog import ProgressDialog

import numpy as np


class TLoopThread(Thread):
    
    model = Instance(BMCSTreeNode)
    
    def run(self): 
        self.model.do_progress()

    