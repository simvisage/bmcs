'''
Created on 21.03.2017

@author: cthoennessen
'''
from PyQt4.Qt import QThread
from traits.api import WeakRef

class TLoopThread(QThread):
    
    model = WeakRef
    
    def run(self): 
        self.model.do_progress()

    