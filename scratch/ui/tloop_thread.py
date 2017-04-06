'''
Created on 21.03.2017

@author: cthoennessen
'''
from PyQt4.Qt import QThread

class TLoopThread(QThread):
    '''
    '''
    
    def __init__(self, model, **args):
        super(TLoopThread, self, **args).__init__()
        self.model = model
    
    def run(self): 
        self.model.do_progress()

    