'''
Created on 21.03.2017

@author: cthoennessen
'''
from threading import Thread


class TLoopThread(Thread):
    '''Time loop thread responsible.
    '''

    def __init__(self, model, *args, **kw):
        super(TLoopThread, self).__init__(*args, **kw)
        self.daemon = True
        self.model = model

    def run(self):
        self.model.running = True
        try:
            self.model.tloop.eval()
        except Exception as e:
            self.model.running = False
            raise
        self.model.running = False
