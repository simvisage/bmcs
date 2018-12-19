'''

This script is used to demonstrate the states of a model.

@author: rch
'''

import time

from .simulator import Simulator

# Construct a Simulator
s = Simulator()
s.tline.step = 0.2

# Start calculation in a thread
print('RUN the calculation thread from t = 0.0')
s.run()
print('WAIT in main thread for 3 secs')
time.sleep(3)
print('PAUSE the calculation thread')
s.pause()
print('RESUME the calculation thread from t = 3.0')
s.run()
print('WAIT in the main thread for 3 secs again')
time.sleep(3)
print('STOP the calculation thread')
s.stop()
print('RUN a new calculation thread from t = 0.0')
s.run()
print('JOIN the calculation thread into main thread to end simultaneously')
s.join()
