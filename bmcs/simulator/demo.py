'''

This script is used to demonstrate the states of a model.

@author: rch
'''

from .simulator import Simulator

s = Simulator()
print('running')
s.run()
print('finshed')
s.pause()
print('paused')
s.stop()
print('stopped')

print('tline', s.tline)
print('model', s.model)
print('tloop', s.tloop)
print('hist', s.hist)