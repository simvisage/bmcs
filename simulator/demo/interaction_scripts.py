'''
Created on Jan 9, 2019

@author: rch
'''
import time


def run_rerun_test(s):
    # Start calculation in a thread
    print('\nRUN the calculation thread from t = 0.0')
    s.run()
    print('\nWAIT in main thread for 3 secs')
    time.sleep(3)
    print('\nPAUSE the calculation thread')
    s.pause()
    print('\nPAUSED wait 1 sec')
    time.sleep(1)
    print('\nRESUME the calculation thread')
    s.run()
    print('\nWAIT in the main thread for 3 secs again')
    time.sleep(3)
    print('\nSTOP the calculation thread')
    s.stop()
    print('\nRUN a new calculation thread from t = 0.0')
    s.run()
    print('\nJOIN the calculation thread into main thread to end simultaneously')
    s.join()
    print('END all threads')
