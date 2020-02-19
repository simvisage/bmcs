'''
Created on Nov 27, 2019

@author: rch
'''

from pstats import Stats

s = Stats("run_stats.txt")
s.sort_stats('cumtime')
s.print_stats(30)