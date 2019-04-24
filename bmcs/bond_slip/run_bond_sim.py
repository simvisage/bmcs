'''
Created on 12.01.2016
@author: ABaktheer, RChudoba, Yingxiong
'''

from .bond_slip_model import BondSlipModel


def run_bond_sim(*args, **kw):
    po = BondSlipModel()
    w = po.get_window()
    w.run()
    w.offline = False
    w.configure_traits(*args, **kw)


if __name__ == '__main__':
    run_bond_sim()
