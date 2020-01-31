
import ibvpy.plugins._mayavi_engine

def set_engine(e):
    ibvpy.plugins._mayavi_engine._engine = e

def get_engine():
    return ibvpy.plugins._mayavi_engine._engine