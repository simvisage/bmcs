import numpy as np
import pandas as pd

L = 0.01
D=np.array([6*L, 12*L])
H=np.array([6*L, 12*L])

_lambda=np.array([1/4, 1/2, 3/4])
mu=np.array([1/4, 1/2, 3/4])

for i in range(len(_lambda)):
    G = np.array(pd.read_table(
        'C:\\Users\\marag\\OneDrive - rwth-aachen.de\\Zylinderscheibe Atena\\parametric_study\\test.cct')).astype('str')
    G[48] = np.array(['      2      ' +np.str(_lambda[i] * D[0] - 0.003)+'      0.0000'])
    G[48] = np.array(['      3      ' + np.str(_lambda[i] * D[0] - 0.003) + '      0.0000'])
    G[48] = np.array(['      4      ' + np.str(_lambda[i] * D[0] - 0.003) + '      0.0000'])
    G[48] = np.array(['      5      ' + np.str(_lambda[i] * D[0] - 0.003) + '      0.0000'])
    G[48] = np.array(['      6      ' + np.str(_lambda[i] * D[0] - 0.003) + '      0.0000'])
    G[48] = np.array(['      7      ' + np.str(_lambda[i] * D[0] - 0.003) + '      0.0000'])
    G[48] = np.array(['      8      ' + np.str(_lambda[i] * D[0] - 0.003) + '      0.0000'])
    G[48] = np.array(['      9      ' + np.str(_lambda[i] * D[0] - 0.003) + '      0.0000'])
    np.savetxt('C:\\Users\\marag\\OneDrive - rwth-aachen.de\\Zylinderscheibe Atena\\parametric_study\\test3.cct', G,
               fmt='%s')
