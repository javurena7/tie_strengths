
"""
Script to analyze if the selection of non-

"""


import numpy as np
import pandas as pd

path = '/scratch/work/urenaj1/full/'
degs = pd.read_table(path + 'neighbors.txt', sep=' ')
del degs['ovrl']
ext_degs = pd.read_table(path + 'extended_neighbors.txt', sep=' ')
del ext_degs['ovrl']
ext_degs.rename({'deg_0': 'edeg_0', 'deg_1':'edeg_1', 'n_ij':'en_ij'}, axis=1, inplace=True)
degs = pd.merge(degs, ext_degs, how='left', on=['0', '1'])
degs['k_nn'] = degs['deg_0'] + degs['deg_1'] - 2 - degs['n_ij']
degs['ek_nn'] = degs['edeg_0'] + degs['edeg_1'] - 2 - degs['en_ij']
degs['k_nn_f'] = degs['k_nn']/degs['ek_nn']
degs['k_n_f'] = degs['n_ij']/degs['en_ij']
degs.to_csv(path + 'neighbors_comparison.txt', index=False, sep=' ')
