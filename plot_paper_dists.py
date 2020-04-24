import matplotlib.pyplot as plt
from latexify import *
from scipy.stats import rankdata
import pandas as pd
import numpy as np
from seaborn import cubehelix_palette


def plot_variable_distr(var, ovrl, w_idx, name=''):
    latexify(8, 1.8, 1)
    fig, axs = plt.subplots(1, 5, sharey=True)
    cols = cubehelix_palette(5)
    ldict = {0: r'$w \in [5, 8]$',\
            1: r'$w \in [9, 14]$',\
            2: r'$w \in [15, 25]$',\
            3: r'$w \in [26, 59]$',\
            4: r'$w \in [60, 10000]$'}

    for i in range(5):
        label = ''
        xlabel= ''
        idx = w_idx == i
        plot_dist(var[idx], ovrl[idx], axs[i], label=ldict[i], xlabel=xlabel, color=cols[i])
    fig.tight_layout()
    fig.savefig(name)

def plot_dist(var, ovrl, ax=None, label='', xlabel='', color=None, step=.1):
    rank_x = lambda x: rankdata(x) / len(x)
    var = rank_x(var)
    rg = list(np.arange(0, 1 + step, step))
    dist = []
    for i0, i1 in zip(rg[:-1], rg[1:]):
        idx = (var >= i0) & (var < i1)
        ovrl_idx = ovrl[idx]
        if len(ovrl_idx) == 0:
            m, m0, m1 = np.nan, np.nan, np.nan
        else:
            m = np.mean(ovrl_idx)
            m0 = np.percentile(ovrl_idx, 20)
            m1 = np.percentile(ovrl_idx, 80)
        dist.append([m0, m, m1])
    dist = np.array(dist).T
    x = [(x1 + x2)/2. for x1, x2 in zip(rg[:-1], rg[1:])]
    dist = pad_distribution(dist)
    ax.fill_between(x, dist[0, ], dist[2, ], alpha=.5, color=color)
    ax.plot(x, dist[1, ], color=color)
    ax.set_title(label)
    return dist

def pad_distribution(dist):
    for i in range(3):
        dst = dist[i,]
        nan_idx = np.isnan(dst)
        for j in range(1, len(dst)-1):
            if nan_idx[j] and not nan_idx[j-1] and not nan_idx[j+1]:
                dist[i,j] = np.mean(dist[i,[j-1, j+1]])
    return dist

#w_idx = pd.qcut(df.w, 5, labels=range(5))








