import matplotlib.pyplot as plt
from latexify import *
from scipy.stats import rankdata
import pandas as pd
import numpy as np
from seaborn import cubehelix_palette


def plot_variable_distr(var, ovrl, w_idx, savepath='', xlabel='', step=.1):
    latexify(8, 1.8, 1)
    fig, axs = plt.subplots(1, 5, sharey=True)
    cols = cubehelix_palette(5)
    ldict = {0: r'$w \in [5, 8]$',\
            1: r'$w \in [9, 14]$',\
            2: r'$w \in [15, 25]$',\
            3: r'$w \in [26, 59]$',\
            4: r'$w \in [60, 10000]$'}

    for i in range(5):
        idx = w_idx == i
        ylabel = r'$\langle O |$' + xlabel + r'$\rangle$' if i == 0 else ''
        plot_dist(var[idx], ovrl[idx], axs[i], label=ldict[i], xlabel=xlabel, \
                ylabel=ylabel, color=cols[i], step=step)
    fig.tight_layout()
    fig.savefig(savepath)

def plot_dist(var, ovrl, ax=None, label='', xlabel='', ylabel='', color=None, step=.1):
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
    ax.fill_between(x, dist[0, ], dist[2, ], alpha=.5, color=color, lw=0.)
    ax.plot(x, dist[1, ], color=color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
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

def plot_variables(data_path='../paper_run/', save_path='../paper_run/distributions/', step=.05, y_var='ovrl'):
    col_labels = {'mu': r'$\bar{\tau}$',
                'sig': r'$\bar{\sigma_{\tau}}$',
                'b': r'$B$',
                'mu_r': r'$\bar{\tau}_R$',
                'r_frsh': r'$\hat{f}$',
                'age': r'$age$',
                't_stb': r'$TS$',
                'm': r'$M$',
                'bt_mu': r'$\bar{E}$',
                'bt_sig': r'$\sigma^{E}$',
                'bt_cv': r'$CV^E$',
                'bt_n': r'$N^E$',
                'bt_tmu': r'$\bar{t}$',
                'bt_tsig': r'$\sigma_{t}$',
                'bt_logt': r'$log(T)$',
                'out_call_div': r'$JSD$',
                'r': r'$r$',
                'w': r'$w$',
                'avg_len': r'$\hat{l}$',
                'len': r'$l$',
                'w_hrs': r'$w_h$',
                'w_day': r'$w_d$'}
    col_labels.update({'c' + str(i): 'C' + r'$' + str(i) + '$' for i in range(1, 16)})
    df = pd.read_csv(data_path + 'full_df_paper.txt', sep=' ')
    df2 = pd.read_csv(data_path + 'clustered_df_paper.txt', sep=' ')
    df = pd.concat([df, df2], axis=1)
    del df2
    w_idx = pd.qcut(df.w, 5, labels=range(5))
    ovrl = df[y_var]
    for var, xlabel in col_labels.items():
        savename = save_path + var + '.pdf'
        plot_variable_distr(df[var], ovrl, w_idx, savename, xlabel, step=step)

if __name__ == '__main__':
    y_var = 'ovrl'
    save_path = '../paper_run/distributions/' + y_var + '/'
    plot_variables(save_path=save_path, step=.1, y_var=y_var)
