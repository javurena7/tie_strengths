from tie_strengths import plots
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt; plt.ion()
from scipy.stats import rankdata

###### DEGREE DISTRIBUTION

run_path = 'data/full_run/'

#Overlap comparison
"""
import pandas as pd
import matplotlib.pyplot as plt; plt.ion()
from tie_strengths import create_plots as cp
from tie_strengths import plots
run_path = 'data/full_run/'
df = pd.read_table(run_path + 'overlap_comparison_sample.txt', sep=' ')
"""
def plot_overlap_histograms(df, n=70):
    fig, ax = plt.subplots(1)
    ax.hist(df.ovrl, n, normed=True, label='Within company network', alpha = .65)
    ax.hist(df.e_ovrl, n, normed=True, label='Full network', alpha=.65)
    ax.set_xlabel('Overlap')
    ax.set_title('Overlap distributions')
    ax.legend()
    fig, ax = plot_overlap_histograms(100)
    fig.savefig(run_path + 'overlap_histograms.png')
    return fig, ax

def plot_overlap_comparison(df, n=50):
    fig, ax = plt.subplots(1)
    ax.plot(df.ovrl, df.e_ovrl, '.', alpha=0.07)
    fig, ax = plots.lin_bin_plot(df.ovrl, df.e_ovrl, n, fig=fig, ax=ax)
    ax.set_xlabel('Within operator network overlap: \n' + r'$O_{op}$')
    ax.set_ylabel('Full network overlap: \n' + r'$\langle O_{full} | O_{op} \rangle$')
    ax.set_title('Within company overlap and \n full network overlap')
    ax.legend()
    fig.savefig(run_path + 'overlap_comparison.png')
    return fig, ax

def plot_overlap_cumulative(df, size=100):
    fig, ax = plots.cumulative_distribution(df.w, df.ovrl, label='Within Company Overlap')
    fig, ax = plots.cumulative_distribution(df.w, df.e_ovrl, label='Full network', fig=fig, ax=ax)
    ax.set_xlabel(r'$P_{>}(w)$')
    ax.set_ylabel(r'$\langle O | P_{>}(w) \rangle$')
    ax.set_title('Overlap as a function of cumulative distribution of number of calls')
    ax.legend()
    fig.savefig(run_path + 'overlap_cumulative_calls.png')
    return fig, ax

def plot_overlap_logbin(df, factor=1.3, limit=8000):
    df = df[df.w < limit]
    fig, ax = plots.log_bin_plot(df.w, df.ovrl, factor, label='Within Company Overlap')
    fig, ax = plots.log_bin_plot(df.w, df.e_ovrl, factor, label='Full network', fig=fig, ax=ax)
    ax.set_xlabel(r'$w$')
    ax.set_ylabel(r'$\langle O | w \rangle$')
    ax.set_title('Overlap as a function of number of calls')
    ax.legend()
    fig.savefig(run_path + 'overlap_logbin_calls.png')
    return fig, ax

def plot_overlap_linbin(df, factor=100, limit=8000):
    df = df[df.w < limit]
    w = rankdata(df.w)/df.shape[0]
    fig, ax = plots.lin_bin_plot(w, rankdata(df.ovrl)/df.shape[0], factor, label='Within Company Overlap', arg='.')
    fig, ax = plots.lin_bin_plot(w, rankdata(df.e_ovrl)/df.shape[0], factor, label='Full network', fig=fig, ax=ax, arg='.')
    ax.set_xlabel(r'$Rank(w)$')
    ax.set_ylabel(r'$\langle Rank(O) | Rank(w) \rangle$')
    ax.set_title('Rank of overlap as a function of the\n rank of the number of calls')
    ax.legend()
    fig.savefig(run_path + 'overlap_rank_calls.png')
    return fig, ax

