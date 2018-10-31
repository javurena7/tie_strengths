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
df = pd.read_table(run_path + 'overlap_comparison.txt', sep=' ')
"""
def plot_overlap_histograms(df, n=70):
    fig, ax = plt.subplots(1)
    ax.hist(df.ovrl, n, normed=True, label='Within company network', alpha = .65)
    ax.hist(df.e_ovrl, n, normed=True, label='Full network', alpha=.65)
    ax.set_xlabel('Overlap')
    ax.set_title('Overlap distributions')
    ax.legend()
    fig.savefig(run_path + 'overlap_histograms.png')
    return fig, ax

#fig, ax = plot_overlap_histograms(df)

def plot_overlap_comparison(df, n=50):
    fig, ax = plt.subplots(1)
    ax.plot(df.ovrl, df.e_ovrl, '.', alpha=0.07)
    fig, ax = plots.lin_bin_plot(df.ovrl, df.e_ovrl, n, fig=fig, ax=ax)
    ax.set_xlabel('Within operator network overlap: \n' + r'$O_{op}$')
    ax.set_ylabel('Full network overlap: \n' + r'$\langle O_{full} | O_{op} \rangle$')
    ax.set_title('Within company overlap and \n full network overlap')
    fig.savefig(run_path + 'overlap_comparison.png')
    return fig, ax

def plot_overlap_cumulative(df, size=100):
    fig, ax = plots.cumulative_distribution(df.w, df.ovrl, label='Within Company Overlap')
    fig, ax = plots.cumulative_distribution(df.w, df.e_ovrl, label='Full network', fig=fig, ax=ax)
    ax.set_xlabel(r'$P_{>}(w)$')
    ax.set_ylabel(r'$\langle O | P_{>}(w) \rangle$')
    ax.set_title('Overlap as a function of cumulative distribution of number of calls')
    ax.legend(loc=0)
    fig.savefig(run_path + 'overlap_cumulative_calls.png')
    return fig, ax

def plot_overlap_rank_cumulative(df, size=100):
    fig, ax = plots.cumulative_distribution(df.w, rankdata(df.ovrl)/df.shape[0], label='Within Company Overlap')
    fig, ax = plots.cumulative_distribution(df.w, rankdata(df.e_ovrl)/df.shape[0], label='Full network', fig=fig, ax=ax)
    ax.set_xlabel(r'$P_{>}(w)$')
    ax.set_ylabel(r'$\langle R(O) | P_{>}(w) \rangle$')
    ax.set_title('Rank of overlap as a function of \n cumulative distribution of number of calls')
    ax.legend(loc=0)
    fig.savefig(run_path + 'overlap_rank_cumulative_calls.png')
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

def plot_overlap_rank_logbin(df, factor=1.3, limit=8000, overlap_rank=True):
    df = df[df.w < limit]
    fig, ax = plots.log_bin_plot(df.w, rankdata(df.ovrl)/df.shape[0], factor, label='Within Company Overlap')
    fig, ax = plots.log_bin_plot(df.w, rankdata(df.e_ovrl)/df.shape[0], factor, label='Full network', fig=fig, ax=ax)
    ax.set_xlabel(r'$w$')
    ax.set_ylabel(r'$\langle Rank(O) | w \rangle$')
    ax.set_title('Rank of overlap as a function of number of calls')
    ax.legend()
    fig.savefig(run_path + 'overlap_rank_logbin_calls.png')
    return fig, ax

def plot_overlap_rank_linbin(df, factor=100, limit=8000):
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


"""
import pandas as pd
import matplotlib.pyplot as plt; plt.ion()
from tie_strengths import create_plots as cp
from tie_strengths import plots
run_path = 'data/full_run/'
df = pd.read_table(run_path + 'full_df_sample.txt', sep=' ')
df = df[df.c_iet_mu_na.notnull()]

"""

def plot_iet_mean(df, factor=1.3):
    idx = df.c_iet_mu_na.notnull()
    fig, ax = plots.log_bin_plot(df.c_iet_mu_na[idx], df.ovrl[idx], factor, label='Naive')
    idx = df.c_iet_mu_km.notnull()
    fig, ax = plots.log_bin_plot(df.c_iet_mu_km[idx], df.ovrl[idx], factor, label='KM', fig=fig, ax=ax)
    ax.legend()
    ax.set_xlabel(r'Mean inter-event time, $\tau$')
    ax.set_ylabel(r'$\langle O | \tau \rangle$')
    ax.set_title('Overlap as a function of mean inter-event time')
    fig.savefig(run_path + 'mean_iets.png')
    return fig, ax


def plot_iet_mean_cumulative(df):
    idx = df.c_iet_mu_na.notnull()
    fig, ax = plots.cumulative_distribution(df.c_iet_mu_na[idx], df.ovrl[idx], label='Naive')
    idx = df.c_iet_mu_km.notnull()
    fig, ax = plots.cumulative_distribution(df.c_iet_mu_km[idx], df.ovrl[idx], label='KM', fig=fig, ax=ax)
    ax.legend()
    ax.set_xlabel(r'$P_{>}(\tau)$')
    ax.set_ylabel(r'$\langle O | P_{>}(\tau) \rangle$')
    ax.set_title('Overlap as a function of cumulative mean inter-event time ')
    fig.savefig(run_path + 'mean_iet_cumulative.png')
    return fig, ax


def plot_iet_bur(df, factor=50):
    idx = df.c_iet_bur_na.notnull()
    fig, ax = plots.lin_bin_plot(df.c_iet_bur_na[idx], df.ovrl[idx], factor, label='Naive')
    idx = df.c_iet_bur_km.notnull()
    fig, ax = plots.lin_bin_plot(df.c_iet_bur_km[idx], df.ovrl[idx], factor, label='KM', fig=fig, ax=ax)
    idx = df.c_iet_bur_c_na.notnull()
    fig, ax = plots.lin_bin_plot(df.c_iet_bur_c_na[idx], df.ovrl[idx], factor, label='C-Naive', fig=fig, ax=ax)
    idx = df.c_iet_bur_c_km.notnull()
    fig, ax = plots.lin_bin_plot(df.c_iet_bur_c_km[idx], df.ovrl[idx], factor, label='C-KM', fig=fig, ax=ax)
    ax.legend()
    ax.set_xlabel(r'Burstiness, $B$')
    ax.set_ylabel(r'$\langle O | B \rangle$')
    ax.set_title('Overlap as a function of burstiness ')
    fig.savefig(run_path + 'bur_iet.png')
    return fig, ax


def plot_iet_bur_cumulative(df, size=20, arg='-'):
    idx = df.c_iet_bur_na.notnull()
    fig, ax = plots.cumulative_distribution(df.c_iet_bur_na[idx], df.ovrl[idx], label='Naive', size=size, arg=arg)
    idx = df.c_iet_bur_km.notnull()
    fig, ax = plots.cumulative_distribution(df.c_iet_bur_km[idx], df.ovrl[idx], label='KM', fig=fig, ax=ax, size=size, arg=arg)
    idx = df.c_iet_bur_c_na.notnull()
    fig, ax = plots.cumulative_distribution(df.c_iet_bur_c_na[idx], df.ovrl[idx], label='C-Naive', fig=fig, ax=ax, size=size, arg=arg)
    idx = df.c_iet_bur_c_km.notnull()
    fig, ax = plots.cumulative_distribution(df.c_iet_bur_c_km[idx], df.ovrl[idx], label='C-KM', fig=fig, ax=ax, size=size, arg=arg)
    ax.legend()
    ax.set_xlabel(r'$P_{>}(B)$')
    ax.set_ylabel(r'$\langle O | P_{>}(B) \rangle$')
    ax.set_title('Overlap as a function of cumulative burstiness')
    fig.savefig(run_path + 'bur_iet_cumulative.png')
    return fig, ax


def plot_mu_sig_bur_na(df):
    idx = df.c_iet_bur_na.notnull()
    fig, ax = plots.loglogheatmap(df.c_iet_mu_na[idx], df.c_iet_sig_na[idx], df.c_iet_bur_na[idx])


def plot_mu_sig_bur_km(df):
    idx = df.c_iet_bur_km.notnull()
    fig, ax = plots.loglogheatmap(df.c_iet_mu_km[idx], df.c_iet_sig_km[idx], df.c_iet_bur_km[idx])


def plot_iet_sig(df, factor=1.2):
    idx = df.c_iet_sig_na.notnull()
    fig, ax = plots.log_bin_plot(df.c_iet_sig_na[idx], df.ovrl[idx], factor, label='Naive')
    idx = df.c_iet_sig_km.notnull()
    fig, ax = plots.log_bin_plot(df.c_iet_sig_km[idx], df.ovrl[idx], factor, label='KM', fig=fig, ax=ax)
    ax.legend()
    ax.set_xlabel(r'Inter-event time standard deviation, $\sigma_{\tau}$')
    ax.set_ylabel(r'$\langle O | \sigma_{\tau}\rangle$')
    ax.set_title('Overlap as a function of inter-event time standard deviation')
    fig.savefig(run_path + 'std_iets.png')
    return fig, ax


def plot_iet_sig_cumulative(df, factor=1.1):
    idx = df.c_iet_sig_na.notnull()
    fig, ax = plots.cumulative_distribution(df.c_iet_sig_na[idx], df.ovrl[idx], label='Naive')
    idx = df.c_iet_sig_km.notnull()
    fig, ax = plots.cumulative_distribution(df.c_iet_sig_km[idx], df.ovrl[idx], label='KM', fig=fig, ax=ax)
    ax.legend()
    ax.set_xlabel(r'$P_>(\sigma_{\tau})$')
    ax.set_ylabel(r'$\langle O | P_>(\sigma_{\tau}) \rangle$')
    ax.set_title('Overlap as a function of cumulative \n inter-event time standard deviation')
    fig.savefig(run_path + 'std_iets_cumulative.png')
    return fig, ax


def plot_iet_rfresh(df, factor=1.4):
    idx = df.c_iet_rfsh_na.notnull()
    fig, ax = plots.log_bin_plot(df.c_iet_rfsh_na[idx], df.ovrl[idx], factor)
    ax.set_xlabel(r'Relative freshness, $f$')
    ax.set_ylabel(r'$\langle O | f \rangle$')
    ax.set_title('Overlap as a function of relative freshness')
    fig.savefig(run_path + 'rfsh_iets.png')
    return fig, ax


def plot_iet_rfresh_cumulative(df, factor=1.1):
    idx = df.c_iet_rfsh_na.notnull()
    fig, ax = plots.cumulative_distribution(df.c_iet_rfsh_na[idx], df.ovrl[idx])
    ax.set_xlabel(r'$P_>(f)$')
    ax.set_ylabel(r'$\langle O | P_>(f) \rangle$')
    ax.set_title('Overlap as a function of cumulative \n relative freshness')
    fig.savefig(run_path + 'rfsh_iets_cumulative.png')
    return fig, ax


def plot_iet_aget(df, factor=1.4):
    idx = df.c_iet_age_na.notnull()
    fig, ax = plots.log_bin_plot(df.c_iet_age_na[idx], df.ovrl[idx], factor)
    ax.set_xlabel(r'Residual Waiting Time, $\tau_r$')
    ax.set_ylabel(r'$\langle O | \tau_r \rangle$')
    ax.set_title('Overlap as a function of residual waiting time')
    fig.savefig(run_path + 'age_iets.png')
    return fig, ax


def plot_iet_aget_cumulative(df, size=100):
    idx = df.c_iet_age_na.notnull()
    fig, ax = plots.cumulative_distribution(1/df.c_iet_age_na[idx], df.ovrl[idx], size=100)
    ax.set_xlabel(r'$P_>(\tau_r^{-1})$')
    ax.set_ylabel(r'$\langle O | P_>(\tau_r^{-1}) \rangle$')
    ax.set_title('Overlap as a function of cumulative \n inverse residual waiting time')
    fig.savefig(run_path + 'age_iets_cumulative.png')
    return fig, ax


def plot_btrns(df, factor=1.4):
    idx = df.c_brtrn.notnull()
    fig, ax = plots.log_bin_plot(df.c_brtrn[idx], df.ovrl[idx], factor, label='Bursty Trains')
    fig, ax = plots.log_bin_plot(df.c_wkn_t[idx], df.ovrl[idx], factor, label='Number of Calls', fig=fig, ax=ax)
    ax.set_xlabel(r'Residual Waiting Time, $\tau_r$')
    ax.set_ylabel(r'$\langle O | \tau_r \rangle$')
    ax.set_title('Overlap as a function of residual waiting time')
    ax.legend()
    fig.savefig(run_path + 'brtrn.png')
    return fig, ax


def plot_btrns_cumulative(df, size=100):
    idx = df.c_brtrn.notnull()
    fig, ax = plots.cumulative_distribution(df.c_brtrn[idx], df.ovrl[idx], size=100)
    fig, ax = plots.cumulative_distribution(df.c_wkn_t[idx], df.ovrl[idx], size=100, fig=fig, ax=ax)
    ax.set_xlabel(r'$P_>(\tau_r^{-1})$')
    ax.set_ylabel(r'$\langle O | P_>(\tau_r^{-1}) \rangle$')
    ax.set_title('Overlap as a function of cumulative \n inverse residual waiting time')
    ax.legend()
    fig.savefig(run_path + 'brtrn_cumulative.png')
    return fig, ax
