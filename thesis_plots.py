from tie_strengths import plots
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata
import seaborn as sb
from scipy.stats import pearsonr, spearmanr


def weight_vs_overlap(w, o, name='full_run/figs/weight_overlap.pdf'):
    g = plots.loglinjointdistr(w, o, height=5)
    g.savefig(name)
    return g

def overlaps_vs_weight(w, o, eo, factor=1.2, name='full_run/figs/weight_overlap_comp.pdf'):
    fig, ax = plots.log_bin_plot(w, o, factor, xlabel=None, ylabel=None, title=None, label=r'$O^c_{ij}$')
    fig, ax = plots.log_bin_plot(w, eo, factor, xlabel=None, ylabel=None, title=None, label=r'$O^f_{ij}$', fig=fig, ax=ax)
    ax.set_ylabel(r'$O_{ij}$')
    ax.set_xlabel(r'$w_{ij}$')
    ax.set_title('')
    ax.legend(loc=0)
    fig.tight_layout()
    fig.savefig(name)
    return fig, ax

def overlaps_vs_weight_ranks(w, o, eo, bins=30, name='full_run/figs/weight_overlap_rank.pdf'):
    w = rankdata(w)/w.shape[0]
    fig, ax = plots.lin_bin_plot(w, o, bins, xlabel=None, ylabel=None, title=None, label=r'$O^c_{ij}$', arg='.')
    fig, ax = plots.lin_bin_plot(w, eo, bins, xlabel=None, ylabel=None, title=None, label=r'$O^f_{ij}$', fig=fig, ax=ax, arg='.')
    ax.set_ylabel(r'$O_{ij}$')
    ax.set_xlabel(r'$Rank(w_{ij})$')
    ax.set_title('')
    ax.legend(loc=0)
    fig.tight_layout()
    fig.savefig(name)
    return fig, ax

def recip_comparison(r, o, bins, name='full_run/figs/recip_comp.pdf'):
    idx = r < .5
    fig, ax = plt.subplots(1)
    ax.hist(o[idx], bins, alpha=.5, normed=True, label=r'$r_{ij} < 0.5$')
    ax.hist(o[~idx], bins, alpha=.5, normed=True, label=r'$r_{ij} = 0.5$')
    ax.set_xlabel(r'$O_{ij}$')
    ax.set_ylabel('Frequencies')
    ax.legend(loc=0)
    fig.tight_layout()
    fig.savefig(name)
    return fig, ax

def recip_overl_weight_ranks(r, w, o, bins=30, name='full_run/figs/recip_comp_weight.pdf'):
    idx = r < .5
    w = rankdata(w)/w.shape[0]
    fig, ax = plots.lin_bin_plot(w[idx], o[idx], bins, xlabel=None, ylabel=None, title=None, label=r'$r_{ij} < 0.5$', arg='.')
    fig, ax = plots.lin_bin_plot(w[~idx], o[~idx], bins, xlabel=None, ylabel=None, title=None, label=r'$r_{ij} = 0.5$', fig=fig, ax=ax, arg='.')
    ax.set_ylabel(r'$\langle O_{ij} | Rank(w_{ij}), r_{ij} \rangle$')
    ax.set_xlabel(r'$Rank(w_{ij})$')
    ax.set_title('')
    ax.legend(loc=0)
    fig.tight_layout()
    fig.savefig(name)
    return fig, ax

def mu_vs_overlap(mu, o, name='full_run/figs/mu-iet_overlap.pdf'):
    idx = mu > 0
    # Note, we are removing data with zero mu, which are 730000 links (mean w=4.2, and mean ovrl=.03)
    g = plots.loglinjointdistr(mu[idx] + 1, o[idx], height=5, xlabel=r'$\bar{\tau}_{ij}$', xlim=(1, 100))
    g.savefig(name)
    return g

def mu_htmap(age, w, o, name='full_run/figs/mu_htmap.pdf'):
    plots.latexify(4.2, 4, 2)
    n = len(o)
    g = plots.linlinsbmap(rankdata(age)/n, rankdata(w)/n, o, y_bins=45, z_label=r'$O_{ij}$')
    g.axes.set_ylabel(r'$Rank(w_{ij})$')
    g.axes.set_xlabel(r'$Rank(\bar{\tau_{ij}})$')
    g.axes = plots.format_axes(g.axes)
    g.get_figure().savefig(name)


def sig_vs_overlap(sig, o, name='full_run/figs/sig-iet_overlap.pdf'):
    idx = sig > 0
    # Note, we are removing data with zero mu, which are 730000 links (mean w=4.2, and mean ovrl=.03)
    g = plots.loglinjointdistr(sig[idx] , o[idx], height=5, xlabel=r'$\sigma_{\tau}_{ij}$', xlim=(1, 100))
    g.savefig(name)
    return g

def sig_htmap(sig, w, o, name='full_run/figs/sig_htmap.pdf'):
    plots.latexify(4.2, 4, 2)
    n = len(o)
    g = plots.linlinsbmap(rankdata(sig)/n, rankdata(w)/n, o, y_bins=45, z_label=r'$O_{ij}$')
    g.axes.set_ylabel(r'$Rank(w_{ij})$')
    g.axes.set_xlabel(r'$Rank(\bar{\tau_{ij}})$')
    g.axes = plots.format_axes(g.axes)
    g.get_figure().savefig(name)


def b_vs_overlap(b, o, name='full_run/figs/b_overlap.pdf'):
    idx = b.notnull()
    g = plots.linlinjointdistr(b[idx], o[idx], height=5, xlabel=r'$B_{ij}$', xlim=(-.85, .85), gridsize=(35, 45))
    g.savefig(name)
    return g

def b_htmap(age, w, o, name='full_run/figs/b_htmap.pdf'):
    plots.latexify(4.2, 4, 2)
    n = len(o)
    g = plots.linlinsbmap(rankdata(age)/n, rankdata(w)/n, o, y_bins=45, z_label=r'$O_{ij}$')
    g.axes.set_ylabel(r'$Rank(w_{ij})$')
    g.axes.set_xlabel(r'$Rank(B_{ij})$')
    g.axes = plots.format_axes(g.axes)
    g.get_figure().savefig(name)


def avg_relay_vs_overlap(mu_r, o, name='full_run/figs/mu_r_overlap.pdf'):
    idx = (mu_r.notnull()) & (mu_r < np.inf)
    g = plots.loglinjointdistr(mu_r[idx] + .5, o[idx], height=5, xlabel=r'$\bar{\tau}_R$', xlim=(.9, 70), gridsize=(25, 35))
    g.savefig(name)


def avg_relay_htmap(age, w, o, name='full_run/figs/mu-r_htmap.pdf'):
    plots.latexify(4.2, 4, 2)
    n = len(o)
    g = plots.linlinsbmap(rankdata(age)/n, rankdata(w)/n, o, y_bins=45, z_label=r'$O_{ij}$')
    g.axes.set_ylabel(r'$Rank(w_{ij})$')
    g.axes.set_xlabel(r'$Rank(\bar{\tau}_R)$')
    g.axes = plots.format_axes(g.axes)
    g.get_figure().savefig(name)

def rel_frsh_vs_overlap(r, o, name='full_run/figs/r-fresh_overlap.pdf'):
    idx = (r.notnull()) & (r < np.inf)
    g = plots.loglinjointdistr(r[idx] + 1, o[idx], height=5, xlabel=r'$\hat{f_r}$', xlim=(1, 1300))
    g.savefig(name)

def rel_frsh_htmap(age, w, o, name='full_run/figs/r-fresh_htmap.pdf'):
    plots.latexify(4.2, 4, 2)
    n = len(o)
    g = plots.linlinsbmap(rankdata(age)/n, rankdata(w)/n, o, y_bins=45, z_label=r'$O_{ij}$')
    g.axes.set_ylabel(r'$Rank(w_{ij})$')
    g.axes.set_xlabel(r'$Rank(\hat{f_r})$')
    g.axes = plots.format_axes(g.axes)
    g.get_figure().savefig(name)

def age_vs_overlap(age, o, name='full_run/figs/age_overlap.pdf'):
    g = plots.linlinjointdistr(age, o, height=5, xlabel=r'$age$', xlim=(0, max(age)))
    g.savefig(name)


def age_htmap(age, w, o, name='full_run/figs/age_htmap.pdf'):
    plots.latexify(4.1, 4.2, 2)
    n = len(o)
    g = plots.linlinsbmap(rankdata(age)/n, rankdata(w)/n, o, y_bins=45, z_label=r'$O_{ij}$')
    g.axes.set_ylabel(r'$Rank(w_{ij})$')
    g.axes.set_xlabel(r'$Rank(age)$')
    g.axes = plots.format_axes(g.axes)
    g.get_figure().savefig(name)


def tmp_stab_vs_overlap(t, o, name='full_run/figs/t-stab_overlap.pdf'):
    g = plots.linlinjointdistr(t, o, height=5, xlabel=r'$TS$', xlim=(-1, 120))
    g.savefig(name)


def tmp_stab_htmap(t, w, o, name='full_run/figs/t-stab_htmap.pdf'):
    plots.latexify(4.2, 4, 2)
    n = len(o)
    g = plots.linlinsbmap(rankdata(t)/n, rankdata(w)/n, o, y_bins=45, z_label=r'$O_{ij}$')
    g.axes.set_ylabel(r'$Rank(w_{ij})$')
    g.axes.set_xlabel(r'$Rank(TS_{ij})$')
    g.axes = plots.format_axes(g.axes)
    g.get_figure().savefig(name)

def m_vs_overlap(m, o, name='full_run/figs/iet-m_overlap.pdf'):
    idx = m.notnull()
    g = plots.linlinjointdistr(m[idx], o[idx], height=5, xlabel=r'$M_{ij}$', xlim=(-1, 1))
    g.savefig(name)

def m_htmap(m, w, o, name='full_run/figs/iet-m_htmap.pdf'):
    plots.latexify(4.2, 4, 2)
    n = len(o)
    g = plots.linlinsbmap(rankdata(m)/n, rankdata(w)/n, o, y_bins=45, z_label=r'$O_{ij}$')
    g.axes.set_ylabel(r'$Rank(w_{ij})$')
    g.axes.set_xlabel(r'$Rank(M_{ij})$')
    g.axes = plots.format_axes(g.axes)
    g.get_figure().savefig(name)

def bt_mu_vs_overlap(mu, o, name='full_run/figs/mu-bt_overlap.pdf'):
    g = plots.loglinjointdistr(1/mu, o, height=5, xlabel=r'$\frac{1}{\bar{E}}$', xlim=(1./15, 1.1), gridsize=(45, 55))
    g.savefig(name)


def bt_mu_htmap(mu, w, o, name='full_run/figs/bt-mu_htmap.pdf'):
    plots.latexify(4.2, 4, 2)
    n = len(o)
    g = plots.linlinsbmap(rankdata(1/mu)/n, rankdata(w)/n, o, y_bins=45, z_label=r'$O_{ij}$')
    g.axes.set_ylabel(r'$Rank(w_{ij})$')
    g.axes.set_xlabel(r'$Rank(\frac{1}{\bar{E}})$')
    g.axes = plots.format_axes(g.axes)
    g.get_figure().savefig(name)


def bt_cv_vs_overlap(cv, o, name='full_run/figs/cv-bt_overlap.pdf'):
    idx = (cv.notnull()) & (cv < np.inf)
    g = plots.loglinjointdistr(cv[idx] + 1, o[idx], height=5, xlabel=r'$CV_E$', xlim=(1, 8), gridsize=(45, 55))
    g.savefig(name)


def bt_cv_htmap(cv, w, o, name='full_run/figs/bt-cv_htmap.pdf'):
    plots.latexify(4.2, 4, 2)
    n = len(o)
    g = plots.linlinsbmap(rankdata(cv)/n, rankdata(w)/n, o, y_bins=45, z_label=r'$O_{ij}$')
    g.axes.set_ylabel(r'$Rank(w_{ij})$')
    g.axes.set_xlabel(r'$Rank(CV_E)$')
    g.axes = plots.format_axes(g.axes)
    g.get_figure().savefig(name)


def bt_n_vs_overlap(n, o, name='full_run/figs/bt-n_overlap.pdf'):
    g = plots.loglinjointdistr(n+1, o, height=5, xlim=(2, 1000), gridsize=(25, 45), xlabel=r'$N_E$')
    g.savefig(name)


def bt_n_htmap(en, w, o, name='full_run/figs/bt-n_htmap.pdf'):
    plots.latexify(4.2, 4, 2)
    n = len(o)
    g = plots.linlinsbmap(rankdata(en)/n, rankdata(w)/n, o, y_bins=45, z_label=r'$O_{ij}$')
    g.axes.set_ylabel(r'$Rank(w_{ij})$')
    g.axes.set_xlabel(r'$Rank(N_E)$')
    g.axes = plots.format_axes(g.axes)
    g.get_figure().savefig(name)


def bt_tmu_vs_overlap(tmu, o, name='full_run/figs/bt-tmu_overlap.pdf'):
    g = plots.linlinjointdistr(tmu, o, xlim=(0, 1), xlabel=r'$\bar{t^b}$', ylim=(0, .3))
    g.savefig(name)


def bt_tmu_htmap(tmu, w, o, name='full_run/figs/bt-tmu_htmap.pdf'):
    plots.latexify(4.2, 4, 2)
    n = len(o)
    g = plots.linlinsbmap(rankdata(tmu)/n, rankdata(w)/n, o, y_bins=45, z_label=r'$O_{ij}$')
    g.axes.set_ylabel(r'$Rank(w_{ij})$')
    g.axes.set_xlabel(r'$Rank(\bar{t^b})$')
    g.axes = plots.format_axes(g.axes)
    g.get_figure().savefig(name)


def bt_tsig_vs_overlap(tsig, o, name='full_run/figs/bt-tsig_overlap.pdf'):
    idx = tsig.notnull()
    g = plots.linlinjointdistr(tsig[idx], o[idx], xlim=(0, .5), xlabel=r'$\sigma_{t^b}$')
    g.savefig(name)

def bt_tsig_htmap(tsig, w, o, name='full_run/figs/bt-tsig_htmap.pdf'):
    plots.latexify(4.2, 4, 2)
    n = len(o)
    g = plots.linlinsbmap(rankdata(tsig)/n, rankdata(w)/n, o, y_bins=45, z_label=r'$O_{ij}$')
    g.axes.set_ylabel(r'$Rank(w_{ij})$')
    g.axes.set_xlabel(r'$Rank(\sigma_{t^b})$')
    g.axes = plots.format_axes(g.axes)
    g.get_figure().savefig(name)

def bt_tlog_vs_overlap(tlog, o, name='full_run/figs/bt-tlog_overlap.pdf'):
    idx = (tlog.notnull()) & (tlog < np.inf)
    g = plots.linlinjointdistr(tlog[idx], o[idx], xlim=(-5, 6), xlabel=r'$\log{(T_{t^b})}$', gridsize=(45, 55))
    g.savefig(name)

def bt_tlog_htmap(tlog, w, o, name='full_run/figs/bt-tlog_htmap.pdf'):
    plots.latexify(4.2, 4, 2)
    n = len(o)
    g = plots.linlinsbmap(rankdata(tlog)/n, rankdata(w)/n, o, y_bins=45, z_label=r'$O_{ij}$')
    g.axes.set_ylabel(r'$Rank(w_{ij})$')
    g.axes.set_xlabel(r'$Rank(\log{(T_{t^b})})$')
    g.axes = plots.format_axes(g.axes)
    g.get_figure().savefig(name)

def recipt_vs_overlap(re, o, name='full_run/figs/recip_overlap.pdf'):
    g = plots.linlinjointdistr(re, o, xlim=(0, .5), xlabel=r'$r_{ij}$')
    g.savefig(name)


def recipt_htmap(re, w, o, name='full_run/figs/recip_htmap.pdf'):
    plots.latexify(4.2, 4, 2)
    n = len(o)
    g = plots.linlinsbmap(rankdata(re)/n, rankdata(w)/n, o, y_bins=45, z_label=r'$O_{ij}$')
    g.axes.set_ylabel(r'$Rank(w_{ij})$')
    g.axes.set_xlabel(r'$Rank(r_{ij})$')
    g.axes = plots.format_axes(g.axes)
    g.get_figure().savefig(name)


def jsd_vs_overlap(jsd, o, name='full_run/figs/jsd_overlap.pdf'):
    g = plots.loglinjointdistr(jsd + .9, o, xlim=(.9, 4500), xlabel=r'$JSD_{ij}$')
    g.savefig(name)

def jsd_diff_vs_overlap(j1, j2, o, name='full_run/figs/jsd_diff_overlap.pdf'):
    g = plots.loglinjointdistr(np.abs(j1 - j2) + 1, o, xlim=(1, 3000), xlabel=r'$|JSD_{i} - JSD_j|$')
    g.savefig(name)

def jsd_mean_vs_overlap(j1, j2, o, name='full_run/figs/jsd_mean_overlap.pdf'):
    g = plots.loglinjointdistr(np.sqrt(j1*j2)+1, o, xlim=(1, 5000), xlabel=r'$\sqrt{JSD_{i} * JSD_j}$')
    g.savefig(name)

def jsd_htmap(jsd, w, o, name='full_run/figs/jsd_htmap.pdf'):
    plots.latexify(4.2, 4, 2)
    n = len(jsd)
    g = plots.linlinsbmap(rankdata(jsd)/n, rankdata(w)/n, o, y_bins=45, z_label=r'$O_{ij}$')
    g.axes.set_ylabel(r'$Rank(w_{ij})$')
    g.axes.set_xlabel(r'$Rank(JSD_{ij})$')
    g.axes = plots.format_axes(g.axes)
    g.get_figure().savefig(name)

def jsd_all_diff_map(jsd, j1, j2, o, name='full_run/figs/jsd_all_htmap.pdf'):
    plots.latexify(4.2, 4, 2)
    n = len(jsd)
    g = plots.linlinsbmap(rankdata(jsd)/n, rankdata(np.abs(j1 - j2))/n, o, y_bins=45, z_label=r'$O_{ij}$')
    g.axes.set_ylabel(r'$Rank(|JSD_{i}- JSD_j|)$')
    g.axes.set_xlabel(r'$Rank(JSD_{ij})$')
    g.axes = plots.format_axes(g.axes)
    g.get_figure().savefig(name)

def jsd_all_weight_map(jsd, j1, j2, w, name='full_run/figs/jsd_all_weight_htmap.pdf'):
    plots.latexify(4.2, 4, 2)
    n = len(jsd)
    g = plots.linlinsbmap(rankdata(jsd)/n, rankdata(np.abs(j1 - j2))/n, np.log(w), y_bins=45, z_label=r'$\log(w_{ij})$')
    g.axes.set_ylabel(r'$Rank(|JSD_{i}- JSD_j|)$')
    g.axes.set_xlabel(r'$Rank(JSD_{ij})$')
    g.axes = plots.format_axes(g.axes)
    g.get_figure().savefig(name)

def jsd_diff_htmap(j1, j2, w, o, name='full_run/figs/jsd_diff_htmap.pdf'):
    plots.latexify(4.2, 4, 2)
    n = len(j1)
    g = plots.linlinsbmap(rankdata(np.abs(j1 - j2))/n, rankdata(w)/n, o, y_bins=45, z_label=r'$O_{ij}$')
    g.axes.set_ylabel(r'$Rank(w_{ij})$')
    g.axes.set_xlabel(r'$Rank(|JSD_{i} - JSD_j|)$')
    g.axes = plots.format_axes(g.axes)
    g.get_figure().savefig(name)

def weekly_corrs(corrs, name='full_run/figs/weekly_correlations.pdf'):

    plots.latexify(5.5, 5, 1)
    g = sb.heatmap(corrs, xticklabels=8, yticklabels=8)
    ticklabels = get_week_labels(8)
    g.axes.set_xticklabels(ticklabels)
    g.axes.set_yticklabels(ticklabels, rotation=0)
    g.get_figure().savefig(name)

def get_week_labels(step=8):
    """
    step: gap (in hours) between labels
    """
    from itertools import product
    def ifelse(a, b, c):
        if a:
            return b
        else:
            return c
    days = ['Mo.', 'Tu.', 'We.', 'Th.', 'Fr.', 'Sa.', 'Su.']
    hours = [ifelse(np.mod(i, step)==0, str(i) + 'h', '') for i in range(24)]
    ticklabels = [d + '  ' + h for d, h in product(days, hours) if len(h) > 0]
    return ticklabels

def hourly_correlation(df, y, name='full_run/figs/hourly_correlation.pdf'):
    step = 8
    ticklabels = get_week_labels(step)
    av = [spearmanr(y, df.iloc[:, i])[0] for i in range(df.shape[1])]
    plots.latexify(6, 2.2, 1)
    fig, ax = plt.subplots(1)
    ax.plot(av)
    ax.set_xticks(range(0, df.shape[1], step))
    ax.set_xticklabels(ticklabels, rotation=90)
    ax.set_ylabel(r'$O_{ij}$')
    fig.tight_layout()
    fig.savefig(name)


def hours_distributions(df, w, y, name='full_run/figs/hourly_distributions.pdf'):
    plots.latexify(6, 6, 1)
    fig, axn = plt.subplots(4, 1, sharex=True)

    wns = [np.mean(w * df.iloc[:, i].values) for i in range(df.shape[1])]
    axn[0].plot(wns)
    axn[0].set_ylabel(r'$\langle w^h_{ij} \rangle$')
    axn[0].grid()

    corrs = [spearmanr(y, df.iloc[:, i])[0] for i in range(df.shape[1])]
    axn[2].plot(corrs)


    step = 8
    ticklabels = get_week_labels(step)
    axn[2].set_xticks(range(0, df.shape[1], step))
    axn[2].set_ylabel(r'$Spearman(O_{ij}, \phi^h_{ij})$')
    axn[2].grid()

    y_avg = [np.average(y, weights=df.iloc[:, i].values) for i in range(df.shape[1])]
    df = df.astype(bool)
    n = float(df.shape[0])
    pct_calls = [len(y[df.iloc[:, i].values])/n for i in range(df.shape[1])]
    axn[3].plot(y_avg)
    axn[3].set_ylabel(r'$\langle O_{ij}| \phi^h_{ij} \rangle$')
    axn[3].grid()

    axn[1].plot(pct_calls)
    axn[1].set_ylabel(r'$\%$' + ' active' )
    axn[1].grid()

    axn[3].set_xticklabels(ticklabels, rotation=90)
    axn[3].set_xlabel(r'$h$')

    fig.tight_layout()
    fig.savefig(name)


if __name__=="__main__":
    plt.ion()
    plots.latexify(4, 4, 2)
    net = pd.read_csv('full_run/net.edg', sep=' ')
    idx = net.w > 2
    net = net[idx]
    weight_vs_overlap(net.w, net.ovrl)
    iet = pd.read_csv('full_run/ietd_stats.txt', sep=' ')
    iet = iet[idx]
    mu_vs_overlap(iet.mu, net.ov)
#   Note: burstiness requires w > 3, or there is high bias)
    idx2 = net.w > 3
    b_vs_overlap(iet.b[idx2], net.ovrl[idx2])
    avg_relay_vs_overlap(iet.mu_r[idx], net.ovrl[idx])
    rel_frsh_vs_overlap(iet.r_frsh[idx2], net.ovrl[idx2])
    age_vs_overlap(iet.age, net.ovrl)
    tmp_stab_vs_overlap(iet.t_stb, net.ovrl)
    m_vs_overlap(iet.m, net.ovrl)
    bt = pd.read_csv('full_run/bursty_train_stats.txt', sep=' ')
    bt = bt[idx]
# Note: for bt_tsig use w > 4i
    rp = pd.read_csv('full_run/reciprocity.txt', sep=' ', names=['0', '1', 'r'])

    dc = pd.read_csv('full_run/daily_cycles_comp.txt', sep=' ')

    net = pd.read_csv('full_run/net.edg', sep=' ')
    tmp = pd.read_csv('full_run/temporal_overlap.txt', sep=' ')
    max_idx = tmp.shape[0]
    net = net.iloc[range(max_idx)]

    idx = (tmp.ov_mean > 0) & (tmp.ovrl > 0)
    g = plots.loglogjointdistr(tmp.ovrl[idx], tmp.ov_mean[idx], xlim=(.01, 1), ylim=(.01, 1), xlabel=r'$O_{ij}$', ylabel=r'$\bar{O^t_{ij}}$', gridsize=(55, 35))
    g.savefig('full_run/figs/overlap_joint.pdf')

    fig, ax = plt.subplots(1)
    ax.hist(tmp.ov_mean, 100, alpha=.3, label=r'$\bar{O^t_{ij}}$', log=True, normed=True)
    ax.hist(tmp.ov_mean[tmp.all_t_comm > 0], 100, alpha=.3, label=r'$All neighbors$', log=True, normed=True)
    ax.hist(tmp.ov_mean[tmp.no_t_comm > 0], 100, alpha=.3, label=r'$No Neighbors$', log=True, normed=True)
    ax.legend(loc=0)

    tmp['noc_frac'] = (tmp.no_t_comm + .0)/(tmp.all_t_comm + tmp.some_t_comm + tmp.no_t_comm)
    plots.linlinjointdistr(tmp.noc_frac[idx], tmp.ovrl[idx], xlim=(0, 1), ylim=(0., .3))

    idx = tmp.w > 2
    possible = tmp[idx].index
    samp = np.random.choice(possible, 1000000)

    tmp = tmp.iloc[samp]
    tmp.to_csv('full_run/tmp_samp2.txt', index=False, sep=' ')
###### TODO: DELETE
# This is for obtaining a sample for week vec. So far we got the first i=3353197
    w = open('full_run/wv_samp.txt', 'w')
    r = open('full_run/week_vec_call.txt', 'r')
    i = 0
    row = r.readline()
    while row:
        if i in samp:
            w.write(row)
        i += 1
        row = r.readline()
    w.close()
    r.close()


