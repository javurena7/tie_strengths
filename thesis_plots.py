from tie_strengths import plots
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata
import seaborn as sb

def weight_vs_overlap(w, o, name='full_run/figs/weight_overlap.pdf'):
    g = plots.loglinjointdistr(w, o, height=5)
    g.savefig(name)
    return g


def mu_vs_overlap(mu, o, name='full_run/figs/mu-iet_overlap.pdf'):
    idx = mu > 0
    # Note, we are removing data with zero mu, which are 730000 links (mean w=4.2, and mean ovrl=.03)
    g = plots.loglinjointdistr(mu[idx] + 1, o[idx], height=5, xlabel=r'$\bar{\tau}_{ij}$', xlim=(1, 100))
    g.savefig(name)
    return g


def b_vs_overlap(b, o, name='full_run/figs/b_overlap.pdf'):
    idx = b.notnull()
    g = plots.linlinjointdistr(b[idx], o[idx], height=5, xlabel=r'$B_{ij}$', xlim=(-.85, .85), gridsize=(35, 45))
    g.savefig(name)
    return g


def avg_relay_vs_overlap(mu_r, o, name='full_run/figs/mu_r_overlap.pdf'):
    idx = (mu_r.notnull()) & (mu_r < np.inf)
    g = plots.loglinjointdistr(mu_r[idx] + .5, o[idx], height=5, xlabel=r'$\bar{\tau}_R$', xlim=(.9, 70), gridsize=(25, 35))
    g.savefig(name)


def rel_frsh_vs_overlap(r, o, name='full_run/figs/r-fresh_overlap.pdf'):
    idx = (r.notnull()) & (r < np.inf)
    g = plots.loglinjointdistr(r[idx] + 1, o[idx], height=5, xlabel=r'$\hat{f_r}$', xlim=(1, 1300))
    g.savefig(name)


def age_vs_overlap(age, o, name='full_run/figs/age_overlap.pdf'):
    g = plots.linlinjointdistr(age, o, height=5, xlabel=r'$age$', xlim=(0, max(age)))
    g.savefig(name)


def tmp_stab_vs_overlap(t, o, name='full_run/figs/t-stab_overlap.pdf'):
    g = plots.linlinjointdistr(t, o, height=5, xlabel=r'$TS$', xlim=(-1, 120))
    g.savefig(name)


def m_vs_overlap(m, o, name='full_run/figs/m_overlap.pdf'):
    idx = m.notnull()
    g = plots.linlinjointdistr(m[idx], o[idx], height=5, xlabel=r'$M_{ij}$', xlim=(-1, 1))
    g.savefig(name)


def bt_mu_vs_overlap(mu, o, name='full_run/figs/mu-bt_overlap.pdf'):
    g = plots.loglinjointdistr(1/mu, o, height=5, xlabel=r'$\frac{1}{\bar{E}}$', xlim=(1./15, 1.1), gridsize=(45, 55))
    g.savefig(name)


def bt_cv_vs_overlap(cv, o, name='full_run/figs/cv-bt_overlap.pdf'):
    idx = (cv.notnull()) & (cv < np.inf)
    g = plots.loglinjointdistr(cv[idx] + 1, o[idx], height=5, xlabel=r'$CV_E$', xlim=(1, 8), gridsize=(45, 55))
    g.savefig(name)


def bt_n_vs_overlap(n, o, name='full_run/figs/n-bt_overlap.pdf'):
    g = plots.loglinjointdistr(n+1, o, height=5, xlim=(1, 700), gridsize=(40, 60), xlabel=r'$N_E$')
    g.savefig(name)


def bt_tmu_vs_overlap(tmu, o, name='full_run/figs/tmu_bt_overlap.pdf'):
    g = plots.linlinjointdistr(tmu, o, xlim=(0, 1), xlabel=r'$\bar{t^b}$')
    g.savefig(name)


def bt_tsig_vs_overlap(tsig, o, name='full_run/figs/tmu_bt_overlap.pdf'):
    idx = tsig.notnull()
    g = plots.linlinjointdistr(tsig[idx], o[idx], xlim=(0, .5), xlabel=r'$\sigma_{t^b}$')
    g.savefig(name)


if __name__=="__main__":
    plots.latexify(4, 4, 2)
    net = pd.read_csv('full_run/net.edg', sep=' ')
    idx = net.w > 2
    net = net[idx]
    weight_vs_overlap(net.w, net.ovrl)
    iet = pd.read_csv('full_run/ietd_stats.txt', sep=' ')
    iet = iet[idx]
    mu_vs_overlap(iet.mu, net.ov)
#   Note: burstiness requires w > 3, or there is high bias)
    idx = net.w > 3
    b_vs_overlap(iet.b[idx], net.ovrl[idx])
    avg_relay_vs_overlap(iet.mu_r[idx], net.ovrl[idx])
    rel_frsh_vs_overlap(iet.r_frsh[idx], net.ovrl[idx])
    age_vs_overlap(iet.age, net.ovrl)
    tmp_stab_vs_overlap(iet.t_stb, net.ovrl)
    m_vs_overlap(iet.m, net.ovrl)
    bt = pd.read_csv('full_run/bursty_train_stats.txt', sep=' ')
    bt = bt[idx]
# Note: for bt_tsig use w > 4

