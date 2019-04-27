import numpy as np:
import pandas as pd
from scipy.stats import ks_2samp


df = pd.read_csv('full_run/overlap_sampling_sample.txt')
df = df[df.ek_nn > 0]
df = df[df.en_ij > 0]

df['pk'] = df['k_nn']/df['ek_nn']
df['pn'] = df['n_ij']/df['en_ij']




plots.latexify(4, 4, 2)
fig, ax = plt.subplots(1)
ax.hist(df.pk, 50, alpha=.5, label=r'$p_K$', color='b', normed=True)
ax.hist(df.pn, 50, alpha=.5, label=r'$p_N$', color='r', normed=True)
ax.set_xlabel('Selection Probabilities ' + r'$p_N$' + ', ' + r'$p_K$')
ax.set_ylabel('Frequencies')
ax.legend(loc=0)
fig.savefig('full_run/figs/pn_pk_distribs.pdf')

df['pn_samp'] = df.apply(lambda x: round(np.random.binomial(x[1], (x[0] + x[2])/(x[1] + x[3]))/x[1], 3), axis=1)
df['pk_samp'] = df.apply(lambda x: round(np.random.binomial(x[3], (x[0] + x[2])/(x[1] + x[3]))/x[3], 3), axis=1)


fig, ax = plt.subplots(1)
ax.hist(df.pk_samp, 50, alpha=.5, label=r'$p^S_K$', color='b', normed=True)
ax.hist(df.pn_samp, 50, alpha=.5, label=r'$p^S_N$', color='r', normed=True)
ax.set_xlabel('Sampled Probabilities ' + r'$p^S_N$' + ', ' + r'$p^S_K$')
ax.set_ylabel('Frequencies')
ax.legend(loc=0)
fig.savefig('full_run/figs/pn_pk_samps.pdf')



plots.latexify(4, 4, 2)
fig, ax = plt.subplots(1)
num_bins = 50
counts, bin_edges = np.histogram (df.pk, bins=num_bins, normed=True)
cdf = np.cumsum (counts)
ax.plot(bin_edges[1:], cdf/cdf[-1], label=r'$p_K$', color='b')

counts, bin_edges = np.histogram (df.pk_samp, bins=num_bins, normed=True)
cdf = np.cumsum (counts)
ax.plot(bin_edges[1:], cdf/cdf[-1], label=r'$p^S_K$', color='k')

ax.legend(loc=0)
ax.set_xlabel(r'$p_K$')
ax.set_ylabel(r'$F(p_K)$')
fig.savefig('full_run/figs/bias_pk.pdf')

fig, ax = plt.subplots(1)

counts, bin_edges = np.histogram (df.pn, bins=num_bins, normed=True)
cdf = np.cumsum (counts)
ax.plot(bin_edges[1:], cdf/cdf[-1], label=r'$p_N$', color='b')

counts, bin_edges = np.histogram (df.pn_samp, bins=num_bins, normed=True)
cdf = np.cumsum (counts)
ax.plot(bin_edges[1:], cdf/cdf[-1], label=r'$p^S_N$', color='k')

ax.legend(loc=0)
ax.set_xlabel(r'$p_N$')
ax.set_ylabel(r'$F(p_N)$')
fig.savefig('full_run/figs/bias_pn.pdf')

# Bayesian estimates
df = pd.read_csv('full_run/overlap_sampling_sample.txt')
df = df[df.ek_nn > 0]
df = df[df.en_ij > 0]
df = df[df.n_ij <= df.en_ij]
df = df[df.k_nn <= df.ek_nn]


df['pk'] = (df['k_nn'] + 1)/(df['ek_nn'] + 2)
df['pn'] = (df['n_ij'] + 1)/(df['en_ij'] + 2)
plots.latexify(4, 4, 2)
fig, ax = plt.subplots(1)
ax.hist(df.pk, 50, alpha=.5, label=r'$p_K$', color='b', normed=True)
ax.hist(df.pn, 50, alpha=.5, label=r'$p_N$', color='r', normed=True)
ax.set_xlabel('Selection Probabilities ' + r'$p_N$' + ', ' + r'$p_K$')
ax.set_ylabel('Frequencies')
ax.text(.7, 5, r'$\alpha = 1, \beta = 1$')
ax.legend(loc=0)
fig.savefig('full_run/figs/pn_pk_11.pdf')

df['pk'] = (df['k_nn'] + 1)/(df['ek_nn'] + 5)
df['pn'] = (df['n_ij'] + 1)/(df['en_ij'] + 5)
plots.latexify(4, 4, 2)
fig, ax = plt.subplots(1)
ax.hist(df.pk, 50, alpha=.5, label=r'$p_K$', color='b', normed=True)
ax.hist(df.pn, 50, alpha=.5, label=r'$p_N$', color='r', normed=True)
ax.set_xlabel('Selection Probabilities ' + r'$p_N$' + ', ' + r'$p_K$')
ax.set_ylabel('Frequencies')
ax.text(.7, 5, r'$\alpha = 1, \beta = 4$')
ax.legend(loc=0)
fig.savefig('full_run/figs/pn_pk_14.pdf')

df['pk'] = (df['k_nn'] + 1)/(df['ek_nn'] + 7)
df['pn'] = (df['n_ij'] + 1)/(df['en_ij'] + 7)
plots.latexify(4, 4, 2)
fig, ax = plt.subplots(1)
ax.hist(df.pk, 50, alpha=.5, label=r'$p_K$', color='b', normed=True)
ax.hist(df.pn, 50, alpha=.5, label=r'$p_N$', color='r', normed=True)
ax.set_xlabel('Selection Probabilities ' + r'$p_N$' + ', ' + r'$p_K$')
ax.set_ylabel('Frequencies')
ax.text(.7, 5, r'$\alpha = 1, \beta = 6$')
ax.legend(loc=0)
fig.savefig('full_run/figs/pn_pk_16.pdf')


df['pk'] = (df['k_nn'])/(df['ek_nn'])
df['pn'] = (df['n_ij'])/(df['en_ij'])
df = df[df.n_ij > 0]
plots.latexify(4, 4, 2)
fig, ax = plt.subplots(1)
ax.hist(df.pk, 50, alpha=.5, label=r'$p_K$', color='b', normed=True)
ax.hist(df.pn, 50, alpha=.5, label=r'$p_N$', color='r', normed=True)
ax.set_xlabel('Selection Probabilities ' + r'$p_N$' + ', ' + r'$p_K$')
ax.set_ylabel('Frequencies')
ax.text(.55, 4.5, r'$\alpha = 0, \beta = 0, n_{ij} > 0$')
ax.legend(loc=0)
fig.savefig('full_run/figs/pn_pk_00.pdf')


df = pd.read_csv('full_run/overlap_tests.csv')

fig, ax = plt.subplots(1)
ax.hist(df.p_greater, 50, alpha=.5, color='g', normed=True, label=r'$\alpha=1, \beta=1$')
ax.hist(df.p_greater_1_4, 50, alpha=.5, normed=True, label=r'$\alpha=1, \beta=4$')
ax.hist(df.p_greater_1_6, 50, alpha=.5, normed=True, label=r'$\alpha=1, \beta=6$')
ax.set_xlabel(r'$P(p_{N_{ij}} > p_{K_{ij}})$')
ax.set_ylabel('Frequencies')
ax.legend(loc=0)
fig.savefig('full_run/figs/probs_greater.pdf')



df = pd.read_csv('full_run/overlap_test_0_0.csv')
df1 = pd.read_csv('full_run/overlap_test.csv')
df1 = df1[df1.n_ij > 0]
fig, ax = plt.subplots(1)
ax.hist(df.p_greater, 50, alpha=.6, color='g', normed=True, label=r'$\alpha=1, \beta=1$')
ax.hist(df.p_greater_1_4, 50, alpha=.5, normed=True, label=r'$\alpha=1, \beta=4$')
ax.hist(df1.p_greater_1_6, 50, alpha=.4, color='r', normed=True, label=r'$\alpha=1, \beta=6$')
ax.hist(df.p_greater_0_0, 50, alpha=.5, normed=True, label=r'$\alpha=0, \beta=0$')
ax.set_xlabel(r'$P(p_{N_{ij}} > p_{K_{ij}}| n_{ij} > 0)$')
ax.set_ylabel('Frequencies')
ax.legend(loc=0)
fig.savefig('full_run/figs/probs_greater_00.pdf')


