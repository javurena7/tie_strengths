
"""
Script to analyze if the selection of non-

"""


import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

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
#degs.to_csv(path + 'neighbors_comparison.txt', index=False, sep=' ')


df = pd.read_table('data/neighbors_comparison_sample.txt', sep=' ')


df['ki_kj_no_nn'] = df['deg_0'] + df['deg_1'] -2 -2*df['n_ij']
df['eki_kj_no_nn'] = df['edeg_0'] + df['edeg_1'] -2 -2*df['en_ij']
df['fki_kj_no_nn'] = df['ki_kj_no_nn']/df['eki_kj_no_nn']


# Delete extreme values
df = df[df.eki_kj_no_nn < 10000]

# Fisher test
df['fisher_test'] = df.apply(lambda r: fisher_exact([[r.n_ij, r.en_ij - r.n_ij], [r.ki_kj_no_nn, r.eki_kj_no_nn - r.ki_kj_no_nn]]), axis=1)

df['f_ratio'] = df.fisher_test.apply(lambda r: r[0])
df['f_pval'] = df.fisher_test.apply(lambda r: r[1])
del df['fisher_test']


ks_2_samp(df.k_n_f, df.fki_kj_no_nn)

# unpooled z test
df['sigma_d'] = df.apply(lambda r: np.sqrt(r.k_n_f*(1 - r.k_n_f)/r.en_ij + r.fki_kj_no_nn*(1-r.fki_kj_no_nn)/r.eki_kj_no_nn), axis=1)


#bayesian prior
#
a = .25
df['fn_ij_prior'] = df.apply(lambda r: (a + r.n_ij)/(5*a + r.en_ij), axis=1)
df['fki_kj_prior'] = df.apply(lambda r: (a + r.ki_kj_no_nn)/(5*a + r.eki_kj_no_nn), axis=1)

# Plots
df = df[df.en_ij > 0] # remove data with no estimation errors
df = df[df.eki_kj_no_nn > 0] #remove data with no estimation errors


# Plot empirical distributions
plt.hist(df.fki_kj_no_nn, 50, normed=True, alpha=.5, label='Not-common neighbors')
plt.hist(df.k_n_f, 50, normed=True, alpha=.5, label='Common neighbors')
plt.xlabel('Probability of Observation')
plt.ylabel('Frequency')
plt.legend(loc='best')


# Plot empirical distribution s.t. n_ij > 0
idx = df.n_ij > 0
plt.hist(df.fki_kj_no_nn[idx], 50, normed=True, alpha=.5, label='Not-common neighbors')
plt.hist(df.k_n_f[idx], 50, normed=True, alpha=.5, label='Common neighbors')
plt.xlabel('Probability of Observation')
plt.ylabel('Frequency')
plt.legend(loc='best')

np.mean(df.k_n_f[idx])
np.mean(df.fki_kj_no_nn[idx])

# Create heatmap of probabilities
#
#heatmap, xedges, yedges = np.histogram2d(x, y, bins=50)
heatmap, xedges, yedges = np.histogram2d(df.fki_kj_no_nn, df.k_n_f, bins=50)
heatmap = np.log(heatmap + 1)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
plt.clf()
plt.imshow(heatmap, extent=extent, origin='lower')
plt.ylabel('Probability of Observing Not-Common Neighbors')
plt.xlabel('Probability of Observing Common Neighbors')


heatmap, xedges, yedges = np.histogram2d(df.fki_kj_no_nn[idx], df.k_n_f[idx], bins=50)
heatmap = np.log(heatmap + 1)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
plt.clf()
plt.imshow(heatmap, extent=extent, origin='lower')
plt.ylabel('Probability of Observing Not-Common Neighbors')
plt.xlabel('Probability of Observing Common Neighbors')

# Using Bayesian priors
a = .25
df['fn_ij_prior'] = df.apply(lambda r: (a + r.n_ij)/(5*a + r.en_ij), axis=1)
df['fki_kj_prior'] = df.apply(lambda r: (a + r.ki_kj_no_nn)/(5*a + r.eki_kj_no_nn), axis=1)

plt.hist(df.fki_kj_prior, 50, normed=True, alpha=.5, label='Not-common neighbors')
plt.hist(df.fn_ij_prior, 50, normed=True, alpha=.5, label='Common neighbors')
plt.xlabel('Probability of Observation')
plt.ylabel('Frequency')
plt.legend(loc='best')

plt.hist(df.fki_kj_prior[idx], 50, normed=True, alpha=.5, label='Not-common neighbors')
plt.hist(df.fn_ij_prior[idx], 50, normed=True, alpha=.5, label='Common neighbors')
plt.xlabel('Probability of Observation')
plt.ylabel('Frequency')
plt.legend(loc='best')


a = 2
df['fn_ij_prior2'] = df.apply(lambda r: (a + r.n_ij)/(5*a + r.en_ij), axis=1)
df['fki_kj_prior2'] = df.apply(lambda r: (a + r.ki_kj_no_nn)/(5*a + r.eki_kj_no_nn), axis=1)

plt.hist(df.fki_kj_prior2, 50, normed=True, alpha=.5, label='Not-common neighbors')
plt.hist(df.fn_ij_prior2, 50, normed=True, alpha=.5, label='Common neighbors')
plt.xlabel('Probability of Observation')
plt.ylabel('Frequency')
plt.legend(loc='best')

plt.hist(df.fki_kj_prior2[idx], 50, normed=True, alpha=.5, label='Not-common neighbors')
plt.hist(df.fn_ij_prior2[idx], 50, normed=True, alpha=.5, label='Common neighbors')
plt.xlabel('Probability of Observation')
plt.ylabel('Frequency')
plt.legend(loc='best')


heatmap, xedges, yedges = np.histogram2d(df.fki_kj_prior[idx], df.fn_ij_prior[idx], bins=50)
heatmap = np.log(heatmap + 1)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
plt.clf()
plt.imshow(heatmap, extent=extent, origin='lower')
plt.ylabel('Probability of Observing Not-Common Neighbors (soft-prior)')
plt.xlabel('Probability of Observing Common Neighbors (soft-prior)')
