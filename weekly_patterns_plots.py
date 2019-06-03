import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from scipy.stats import rankdata, pearsonr, spearmanr
#from tie_strengths import plots
from itertools import product

from sklearn.decomposition import PCA
plt.ion()

"""
df = pd.read_csv('full_run/week_vec_call_sample.txt', sep=' ')
del df['1']
del df['0']
cols = df.columns

w = pd.read_csv('full_run/net.edg', sep=' ')
idx = w.w > 5
y = w[idx]
y = y.ovrl.iloc[:400000]
w = w[idx]
w = w.w.iloc[:400000]

ccorrs = pd.read_csv('full_run/weekly_pattern_corrs.txt', sep=' ')
mat = ccorrs.values

"""

def evaluate_clusters(w1, w2, o, idx):
    c1 = pearsonr(w1, o)[0]
    c2 = pearsonr(w2, o)[0]
    #c1 = np.var(o[w1.values])
    #c2 = np.var(o[w2.values])
    w = (w1 + w2)/2
    c = pearsonr(w, o)[0]
    #c = np.var(o[w.values])
    if c >= 1.05 * max([c1, c2]):
    #if c < c1 + c2:
        new = min(idx)
        de = max(idx)
        return new, de, w
    else:
        return idx, None, None

def get_max_corr_idx(corrs, explored, cols):
    for ex in explored:
        corrs[ex[0]][ex[1]] = 0
        corrs[ex[1]][ex[0]] = 0
    corrs = np.triu(corrs, 1)
    idx = np.unravel_index(np.argmax(corrs), corrs.shape)
    mcorr = corrs[idx]
    idx = [cols[idx[0]], cols[idx[1]]]
    return idx, mcorr

def get_max_corr_idx_ordered(corrs, explored, cols):
    for ex in explored:
        corrs[ex[0]][ex[1]] = 0
        corrs[ex[1]][ex[0]] = 0

    corrs = np.triu(corrs, 1)
    idx = np.unravel_index(np.argmax(corrs), corrs.shape)
    mcorr = corrs[idx]
    idx = [cols[idx[0]], cols[idx[1]]]
    return idx, mcorr

def update_explored(explored, del_idx):
    explored = [ex for ex in explored if del_idx not in ex]
    return explored

def iterate_corrs(df, o):
    corrs = df.corr()
    explored = []
    idx, mcorr = get_max_corr_idx(corrs, explored, df.columns)
    idx_temp = {}
    ids = {i:[i] for i in df.columns}
    i = 1
    while mcorr > 0:
        idx_n, del_idx, w = evaluate_clusters(df[idx[0]], df[idx[1]], o, idx)
        if w is not None:
            df[idx_n] = w
            df.drop(del_idx, axis=1, inplace=True)
            tmp = ids.pop(del_idx)
            ids[idx_n] += tmp
            explored = update_explored(explored, del_idx)
            corrs.drop(del_idx, axis=0, inplace=True)
            corrs.drop(del_idx, axis=1, inplace=True)
        else:
            explored.append(idx)
        if np.mod(i, 1) == 0:
            corrs = df.corr()
        i = i + 1
        idx, mcorr = get_max_corr_idx(corrs, explored, df.columns)
        print(str(mcorr))
    return df, ids

def mat_from_ids(ids, cols):
    col_ids = {k:i for i, k in enumerate(cols)}
    i = 10
    z = np.zeros((168, 168))
    for k, v in ids.iteritems():
        hlp = [col_ids[rs] for rs in v]
        #if len(v) == 1:
        #    z[idx] = 0
        #else:
        for idx in product(hlp, hlp):
            z[idx] = i
        i += 1
    return z

def pca_clustering(pca_comps, pca_vars, num_comps):
    ds = []
    pca_vars = np.exp(pca_vars[:num_comps])
    pca_vars = pca_vars/sum(pca_vars)
    for k in range(num_comps):
        d = np.zeros((168, 168))
        for i in range(168):
            for j in range(168):
                ki = pca_comps[k][i]
                kj = pca_comps[k][j]
                d[i][j] = np.abs(ki - kj) #/np.abs(ki + kj)
        ds.append(pca_vars[k]*d)
    return ds

def get_covs(df):
    covs = np.cov(df, rowvar=False)
    # upper triangular
    covs = np.triu(covs, 1)
    return covs

def plot_pca(a1, a2, name='full_run/figs/pca.pdf'):
    """
    Plot percentage of varaince explained by PCA for binary and percentage (a1 and a2 are variance explained)
    """
    plots.latexify(3, 3, 2)
    fig, ax = plt.subplots(1)
    ax.plot(a1, label=r'$\phi^h_{ij}$')
    ax.plot(a2, label=r'$I_{\phi^h_{ij} > 0}$')
    #ax.set_yscale('log')
    ax.set_ylabel('Variance Explained')
    ax.set_xlabel('PC')
    ax.legend(loc=0)
    fig.tight_layout()
    fig.savefig(name)

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

def plot_pca_c1(b1, b2, name='full_run/figs/pca_c1.pdf'):
    """
    Plot first PCA component
    """
    plots.latexify(5, 3, 2)
    fig, ax = plt.subplots(1)
    ax.plot(b1, label=r'$\phi^h_{ij}$')
    ax.plot(b2, label=r'$I_{\phi^h_{ij} > 0}$')
    ticklabels = get_week_labels(12)
    ax.set_xticks(range(0, 168, 12))
    ax.set_xticklabels(ticklabels, rotation=90)
    ax.set_ylabel(r'$h$' + ' Weight')
    ax.set_xlabel(r'$h$')
    ax.legend(loc=0)
    fig.tight_layout()
    fig.savefig(name)

def plot_pca_c(b, bi, name='full_run/figs/pca_comps.pdf'):
    """
    Plot PCA components
    """

    plots.latexify(6, 7, 1)
    fig, axn = plt.subplots(5, 1, sharex=True)
    labels = [r'$\phi^h_{ij}$', r'$I_{\phi^h_{ij} > 0}$']
    zero = [0]*168

    for i in range(5):
        axn[i].plot(zero, alpha=.2, color='grey')
        axn[i].plot(b[i], label=labels[0])
        axn[i].plot(bi[i], label=labels[1])
        axn[i].set_ylabel(r'$PC$' + ' ' + str(i+1) + ' ' + r'$w$')
        axn[i].grid()
    axn[0].legend(loc=0)
    ticklabels = get_week_labels(8)
    axn[i].set_xticks(range(0, 168, 8))
    axn[i].set_xticklabels(ticklabels, rotation=90)
    fig.tight_layout()
    fig.savefig(name)

    """
def mca_clustering(mat, cutoff):
    import markov_clustering as mc
    mat = mat.copy()
    mat[mat <= cutoff] = 0
    mat[mat > cutoff] = 1

    result  = mc.run_mcl(mat)
    clusters = mc.get_clusters(result)

    clusters = {i:[cols[j] for j in c] for i, c in enumerate(clusters)}
    corrs = []
    n = df.shape[0] + 0.
    for i in range(len(clusters)):

        dfc = df[clusters[i]].sum(1)
        idx = dfc > 0
        corrs.append((len(dfc[idx])/n, spearmanr(dfc[idx], y.values[idx])[0], np.average(y, weights=dfc.values)))
    #corrs = [(i, spearmanr(df[clusters[i]].sum(1), y)[0]) for i in range(len(clusters))]

    return result, clusters, corrs
    """


def decouple_overlap(w, y):
    """
    Decouple w and y by rescaling y according to similar w values (for a binning of w, we obtain the average overlap of the bin; we rescale all overlap values according to this)
    """
    n_cuts = 11
    w_c = pd.qcut(w, n_cuts, labels=False)
    av_ovs = {k: np.mean(y[w_c == k]) for k in range(n_cuts)}
    w_c = w_c.map(av_ovs)
    y_r = y/w_c

    return y_r

def arrange_cluster_results(result, ccorrs):
    result[result > 0] = 1
    result = np.unique([r for r in result if sum(r) > 0], axis=0)
    lr = result.shape[0]

    for i, r in enumerate(result):
        r *= ccorrs[lr - i - 1][1]
    result[result==0] = np.nan

    return result


def get_diff_clusts(mat, cutoffs):
    results = []
    #clusters = []
    ccorrs = []
    for ct in cutoffs:
        res, cls, ccrs = mca_clustering(mat, ct)
        res = arrange_cluster_results(res, ccrs)
        results.append(res)
        c = [x[1] for x in ccrs]
        cx = [np.abs(x[0]*x[1]) for x in ccrs]
        cm = [x[2] for x in ccrs]
        ccorrs.append((np.mean(c), np.percentile(c, 5), np.percentile(c, 95), np.max(cx), np.mean(cx), np.mean(cm), np.percentile(cm, 5), np.percentile(cm, 95), np.percentile(cx, 5), np.percentile(cx, 95)))
    return results, ccorrs


def plot_different_clusters(results, cutoffs, idx):
    fig, axn = plt.subplots(len(idx), 1, sharex=True)
    vmin = min([np.nanmin(r) for r in results])
    vmax = max([np.nanmax(r) for r in results])
    vmin = min([vmin, -vmax])
    vmax = max([-vmin, vmax])
    for i, idx_n in enumerate(idx):
        im = axn[i].imshow(results[idx_n], origin='lower', cmap='coolwarm', vmin=vmin, vmax=vmax)
        axn[i].set_yticks([])
    axn[i].set_xticks(range(0, 168, 8))
    ticklabels = get_week_labels(8)
    axn[i].set_xticklabels(ticklabels, rotation=90)
    fig.tight_layout()
    fig.colorbar(im, ax=axn)
    return fig, axn

def plot_cluster_overlap(cutoffs, ccorrs, name='full_run/figs_reclust/cluster_overlap.pdf'):
    fig, ax = plt.subplots()
    ax.plot(cutoffs, [x[5] for x in ccorrs], color='r')
    ax.plot(cutoffs, [x[6] for x in ccorrs], color='grey', alpha=.6)
    ax.plot(cutoffs, [x[7] for x in ccorrs], color='gray', alpha=.6)
    ax.set_ylabel(r'$\langle O_{ij} | \{\phi^c_{ij}\}_c\rangle$')
    ax.set_xlabel(r'$\psi$')
    fig.tight_layout()
    fig.savefig(name)
    return fig, ax


def plot_cluster_correlation_weights(cutoffs, ccorrs, name='full_run/figs_reclust/cluster_overlap_correlation.pdf'):
    fig, ax = plt.subplots()
    ax.plot(cutoffs, [x[4] for x in ccorrs], color='r')
    ax.plot(cutoffs, [x[8] for x in ccorrs], color='grey', alpha=.6)
    ax.plot(cutoffs, [x[3] for x in ccorrs], '.',color='grey', alpha=.6)
    ax.plot(cutoffs, [x[9] for x in ccorrs], color='gray', alpha=.6)
    ax.set_ylabel(r'$WS^c$')
    ax.set_xlabel(r'$\psi$')
    fig.tight_layout()
    fig.savefig(name)
    return fig, ax
# use cmap='gnuplot2'


def cluster_to_heatmap(clusters):
    ht = np.zeros((7, 24))
    for k, v in clusters.items():
        for d in v:
            dd = d.split('_')
            ht[int(dd[0]), int(dd[1])] = int(k) + 1
    return ht

def plot_clusters(ht, name='full_run/figs_reclust/clusters.pdf', ncolors=12):
    #plots.latexify(6, 3, 1)
    y_ticks = ['Mo.', 'Tu.', 'We.', 'Th.', 'Fr.', 'Sa.', 'Su.']
    x_ticks = [x for x in range(0, 24)]
    g = sns.heatmap(ht, cmap=sns.color_palette('Paired', ncolors, desat=.5), annot=True, cbar=False)
    g.axes.set_xticks(x_ticks)
    g.axes.set_yticks(range(0, 7))
    g.axes.set_yticklabels(y_ticks)
    g.axes.set_xticklabels(x_ticks)
    g.axes.set_xlabel('Hour')
    g.axes.set_ylabel('Day')
    g.get_figure().tight_layout()
    g.get_figure().savefig(name)
    return g

def clusters_to_df(df, clusters):
    df_new = pd.DataFrame()
    for k, v in clusters.items():
        df_new['c' + str(k + 1)] = df[v].sum(1)
    return df_new


"""
name = 'full_run/weekly_cluster_ids_spearman.p'
with open(name, 'w') as fp:
    json.dump(ids, fp)
"""

