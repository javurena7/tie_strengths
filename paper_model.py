from sklearn.model_selection import train_test_split
import numpy as np; from numpy import inf
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import matthews_corrcoef
import pickle
from os import listdir

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier


class PredictTieStrength(object):
    def __init__(self, y_var, data_path='../paper_run/sample/', models=['SVC', 'LR'], remove=['deg_0', 'deg_1', 'n_ij', 'ov_mean', 'e0_div', 'e1_div', 'bt_tsig1'], save_prefix='../paper/'):
        self.save_prefix = save_prefix
        self._init_models(models)
        if data_path:
            self.read_tables(data_path, y_var, remove)
            self.single_scores = self._init_scores()
        else:
            self.variables = []
            self.x, self.y, self.scores = None, None, {}
        self.dual_scores = {}

        self.col_labels = {'mu': r'$\bar{\tau}$', 'sig': r'$\bar{\sigma_{\tau}}$', 'b': r'$B$', 'mu_r': r'$\bar{\tau}_R$', 'r_frsh': r'$\hat{f}$', 'age': r'$age$', 't_stb': r'$TS$', 'm': r'$M$', 'bt_mu': r'$\bar{E}$', 'bt_sig': r'$\sigma^{E}$', 'bt_cv': r'$CV^E$', 'bt_n': r'$N^E$', 'bt_tmu': r'$\bar{t}$', 'bt_tsig': r'$\sigma_{t}$', 'bt_logt': r'$\log(T)$', 'out_call_div': r'$JSD$', 'r': r'$r$', 'w': r'$w$', 'e0_div': r'$JSD_{diff}$', r'ovrl': 'HT'}
        self.col_labels.update({'c' + str(i): 'C' + r'$' + str(i) + '$' for i in range(1, 16)})

    def _init_models(self, models):
        available_models = {'SVC': 'LinearSVC',
                'LR': 'LogisticRegression',
                'RF': 'RadomForestClassifier',
                'GP': 'GaussianProcessClassifier',
                'ABC': 'AdaBoostClassifier',
                'RBF': 'RBF',
                'QDA': 'QuadraticDiscriminantAnalysis',
                'MLP': 'MLPClassifier'}
        self.models = []
        for model in models:
            model_str = available_models[model] + '()'
            self.models.append((model, eval(model_str)))


    def _init_scores(self):
        return {kind[0]: {var: [] for var in self.variables} for kind in self.models}

    def _init_dual_scores(self, fvar):
        self.dual_scores[fvar] = self.init_scores()

    def read_tables(self, path, y_var, remove):
        if 'delta_t' in path:
            deltas = pickle.load(open(path + 'deltas.p', 'rb'))
            colnames = ['ovrl', '0', '1'] + sorted(deltas, key=lambda x: int(x))
            df = pd.read_csv(path + 'new_full_bt_n.txt', sep=' ', names=colnames, index_col=['0', '1'], skiprows=1)
        else:

            df = pd.read_csv(path + 'full_df_paper.txt', sep=' ', index_col=['0', '1'])
            df_wc = pd.read_csv(path + 'clustered_df_paper.txt', sep=' ', index_col=['0', '1'])
            df = pd.concat([df, df_wc], axis=1)
            for col in remove:
                df.drop(col, axis=1, inplace=True)

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        self.y = df[y_var]
        df.drop(y_var, axis=1, inplace=True)
        self.variables = list(df.columns)
        self.x = df

    def get_alphas(self):
        self.alphas = [0] + [np.percentile(self.y[self.y > 0], i) for i in range(5, 100, 5)]

    def get_y_class(self, alpha):
        yb = self.y.copy()
        idx = self.y > alpha
        yb[idx] = 1
        yb[~idx] = 0
        self.yb = yb

    def get_training_data(self):
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.yb, test_size=0.33, stratify=self.yb)
        self.x_train = x_train
        self.y_train = y_train.values
        self.x_test = x_test
        self.y_test = y_test.values

    def eval_model(self, model):
        model.fit(self.x_train, self.y_train)
        y_pred = model.predict(self.y_test)
        return matthews_corrcoef(self.y_test, y_pred)

    def eval_single_var(self, model):
        for var, data in self.x_train.iteritems():
            idx = data.notnull()
            x = data.values.reshape(-1, 1)
            xt = self.x_test[var].values.reshape(-1, 1)
            model[1].fit(x, self.y_train)
            y_pred = model[1].predict(xt)
            self.single_scores[model[0]][var].append(matthews_corrcoef(self.y_test, y_pred))

        self.save_scores(self.single_scores[model[0]], 'single_scores.{}.p'.format(model[0]))

    def eval_dual_var(self, model, fvar):
        for var in self.x_train.columns:
            if var == fvar:
                x = self.x_train[var].reshape(-1, 1)
                xt = self.x_test[var].reshape(-1, 1)
            else:
                x = self.x_train[[fvar, var]]
                xt = self.x_test[[fvar, var]]
            model[1].fit(x, self.y_train)
            y_pred = model[1].predict(xt)
            self.dual_scores[fvar][model[0]][var].append(matthews_corrcoef(self.y_test, y_pred))

        self.save_scores(self.dual_scores[fvar][model[0]], 'dual_scores.{}.{}.p'.format(fvar, model[0]))

    def run_alphas(self, fvars=[], single_var=True):
        for fvar in fvars:
            self._init_dual_scores(fvar)
        for alpha in self.alphas:
            print('--- alpha = {} --- '.format(alpha))
            self.get_y_class(alpha)
            self.get_training_data()
            for model in self.models:
                if single_var == True:
                    self.eval_single_var(model)
                for fvar in fvars:
                    self.eval_dual_var(fvar)


    def save_scores(self, obj, name):
        pickle.dump(obj, open(self.save_prefix + name, 'wb'))

    def load_scores(self):
        single_files = [f for f in listdir(self.save_prefix) if 'single_scores' in f]
        dual_files = [f for f in listdir(self.save_prefix) if 'dual_scores' in f]

        for sf in single_files:
            model = sf.split('.')[1]
            self.single_scores[model] = pickle.load(open(self.save_prefix + sf, 'rb'))
        for df in dual_files:
            fvar, model = sf.split('.')[1:2]
            if self.dual_scores.get(fvar, 0) == 0:
                self.dual_scores[fvar] = {}
            self.dual_scores[fvar][model] = pickle.load(open(self.save_prefix + 'dual_scores.p', 'rb'))

    def plot_singlevar_mcc(self):
        htmap_data = [0] * len(alphas)
        for i in range(len(alphas)):
            htmap_data[i] = np.max(scores[i], 0)[0:88:2]

        htmap_data = np.array(htmap_data)
        args = np.argsort(np.mean(htmap_data, 0))

        ticklabels = [col_labels[cols[i]] for i in args]
        htmap_sorted = htmap_data.T[args].T

        g = sns.heatmap(htmap_sorted, cmap='GnBu', center=0, cbar_kws={'label':r'$MCC$'})
        xticks = [i + .5 for i in range(len(args))]
        g.axes.set_xticks(xticks)
        g.axes.set_xticklabels(ticklabels, rotation=90)
        alphas = [round(a, 3) for a in alphas]

        g.axes.set_yticks(range(0, 20, 4))
        g.axes.set_yticklabels(alphas[0:20:4], rotation=0)
        g.axes.set_ylabel(r'$O_{\alpha}$')
        g.axes.set_title(title, loc='left')
        g.get_figure().tight_layout()
        g.get_figure().savefig(name)
        return args


if __name__ == '__main__':
    PTS = pm.PredictTieStrength(y_var='ovrl', data_path='../paper_run/sample/', save_prefix='../paper_run/sample/')
    PTS.get_alphas()
    PTS.run_alphas()

"""

#to = pd.read_csv('full_run/reduced_')
# FOR TEMPORAL OVERLAP
to = pd.read_csv('full_run/temporal_overlap_reduced.txt', sep=' ')
to = to[to.w > 5]
to = to.head(400000)
y = to.ov_mean

x_train, x_test, y_train, y_test = train_test_split(iet, y, test_size=0.3)
n = x_train.shape[0]
rank_train = lambda x: rankdata(x)/n
x_train = x_train.apply(rank_train)
#y_train = rank_train(y_train)
n = x_test.shape[0]
rank_test = lambda x: rankdata(x)/n
x_test = x_test.apply(rank_test)
#y_test = rank_test(y_test)

RF = RandomForestRegressor(35)
RF.fit(x_train, y_train)
sc1 = RF.score(x_test, y_test)
#sc1b = RF.score(x_test, y_test.ov_mean)
f1 = {x: y for x, y in zip(iet.columns, RF.feature_importances_)}
f1 = RF.feature_importances_

ET = ExtraTreesRegressor()
ET.fit(x_train, y_train)
f2 = {x: y for x, y in zip(iet.columns, ET.feature_importances_)}
f2 = ET.feature_importances_


f = list(f1) + list(f2)
model = ['RF'] * (len(f1)) + ['ET'] * (len(f2))

df = pd.DataFrame()
df['var'] = list(iet.columns) * 2
df['I'] = f
df['model'] = model
df['I'].iloc[11] = 0.18
plots.latexify(6, 2.5, 2)
sns.set(rc={'figure.figsize':(6, 2.5)})
g = sns.catplot(x="var", y="I", hue="model", data=df, kind="bar", legend=False)
g.ax.legend(loc=0)



#args = np.argsort(np.mean(np.abs(fis[2]), 0))
ticklabels = [col_labels[cols[i]] for i in args]
g.set_xticklabels(tick_labels, rotation=40)
g.set_xlabels('(e)')
g.set_ylabels('')
g.fig.set_size_inches(6, 2.5)
g.fig.tight_layout()
g.savefig('full_run/figs/model_eval.pdf')

# To reorder columns
#reorder = ['w', 'r', 'mu', 'sig', 'b', 'mu_r', 'm','bt_n', 'bt_cv', 'bt_mu', 'r_frsh', 'age', 't_stb', 'bt_tmu', 'bt_tsig', 'bt_logt', 'out_call_div', 'e0_div'] + ['c' + str(i) for i in range(25)]

def plot_corrs(iet, name='full_run/figs/correlations.pdf'):
    corrs = iet.corr('spearman').values
    n = corrs.shape[0]
    #for i in range(n):
    #    corrs[i][i] = 0

    ticklabels = [col_labels[i] for i in iet.columns]
    g = sns.heatmap(corrs, center=0)
    g.axes.set_yticks(range(n))
    g.axes.set_xticks(range(n))
    g.axes.set_yticklabels(ticklabels, rotation=0)
    g.axes.set_xticklabels(ticklabels, rotation=90)
    g.get_figure().tight_layout()
    g.get_figure().savefig(name)
    return g


Start doing the Model evaluantion for three models

def evaluate_model(model, x_train, y_train, x_test, y_test, scores, fi, kind):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    sc = [accuracy_score(y_test, y_pred), matthews_corrcoef(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred)]
    scores.append(sc)

    if kind != 'L':
        fi.append(model.feature_importances_)
    else:
        fi.append(model.coef_[0])
alphas = [0] + [np.percentile(y[y > 0], i) for i in range(5, 100, 5)]
scores = []
fi1 = []
fi2 = []
fi3 = []

fis = [fi1, fi2, fi3]

for alpha in alphas:
    print(alpha)
    #yb_train = y_train.copy()

    #idx = yb_train > alpha
    #yb_train[idx] = 1
    #yb_train[~idx] = 0

    #yb_test = y_test.copy()
    #idx = yb_test > alpha
    #yb_test[idx] = 1
    #yb_test[~idx] = 0

    yb = y.copy()

    idx = y > alpha
    yb[idx] = 1
    yb[~idx] = 0

    x_train, x_test, yb_train, yb_test = train_test_split(iet, yb, test_size=0.3, stratify=yb)
    n = x_train.shape[0]
    rank_train = lambda x: rankdata(x)/n
    x_train = x_train.apply(rank_train)
    n = x_test.shape[0]
    rank_test = lambda x: rankdata(x)/n
    x_test = x_test.apply(rank_test)

    models = [RandomForestClassifier(), ExtraTreesClassifier(), LogisticRegression()]
    kinds = ['RF', 'ET', 'L']
    s = []
    for model, kind, fi in zip(models, kinds, fis):
        evaluate_model(model, x_train, yb_train, x_test, yb_test, s, fi, kind)
    print(s)
    scores.append(s)
np.save('full_run/figs_reclust/full_model_scores_temp.npy', scores)
np.save('full_run/figs_reclust/full_model_feature_importances_temp.npy', fis)
cols = iet.columns

#DO FULL MODEL EVALUATION


def evaluate_singlevar_model(model, x_train, y_train, x_test, y_test, scores, fi, kind):
    sc = []
    feat_imp = []
    for c in x_train.columns:
        x = x_train[c].reshape(-1, 1)
        model.fit(x, y_train)
        xt = x_test[c].reshape(-1, 1)
        y_pred = model.predict(xt)
        sc.append(matthews_corrcoef(y_test, y_pred))

        cols = np.unique([c, 'w'])
        model.fit(x_train[cols], y_train)
        y_pred = model.predict(x_test[cols])
        sc.append(matthews_corrcoef(y_test, y_pred))

        if kind != 'L':
            feat_imp.append(model.feature_importances_)
        else:
            feat_imp.append(model.coef_[0])
    scores.append(sc)
    fi.append(feat_imp)

alphas = [0] + [np.percentile(y[y > 0], i) for i in range(5, 100, 5)]
scores = []
fi1 = []
fi2 = []
fi3 = []

fis = [fi1, fi2, fi3]

for alpha in alphas:
    print(alpha)
    yb = y.copy()

    idx = y > alpha
    yb[idx] = 1
    yb[~idx] = 0

    x_train, x_test, yb_train, yb_test = train_test_split(iet, yb, test_size=0.5, stratify=yb)
    n = x_train.shape[0]
    rank_train = lambda x: rankdata(x)/n
    x_train = x_train.apply(rank_train)
    n = x_test.shape[0]
    rank_test = lambda x: rankdata(x)/n
    x_test = x_test.apply(rank_test)

    models = [RandomForestClassifier(), ExtraTreesClassifier(), LogisticRegression()]
    kinds = ['RF', 'ET', 'L']
    s = []
    for model, kind, fi in zip(models, kinds, fis):
        evaluate_singlevar_model(model, x_train, yb_train, x_test, yb_test, s, fi, kind)
    print(s)
    scores.append(s)
np.save('full_run/figs_reclust/model_scores_single_var.npy', scores)
np.save('full_run/figs_reclust/model_feature_importances_single_var.npy', fis)
cols = iet.columns



#PLOTS

def plot_scores(scores, alphas, name='full_run/figs/scores_accuracy.pdf'):
    plots.latexify(4, 4, 2)
    fig, ax = plt.subplots()
    labels = ['RF', 'ET', 'LR']
    for i in range(3):
        ax.plot(alphas, [scores[j][i][0] for j in range(len(scores))], label=labels[i])
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel('Accuracy')
    ax.legend(loc=0)
    fig.tight_layout()
    fig.savefig(name)
    return fig, ax

def plot_matthews_score(scores, alphas, name='full_run/figs/scores_matth.pdf'):
    plots.latexify(4, 4, 2)
    fig, ax = plt.subplots()
    labels = ['RF', 'ET', 'LR']
    for i in range(3):
        ax.plot(alphas, [scores[j][i][1] for j in range(len(scores))], label=labels[i])
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel('MCC')
    ax.legend(loc=0)
    fig.tight_layout()
    fig.savefig(name)
    return fig, ax

def plot_precision_score(scores, alphas, name='full_run/figs/scores_precision.pdf'):
    plots.latexify(4, 4, 2)
    fig, ax = plt.subplots()
    labels = ['RF', 'ET', 'LR']
    for i in range(3):
        ax.plot(alphas, [scores[j][i][2] for j in range(len(scores))], label=labels[i])
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel('Precision')
    ax.legend(loc=0)
    fig.tight_layout()
    fig.savefig(name)
    return fig, ax

def plot_recall_score(scores, alphas, name='full_run/figs/scores_recall.pdf'):
    plots.latexify(4, 4, 2)
    fig, ax = plt.subplots()
    labels = ['RF', 'ET', 'LR']
    for i in range(3):
        ax.plot(alphas, [scores[j][i][3] for j in range(len(scores))], label=labels[i])
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel('Recall')
    ax.legend(loc=0)
    fig.tight_layout()
    fig.savefig(name)
    return fig, ax

def plot_f1_score(scores, alphas, name='full_run/figs/scores_f1.pdf'):
    plots.latexify(4, 4, 2)
    fig, ax = plt.subplots()
    labels = ['RF', 'ET', 'LR']
    for i in range(3):
        ax.plot(alphas, [scores[j][i][4] for j in range(len(scores))], label=labels[i])
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel('F1')
    ax.legend(loc=0)
    fig.tight_layout()
    fig.savefig(name)
    return fig, ax

def plot_feature_importances(fis, cols, alphas, path='full_run/figs_reclust/'):
    #Function to plot feature importantces, where fis is a list of arrays containing feature importances of RandomForest, ExtraTrees and LogReg for each value of alphas.
    names = [path + 'feature_imp_' + i for i in ['rf.pdf', 'et.pdf', 'lr.pdf']]
    plots.latexify(6, 2, 1)
    col_labels = {'mu': r'$\bar{\tau}$', 'sig': r'$\bar{\sigma_{\tau}}$', 'b': r'$B$', 'mu_r': r'$\bar{\tau}_R$', 'r_frsh': r'$\hat{f}$', 'age': r'$age$', 't_stb': r'$TS$', 'm': r'$M$', 'bt_mu': r'$\bar{E}$', 'bt_sig': r'$\sigma^{E}$', 'bt_cv': r'$CV^E$', 'bt_n': r'$N^E$', 'bt_tmu': r'$\bar{t}$', 'bt_tsig': r'$\sigma_{t}$', 'bt_logt': r'$\log(T)$', 'out_call_div': r'$JSD$', 'r': r'$r$', 'w': r'$w$', 'e0_div': r'$JSD_{diff}$', r'ovrl': 'HT'}
    col_labels.update({'c' + str(i): 'C' + r'$' + str(i) + '$' for i in range(25)})

    #args = np.argsort(np.mean(np.abs(fis[2]), 0))
    ticklabels = [col_labels[cols[i]] for i in args]
    centers = [1./len(args), 1./len(args), 0]
    cmaps = ['PuOr', 'PuOr', 'BrBG']
    for k in range(3):
        fi_sorted = np.array([fis[k][i][args] for i in range(20)])

        plt.clf()
        center = centers[k]
        g = sns.heatmap(fi_sorted, center=center, cmap=cmaps[k])
        g.axes.set_xticks([x + .5 for x in range(0, len(args))])
        g.axes.set_xticklabels(ticklabels, rotation=90)
        alphas = [round(a, 3) for a in alphas]

        g.axes.set_yticks(range(0, 20, 4))
        g.axes.set_yticklabels(alphas[0:20:4], rotation=0)
        g.axes.set_ylabel(r'$O_{\alpha}$')
        g.get_figure().tight_layout()
        g.get_figure().savefig(names[k])




def plot_twovar_mcc(scores, fis, alphas, size=10, model_idx=0, name='full_run/figs_reclust/twovar_mcc.pdf', title=''):
    plots.latexify(6, 2, 1)
    htmap_data = []
    fi = []
    for i in range(len(alphas)):
        htmap_data.append(scores[i][model_idx][1:88:2])
        fi.append([a[0] for a in fis[0][model_idx]])

    fia = np.array(fi)
    fia = fia.T[args].T
    fia = np.abs(fia.reshape(1, -1))

    htmapa = np.array(htmap_data)
    htmapa = htmapa.T[args].T
    htmapa = htmapa.reshape(1, -1)

    largs = len(args)
    ralphas = np.repeat(range(20), largs)
    rvars = range(largs) * 20

    fig, ax = plt.subplots()
    ax.scatter(rvars, ralphas, c=htmapa, s=fia*size, cmap='GnBu')
    cb = fig.colorbar(ax.collections[0], label=r'$MCC$')
    #cb.outline.set_edgecolor('white')
    cb.outline.set_visible(False)

    col_labels = {'mu': r'$\bar{\tau}$', 'sig': r'$\bar{\sigma_{\tau}}$', 'b': r'$B$', 'mu_r': r'$\bar{\tau}_R$', 'r_frsh': r'$\hat{f}$', 'age': r'$age$', 't_stb': r'$TS$', 'm': r'$M$', 'bt_mu': r'$\bar{E}$', 'bt_sig': r'$\sigma^{E}$', 'bt_cv': r'$CV^E$', 'bt_n': r'$N^E$', 'bt_tmu': r'$\bar{t}$', 'bt_tsig': r'$\sigma_{t}$', 'bt_logt': r'$\log(T)$', 'out_call_div': r'$JSD$', 'r': r'$r$', 'w': r'$w$', 'e0_div': r'$JSD_{diff}$', r'ovrl': 'HT'}
    col_labels.update({'c' + str(i): 'C' + r'$' + str(i) + '$' for i in range(25)})
    ax.set_xticks(range(0, len(args)))
    ticklabels = [col_labels[cols[i]] for i in args]
    ax.set_xticklabels(ticklabels, rotation=90)
    alphas = [round(a, 3) for a in alphas]

    ax.set_yticks(range(0, 20, 4))
    ax.set_yticklabels(alphas[0:20:4], rotation=0)
    ax.set_ylabel(r'$O_{\alpha}$')
    ax.invert_yaxis()
    ax.set_title(title, loc='left')
    fig.tight_layout()
    fig.savefig(name, transparent=True)

    return fig, ax


def plot_single_vs_twovar(scores, alphas, size=10, model_idx=0, name='full_run/figs_reclust/twovar_mcc.pdf', title=''):
    plots.latexify(6, 2, 1)
    scores_1 = []
    scores_2 = []
    for i in range(len(alphas)):
        scores_1.append(np.max(scores[i], 0)[0:88:2])
        scores_2.append(np.max(scores[i], 0)[1:88:2])
    scores_1 = np.mean(scores_1, 0)
    scores_2 = np.mean(scores_2, 0)
    fig, ax = plt.subplots()
    largs = len(args)
    ax.scatter(range(largs), scores_1[args], label='Single var', alpha=.5)
    ax.scatter(range(largs), scores_2[args], label='Double var', alpha=.5)

    col_labels = {'mu': r'$\bar{\tau}$', 'sig': r'$\bar{\sigma_{\tau}}$', 'b': r'$B$', 'mu_r': r'$\bar{\tau}_R$', 'r_frsh': r'$\hat{f}$', 'age': r'$age$', 't_stb': r'$TS$', 'm': r'$M$', 'bt_mu': r'$\bar{E}$', 'bt_sig': r'$\sigma^{E}$', 'bt_cv': r'$CV^E$', 'bt_n': r'$N^E$', 'bt_tmu': r'$\bar{t}$', 'bt_tsig': r'$\sigma_{t}$', 'bt_logt': r'$\log(T)$', 'out_call_div': r'$JSD$', 'r': r'$r$', 'w': r'$w$', 'e0_div': r'$JSD_{diff}$', r'ovrl': 'HT'}
    col_labels.update({'c' + str(i): 'C' + r'$' + str(i) + '$' for i in range(25)})
    ax.set_xticks(range(0, len(args)))
    ticklabels = [col_labels[cols[i]] for i in args]
    ax.set_xticklabels(ticklabels, rotation=90)
    ax.set_ylabel(r'$MCC$')
    ax.legend(loc=0)
    ax.set_title(title, loc='left')
    fig.tight_layout()
    fig.savefig(name, transparent=True)

f = list(f1) + list(f2)
model = ['RF'] * (len(f1)) + ['ET'] * (len(f2))

df = pd.DataFrame()
df['var'] = list(iet.columns) * 2
df['I'] = f
df['model'] = model
df['I'].iloc[11] = 0.18
plots.latexify(6, 2.5, 2)
sns.set(rc={'figure.figsize':(3, 2.5)})
g = sns.catplot(x="var", y="I", hue="model", data=df, kind="bar", legend=False)
g.ax.legend(loc=0)
tick_labels = [r'$\bar{\tau}$', r'$\bar{\sigma_{\tau}}$', r'$B$', r'$\bar{\tau}_R$', r'$\hat{f}$', r'$age$', r'$TS$', r'$M$', r'$\bar{E}$', r'$\sigma^{E}$', r'$CV^E$', r'$N^E$', r'$\bar{t}$', r'$\sigma_{t}$', r'$\log(T)$', r'$JSD$', r'$\bar{r}$', r'$w$']
g.set_xticklabels(tick_labels, rotation=40)
g.set_xlabels('(e)')
g.set_ylabels('')
g.fig.set_size_inches(6, 2.5)
g.fig.tight_layout()
g.savefig('full_run/figs/model_eval.pdf')



#def write_results(self, w, comb, sms, n_row, score, model):
#     ltw = ['_'.join(r) for r in comb] + [str(sms), str(n_row), str(score), model]
#     w = open(self.paths['cv_stats'], 'a')
#     w.write(' '.join(ltw) + '\n')
#     w.close()

iet = pd.read_csv('full_run/model_x_data.txt', sep=' ')
tmp = pd.read_csv('full_run/model_y_data.txt', sep=' ')
iet['w'] = tmp.w
del iet['jsd_mean']
del iet['jsd_diff']
del iet['bt_tsig1']

x_train, x_test, y_train, y_test = train_test_split(iet, y, test_size=0.3)
n = x_train.shape[0]
rank_train = lambda x: rankdata(x)/n
x_train = x_train.apply(rank_train)
y_train = rank_train(y_train)
n = x_test.shape[0]
rank_test = lambda x: rankdata(x)/n
x_test = x_test.apply(rank_test)
y_test = rank_test(y_test)

RF = RandomForestRegressor()
RF.fit(x_train, y_train)
sc1 = RF.score(x_test, y_test)
#sc1b = RF.score(x_test, y_test.ov_mean)
f1 = list(RF.feature_importances_)

#RF = RandomForestRegressor()
#RF.fit(x_train, y_train.ov_mean)
#sc2 = RF.score(x_test, y_test.ov_mean)
#f2 = list(RF.feature_importances_)

RF = ExtraTreesRegressor()
RF.fit(x_train, y_train)
sc3 = RF.score(x_test, y_test)
#sc3b = RF.score(x_test, y_test.ov_mean)
f3 = list(RF.feature_importances_)

#RF = ExtraTreesRegressor()
#RF.fit(x_train, y_train.ov_mean)
#sc4 = RF.score(x_test, y_test.ov_mean)
#f4 = list(RF.feature_importances_)


var = range(len(iet.columns)) *2
f = f1 + f3
model = ['RF'] * (len(var)/2) + ['ET'] * (len(var)/2)

df = pd.DataFrame()
df['var'] = var
df['I'] = f
df['model'] = model

#idf.to_csv('full_run/model_summary.txt')

plots.latexify(6, 2.2, 2)
sns.set(rc={'figure.figsize':(6, 2)})
g = sns.catplot(x="var", y="I", hue="model", data=df, kind="bar", legend=False)
g.ax.legend(loc=0)
tick_labels = [r'$\bar{\tau}$', r'$\bar{\sigma_{\tau}}$', r'$B$', r'$\bar{\tau}_R$', r'$\hat{f}$', r'$age$', r'$TS$', r'$M$', r'$\bar{E}$', r'$\sigma^{E}$', r'$CV^E$', r'$N^E$', r'$\bar{t}$', r'$\sigma_{t}$', r'$\log(T)$', r'$JSD$', r'$\bar{r}$', r'$w$']
g.set_xticklabels(tick_labels, rotation=40)
g.set_xlabels('(e)')
g.set_ylabels(r'$I$')
g.fig.set_size_inches(6, 2.2)
g.fig.tight_layout()
g.savefig('full_run/figs/models.pdf')

# Without w
# Score for ovrl: .2317
# Score for rank(ovrl): .2705
# Score for ov_mean: .2503
# Score for rank(ov_mean): .30

# With w
# Score for ovrl: .228
# SCore for rank(ovrl): .2696
# Score for ov_mean: .2504
# Score for rank(ov_mean): .297

# About temporal overlap: higher temporal std implies higher overlap (Does this imply that neighbors)
# Correlation between temporal stability and temporal overlap
# Nice result: correlation between overlap trend and tmp_mass

rank_train = lambda x: rankdata(x)/350000
rank_test = lambda x: rankdata(x)/150000
x_train['w'] = rank_train(y_train.w)
x_test['w'] = rank_test(y_test.w)

#write_results(w, comb, 0, proc_df.shape[0], sc, 'RF')
# TODO: table with feature importance for mean overlap and temporal overlap

"""

