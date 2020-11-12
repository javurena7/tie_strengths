from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np; from numpy import inf
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import matthews_corrcoef, roc_auc_score, f1_score
import pickle
from os import listdir
from collections import OrderedDict
from itertools import product
import matplotlib.pyplot as plt ; plt.ion()
import seaborn as sns

from latexify import *
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample
from sklearn.inspection import partial_dependence


class PredictTieStrength(object):
    def __init__(self, y_var, data_path='../paper_run/sample/', models=['SVC', 'LR'], remove=['deg_0', 'deg_1', 'n_ij', 'e0_div', 'e1_div', 'bt_tsig1'], save_prefix='../paper/', k=3, alpha_step=5, ranked=False, cluster_only=False, c_star=False, df_name=''):
        self.save_prefix = save_prefix
        self.data_path = data_path
        self._init_models(models)
        self.k = k
        self.c_star = c_star
        self.col_labels = {'mu': r'$\bar{\tau}$',
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
                'c_star': r'$C^*$',
                'e0_div': r'$JSD_{diff}$',
                'ovrl': r'$O$',
                'avg_len': r'$\hat{l}$',
                'len': r'$l$',
                'w_hrs': r'$a_h$',
                'w_day': r'$a_d$'}
        self.col_labels.update({'c' + str(i): 'C' + r'$' + str(i) + '$' for i in range(1, 16)})
        self.col_labels_full = {'mu': 'IET avg., ' + r'$\bar{\tau}$',
                'sig': 'IET std., ' + r'$\bar{\sigma_{\tau}}$',
                'b': 'Burstiness, ' + r'$B$',
                'mu_r': 'Avg. relay time, ' + r'$\bar{\tau}_R$',
                'r_frsh': 'Relat. freshness, ' + r'$\hat{f}$',
                'age': 'Age, ' + r'$age$',
                't_stb': 'Temporal stability, ' + r'$TS$',
                'm': 'Memory, ' + r'$M$',
                'bt_mu': 'Avg. events/burst, ' + r'$\bar{E}$',
                'bt_sig': 'Std. events/burst, ' + r'$\sigma^{E}$',
                'bt_cv': 'CV events/burst, ' + r'$CV^E$',
                'bt_n': 'Num. Bursts, ' + r'$N^E$',
                'bt_tmu': 'Avg. inter. t., ' + r'$\bar{t}$',
                'bt_tsig': 'Std. inter. t., ' + r'$\sigma_{t}$',
                'bt_logt': 'Test inter. t., 'r'$log(T)$',
                'out_call_div': 'Daily behaviour, ' + r'$JSD$',
                'r': 'Reciprocity, ' + r'$r$',
                'w': r'\textbf{Num. Calls}, ' + r'$\mathbf{w}$',
                'c_star': 'Weekly profile, ' + r'$C^*$',
                'e0_div': r'$JSD_{diff}$',
                'ovrl': r'$O$',
                'avg_len': 'Avg. length, ' + r'$\hat{l}$',
                'len': 'Total length, ' + r'$l$',
                'w_hrs': 'Active hours, ' + r'$a_h$',
                'w_day': 'Active days, ' + r'$a_d$',
                'c1': 'Night, ' + r'$C1$',
                'c2': 'Monday morning, ' + r'$C2$',
                'c3': 'Monday morning, ' + r'$C3$',
                'c4': 'Weekday 7am, ' + r'$C4$',
                'c5': 'Weekday afternoon, ' + r'$C5$',
                'c6': 'Weekday evening, ' + r'$C6$',
                'c7': 'Weekay morning, ' + r'$C7$',
                'c8': 'Thursday morning, ' + r'$C8$',
                'c9': 'Weekend evening, ' + r'$C9$',
                'c10': 'Weekend morning, ' + r'$C10$',
                'c11': 'Saturday. morning, ' + r'$C11$',
                'c12': 'Weekend afternoon, ' + r'$C12$',
                'c13': 'Saturday afternoon, ' + r'$C13$',
                'c14': 'Sunday morning, ' + r'$C14$',
                'c15': 'Sunday afternoon, ' + r'$C15$',
                }
        #self.col_labels_full.update({'c' + str(i): 'C' + r'$' + str(i) + '$' for i in range(1, 16)})

        if data_path:
            self.read_tables(data_path, y_var, remove, cluster_only, df_name)
            self.single_scores = self._init_scores()
            self.boot_scores = self._init_scores()
            self.auc_scores = self._init_scores()
            self.f1_scores = self._init_scores()
            self.partial_dep = self._init_scores()
        else:
            self.variables = []
            self.x, self.y, self.scores = None, None, {}
        self.dual_scores = {}
        self.boot_dual_scores = {}
        self.alpha_step = alpha_step
        self.ranked = ranked
        self._init_full_scores()
        self._init_model_params()


    def _init_models(self, models):
        available_models = {'SVC': 'LinearSVC',
                'LR': 'LogisticRegression',
                'RF': 'RandomForestClassifier',
                'ABC': 'AdaBoostClassifier',
                'QDA': 'QuadraticDiscriminantAnalysis'}
        self.models = []
        for model in models:
            model_str = available_models[model] + '()'
            self.models.append((model, model_str))


    def _init_scores(self):
        return {kind[0]: OrderedDict((var, []) for var in self.variables) for kind in self.models}

    def _init_dual_scores(self, fvar):
        self.dual_scores[fvar] = self._init_scores()
        self.boot_dual_scores[fvar] = self._init_scores()

    def _init_full_scores(self):
        self.feature_imp = {kind[0]: [] for kind in self.models}
        self.full_scores = {kind[0]: [] for kind in self.models}
        self.imp_df = {}

    def _init_model_params(self):
        smodel_params = {'LR': {'C': 1, 'fit_intercept': True},
                'QDA': {},
                'RF': {'criterion': 'gini', 'n_estimators' : 20},
                'ABC': {}}

        fmodel_params = {'LR': {'C': .5, 'fit_intercept': True},
                'QDA': {'reg_param' : .75},
                'RF': {'max_features': 'sqrt', 'criterion': 'gini', 'n_estimators': 55},
                'ABC': {}}

        self.smodel_params = smodel_params
        self.fmodel_params = fmodel_params

    def read_tables(self, path, y_var, remove, cluster_only, df_name=''):
        if 'delta_t' in path:
            deltas = pickle.load(open(path + 'deltas.p', 'rb'))
            colnames = ['ov_mean', 'ovrl', '0', '1'] + sorted(deltas, key=lambda x: int(x))
            df = pd.read_csv(path + 'new_full_bt_n.txt', sep=' ', names=colnames, index_col=['0', '1'], skiprows=1)
            self.update_delta_colnames(deltas)
        else:
            if len(df_name) < 1:
                df_name = 'full_df_paper.txt'
            df = pd.read_csv(path + df_name, sep=' ', index_col=['0', '1'])
            try:
                df_wc = pd.read_csv(path + 'clustered_df_paper.txt', sep=' ', index_col=['0', '1'])
                df = pd.concat([df, df_wc], axis=1)
            except:
                pass
            for col in remove:
                if col in df.columns:
                    df.drop(col, axis=1, inplace=True)

        if y_var == 'ovrl' and 'ov_mean' in df.columns:
            df.drop('ov_mean', axis=1, inplace=True)
        elif y_var == 'ov_mean' and 'ovrl' in df.columns:
            df.drop('ovrl', axis=1, inplace=True)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        self.y = df[y_var]
        df.drop(y_var, axis=1, inplace=True)
        if cluster_only:
            from re import match
            columns = [col for col in df.columns if match('c\d', col)]
            df = df[columns]
        self.variables = list(df.columns)
        self.variables += ['c_star'] if self.c_star else []
        self.x = df

    def update_delta_colnames(self, deltas):
        self.col_labels = {}
        for delta in deltas:
            dta = int(delta)
            if dta < 3600:
                self.col_labels[delta] = str(dta / 60) + 'm'
            elif dta < 86400:
                self.col_labels[delta] = str(dta / 3600) + 'h'
            else:
                days, hrs = dta / 86400, (dta % 86400) / 3600
                self.col_labels[delta] = "{}d {}h".format(days, hrs)


    def get_alphas(self):
        self.alphas = [0] + [np.percentile(self.y[self.y > 0], i) for i in range(5, 100, self.alpha_step)]

    def get_y_class(self, alpha):
        yb = self.y.copy()
        idx = self.y > alpha
        yb[idx] = 1
        yb[~idx] = 0
        self.yb = yb

    def eval_model(self, model):
        model.fit(self.x_train, self.y_train)
        y_pred = model.predict(self.y_test)
        return matthews_corrcoef(self.y_test, y_pred)

    def kfold(self):
        skf = StratifiedKFold(n_splits=self.k, random_state=0)
        return skf.split(self.x, self.yb)

    def resample_data(self, x, train_idx):
        y = self.yb.iloc[train_idx]

        y_pos = y[y == 1]
        y_neg = y[y == 0]

        x_pos = x[y == 1]
        x_neg = x[y == 0]

        if len(y_pos) > len(y_neg):
            x_samp, y_samp = resample(x_pos, y_pos, replace=False, n_samples=len(y_neg), random_state=29)
            x_new = np.concatenate((x_samp, x_neg), axis=0)
            y_new = np.concatenate((y_samp, y_neg), axis=0)
        else:
            x_samp, y_samp = resample(x_neg, y_neg, replace=False, n_samples=len(y_pos), random_state=29)
            x_new = np.concatenate((x_samp, x_pos), axis=0)
            y_new = np.concatenate((y_samp, y_pos), axis=0)

        return x_new, y_new

    def eval_single_var(self, model, resampled=False):
        #Note: resampled works better with python2
        var_set = list(self.x.columns)
        if self.c_star:
            from re import match
            c_vars = [c for c in var_set if match('c\d', c)]
            var_set += ['c_star']
        for var in var_set:
            scores, aucs, f1s = [], [], []
            y_preds = []
            yb = []
            pdep = None

            for train_idx, test_idx in self.kfold():
                if var == 'c_star':
                    mod = eval(model[1])
                    mod.set_params(**self.fmodel_params[model[0]])
                    x = self.x.iloc[train_idx][c_vars]
                    xt = self.x.iloc[test_idx][c_vars]
                    if self.ranked:
                        rank_x = lambda x: rankdata(x, method='min') / (x.shape[0] + 0.)
                        x = x.apply(rank_x, 0)
                        xt = xt.apply(rank_x, 0)
                else:
                    mod = eval(model[1])
                    mod.set_params(**self.smodel_params[model[0]])
                    if not self.ranked:
                        x = self.x.iloc[train_idx][var].values.reshape(-1, 1)
                        xt = self.x.iloc[test_idx][var].values.reshape(-1, 1)
                    else:
                        rank_x = lambda x: rankdata(x, 'min') / (x.shape[0] + 0.)
                        x = rank_x(self.x.iloc[train_idx][var].values).reshape(-1, 1)
                        xt = rank_x(self.x.iloc[test_idx][var].values).reshape(-1, 1)
                if resampled:
                    x, y_train = self.resample_data(x, train_idx)
                else:
                    y_train = self.yb.iloc[train_idx]
                mod.fit(x, y_train)
                y_pred = mod.predict(xt)
                y_prob = mod.predict_proba(xt)[:, 0]
                if pdep is None:
                    pdep = (0, 0) #partial_dependence(mod, x, [0])

                scores.append(matthews_corrcoef(self.yb.iloc[test_idx], y_pred))
                yb = np.concatenate((yb, self.yb.iloc[test_idx].values)) if len(yb) > 0 else self.yb.iloc[test_idx].values
                y_preds = np.concatenate((y_preds, y_pred)) if len(y_preds) > 0 else y_pred
                aucs.append(roc_auc_score(self.yb.iloc[test_idx], y_prob))
                f1s.append(f1_score(self.yb.iloc[test_idx], y_pred))

            try:
                self.single_scores[model[0]][var].append(np.mean(scores))
            except:
                import pdb; pdb.set_trace()
            self.boot_scores[model[0]][var].append(self.bootstrap_ci(yb, y_preds))
            self.auc_scores[model[0]][var].append(np.mean(aucs))
            self.f1_scores[model[0]][var].append(np.mean(f1s))
            self.partial_dep[model[0]][var].append(pdep)
        print('Number of alphas for model {}: {}'.format(model[0], len(self.single_scores[model[0]][var])))
        self.save_scores(self.single_scores[model[0]], 'single_scores.{}.p'.format(model[0]))
        self.save_scores(self.boot_scores[model[0]], 'boot_scores.{}.p'.format(model[0]))
        self.save_scores(self.auc_scores[model[0]], 'auc_scores.{}.p'.format(model[0]))
        self.save_scores(self.f1_scores[model[0]], 'f1_scores.{}.p'.format(model[0]))
        self.save_scores(self.partial_dep[model[0]], 'part_dep.{}.p'.format(model[0]))


    def bootstrap_ci(self, yb, y_preds, n_samp=100):
        n = len(y_preds)
        mscores = []
        mean_score = matthews_corrcoef(yb, y_preds)
        for i in range(n_samp):
            samp_idx = np.random.choice(n, n, replace=True)
            mscores.append(matthews_corrcoef(yb[samp_idx], y_preds[samp_idx]))
        mscores = sorted(mscores)
        min_idx = int(n_samp * .05)
        max_idx = int(n_samp * .95)
        var = np.var(mscores)
        return (mscores[min_idx], mscores[max_idx], var)


    def eval_dual_var(self, model, fvar, resampled=True):
        """
        If c_star, calculate a model with all clusters
        """

        var_set = list(self.x.columns)
        rank_x = lambda x: rankdata(x, 'min') / (x.shape[0] + 0.)
        if self.c_star:
            from re import match
            c_vars = [fvar] + [c for c in var_set if match('c\d', c)]
            var_set += ['c_star']
        for var in var_set:
            scores = []
            yb = []
            y_preds = []
            mod = eval(model[1]) #Initialize model
            for train_idx, test_idx in self.kfold():
                if var == fvar:
                    x = self.x.iloc[train_idx][var].values.reshape(-1, 1)
                    xt = self.x.iloc[test_idx][var].values.reshape(-1, 1)
                    mod.set_params(**self.smodel_params[model[0]])
                    if self.ranked:
                        x = rank_x(x).reshape(-1, 1)
                        xt = rank_x(xt).reshape(-1, 1)
                elif var == 'c_star':
                    x = self.x.iloc[train_idx][c_vars]
                    xt = self.x.iloc[test_idx][c_vars]
                    mod.set_params(**self.fmodel_params[model[0]])
                    if self.ranked:
                        x = x.apply(rank_x, 0)
                        xt = xt.apply(rank_x, 0)
                else:
                    x = self.x.iloc[train_idx][[fvar, var]]
                    xt = self.x.iloc[test_idx][[fvar, var]]
                    mod.set_params(**self.smodel_params[model[0]])
                    if self.ranked:
                        x = x.apply(rank_x, 0)
                        xt = xt.apply(rank_x, 0)

                if resampled:
                    x, y_train = self.resample_data(x, train_idx)
                else:
                    y_train = self.yb[train_idx]
                mod.fit(x, y_train)
                y_pred = mod.predict(xt)
                #y_prob = mod.predict_proba(xt)

                scores.append(matthews_corrcoef(self.yb[test_idx], y_pred))
                yb = np.concatenate((yb, self.yb[test_idx].values)) if len(yb) > 0 else self.yb[test_idx].values
                y_preds = np.concatenate((y_preds, y_pred)) if len(y_preds) > 0 else y_pred
            self.single_scores[model[0]][var].append(np.mean(scores))
            self.boot_dual_scores[fvar][model[0]][var].append(self.bootstrap_ci(yb, y_preds))
            self.dual_scores[fvar][model[0]][var].append(np.mean(scores))
            #self.dual_probas[fvar][model[0]][var].append(np.mean(scores))

        self.save_scores(self.dual_scores[fvar][model[0]], 'dual_scores.{}.{}.p'.format(fvar, model[0]))
        self.save_scores(self.boot_dual_scores[fvar][model[0]], 'boot_dual_scores.{}.p'.format(model[0]))


    def eval_full(self, model, resampled=True):
        scores = []
        feat_imports_k = []
        mod = eval(model[1])
        mod.set_params(**self.fmodel_params[model[0]])
        for train_idx, test_idx in self.kfold():
            x = self.x.iloc[train_idx]
            xt = self.x.iloc[test_idx]
            if self.ranked:
                rank_x = lambda x: rankdata(x, 'min') / (x.shape[0] + 0.)
                x = x.apply(rank_x, 0)
                xt = xt.apply(rank_x, 0)
            if resampled:
                x, y_train = self.resample_data(x, train_idx)
            else:
                y_train = self.yb[train_idx]
            mod.fit(x, y_train)
            y_pred = mod.predict(xt)
            scores.append(matthews_corrcoef(self.yb[test_idx], y_pred))
            self.get_feat_imports(model[0], mod, feat_imports_k)
        score = np.mean(scores)
        feat_imports = np.mean(feat_imports_k, 0)
        self.full_scores[model[0]].append(score)
        self.feature_imp[model[0]].append(feat_imports)

        self.save_scores(self.full_scores[model[0]], 'full_scores.{}.p'.format(model[0]))
        self.save_scores(self.feature_imp[model[0]], 'feature_imp.{}.p'.format(model[0]))

    def get_feat_imports(self, model, mod, feat_imports):
        if model in ['LR', 'SVC']:
            feat_imports.append(mod.coef_[0])
        elif model in ['RF', 'ABC']:
            feat_imports.append(mod.feature_importances_)
        else:
            feat_imports.append([])


    def run_alphas(self, fvars=[], single_var=True, full_vars=False):
        for fvar in fvars:
            self._init_dual_scores(fvar)

        if full_vars:
            self.feature_imp['vars'] = self.x.columns.values
            self.save_scores(self.feature_imp['vars'], 'feature_imp.vars.p')
        for alpha in self.alphas:
            print('--- alpha = {} --- '.format(alpha))
            self.get_y_class(alpha)

            for model in self.models:
                if single_var == True:
                    self.eval_single_var(model)
                for fvar in fvars:
                    self.eval_dual_var(model, fvar)
                if full_vars == True:
                    self.eval_full(model)


    def save_scores(self, obj, name):
        with open(self.save_prefix + name, 'wb') as opn:
            pickle.dump(obj, opn)


    def load_scores(self, single=False, dual=False, full=False):
        single_files = [f for f in listdir(self.save_prefix) if 'single_scores' in f] if single else []
        dual_files = [f for f in listdir(self.save_prefix) if f.startswith('dual_scores')] if dual else []
        full_files = [f for f in listdir(self.save_prefix) if 'full_scores' in f] if full else []
        imps_files = [f for f in listdir(self.save_prefix) if 'feature_imp' in f] if full else []
        boot_files = [f for f in listdir(self.save_prefix) if 'boot_scores' in f] if single else []
        boot_dual_files = [f for f in listdir(self.save_prefix) if 'boot_dual' in f] if dual else []
        auc_files = [f for f in listdir(self.save_prefix) if 'auc_scores' in f] if single else []
        f1_files = [f for f in listdir(self.save_prefix) if 'f1_scores' in f] if single else []

        self.single_scores = {}
        for sf in single_files:
            model = sf.split('.')[1]
            try:
                self.single_scores[model] = pickle.load(open(self.save_prefix + sf, 'rb'))
            except:
                self.single_scores[model] = pickle.load(open(self.save_prefix + sf, 'rb'), encoding='bytes')


        self.boot_scores = {}
        for bf in boot_files:
            model = bf.split('.')[1]
            try:
                self.boot_scores[model] = pickle.load(open(self.save_prefix + bf, 'rb'))
            except:
                self.boot_scores[model] = pickle.load(open(self.save_prefix + bf, 'rb'), encoding='bytes')

        self.auc_scores = {}
        for af in auc_files:
            model = af.split('.')[1]
            try:
                self.auc_scores[model] = pickle.load(open(self.save_prefix + af, 'rb'))
            except:
                self.auc_scores[model] = pickle.load(open(self.save_prefix + af, 'rb'), encoding='bytes')

        self.f1_scores = {}
        for ff in f1_files:
            model = ff.split('.')[1]
            try:
                self.f1_scores[model] = pickle.load(open(self.save_prefix + ff, 'rb'))
            except:
                self.f1_scores[model] = pickle.load(open(self.save_prefix + ff, 'rb'), encoding='bytes')


        self.dual_scores = {}
        for df in dual_files:
            fvar, model = df.split('.')[1:3]
            #if model == 'ABC':
            if self.dual_scores.get(fvar, 0) == 0:
                self.dual_scores[fvar] = {}
            try:
                self.dual_scores[fvar][model] = pickle.load(open(self.save_prefix + df, 'rb'))
            except:
                self.dual_scores[fvar][model] = pickle.load(open(self.save_prefix + df, 'rb'), encoding='bytes')


        self.dual_boot_scores = {}
        for df in boot_dual_files:
            fvar, model = df.split('.')[1:3]
            if self.dual_boot_scores.get(fvar, 0) == 0:
                self.dual_boot_scores[fvar] = {}
            try:
                self.dual_boot_scores[fvar][model] = pickle.load(open(self.save_prefix + df, 'rb'))
            except:
                self.dual_boot_scores[fvar][model] = pickle.load(open(self.save_prefix + df, 'rb'), encoding='bytes')


        self.feature_imp = {}
        self.full_scores = {}
        for ff in full_files:
#### This might fail in python 3 bc of encoding, see other pickle loads
            model = ff.split('.')[1]
            self.full_scores[model] = pickle.load(open(self.save_prefix + ff, 'rb'))
        for imf in imps_files:
            model = imf.split('.')[1]
            self.feature_imp[model] = pickle.load(open(self.save_prefix + imf, 'rb'))


    def model_variance(self, mat, case='AVG', boot_scores=[]):
        """
        Obtain varaince induced by model and sampling variance (if boot_scores)
        """
        #TODO: this only contains model variance, instead add alpha and sampling variance
        if case == 'AVG':
            weights = [m.mean(0) for m in mat] #Check if this gets the average perf per model
            w = sum(dms)
            model_weights = [dm / w for dm in dms]
        elif case == 'MAX':
            w = len(mat) + 0.
            model_weights = [1 / w for dm in dms]
        #Average performance per variable and alpha
        alpha_avgs = sum([weight.reshape(-1, 1) * model for weight, model in zip(model_weights, mat)])

        model_avgs = [model.mean(1) for model in mat] #Get average perfromance over all alphas
        total_avg = sum([weight * avg for weight, avg in zip(model_weights, model_avgs)])
        model_var = sum([weight * (model_avg - total_avg)**2 for weight, model_avg in \
                zip(model_weights, model_avgs)]) #Model variance over all alphas

        if boot_scores:
            # Variance induced by sampling
            samp_var = []
            #for bscore in boot_scores:
            #    boot_vars = OrderedDict((k, [vi[2] for vi in v]) for k, v in bscore.items())
            #    d_boot = pd.DataFrame(boot_vars).as_matrix()
            #    boot_var = []
            samp_var.append([np.array([w * w * di for w, di in zip(model_weights, d_boot.T)])])
            return alpha_avgs, model_var, samp_var
        else:
            return alpha_avgs, model_var

    def max_model_var(self, mat, var_mat):
        max_vals = mat.max(0)
        max_args = mat.argmax(0)
        max_vars = self.argmax_to_max(var_mat, max_args, axis=0)
        return max_vals, max_vars


    def scores_to_array(self, scores, variance=False):
        """
        Convert scores to tensor of shape (n_models)x(n_alphas)x(n_variables)
        """

        models = ['ABC', 'RF', 'LR', 'QDA']
        full_scores = []
        for model in models:
            if variance and (model in scores):
                bscore = scores[model]
                boot_vars = OrderedDict((k, [vi[2] for vi in v]) for k, v in bscore.items())
                d_boot = pd.DataFrame(boot_vars).as_matrix()
                full_scores.append(d_boot)
            else:
                if model in scores:
                    full_scores.append(pd.DataFrame(scores[model]).as_matrix())

        return np.array(full_scores)


    def argmax_to_max(self, arr, argmax, axis=0):
        """
        Given an array (variances), and armax indeces (from scores) on an axis, select
        the values or array from the argmax indeces

        argmax_to_max(arr, arr.argmax(axis), axis) == arr.max(axis

        """
        new_shape = list(arr.shape)
        del new_shape[axis]

        grid = np.ogrid[tuple(map(slice, new_shape))]
        grid.insert(axis, argmax)
        return arr[tuple(grid)]

    def merge_score_and_variance(self):
        self.single_score_array = self.scores_to_array(self.single_scores)
        self.single_var_array = self.scores_to_array(self.boot_scores, variance=True)


        self.dual_score_array = self.scores_to_array(self.single_scores)
        self.dual_var_arry = self.scores_to_array(self.dual_boot_score, variance=True)

    def max_and_var(self, mat, boot_scores):
        max_score = np.max(mat, 0)
        max_idx = np.argmax(mat, 0)
        max_var = np.zeros(max_idx.shape)
        for i, j in product(range(max_idx.shape[0]), range(max_idx.shape[1])):
            mod = max_idx[i, j]
            max_var[i, j] = boot_scores[mod][i, j]
        return max_score, max_var

    def get_variance(self, boot_score):
        var_dict = OrderedDict((k, [vi[2] for vi in v]) for k, v in boot_score.items())
        var_mat = pd.DataFrame(var_dict).values
        return var_mat

    def merge_scores(self):
        mat = [] #List of model score df's (each df is an alphas-ftrs score mat)
        mat_var = [] #List of model variances df's (each df is an alphas-ftrs mat with sampling variances)
        total_weight = np.zeros(len(self.variables)) #
        weights = []
        model_xs = []
        boot_scores = []

        models = ['ABC', 'RF', 'LR', 'QDA']
        auc_mat = []
        f1_mat = []
        for model in models:
            scores = self.single_scores[model]
            colnames = scores.keys()
            scores = pd.DataFrame(scores).values
            auc = self.auc_scores.get(model, {})
            auc_scores = pd.DataFrame(auc).values
            f1 = self.f1_scores.get(model, {})
            f1_scores = pd.DataFrame(auc).values

            if scores.shape[0] < 20:
                r, c = scores.shape # REMOVE THIS,
                scores = np.concatenate((scores, np.zeros((20 - r, c))), axis=0)
                auc_scores = np.concatenate((auc_scores, np.zeros((20 - r, c))), axis=0)
                f1_scores = np.concatenate((f1_scores, np.zeros((20 - r, c))), axis=0)
                print(f1_scores.shape)

            mat.append(scores)
            var_mat = self.get_variance(self.boot_scores[model])
            boot_scores.append(var_mat)
            auc_mat.append(auc_scores)
            f1_mat.append(f1_scores)

        max_score = np.max(mat, 0) # REMOVE THIS
        #max_score, max_variance = self.max_and_var(mat, boot_scores)
        max_auc = np.max(auc_mat, 0)
        max_f1 = np.max(f1_mat, 0)

        self.average_score = pd.DataFrame(max_score, columns=colnames)
        #self.average_auc = pd.DataFrame(max_auc, columns=colnames)
        #self.average_f1 = pd.DataFrame(max_f1, columns=colnames)



        #CASE: test how max plot looks like
        #self.single_scores['MAX'] = pd.DataFrame(np.max(mat, 0).T, columns=colnames)


                #mat_var = np.array([mi / (dsi)**2 for mi, dsi in zip(sum(mat_var), ds)]).T

                #self.average_score = pd.DataFrame(average_score, columns=colnames)
                #model_variance = self.average_score.mean()
                #self.boot_vars = pd.DataFrame(mat_var, columns=colnames)

        #else:
        #    for scores in self.single_scores.values():
        #        colnames = scores.keys()
        #        d = pd.DataFrame(scores).as_matrix()
        #        dm = d.mean(0)
        #        mat.append(np.array([dmi * di for dmi, di in zip(dm, d.T)]))
        #        ds += dm
            #For each variable, we get the weighted average of the performance
        #    if mat:
        #        self.single_scores['MAX'] = pd.DataFrame(np.max(mat, 0).T, columns=colnames)
        #        mat = np.array([mi / dsi for mi, dsi in zip(sum(mat), ds)]).T
        #        self.average_score = pd.DataFrame(mat, columns=colnames)

        self.average_dual_score = {}
        for var, var_models in self.dual_scores.items():
            mat = []
            ds = np.zeros(len(self.variables))
            for scores in var_models.values():
                colnames = scores.keys()
                scrs = pd.DataFrame(scores).as_matrix()
                mat.append(scrs)
            max_score = np.max(mat, 0)
            self.average_dual_score[var] = pd.DataFrame(max_score, columns=colnames)

        if self.feature_imp.get('vars', None):
            colnames = self.feature_imp['vars']
            self.imp_df = {}
        for model, scores in self.feature_imp.items():
            if model not in ['vars', 'QDA']:
                df = pd.DataFrame(scores, columns=colnames)
                self.imp_df[model] = df


    def plot_feature_imp(self, case='LR', title=''):
        plt.close('all')
        latexify(6, 2.2, 2, usetex=True)
        assert self.imp_df, "run merge_scores for feature imp"
        df = self.imp_df[case]
        df = df.reindex_axis(df.abs().mean().sort_values().index, axis=1)
        if case in ['LR', 'SVC']:
            center = 0
        else:
            center = 1. / len(df.columns)
        g = sns.heatmap(df, cmap='BrBG', center=center, cbar_kws = {'label': r'$FI$'})
        xticks = [i + .5 for i in range(len(df.columns))]
        ticklabels = [self.col_labels[col] for col in df]
        g.axes.set_xticks(xticks)
        g.axes.set_xticklabels(ticklabels, rotation=90)

        alphas = [round(a, 3) for a in self.alphas]
        g.axes.set_yticks(np.arange(.5, len(alphas) + .5, 3))
        g.axes.set_yticklabels(alphas[0:len(alphas):3], rotation=0)

        if self.y.name == 'ovrl':
            g.axes.set_ylabel(r'$O_{\alpha}$')
        elif self.y.name == 'ov_mean':
            g.axes.set_ylabel(r'$\hat{O}^t_{\alpha}$')
        g.axes.set_title(title, loc='left')
        g.get_figure().tight_layout()

        name = self.save_prefix + 'fimp_{}.pdf'.format(case)
        g.get_figure().savefig(name)
        plt.clf()
        plt.close()

    def load_debug(self):
        self.get_alphas()
        self.load_scores(single=True)
        self.merge_scores()
        self.plot_singlevar_mcc()

    def result_table(self):
        # TABLE for paper including ovlp quantiles .25, .5, .75, avg, max.
        self.load_scores(single=True)
        self.merge_scores()
        data = {}
        data['q1'] = self.average_score.loc[4]
        data['q2'] = self.average_score.loc[10]
        data['q3'] = self.average_score.loc[15]
        data['mean'] = self.average_score.mean()
        data['max'] = self.average_score.max()
        data = np.round(pd.DataFrame(data), 3)
        dfname = self.save_prefix + 'sscore_summary.csv'
        data.to_csv(dfname, sep=' ')

    def plot_singlevar_mcc(self, case='AVG', sort_order='average', title=''):
        plt.close('all')
        latexify(6, 4, 2, usetex=True)

        if case == 'AVG':
            df = self.average_score
            try:
                df.columns[0].encode('ascii')
            except AttributeError:
                df.columns = [col.decode('ascii') for col in df.columns]
            if sort_order in ['average', 'self']:
                df = df.reindex(df.mean().sort_values().index, axis=1)
        elif case == 'MAX':
            df = self.single_scores[case]
            if sort_order == 'average':
                df_a = self.average_score
                df = df.reindex(df_a.mean().sort_values().index, axis=1)
            elif sort_order == 'self':
                df = df.reindex(df.mean().sort_values().index, axis=1)
        else:
            if sort_order == 'average':
                df_a = self.average_score
                df = pd.DataFrame(self.single_scores[case])
                df = df.reindex(df_a.mean().sort_values().index, axis=1)
            elif sort_order == 'self':
                df = pd.DataFrame(self.single_scores[case])
                df = df.reindex(df.mean().sort_values().index, axis=1)
            else:
                df = pd.DataFrame(self.single_scores[case])

        df_max = df.max()
        from re import match
        for var in df.columns:
            if (df_max[var] < .1) and match('c\d', var):
                df.drop(var, axis=1, inplace=True)
        fig, axs = plt.subplots(3, 2, gridspec_kw={'height_ratios': [.5, 1, 4], 'width_ratios': [.95, .03]}) #Generate four axes ([[baseline, empty], [heatmap, cbar]])
        #cbar_ax = fig.add_axes([.905, .3, .05, .3])

        # Plot baseline violin plot
        df_baseline = df.sub(df.w, axis=0)
        df_baseline[df.w == 0] = df[df.w == 0] # Replace 0's with actual value
        #axs[0,0].axhline(0, linestyle='-', alpha=1, color='grey')
        for loc in ['top', 'right', 'left', 'bottom']:
            axs[1,0].spines[loc].set_visible(False)
        g0 = sns.violinplot(data=df_baseline, color='salmon', orient='v', ax=axs[1,0], linewidth=1, inner=None, cut=0)
        axs[1,0].axhline(0, linestyle=':', alpha=.5, color='red')
        g0.vlines(np.arange(.5, df.shape[1] + .5), *g0.get_ylim(), color='grey', alpha=.3)
        g0.set_xticklabels([])
        g0.set_xticks([])
        g0.set_xlim([-.5, df.shape[1]-.5])
        g0.set_ylabel(r'$MCC_*-$' + '\n' + r'$MCC_w$')
        # Add "1" to array
        yticks = [-1, 0, 1] #list(g0.get_yticks()) + [1.]
        #g0.set_yticks(yticks) #sorted(yticks)[1:-1])

        axs[0, 1].axis('off') #Erase extra axis
        axs[1, 1].axis('off') #Erase extra axis
        #axs[1, 0].axis('off') #Erase extra axis

        # Plot directionality
        self.plot_directionality(df, axs[0, 0])

        # Plot heatmap

        g = sns.heatmap(df, cmap='GnBu', cbar_kws={'label':r'$MCC$'}, ax=axs[2, 0], cbar_ax=axs[2, 1])
        g.invert_yaxis()

        # SPECIAL delta_t case
        if 'delta_t' in self.data_path:
            basis = [0, 12 * 3600, 60, 1800]
            xticks = [i + .5 for i, j in enumerate(df.columns) if int(j) % (3600 * 24) in basis]
            ticklabels = [self.col_labels[j] for i, j in enumerate(df.columns) if i + .5 in xticks]
        else:
            xticks = [i + .5 for i in range(len(df.columns))]
            ticklabels = [self.col_labels_full[col] for col in df]


        g.axes.set_xticks(xticks)
        g.axes.set_xticklabels(ticklabels, rotation=90)
        percentiles = np.arange(0, 100, 5)

        g.axes.set_yticks(np.arange(.5, len(percentiles) + .5, 3))
        g.axes.set_yticklabels(percentiles[0:len(percentiles):3], rotation=0)
        g.vlines(range(df.shape[1] + 1), *g.get_ylim(), color='grey', alpha=.3)
        if self.y.name == 'ovrl':
            g.axes.set_ylabel('Overlap Percentile \n' + r'$\eta\left(O\right)$')
        elif self.y.name == 'ov_mean':
            g.axes.set_ylabel('Overlap Percentile \n' + r'$\eta\left(\hat{O}^t\right)$')
        g.axes.set_title(title, loc='center')

        name = self.save_prefix + 'singlevar_baseline_{}.pdf'.format(case)

        fig.tight_layout()
        plt.subplots_adjust(hspace=.2)
        fig.savefig(name)
        plt.clf()

    def plot_directionality(self, df, ax):

        dts = {var: [(0, 1)] for var in df.columns}
        dts['avg_len'] = [(0, -1)]
        dts['avg_len'].append((0, .2))
        dts['mu'] = [(0, -1)]
        dts['sig'] = [(0, -1)]
        dts['b'] = [(0, -.5)]
        dts['b'].append((0, .5))
        dts['mu_r'] = [(0, -1)]
        dts['r_frsh'] = [(0, -1)]
        dts['age'] = [(0, -1)]
        dts['m'].append((0, -.2))
        dts['bt_mu'] = [(0, -.5)]
        dts['bt_mu'].append((0, .5))
        dts['bt_tmu'] = [(0, -.5)]
        dts['bt_tmu'].append((0, .5))
        dts['bt_logt'] = [(0, -1)]
        dts['r'] = [(0, -1)]
        dts['out_call_div'] = [(0, -1)]
        dts['c1'] = [(0, -1)]
        dts['c1'].append((0, .1))
        dts['c6'].append((0, -.2))
        dts['c5'].append((0, -.2))
        dts['cstar'] = [(None, None)]
        ax.set_ylim(-1.5, 1.5)
        ax.axhline(y=0, color='grey', ls='--', alpha=.3, lw=1) #set_ylim(-1.3, 1.3)
        ax.set_xlim(0, len(df.columns))
        ax.set_ylabel('Direct.')
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['top'].set_visible(False)
        #ax.axis('off')
        ax.tick_params(axis='both', which='both', bottom='off', left='off', labelbottom='off', top='off', right='off', labelleft='off')
        ax.set_yticks([])
        ax.set_xticks([])
        for i, var in enumerate(df.columns):
            for arr in dts[var]:
                col = '#347768' if arr[1] > 0 else '#6B273D'
                col = 'g' if arr[1] > 0 else 'r'
                ax.arrow(i + .5, #x start
                        0, # y start
                        0, # change in x
                        arr[1], #change in y
                        width=.2, head_width=.3,
                        fc=col, ec=col)


    def plot_heatmap(self, df, sort_order='average', title='', savename=''):
        plt.close('all')
        latexify(6, 2.2, 2, usetex=True)

        g = sns.heatmap(df, cmap='GnBu', cbar_kws={'label':r'$F1$'})
        g.invert_yaxis()
        xticks = [i + .5 for i in range(len(df.columns))]
        ticklabels = [self.col_labels[col] for col in df]

        g.axes.set_xticks(xticks)
        g.axes.set_xticklabels(ticklabels, rotation=90)
        percentiles = np.arange(0, 100, 5)

        g.axes.set_yticks(np.arange(.5, len(percentiles) + .5, 3))
        g.axes.set_yticklabels(percentiles[0:len(percentiles):3], rotation=0)
        g.vlines(range(df.shape[1] + 1), *g.get_ylim(), color='grey', alpha=.3)
        if self.y.name == 'ovrl':
            g.axes.set_ylabel('Overlap Percentile \n' + r'$\eta\left(O\right)$')
        elif self.y.name == 'ov_mean':
            g.axes.set_ylabel('Overlap Percentile \n' + r'$\eta\left(\hat{O}^t\right)$')
        g.axes.set_title(title, loc='center')

        g.get_figure().tight_layout()
        plt.subplots_adjust(hspace=.2)
        g.get_figure().savefig(savename)
        plt.clf()


    def plot_singlevar_f1(self, case='AVG', sort_order='average', title=''):
        plt.close('all')
        latexify(6, 2.2, 2, usetex=True)
        df_score = self.average_score
        if case == 'AVG':
            df = 1-self.average_f1
            if sort_order in ['average', 'self']:
                df = df.reindex(df_score.mean().sort_values().index, axis=1)
        elif case == 'MAX':
            df = self.single_scores[case]
            if sort_order == 'average':
                df_a = self.average_score
                df = df.reindex_axis(df_a.mean().sort_values().index, axis=1)
            elif sort_order == 'self':
                df = df.reindex_axis(df.mean().sort_values().index, axis=1)
        else:
            if sort_order == 'average':
                df_a = self.average_score
                df = pd.DataFrame(self.single_scores[case])
                df = df.reindex_axis(df_a.mean().sort_values().index, axis=1)
            elif sort_order == 'self':
                df = pd.DataFrame(self.single_scores[case])
                df = df.reindex_axis(df.mean().sort_values().index, axis=1)
            else:
                df = pd.DataFrame(self.single_scores[case])
        savename = 'singlevar_f1_{}.pdf'.format(case)
        self.plot_heatmap(df, sort_order, title, savename)


    def plot_singlevar_auc(self, case='AVG', sort_order='average', title=''):
        plt.close('all')
        latexify(6, 2.2, 2, usetex=True)
        df_score = self.average_score
        if case == 'AVG':
            df = 1-self.average_auc
            if sort_order in ['average', 'self']:
                df = df.reindex(df_score.mean().sort_values().index, axis=1)
        elif case == 'MAX':
            df = self.single_scores[case]
            if sort_order == 'average':
                df_a = self.average_score
                df = df.reindex_axis(df_a.mean().sort_values().index, axis=1)
            elif sort_order == 'self':
                df = df.reindex_axis(df.mean().sort_values().index, axis=1)
        else:
            if sort_order == 'average':
                df_a = self.average_score
                df = pd.DataFrame(self.single_scores[case])
                df = df.reindex_axis(df_a.mean().sort_values().index, axis=1)
            elif sort_order == 'self':
                df = pd.DataFrame(self.single_scores[case])
                df = df.reindex_axis(df.mean().sort_values().index, axis=1)
            else:
                df = pd.DataFrame(self.single_scores[case])
        savename = 'singlevar_auc_{}.pdf'.format(case)
        self.plot_heatmap(df, sort_order, title, savename)


    def plot_dual_var(self, title=r'(ii)'):
        plt.close('all')
        latexify(5.2, 2.2, 1, usetex=True)
        index = self.average_score.mean().sort_values().index
        markers = {'w_day': 'x', 'w_hrs': '+', 'bt_n': '1', 'w': 'v', 'x': '.'}
        fig, ax = plt.subplots()
        name = r'${}$'
        #if 'w_hrs' in self.average_dual_score:
        #    self.average_dual_score.pop('w_hrs')
        for var, score in self.average_dual_score.items():
            score = score.reindex_axis(index, axis=1)
            values = np.mean(score, 0)
            label = self.col_labels[var] #' #name.format(self.col_labels[var])
            ax.scatter(range(len(values)), values, label=label, marker=markers[var], alpha=.5)
        else:
            df = self.average_score.copy()
            df = df.reindex_axis(index, axis=1)
            values = np.mean(df)
            label = r'$X$'
            if self.boot_scores:
                error = self.boot_var.copy()
                n_alph = error.shape[0]
                error = error.reindex_axis(index, axis=1)
                error = error.sum()
                error = 1.96* np.sqrt(error)
                ax.errorbar(range(len(values)), values, yerr=error, label=label, marker=markers['x'], alpha=.5, color='#9467bd', linestyle='')
            else:
                ax.scatter(range(len(values)), values, label=label, marker=markers['x'], alpha=.5)


        xticks = [i + .1 for i in range(len(df.columns))]
        ax.set_xticks(xticks)
        ticklabels = [self.col_labels[col] for col in df]
        ax.set_xticklabels(ticklabels, rotation=90)
        ax.set_ylabel(r'$MCC$')
        ax.legend(loc=0)
        ax.set_title(title, loc='center')
        name = self.save_prefix + 'doublevar.pdf'
        fig.tight_layout()
        fig.savefig(name)

    def plot_full_scores(self, title=''):
        plt.close('all')
        assert self.full_scores is not None, 'Run or load full scores first'
        latexify(3, 3, 1, usetex=True)
        fig, ax = plt.subplots()
        for model, scores in self.full_scores.items():
            ax.plot(scores, label=model)
        ax.legend(loc=0)
        ax.set_title(title)
        if self.y.name == 'ovrl':
            ax.set_xlabel(r'$O_{\alpha}$')
        elif self.y.name == 'ov_mean':
            ax.set_xlabel(r'$\hat{O}^t_{\alpha}$')
        alphas = [round(a, 3) for a in self.alphas]

        ax.set_xticks(np.arange(.5, len(alphas) + .5, 3))
        ax.set_xticklabels(alphas[0:len(alphas):3], rotation=0)
        ax.set_ylabel('MCC')
        name = self.save_prefix + 'fscores.pdf'
        fig.tight_layout()
        fig.savefig(name)


    def plot_all_single(self, sort_order='self'):
        title = {'AVG': r'(i)', 'LR': r'(i)', 'QDA':'(ii)', 'RF':'(iii)', 'ABC':'(iv)'}
        for case in ['AVG'] + [k for k in self.single_scores.keys()]:
            self.plot_singlevar_mcc(case=case, sort_order=sort_order, title=title[case])

    def plot_corrs(self):
        latexify(8, 3.75, 1, usetex=True)
        df = self.x.copy()
        if 'delta_t' in self.data_path:
            reindex = self.x.columns
            basis = [0, 12 * 3600, 60, 1800]
            xticks = [i for i, j in enumerate(df.columns) if int(j) % (3600 * 24) in basis]
            ticklabels = [self.col_labels[j] if i in xticks else '' for i, j in enumerate(df.columns)]
        else:
            reindex = ['w', 'w_day', 'w_hrs', 'len', 'avg_len', 'r', 'mu', 'sig', 'b', 'mu_r', 'm', 'r_frsh', 'age', 't_stb', 'bt_n', 'bt_mu', 'bt_sig', 'bt_cv', 'bt_tmu', 'bt_tsig', 'bt_logt', 'out_call_div'] + ['c' + str(i) for i in range(1, 16)]
            xticks = [i for i in range(len(reindex))]
            ticklabels = [self.col_labels[col] for col in reindex]
        df = df.reindex_axis(reindex, axis=1)
        fig, axs = plt.subplots(1, 2, sharey=True, sharex=True)
        cbar_ax = fig.add_axes([.91, .3, .03, .4])

        dfc = df.corr('pearson')
        g1 = sns.heatmap(dfc, ax=axs[0], center=0, cbar=False, cbar_ax=None,
                xticklabels=ticklabels, yticklabels=ticklabels)
        g1.set_xticklabels(ticklabels)
        g1.set_yticklabels(ticklabels)

        dfc = df.corr('spearman')
        g2 = sns.heatmap(dfc, ax=axs[1], center=0, cbar=True, cbar_ax=cbar_ax,
                vmin=-1, vmax=1, xticklabels=ticklabels, yticklabels=ticklabels)

        fig.tight_layout(rect=[0, 0, .9, 1])
        fig.savefig(self.save_prefix + 'correlations.pdf')


if __name__ == '__main__':
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/scratch/work/urenaj1/full/")
    parser.add_argument("--y_var", default="ovrl")
    parser.add_argument("--ranked", default="False")
    parser.add_argument("--models", default=["QDA"], nargs="+") #nargs=+, take 1 or more arguments
    parser.add_argument("--save_path", default="./")
    parser.add_argument("--cluster_only", default="False")
    parser.add_argument("--single_var", default="True")
    parser.add_argument("--fvars", default=[], nargs="*") #nargs=*, take 0 or more arguments
    parser.add_argument("--full_vars", default="False")
    parser.add_argument("--c_star", default="False")

    remove_default = ['deg_0', 'deg_1', 'n_ij', 'e0_div', 'e1_div', 'bt_tsig1']

    parser.add_argument("--remove", default=remove_default, nargs="*")
    pargs = parser.parse_args()

    remove = pargs.remove
    #if pargs.y_var == 'ovrl':
    #    remove.append('ov_mean')
    #elif pargs.y_var == 'ov_mean':
    #    remove.append('ovrl')

    r = "r" if eval(pargs.ranked) else "nr"
    save_path = os.path.join(pargs.save_path, "{}-{}/".format(pargs.y_var, r))
    cluster_only = eval(pargs.cluster_only)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if cluster_only:
        save_path  = os.path.join(save_path, 'cluster/')
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    PTS = PredictTieStrength(y_var=pargs.y_var, data_path=pargs.data_path, save_prefix=save_path, models=pargs.models, k=3, alpha_step=5, ranked=eval(pargs.ranked), remove=remove, cluster_only=cluster_only, c_star=eval(pargs.c_star))
    PTS.get_alphas()
    PTS.run_alphas(single_var=eval(pargs.single_var), fvars=pargs.fvars, full_vars=eval(pargs.full_vars))

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

RF = RandomForestRegressor(35a
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

