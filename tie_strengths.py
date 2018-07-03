from nets import *
import analysis_tools as at
import numpy as np
import pandas as pd
import plots
from pandas import read_pickle
from pickle import dump as dump_pickle
from scipy.stats import rankdata
from itertools import chain, combinations, product
from verkko.binner import bins as binner
from scipy.stats import binned_statistic_dd
from sklearn import linear_model
import powerlaw
import os
import yaml


class TieStrengths(object):
    def __init__(self, logs_path, run_path, kaplan=True, extended_log_path=None, delta=3600):
        if not os.path.exists(run_path):
            os.makedirs(run_path)
        self.analysis = {}
        self.paths = {'times_dict': os.path.join(run_path, 'times_dic.txt')}
        self.paths['logs'] = logs_path
        if not os.path.isfile(self.paths['times_dict']):
            times = dict_elements(logs_path)
            utils.write_dic(times, self.paths['times_dict'])
            del times

        self.paths['net'] = os.path.join(run_path, 'net.edg')
        if not os.path.isfile(self.paths['net']):
            net = total_calls_times(self.paths['times_dict'])
            utils.write_edges(net, self.paths['net'])

        if extended_log_path is not None:
            del net
            self.paths['extended_net'] = os.path.join(run_path, 'extended_net.edg')
            self.paths['overlap'] = os.path.join(run_path, 'extended_overlap.edg')
            if not os.path.isfile(self.paths['overlap']):
                net_ext = total_calls(extended_log_path)
                utils.write_edges(net_ext, self.paths['extended_net'])
                overlap = at.get_weight_overlap(net_ext)
                utils.write_dic(overlap, self.paths['overlap'], add = 0.0)

        else:
            self.paths['overlap'] = os.path.join(run_path, 'overlap.edg')
            if not os.path.isfile(self.paths['overlap']):
                net = read_edgelist(self.paths['net'])
                overlap = at.get_weight_overlap(net)
                utils.write_dic(overlap, self.paths['overlap'], add = 0.0)

        self.run_path = run_path
        self.delta = delta
        self.df = None

        self.cv_columns = []
        self.km_variables = []


    def _burstiness(self, kaplan):
        path_key = 'burstiness_' + kaplan[:2]
        path = 'burstiness_{}.edg'.format(kaplan[:2])
        self.paths[path_key] = os.path.join(self.run_path, path)
        if not os.path.isfile(self.paths[path_key]):
            net_burstiness(self.paths['times_dict'], self.paths[path_key], kaplan=kaplan)

    def _mean_inter_event(self, kaplan):
        path_key = 'mean_iet_' + kaplan[:2]
        path = 'mean_iet_{}.edg'.format(kaplan[:2])
        self.paths[path_key] = os.path.join(self.run_path, path)
        if not os.path.isfile(self.paths[path_key]):
            net_residual_times(path=self.paths['times_dict'], output_path=self.paths[path_key], kaplan=kaplan)


    def _calltimes_mode(self):
        self.paths['calltimes'] = os.path.join(self.run_path, 'calltimes.edg')
        if not os.path.isfile(self.paths['calltimes']):
            net_calltimes_mode(self.paths['times_dict'], self.paths['calltimes'])

    def _reciprocity(self):
        self.paths['reciprocity'] = os.path.join(self.run_path, 'reciprocity.edg')
        if not os.path.isfile(self.paths['reciprocity']):
            reciprocity(self.paths['logs'], self.paths['reciprocity'])

    def _bursty_trains(self):
        self.paths['trains'] = os.path.join(self.run_path, 'bursty_trains.edg')
        if not os.path.isfile(self.paths['trains']):
            bursty_trains_to_file(self.paths['times_dict'], self.paths['trains'], self.delta)

    def all_stats(self):

        self._burstiness('km')
        self._burstiness('naive')
        self.km_variables.append('burstiness')

        self._mean_inter_event('km')
        self._mean_inter_event('naive')
        self.km_variables.append('mean_iet')

        self._calltimes_mode()

        self._bursty_trains()

        self._reciprocity()

        self._join_stats()

    def _join_stats(self):
        stats_paths = {x: y for x, y in self.paths.items() if y.endswith('.edg')}
        net_path = stats_paths.pop('net')
        net = read_edgedic(net_path)
        vrbs = ['n_calls']
        for var, path in stats_paths.iteritems():
            vrbs.append(var)
            stat = read_edgedic(path)
            for k, v in net.iteritems():
                try:
                    v.append(stat[k][0])
                except KeyError:
                    v.append(np.nan)

        self.df = pd.DataFrame(net.values(), columns=vrbs)


    def powerset(self, x):
        # Obtain power set of x
        return chain.from_iterable(combinations(x, r) for r in range(len(x)+1))


    def _add_km_variables(self, comb, var, suffix):
        try:
            comb.remove(var)
        except ValueError:
            pass
        comb.append("{}_{}".format(var, suffix))
        return comb


    def _get_variable_combinations(self):
        columns = list(self.df.columns)
        columns.remove('overlap')

        for col, km_var in product(columns, self.km_variables):
            if col.startswith(km_var):
                columns.remove(col)
                if km_var not in columns:
                    columns.append(km_var)
        vrbls = [list(comb) for comb in self.powerset(columns) if len(list(comb)) > 0]
        variables = []
        for comb in vrbls:
            comb_list = [comb]
            for var in self.km_variables:
                if var in comb:
                    for c in comb_list[:]:
                        comb_list.remove(c)
                        comb_list.append(self._add_km_variables(c[:], var, 'km'))
                        comb_list.append(self._add_km_variables(c[:], var, 'na'))
            variables += comb_list

        return variables


    def _get_variable_transformations(self, comb, conf):
        s = ", ".join(["conf['{}']".format(v) for v in comb])
        return eval("product({})".format(s))


    def _transform(self, x, mode):

        if mode == 'raw':
            return x
        if mode == 'rank':
            return rankdata(x)/len(x)
        if mode == 'log':
            try:
                return np.log(np.array(x))
            except:
                print("Error: could not obtain logarithm")
                return x
        else:
            print("Invalid transformation, using raw values")
            return x


    def _transform_variables(self, x, transf):
        return pd.DataFrame({column[0]: self._transform(column[1], mode) for column, mode in zip(x.iteritems(), transf)}, columns=[c[0] for c in x.iteritems()])

    def fit_lm(self, x, y):
        lm = linear_model.LinearRegression().fit(x, y)
        return lm

    def _write_scores(self, scores, n_row, comb, transfs, outputfile):
        f_cols = self.df.columns.tolist()
        with open(outputfile, 'a+') as f:
            for transf, score, row in zip(transfs, scores, n_row):
                comb_dict = {v: t for v, t in zip(comb, transf)}
                row = [comb_dict.get(col, '-') for col in f_cols] + \
                        [str(score), str(row) + '\n']
                f.write(','.join(row))

    def _get_bins(self, x, bin_params, transfs):
        bins, bin_means = [], []
        for column, t in zip(x.iteritems(), transfs):
            col_type = bin_params[column[0]][0]
            if col_type == 'log' and t not in ['rank', 'raw']:
                b = binner.Bins(float, max(1, min(column[1])), max(column[1]), 'log', bin_params.get(column[0], 1.5)[1])
            else:
                if col_type != 'log':
                    n_bin = bin_params[column[0]][1]
                else:
                    n_bin = bin_params[column[0]][2]
                b = binner.Bins(float, min(column[1]), max(column[1]), 'lin', n_bin)
            bins.append(b.bin_limits)
            bin_means.append(b.centers)

        return bins, bin_means

    def _transform_x(self, i, centers, shape):
        if i == 0:
            rep = 1
        else:
            rep = np.prod(shape[:i])
        if i == len(shape):
            til = 1
        else:
            til = np.prod(shape[(i+1):])
        x = np.repeat(centers[i], rep)
        return np.tile(x, til)

    def _bin_and_transform(self, x, y, bins, centers):
        bin_means, _, _ = binned_statistic_dd(x.as_matrix(), y.as_matrix(), bins=bins)
        shape = bin_means.shape
        x_new = []
        for i in range(len(shape)):
            x_new.append(self._transform_x(i, centers, shape))
        x = pd.DataFrame({c: col for c, col in zip(x.columns, x_new)}, columns=x.columns)
        return pd.DataFrame({'overlap': bin_means.reshape(-1)}), x


    def cv(self, config_path='cv_config.yaml'):
        if self.df is None:
            self.all_stats()

        with open(config_path) as f:
            conf = yaml.load(f)

        outputfile = conf['output_file']
        variables = self._get_variable_combinations()
        f_cols = self.df.columns.tolist()
        row = f_cols + ['score', 'n_rows\n']
        with open(outputfile, 'wb') as f:
            f.write(','.join(row))

        for comb in variables:
            transfs = list(self._get_variable_transformations(comb, conf))
            scores, n_row = [], []
            for transf in transfs:
                X = self._transform_variables(self.df[comb], transf)
                idx = pd.notnull(X).all(1) & ~np.isinf(X).any(1)
                X = X[idx]
                bins, centers = self._get_bins(X, conf['bin_params'], transf)
                Y, X = self._bin_and_transform(X, self.df.overlap[idx], bins, centers)
                idx = pd.notnull(Y).all(1)
                X = X[idx]
                Y = Y[idx]
                n_row.append(X.shape[0])
                try:
                    lm = self.fit_lm(X, Y)
                    scores.append(lm.score(X, Y))
                except:
                    scores.append(np.nan)
            self._write_scores(scores, n_row, comb, transfs, outputfile)


if __name__=="__main__":
    logs_path = '../data/mobile_network/canarias/canarias_call_newid_log.txt'
