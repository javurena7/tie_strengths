from nets import *
from sklearn.model_selection import train_test_split
import analysis_tools as at
import numpy as np; from numpy import inf
import pandas as pd
import plots
from pandas import read_pickle
from pickle import dump as dump_pickle
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestRegressor
from itertools import chain, combinations, product
from verkko.binner import bins as binner
from scipy.stats import binned_statistic_dd
from sklearn import linear_model
from re import search as re_search, match
import os
import yaml
import copy


def write_logs(msg, path):
    with open(path, 'a') as f:
        f.write(msg)

class TieStrengths(object):
    def __init__(self, logs_path, run_path, kaplan=True, extended_logs_path=None, delta=3600):
        if not os.path.exists(run_path):
            os.makedirs(run_path)

        self.analysis = {}
        tmp_file = os.path.join(run_path, 'tmp_file.txt')
        self.paths = {'full_times_dict': os.path.join(run_path, 'times_dic.txt')}
        self.paths['call_times'] = os.path.join(run_path, 'call_times.txt')
        self.paths['sms_times'] = os.path.join(run_path, 'sms_times.txt')
        self.paths['logs'] = logs_path
        self.paths['status'] = os.path.join(run_path, 'status.txt')
        self.first_date, self.last_date = get_dates(self.paths['logs'])
        self._obs = (self.last_date - self.first_date)/(60.*60*24)

        """
        with open(self.paths['status'], 'wb') as f:
            f.write('running on ' + run_path + ' \n')
        write_logs('-------------\n', self.paths['status'])

#### Create files for temporal data
        rm = False
        # Create temporal file from logs
        if not all([os.path.isfile(self.paths['full_times_dict']), os.path.isfile(self.paths['sms_times']), os.path.isfile(self.paths['call_times'])]):
            awk_tmp_times(self.paths['logs'], tmp_file, run_path)
            rm = True
        # Create file with times of contact for each edge
        if not os.path.isfile(self.paths['full_times_dict']):
            print('Creating call and sms times dictionary... \n')
            awk_full_times(tmp_file, self.paths['full_times_dict'])
        # Create file with call times and call length for each edge
        if not os.path.isfile(self.paths['call_times']):
            print('Creating calls dictionary...\n')
            awk_calls(tmp_file, self.paths['call_times'])
        # Create file with sms times for each edge
        if not os.path.isfile(self.paths['sms_times']):
            print('Creating sms dictionary...\n')
            awk_sms(tmp_file, self.paths['sms_times'])
        if rm:
            remove_tmp(tmp_file)

## Create net files
        # Create net
        self.paths['net'] = os.path.join(run_path, 'net.edg')
        if not os.path.isfile(self.paths['net']):
            print('Creating net... \n')
            awk_total_calls_from_times(self.paths['full_times_dict'], self.paths['net'])

        # Obtain extended overlap
        if extended_logs_path is not None:
            self.paths['extended_logs'] = os.path.join(extended_logs_path)
            self.paths['extended_net'] = os.path.join(run_path, 'extended_net.edg')
            self.paths['overlap'] = os.path.join(run_path, 'extended_overlap.edg')
            self.paths['degrees'] = os.path.join(run_path, 'extended_degrees.txt')
            self.paths['neighbors'] = os.path.join(run_path, 'neighbors.txt')
            if not os.path.isfile(self.paths['extended_net']):
                awk_total_calls(self.paths['extended_logs'], self.paths['extended_net'])
                write_logs('Creating extended net... \n', self.paths['status'])
            if not os.path.isfile(self.paths['overlap']):
                write_logs('Obtaining net overlap... \n', self.paths['status'])
                write_logs('\t Reading edges... \n', self.paths['status'])
                net_ext = read_edgelist(self.paths['extended_net'])  #netio.loadNet(self.paths['extended_net'])
                write_logs('\t Calculating overlap... \n', self.paths['status'])
                at.net_overlap(net_ext, output_path=self.paths['overlap'], alt_net_path=self.paths['net'])
                write_logs('\t Done. \n', self.paths['status'])
        # Obtain basic overlap
        else:
            self.paths['overlap'] = os.path.join(run_path, 'overlap.edg')
            self.paths['degrees'] = os.path.join(run_path, 'degrees.txt')
            if not os.path.isfile(self.paths['overlap']):
                net = read_edgelist(self.paths['net'])
                at.net_overlap(net, output_path=self.paths['overlap'])
        if not os.path.isfile(self.paths['degrees']):
            print('Obtaining degrees\n')
            awk_degrees(self.paths['net'], self.paths['degrees'])
        if not os.path.isfile(self.paths['neighbors']):
            get_neighbors(self.paths['overlap'], self.paths['degrees'], self.paths['neighbors'])
        """

        self.run_path = run_path
        self.delta = delta
        self.df = None
        self.cv_columns = []
        self.km_variables = []

    def get_stats(self, mode='call'):

        assert mode in ['call', 'sms'], "mode must be either 'call' or 'sms'"
        assert os.path.isfile(self.paths[mode + '_times']), mode + '_times file not found'
        self.paths[mode + '_stats'] = os.path.join(self.run_path, mode + '_stats.txt')
        w = open(self.paths[mode + '_stats'], 'wb')

        colnames = ['0', '1']
        week_stats = [mode[0] + '_wkn_' + str(i) for i in range(12)]; week_stats.append(mode[0] + '_wkn_t')
        colnames.extend(week_stats)
        if mode == 'call':
            len_stats = [mode[0] + '_wkl_' + str(i) for i in range(12)]; len_stats.append(mode[0] + '_wkl_l')
            colnames.extend(len_stats)
        unif_stats = [mode[0] + '_uts_' + i for i in ['mu', 'sig', 'sig0', 'logt']]
        colnames.extend(unif_stats)
        iet_names = [mode[0] + '_iet_' + i for i in ['mu_na', 'sig_na', 'bur_na', 'mu_km', 'sig_km', 'bur_km']]
        colnames.extend(iet_names)
        colnames.append(mode[0] + '_brtrn')
        w.write(' '.join(colnames) + '\n')
        with open(self.paths[mode + '_times'], 'r') as r:
            row = r.readline()
            while row:
                if mode=='call':
                    e0, e1, times, lengths = utils.parse_time_line(row, True)
                    l = [e0, e1]
                    lengths = [ln + 1 for ln in lengths] #Some call lengths are zero
                    week_stats, len_stats = weekday_call_stats(times, lengths)
                    l.extend(week_stats); l.extend(len_stats)
                    unif_call_stats = uniform_time_statistics(times, self.first_date, self.last_date, lengths)
                    l.extend(unif_call_stats)
                elif mode == 'sms':
                    e0, e1, times = utils.parse_time_line(row, False)
                    l = [e0, e1]
                    week_stats, _ = weekday_call_stats(times)
                    l.extend(week_stats)
                    unif_sms_stats = uniform_time_statistics(times, self.first_date, self.last_date)
                    l.extend(unif_sms_stats)

                if len(times) > 1:
                    iet_na = inter_event_times(times, self.last_date, self.first_date, method='naive')
                else:
                    iet_na = [np.nan, np.nan, np.nan]
                l.extend(iet_na)

                iet_km = inter_event_times(times, self.last_date, self.first_date, method='km')
                l.extend(iet_km)

                bursty_trains = number_of_bursty_trains(times, delta=self.delta)
                l.append(bursty_trains)

                l = [str(s) for s in l]
                w.write(' '.join(l) + '\n')
                row = r.readline()
        w.close()

    def _join_dataframes(self, df_list=['neighbors', 'call_stats', 'sms_stats'], mode_list=['outer', 'outer'], return_df = False):
        df = pd.read_table(self.paths[df_list[0]], sep=' ')
        for name, mode in zip(df_list[1:], mode_list):
            df_2 = pd.read_table(self.paths[name], sep=' ')
            df = pd.merge(df, df_2, on=['0', '1'], how=mode)
        self.paths['full_df'] = os.path.join(self.run_path, 'full_df.txt')
        df.to_csv(self.paths['full_df'], sep=' ', index=False)
        if return_df:
            return df

    def df_preprocessing(self, transfs, na_values, df=None):
        if df is None:
            df = pd.read_table(self.paths['full_df'], sep=' ')

        df = df.fillna(value=na_values)
        for col, c in transfs.get('tanh', []):
            df.loc[:, col] = (c*df[col]).apply(np.tanh)

        for col, c in transfs.get('log', []):
            df.loc[:, col] = (df[col] + c).apply(np.log)

        for col, c in transfs.get('sqr', []):
            df.loc[:, col] = ((df[col] - c)**2)

        for col in transfs.get('rank', []):
            l, _ = df.shape
            df.loc[:, col] = rankdata(df[col])/float(l)


        # add uts_mu sampling from the distribution (with uts_mu==1)
        #idx = df.s_wkn_t == 1
        #r_idx = df.s_uts_mu.isnull()
        #df.loc[r_idx, 's_uts_mu'] = np.random.choice(df.s_uts_mu[~r_idx & idx], size=r_idx.sum())
        #idx = df.c_wkn_t == 1
        #r_idx = df.c_uts_mu.isnull()
        #df.loc[r_idx, 'c_uts_mu'] = np.random.choice(df.c_uts_mu[~r_idx & idx], size=r_idx.sum())
        df.replace(np.inf, np.nan, inplace=True)
        df.dropna(inplace=True)
        return df

    def get_variable_transformations(self, cv_params):
        params = copy.deepcopy(cv_params)
        nas = {k:[] for k in params}
        for k, v in params.iteritems():
            for trans in v:
                nas[k].extend(v[trans]['na'])
        for var in params.keys():
            params[var]['raw'] = {'na': []}
            params[var]['rank'] = {'na': nas[var]}

        flt = {}
        for var, var_dict in params.iteritems():
            for transf in var_dict:
                if transf not in ['rank', 'raw']:
                    params[var][transf] = [list(a) for a in product(params[var][transf]['na'], params[var][transf]['c'])]
                elif transf == 'rank':
                    params[var][transf] = [[n] for n in params[var][transf]['na']]
                else:
                    params[var][transf] = [[]]
            flt[var] = [[k] + comb for k, v in var_dict.items() for comb in v]

        return flt

    def params_cross_validation(self, cv_path='tie_strengths/cv_config.yaml'):
        conf = yaml.load(open(cv_path))
        params = self.get_variable_transformations(conf['params'])
        cols_pttrns = params.keys()
        try: #TODO: change this (for db)
            self.paths['full_df']
        except:
            self.paths['full_df'] = os.path.join(self.run_path, 'full_df.txt')
        df = pd.read_table(self.paths['full_df'], sep=' ')
        cols_dic = self.get_cols_dic(cols_pttrns, df.columns) # link cols with patterns

        # TODO: add this to a diff function, it's different preprocessing
        pttrn = '_wk(n|l)_(\d+|t|l)'
        df_nas = {col: 0. for col in df.columns if re_search(pttrn, col)}
        df = df.fillna(value = df_nas)

        wkn_cols = [col for col in df.columns if re_search('c_wkn_\d+', col)]
        wkl_cols = [col for col in df.columns if re_search('c_wkl_\d+', col)]
        wks_cols = [col for col in df.columns if re_search('s_wkn_\d+', col)]
        #df.loc[:, 'c_l_dist'] = df.apply(lambda x: np.dot(x[wkn_cols], x[wkl_cols]), axis=1)
        if len(wkn_cols) == len(wks_cols):
            #df.loc[:, 's_c_dist'] = df.apply(lambda x: np.dot(x[wkn_cols], x[wks_cols]), axis=1)
            wks_cols.append('s_c_dist')
        del df['c_wkn_0']
        del df['c_wkl_0']
        del df['s_wkn_0']
        del df['0']
        del df['1']
        del df['n_ij']
        del df['deg_0']
        del df['deg_1']
        w = open(conf['output_file'], 'wb')
        w.write(';'.join(['conf', 'sms', 'n_row', 'score']) + '\n')

        for comb in product(*params.values()):
            transf, nas = self.parse_variable_combinations(cols_pttrns, cols_dic, comb)
            proc_df = self.df_preprocessing(transf, nas, df)
            y = proc_df['ovrl']; del proc_df['ovrl']
            x_train, x_test, y_train, y_test = train_test_split(proc_df, y, test_size=0.3)
            try:
                rf = RandomForestRegressor()
                rf.fit(x_train, y_train)
            except:
                import pdb; pdb.set_trace()
            sc = rf.score(x_test, y_test)
            self.write_results(w, comb, 1, proc_df.shape[0], sc)

            transf = self.remove_sms_cols(transf)
            proc_df = self.df_preprocessing(transf, nas, df)
            y = proc_df['ovrl']; del proc_df['ovrl']
            x_train, x_test, y_train, y_test = train_test_split(proc_df, y, test_size=0.3)
            rf = RandomForestRegressor()
            rf.fit(x_train, y_train)
            sc = rf.score(x_test, y_test)
            self.write_results(w, comb, 0, proc_df.shape[0], sc)

        w.close()

    def write_results(self, w, comb, sms, n_row, score):
        l = [str(comb), str(sms), str(n_row), str(score)]
        w.write(';'.join(l) + '\n')

    def remove_sms_cols(self, transf):
        transf_new = {k: [] for k in transf}
        for k, v in transf.iteritems():
            for cmb in v:
                if k == 'raw':
                    if not match('s_', cmb):
                        transf_new[k].append(cmb)
                elif not match('s_', cmb[0]):
                    transf_new[k].append(cmb)
        return transf_new


    def get_cols_dic(self, cols_pttrns, columns):
        cols_dic = {pttrn: [] for pttrn in cols_pttrns}
        for col in columns:
            for pttrn in cols_pttrns:
                if re_search(pttrn, col):
                    cols_dic[pttrn].append(col)
                    break
        return cols_dic

    def parse_variable_combinations(self, cols_pttrns, cols_dic, comb):
        transf, nas = {cbn[0]:[] for cbn in comb}, {}
        obs = self._obs
        for pttrn, cbn in zip(cols_pttrns, comb):
            if len(cbn) > 1:
                nas.update({col: eval(cbn[1]) for col in cols_dic[pttrn]})
            else:
                transf['raw'].extend(cols_dic[pttrn])
            if len(cbn) == 2:
                transf[cbn[0]].extend(cols_dic[pttrn])
            if len(cbn) == 3:
                transf[cbn[0]].extend([(col, eval(cbn[2])) for col in cols_dic[pttrn]])

        return transf, nas

    def _ifelse(a, b, c):
        if a:
            return b
        else:
            return c


    def _reciprocity(self):
        self.paths['reciprocity_1'] = os.path.join(self.run_path, 'reciprocity.edg')
        self.paths['reciprocity_2'] = os.path.join(self.run_path, 'reciprocity_2.edg')
        if not os.path.isfile(self.paths['reciprocity_1']):
            reciprocity(self.paths['logs'], self.paths['reciprocity_1'], self.paths['reciprocity_2'])

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

########## new part: random forest


if __name__=="__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor

    logs_path = '../data/mobile_network/canarias/canarias_call_newid_log.txt'
    df = pd.read_csv('data/run/all_stats.csv', sep=' ')
    df = df.drop_na()
    del df['0']
    del df['1']
    y = df['overlap']
    del df['overlap']
    rf = RandomForestRegressor()
    x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.5)
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)
