# -*- coding: utf-8 -*-
import numpy as np
from netpython import *
import datetime as dt
import subprocess
import scipy.integrate as sinteg
from iet import events
from verkko.binner.binhelp import *
from scipy.stats import mode
import utils
import os
import pandas as pd #only for one function

from scipy.stats import entropy

logs_path = '../data/mobile_network/madrid/madrid_call_newid_log.txt'
times_path = 'run/madrid_2504/times_dic.txt'

def get_dates(logs_path):
    cmd_list = ['head', '-1', logs_path]
    p = subprocess.check_output(' '.join(cmd_list), shell=True)
    first_date = int(p.split()[0])

    cmd_list = ['tail', '-1', logs_path]
    p = subprocess.check_output(' '.join(cmd_list), shell=True)
    last_date = int(p.split()[0])

    return first_date, last_date

def total_calls(logs_path=logs_path, id_cols=(1,3)):
    """
    From call logs, crete net where stregnth is the total number of calls
    """
    id_0, id_1 = id_cols
    net = netio.pynet.SymmNet()
    with open(logs_path, 'r') as r:
        row = r.readline()
        while row:
            rs = row.split(' ')
            net[int(rs[id_0]), int(rs[id_1])] += 1
            row = r.readline()
    return  net

def awk_total_calls(logs_path, output_path):
    main_awk = "{($5 > $3) ? p = $3 FS $5 : p = $5 FS $3; print p}"
    main_awk_rearrange = "{if($2 != $3) print $2, $3, $1}"
    cmd_list = ["awk", "'", main_awk, "'", logs_path , "| sort | uniq -c |", "awk", "'", main_awk_rearrange, "'",">", output_path]
    p = subprocess.Popen(' '.join(cmd_list), shell=True)
    p.wait()

def total_calls_times(times_path, output_path):
    """
    Create net with total number of calls using times of calls
    """
    net = netio.pynet.SymmNet()
    w = open(output_path, 'wb')
    with open(times_path, 'r') as r:
        row = r.readline()
        while row:
            n1, n2, times = utils.parse_time_line(row)
            f = ' '.join(n1, n2, str(len(times)))
            w.write(f)
            #net[int(n1), int(n2)] = len(times)
            row = r.readline()
    w.close()

def total_time(logs_path=logs_path, id_cols=(1,3), id_len=4):
    """
    From call logs, create net where strength is the total calling time
    """
    net = netio.pynet.SymmNet()
    id_0, id_1 = id_cols
    with open(logs_path, 'r') as r:
        row = r.readline()
        while row:
            rs = row.split(' ')
            net[int(rs[id_0]), int(rs[id_1])] += float(rs[id_len])
            row = r.readline()
    return net

def awk_tmp_times(logs_path, tmp_file, run_path):
    """
    Use awk to obtain a file with call timestamps in the format:
        id_1 id_2 ts_1 ts_2 ... ts_n
    where id_1 is the min id of the edge, and id_2 is the max id of the edge, and ts_i is the timestamp for the i-th call between id_1 and id_2
    """

    add_tmp_file = os.path.join(run_path, "add_tmp_times_file.txt")
    # First, use awk to resort logs into id_1, id_2, timestamp; where id_1 is the min id, and id_2 is the max
    main_awk = "{($4 > $2) ? p = $2 FS $4 FS $1 FS $3 FS $5: p = $4 FS $2 FS $1 FS $3 FS $5; print p}"
    cmd_list = ["awk", "'", main_awk, "'", logs_path, ">", tmp_file]
    print(' '.join(cmd_list))
    p1 = subprocess.Popen(' '.join(cmd_list), shell=True)
    p1.wait()

    main_awk = "{if ($1 != $2) print}"
    cmd_list = ["awk", "'", main_awk, "'", tmp_file, ">", add_tmp_file, "&& mv", add_tmp_file, tmp_file]
    p2 = subprocess.Popen(' '.join(cmd_list), shell=True)
    p2.wait()


def awk_full_times(tmp_file, output_path):
    # Next, use awk to obtain a file with id_1, id_2 followed by a list of timestamps
    main_awk = "{if (a[$1 FS $2]) a[$1 FS $2]=a[$1 FS $2] FS $3; else a[$1 FS $2] = $3;} END {for (i in a) print i, a[i];}"
    cmd_list = ["awk", "'", main_awk, "'", tmp_file, ">", output_path]
    p = subprocess.Popen(' '.join(cmd_list), shell=True)
    p.wait()

#    cmd_list = ["rm", tmp_file]
#    p2 = subprocess.Popen(' '.join(cmd_list), shell=True)
#    p2.wait()

def awk_sms(tmp_file, output_path):

    main_awk = "{if($4 == 5) {if (a[$1 FS $2]) a[$1 FS $2]=a[$1 FS $2] FS $3; else a[$1 FS $2] = $3;}} END {for (i in a) print i, a[i];}"
    cmd_list = ["awk", "'", main_awk, "'", tmp_file, ">", output_path]
    p = subprocess.Popen(' '.join(cmd_list), shell=True)
    p.wait()


def awk_calls(tmp_file, output_path):

    main_awk = "{if ($4 == 2) {if (a[$1 FS $2]) a[$1 FS $2]=a[$1 FS $2] FS $3 FS $5; else a[$1 FS $2] = $3 FS $5;}} END {for (i in a) print i, a[i];}"
    cmd_list = ["awk", "'", main_awk, "'", tmp_file, ">", output_path]
    p = subprocess.Popen(' '.join(cmd_list), shell=True)
    p.wait()


def remove_tmp(tmp_file):
    cmd_list = ['rm', tmp_file]
    p = subprocess.Popen(' '.join(cmd_list), shell=True)
    p.wait()

def awk_total_calls_from_times(times_path, output_path):

    main_awk = "{print $1, $2, NF-2}"
    cmd_list = ["awk", "'", main_awk, "'", times_path, ">", output_path]
    p = subprocess.Popen(' '.join(cmd_list), shell=True)
    p.wait()

def awk_filter_extended_net(net_path, extended_net_path, output_path):
    """
    Filter extended_net_path so that only elements in net_path are included.
    """
    main_awk = "FNR==NR{a[$1 FS $2]=1;next;}{if (a[$1 FS $2]) print}"
    cmd_list = ["awk", "'", main_awk, "'", net_path, extended_net_path, ">", output_path]
    p = subprocess.Popen(' '.join(cmd_list), shell=True)
    p.wait()

def awk_degrees(net_path, output_path):
    main_awk = "{a[$1]+=1; a[$2]+=1} END {for(i in a) print i, a[i];}"
    cmd_list = ["awk", "'", main_awk, "'", net_path, ">", output_path]
    p = subprocess.Popen(' '.join(cmd_list), shell=True)
    p.wait()

def dict_elements(logs_path=logs_path, id_cols=(1,3), id_store=0, extra_id=None):
    """
    Function for creating a dictionary with all the elements, where each entry
    is an edge, and the values correspond to observations.
    For call logs, id_store = 0 is the call time, while extra_id is the call lengths (if necessary)
    """
    id_0, id_1 = id_cols
    dic = {}
    c = 0
    with open(logs_path, 'r') as r:
        row = r.readline()
        while row:
            rs = row.split(' ')
            v0 = int(rs[id_0])
            v1 = int(rs[id_1])
            key = (min(v0, v1), max(v0, v1))
            try:
                dic[key].append(int(rs[id_store]))
            except:
                dic[key] = [int(rs[id_store])]
            if extra_id:
                dic[key].append(int(rs[extra_id]))
            row = r.readline()
            if c % 1000000 == 0:
                print('row: {}'.format(c))
            c += 1

    return dic


def weekday_call_stats(x, extra_information=None):
    """
    Function for obtaining the proportion of calls/sms according to day of the week and time.
    x: list of timestamps
    extra_information: weight associated to each timestamp

    returns:
        lv: portion of calls for
            - monday to thursday: morning (0 to 6)
            - monday to thursday: daytime (7 to 15)
            - monday to thursday: evening (16 to 24)
            - friday morning (0 to 6)
            - friday daytime (7 to 15)
            - friday evening (16 to 23)
            - saturday morning (0 to 8)
            - saturday daytime (9 to 15)
            - saturday evening (16 to 23)
            - sunday morning (0 to 8)
            - sunday afternoon (9 to 15)
            - sunday evening (16 to 23)
            - total calls
        wv: proportion of weight, as in lv
    """
    dates = [dt.datetime.fromtimestamp(d) for d in x]
    if extra_information is None:
        extra_information = [0] * len(x)
    lv = [0] * 12
    wv = [0] * 12
    for date, extra in zip(dates, extra_information):
        w, h = date.weekday(), date.hour
        if w < 4: #monday to thursday
            _classify_weekday(h, lv, wv, extra, [7, 16], [0, 1, 2])
        elif w == 4: #friday
            _classify_weekday(h, lv, wv, extra, [7, 16], [3, 4, 5])
        elif w == 5: #saturday
            _classify_weekday(h, lv, wv, extra, [8, 16], [6, 7, 8])
        elif w == 6: #sunday
            _classify_weekday(h, lv, wv, extra, [8, 16], [9, 10, 11])

    t_lv = sum(lv)
    t_wv = sum(wv)
    lv = [round(float(t)/t_lv, 4) for t in lv]
    if t_wv > 0:
        wv = [round(float(t)/t_wv, 4) for t in wv]
    lv.append(t_lv); wv.append(t_wv)

    return lv, wv

def _classify_weekday(h, lv, wv, extra, bins, idx):
    if h < bins[0]:
        lv[idx[0]] += 1
        wv[idx[0]] += extra
    elif h < bins[1]:
        lv[idx[1]] += 1
        wv[idx[1]] += extra
    else:
        lv[idx[2]] += 1
        wv[idx[2]] += extra


def bin_ts_idx(x, start_date, bin_len):
    idx = [(t-start_date)/bin_len for t in x]
    return idx

# ASSESS HOW AUTOCORR WORKS - TO BE REMOVED
def assess_autocorr(run_path):
    times = read_timesdic(run_path + 'times_dic_sample.txt')
    times = {k: v for k, v in times.iteritems() if len(v) > 1}
    res = []
    start = 1167609600
    lags = []
    bins_per_day = [1, 4, 12]
    for k, v in times.iteritems():
        r = [k[0], k[1]]
        for bin_per_day in bins_per_day:
            ac = np.array(autocorr_with_lags(v, start, bin_per_day))
            r.append(np.argmax(ac)/float(bin_per_day))
            if sum(ac) > 0:
                r.append(sum(ac*np.linspace(0, 1, len(ac))))
                r.append(sum(ac/sum(ac)*np.linspace(0, 1, len(ac))))
            else:
                r.append(0)
                r.append(np.inf)

        res.append(r)
    return np.array(res)


def assess_fourier(run_path):
    from collections import Counter
    times = read_timesdic(run_path + 'times_dic_sample.txt')
    times = {k: v for k, v in times.iteritems() if len(v) > 1}
    res = []
    start = 1167609600
    bins_per_day = [1, 4, 6, 12]

    for k, v in times.iteritems():
        r = [k[0], k[1]]
        for bin_per_day in bins_per_day:
            n_bin = 31*bin_per_day
            bin_x = bin_ts_idx(v, start, 60*60*24/bin_per_day)
            z = np.zeros(n_bin)
            for j, l in Counter(bin_x).iteritems():
                z[j] = l
            a = np.fft.fft(z)
            r.append(sum(np.abs(np.real(a[1:]))))
            r.append(np.real(a[0]))

        res.append(r)
    return np.array(res)


def weekday_from_bins(x, start_day_weekday, bin_per_day, extra=None):
    n_bins = bin_per_day*7
    bins = [0]*n_bins
    if extra is None:
        for t in x:
            bins[t % n_bins] += 1
    else:
        for t, e in zip(x, extra):
            bins[t % n_bins] += e
    return bins


def autocorr_with_lags(x, start_date, bin_per_day, days=31):
    bin_len = 60*60*24/bin_per_day
    n_bins = bin_per_day*days
    x_bins = set(bin_ts_idx(x, start_date, bin_len))
    s = [_autocorr(x_bins, l, n_bins) for l in range(1, 2*n_bins/3)]
    return s

def autocorr_full(times, start_date, bin_per_day, days):
    r = []
    for k, v in times.iteritems():
        r.append([k[0], k[1]] + autocorr_with_lags(v, start_date, bin_per_day, days))
    return r


def _autocorr(x_bins, lag, n_bins):
    s = 0
    for i in x_bins:
        if i+lag in x_bins: s += 1
    return s/float(n_bins - lag)


def jensen_shannon_divergence(x, y):
    h_1 = entropy(.5*x + .5*y)
    h_2 = .5*(entropy(x) + entropy(y))
    return h_1 - h_2


def uniform_time_statistics(times, start, end, weights=None):
    """
    Obtains statistics to see how uniform the calls/sms
    """
    t_d = float(end - start)
    times = np.array([(t-start)/t_d for t in times])
    mu = np.average(times, weights=weights)
    s0 = np.average((times - .5)**2, weights=weights)**.5
    s1 = np.average((times - mu)**2, weights=weights)**.5
    try:
        t = abs(mu-.5)*len(times)**.5/s1
    except:
        t = np.inf

    return [mu, s1, s0, np.log(t)]


def get_neighbors(ov_path, deg_path, output_path):
    df = pd.read_table(ov_path, sep=' ', names=['0', '1', 'ovrl'])
    degs = pd.read_table(deg_path, sep=' ', names=['n', 'deg'])
    df = pd.merge(df, degs, left_on='0', right_on='n', how='left', copy=False)
    df = pd.merge(df, degs, left_on='1', right_on='n', how='left', suffixes=('_0', '_1'), copy=False)
    del df['n_0']; del df['n_1']
    df.loc[:, 'n_ij'] = df[['ovrl', 'deg_0', 'deg_1']].apply(lambda x: round(x[0]*(x[1] + x[2]-2)/(x[0]+1)), axis=1)
    df.to_csv(output_path, sep=' ', index=False)


def reciprocity(logs_path=logs_path, output_path_1=None, output_path_2=None, id_cols=(1, 3)):
    """
    Reciprocity: ratio of how it is for one node to call the other (0 implies equal number of calls, 1 implies all calls are placed by one node)
    """
    dic = {}
    with open(logs_path, 'r') as r:
        row = r.readline()
        while row:
            rs = row.split(' ')
            v0, v1 = int(rs[id_cols[0]]), int(rs[id_cols[1]])
            key = (min(v0, v1), max(v0, v1))
            try:
                dic[key] = _recip(key, v0, dic[key])
            except:
                dic[key] = _recip(key, v0, [0, 0])
            row = r.readline()

    dic_2 = {key: max(val)/float(sum(val)) for key, val in dic.iteritems()}
    dic = {key: np.sqrt((val[0]*val[1])/float(sum(val)**2)) for key, val in dic.iteritems()}
    if output_path_1:
        utils.write_dic(dic, output_path_1)
        utils.write_dic(dic_2, output_path_2)
    else:
        return dic


def _recip(key, v0, val):
    if key[0] == v0:
        val[0] += 1
    else:
        val[1] += 1
    return val

def read_edgelist(path):

    net = netio.pynet.DictSymmNet()
    with open(path, 'r') as r:
        row = r.readline()
        while row:

            rs = row.split(' ')
            net[int(rs[0]), int(rs[1])] = np.float32(rs[2])
            row = r.readline()

    return net


def number_of_bursty_trains(x, delta):
    if len(x) > 1:
        t = 0.0
        e = 1
        for t0, t1 in zip(x[:-1], x[1:]):
            t += t1 - t0
            if t > delta:
                t = 0.0
                e += 1
        return e
    else:
        return 1


def read_timesdic(path):
    dic = {}
    with open(path, 'r') as r:
        row = r.readline()
        while row:
            rs = row.split(' ')
            rs = [int(s) for s in rs]
            dic[(rs[0], rs[1])] = rs[2:]
            row = r.readline()

    return dic

def read_edgedic(path):
    dic = {}
    with open(path, 'rb') as r:
        row = r.readline()
        while row:
            rs = row.split(' ')
            r_0 = min(int(rs[0]), int(rs[1]))
            r_1 = max(int(rs[0]), int(rs[1]))
            dic[(r_0, r_1)] = [np.float64(rs[2])]
            row = r.readline()
    return dic


def bursty_trains_to_file(path, output_path, delta=3600):
    f = open(output_path, "a")
    with open(path, 'r') as r:
        row = r.readline()
        while row:
            n1, n2, times = utils.parse_time_line(row)
            if len(times) > 1:
                w = number_of_bursty_trains(times, delta)
                s = n1 + " " + n2 + " " + str(w)  + "\n"
                f.write(s)
            row = r.readline()
    f.close()


def mean_inter_event_km(t_seq, mode='km+', max_date=1177970399., moment=1):
    """
    Other mode: naive
    """
    r = events.IntereventTimeEstimator(max_date, 'censorall')
    r.add_time_seq(t_seq)
    if mode == 'km+':
        mode = 'km'
        m = max(r.observed_iets)
        r.forward_censored_iets[m] = 1
        r.backward_censored_iets[m] = 1
    return r.estimate_moment(moment, mode)


def normalize_dates(x):
    """
    Given a timestamp date, normalize it so that it falls between 0 and 1:
    Formula: (x-start)/(end-start)
    """
    return (x - 1167606000)/10364399.0

def km_residual_intervals(x, method='km', max_date=1177970399.):

    estimator = events.IntereventTimeEstimator(max_date, mode='censorall')
    if method == 'km+':
        method = 'km'
    if x[-1] < max_date:
            x.append(max_date)
    estimator.add_time_seq(x)
    m = max(estimator.observed_iets)
    estimator.forward_censored_iets[m] = 1
    estimator.backward_censored_iets[m] = 1
    mu = estimator.estimate_moment(1, method)
    return mu

def inter_event_times(x, end, start, method='km'):

    c_norm = 60*60*24.
    estimator = events.IntereventTimeEstimator((end-start)/c_norm, mode='censorall')
    x = [(t - start)/c_norm for t in x]
    estimator.add_time_seq(x)
    mu = estimator.estimate_moment(1, method)
    try:
        sigma = np.sqrt(estimator.estimate_moment(2, method) - mu**2)
    except:
        sigma = np.inf

    try:
        burst = (sigma - mu)/(sigma + mu)
    except:
        burst = np.nan

    return [mu, sigma, burst]


def km_burstiness(x, method='km', max_date=1177970399.):
    """
    Burstiness estimator with Mikko's library

    method='km' or 'naive', if 'km+', add values to be censored at beginning and end
    """
    estimator = events.IntereventTimeEstimator(max_date, mode='censorall')
    estimator.add_time_seq(x)
    #m = max(estimator.observed_iets)
    #estimator.forward_censored_iets[m] = 1
    #estimator.backward_censored_iets[m] = 1
    mu = estimator.estimate_moment(1, method)
    sigma = np.sqrt(estimator.estimate_moment(2, method) - mu**2)
    return (sigma - mu)/(sigma + mu)



def net_residual_times(dic=None, path=times_path, output_path='', kaplan='naive'):
    """
    Dic must be an edge dictionary, where values are a list of call times
    Otherwise, write a .txt file containing the edges and values
    """
    net = netio.pynet.SymmNet()
    if dic:
        for k, v in dic.iteritems():
            n1, n2 = k
            if len(v) > 1:
                net[n1, n2] = residual_intervals(v, kaplan)
        return net
    else:
        f = open(output_path, "a")
        with open(path, 'r') as r:
            row = r.readline()
            while row:
                n1, n2, times = utils.parse_time_line(row)
                if len(times) > 1:
                    w = mean_inter_event_km(times, kaplan)/86400
                    s = n1 + " " + n2 + " " + str(w)  + "\n"
                    f.write(s)
                row = r.readline()
        f.close()

#### TODO: move, function to check differences between km and naive
def analyze_km(times_path, output_path=''):
    f = open(output_path, 'wb')
    f.write('0 1 calls mu_na_nomin va_na_nomin bu_na_nomin mu_na_min va_na_min bu_na_min mu_km_nomin va_km_nomin bu_km_nomin mu_km_min va_km_min bu_km_min\n')
    with open(times_path, 'r') as r:
        row = r.readline()
        while row:
            n1, n2, times = utils.parse_time_line(row)
            if len(times) > 1:
                s = '{} {} {} '.format(n1, n2, len(times))
                m, v, b = _km_versions(times, 'naive', None)
                s += '{} {} {} '.format(m, v, b)
                try:
                    m, v, b = _km_versions(times, 'naive')
                except:
                    import pdb; pdb.set_trace()
                s += '{} {} {} '.format(m, v, b)
                m, v, b = _km_versions(times, 'km', None)
                s += '{} {} {} '.format(m, v, b)
                m, v, b = _km_versions(times, 'km')
                s += '{} {} {}\n'.format(m, v, b)
                f.write(s)
            row = r.readline()
    f.close()

def _km_versions(x, method, min_date=1167598799, max_date=1177970399.0):
    estimator = events.IntereventTimeEstimator(max_date, mode='censorall')
    if min_date is not None:
        x.insert(0, min_date)
    estimator.add_time_seq(x)
    mu = estimator.estimate_moment(1, method)
    var = estimator.estimate_moment(2, method)
    sigma = np.sqrt(var - mu**2)
    burst = (sigma - mu)/(sigma + mu)
    return round(mu, 2), round(var, 2), round(burst, 4)


def net_burstiness(path=times_path, output_path='', kaplan='naive'):
    """
    Create net with burstiness, kaplan is either 'km', 'km+', 'naive'
    """
    f = open(output_path, "a")
    with open(path, 'r') as r:
        row = r.readline()
        while row:
            n1, n2, times = utils.parse_time_line(row)
            if len(times) > 1:
                w = km_burstiness(times, kaplan)
                s = n1 + " " + n2 + " " + str(w) + "\n"
                f.write(s)
            row = r.readline()
    f.close()

def net_calltimes_mode(path=times_path, output_path=''):
    """
    Write edgelist of net with mode of each call time
    """
    f = open(output_path, "a")
    with open(path, 'r') as r:
        row = r.readline()
        while row:
            n1, n2, times = utils.parse_time_line(row)
            if len(times) > 1:
                w = main_call_time(times)
                s = n1 + " " + n2 + " " + str(w) + "\n"
                f.write(s)
            row = r.readline()
    f.close()


def net_tail(path=times_path, output_path=''):
    """
    Create net with tail probabilitys
    """
    f = open(output_path, "a")
    with open(path, 'r') as r:
        row = r.readline()
        while row:
            n1, n2, times = utils.parse_time_line(row)
            if len(times) > 1:
                w = tail_weight(times)
                s = n1 + " " + n2 + " " + str(w) + "\n"
                f.write(s)
            row = r.readline()
    f.close()


def alt_net_burstiness(samp, edges, dic):
    burst = []
    for edge in samp:
        burst.append(alt_burst(dic[edges[edge]]))

    return burst


def net_worktime(dic, worktime=True):
    net = netio.pynet.SymmNet()
    for k, v in dic.iteritems():
        n1, n2 = k
        net[n1, n2] = worktime_counts(v, worktime)
    return net


def obs_residual_times(times_list):
    times_list = [dt.datetime.fromtimestamp(t) for t in times_list]
    times_list.insert(0, dt.datetime(2007, 1, 1, 1, 0, 0))
    times_list.append(dt.datetime(2007, 5, 1, 1, 0, 0))
    time_diff = np.diff(times_list)
    time_diff = np.array([t.total_seconds() for t in time_diff])/(24*60*60)
    return np.mean(time_diff), np.var(time_diff)


def worktime_counts(times_list, worktime=True):

    times_list = [dt.datetime.fromtimestamp(t) for t in times_list]
    if worktime:
        r = sum([isworktime(date) for date in times_list])
    else:
        r = len(times_list) - sum([isworktime(date) for date in times_list])
    return r


def isworktime(date):
    if date.weekday() < 6: # check weekdays (6-7 are weekend)
        hour = date.hour
        if (hour >= 8 and hour <= 14) or (hour >= 16 and hour <= 19):
            return 1.0
        else:
            return 0.0
    else:
        return 0.0

def sort_times(dic):
    for v in dic.itervalues():
        v.sort()
    return dic


def reformat_intervals(x, start=None, end=None, observed_largest=True):
    """
    From a list of timestamps, return two sets of (durations, censorship),
    where the first set is left-censored and the other is right-censored

    """

    times_list = [dt.datetime.fromtimestamp(t) for t in x]
    if start:
        times_list.insert(0, start)
    else:
        times_list.insert(0, dt.datetime(2007, 1, 1, 1, 0, 0))
    if not end:
        end = dt.datetime(2007, 5, 1, 1, 0, 0)
    times_list.append(None)
    T1, E1 = lls.utils.datetimes_to_durations(times_list[:-2], times_list[1:-1], freq='s')
    E1[0] = False
    T2, E2 = lls.utils.datetimes_to_durations(times_list[1:-1], times_list[2:], freq = 's', fill_date=end)
    # 86400 = 24*60*60 convert to day from seconds
    if not observed_largest:
        E1[np.argmax(T1)] = False
        E2[np.argmax(T2)] = False

    return T1/(86400), E1, T2/(86400), E2

def simple_interval_reformat(x, end=None):
    times_list = [dt.datetime.fromtimestamp(t) for t in x]
    if not end:
        end = dt.datetime(2007, 5, 1, 1, 0, 0)
    times_list.append(None)
    T, E = lls.utils.datetimes_to_durations(times_list[0:-1], times_list[1:], freq = 's', fill_date = end)

    E[np.argmax(T)] = False

    return T/(86400), E

def tail_dependence(x):
    T, E = simple_interval_reformat(x)

    km = lls.KaplanMeierFitter()
    km.fit(T,E)
    t = km.timeline
    s = list(km.survival_function_['KM_estimate'])
    mu = T[E].mean()
    tail = t[-1] * s[-1]
    return mu/tail

def kaplan_meier_double(x):

    t1, e1, t2, e2 = reformat_intervals(x)

    km = lls.KaplanMeierFitter()
    km.fit(t1, e1, left_censorship=True)
    time_1 = km.timeline
    surv_1 = map(lambda x: 1. - x, km.cumulative_density_['KM_estimate'])

    km = lls.KaplanMeierFitter()
    km.fit(t2, e2)
    time_2 = km.timeline
    surv_2 = list(km.survival_function_['KM_estimate'])

    return time_1, surv_1, time_2, surv_2

def residual_intervals(x, kaplan=True):
    """
    Compute the mean resideual waiting time
    """
    if kaplan:
        time_1, surv_1, time_2, surv_2 = kaplan_meier_double(x)
        mu = mean_residual(time_1, surv_1, time_2, surv_2)
    else:
        tl = [dt.datetime.fromtimestamp(t) for t in x]
        #tl.insert(0, dt.datetime(2007, 1, 1, 1, 0, 0))
        #tl.append(dt.datetime(2007, 5, 1, 1, 0, 0))
        days = [(tl[i] - tl[i-1]).total_seconds()/86400.0 for i in range(1, len(tl))]
        mu = np.mean(days)

    return mu

def main_call_time(x):
    """
    Obtain the most likely hour when the two people are calling each other
    """
    x = utils.timestamps_to_dates(x)
    hours = [t.hour for t in x]
    m_h = mode(hours)
    return m_h.mode[0]


def burstiness(x, kaplan=True):

    if kaplan:
        time_1, surv_1, time_2, surv_2 = kaplan_meier_double(x)
        mu = mean_residual(time_1, surv_1, time_2, surv_2)
        mu_2 = var_residual(time_1, surv_1, time_2, surv_2, mu)
    else:
        tl = [dt.datetime.fromtimestamp(t) for t in x]
        days = [(tl[i] - tl[i-1]).total_seconds()/86400.0 for i in range(1, len(tl))]
        mu = np.mean(days)
        mu_2 = np.var(days)
    sigma = np.sqrt(mu_2)

    return (sigma - mu)/(sigma + mu)

def alt_burst(x):

    time_1, surv_1, time_2, surv_2 = kaplan_meier_double(x)
    mu = nth_moment(time_1, surv_1, time_2, surv_2, 1)
    mu_2 = nth_moment(time_1, surv_1, time_2, surv_2, 2)
    sigma = np.sqrt(mu_2 - mu**2)

    return (sigma - mu)/(sigma + mu)

def mean_residual(time_1, surv_1, time_2, surv_2):

    a = sinteg.trapz(surv_1, time_1) + time_1[-1]*surv_1[-1]
    b = sinteg.trapz(surv_2, time_2) + time_2[-1]*surv_2[-1]

    return np.mean([a, b])

def var_residual(time_1, surv_1, time_2, surv_2, mu):

    surv_1 *= np.abs(mu - time_1)
    surv_2 *= np.abs(mu - time_2)

    a = 2*sinteg.trapz(surv_1, time_1) + time_1[-1]**2*surv_1[-1]
    b = 2*sinteg.trapz(surv_2, time_2) + time_2[-1]**2*surv_2[-1]

    return np.max([a, b])


def nth_moment(time_1, surv_1, time_2, surv_2, n = 1):

    if n > 1:
        surv_1 *= time_1 ** (n - 1)
        surv_2 *= time_2 ** (n - 1)

    a = n*sinteg.trapz(surv_1, time_1) + time_1[-1]**n * surv_1[-1]
    b = n*sinteg.trapz(surv_2, time_2) + time_2[-1]**n * surv_2[-1]

    return np.max([a, b])


if __name__ == '__main__':
    import analysis_tools as at
    import plots
    from scipy.stats import rankdata

    times_dic = dict_elements()
    net_residual_times(output_path='run/galicia/mean_residual_times.edg')
    print('Residual Times net created')
    net = total_calls()
    overlap = at.get_weight_overlap(net)
    calls = at.weights_to_dic(net)
    del net
    net_res = read_edgelist('run/galicia/mean_residual_times.edg')
    list_ov, list_rt = at.overlap_list(overlap, net_res)
    list_st, _ = at.overlap_list(calls, net_res)
    ind = (np.array(list_rt) > 0) & (np.array(list_st) < 101)
    list_rt = np.array(list_rt)[ind]
    list_ov = np.array(list_ov)[ind]
    list_st = np.array(list_st)[ind]
    list_rt_rank = max(list_st)*rankdata(list_rt)/len(list_rt)
    fig, ax = plots.loglogheatmap(list_st, list_rt, list_ov, 1.65, 1.6)
    fig.savefig('run/galicia/heatmap.png')

    fig, ax = plots.loglogheatmap(list_st, list_rt_rank, list_ov, 1.6, 1.6)
    fig.savefig('run/galicia/heatmap_rank.png')

