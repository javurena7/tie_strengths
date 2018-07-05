import numpy as np
from netpython import *
import datetime as dt
#import lifelines as lls
import scipy.integrate as sinteg
from iet import events
from verkko.binner.binhelp import *
from scipy.stats import mode
import utils

logs_path = '../data/mobile_network/madrid/madrid_call_newid_log.txt'
times_path = 'run/madrid_2504/times_dic.txt'

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
    cmd = "awk '{($4 > $2) ? p = $2 {sp} $4 : p = $4 {sp} $2; print p}' {logs_path} | sort | uniq -c > output.txt".format(sp = '" "', logs_path = logs_path)


def total_calls_times(times_path, output_path):
    """
    Create net with total number of calls using times of calls
    """
    net = netio.pynet.SymmNet()
    w = open(output_path, 'wb')
    with open(times_path, 'r') as r:
        row = r.readline()
        while row:
            row = r.readline()
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

def awk_times(logs_path, output_path):
    cmd = "SORT caller ids" #'{($4 > $2) ? p = $2 " " $4 " " $1: p = $4 " " $2 " " $1; print p}'
    cmd = "awk '{if(a[$2" "$4])a[$2" "$4]=a[$2" "$4]" "$1; else a[$2" "$4]=$1;}END{for (i in a) print i, a[i];}' {} > {}".format(logs_path, output_path)

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


def reciprocity(logs_path=logs_path, output_path=None, id_cols=(1, 3)):
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

    dic = {key: np.abs(float(val[0]))/val[1] for key, val in dic.iteritems()}
    if output_path:
        utils.write_dic(dic, output_path)
    else:
        return dic


def _recip(key, v0, val):
    if key[0] == v0:
        val[0] += 1
    else:
        val[0] -= 1
    val[1] += 1

    return val

def read_edgelist(path):

    net = netio.pynet.SymmNet()
    with open(path, 'r') as r:
        row = r.readline()
        while row:
            rs = row.split(' ')
            net[int(rs[0]), int(rs[1])] = rs[2]
            row = r.readline()

    return net


def number_of_bursty_trains(x, delta):
    if len(x) > 1:
        t = 0.0
        e = 0.0
        for t0, t1 in zip(x[:-1], x[1:]):
            t += t1 - t0
            if t > delta:
                t = 0.0
                e += 1
        return e
    return 1.0


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

def km_burstiness(x, method='km', max_date=1177970399.):
    """
    Burstiness estimator with Mikko's library

    method='km' or 'naive', if 'km+', add values to be censored at beginning and end
    """
    estimator = events.IntereventTimeEstimator(max_date, mode='censorall')
    estimator.add_time_seq(x)
    m = max(estimator.observed_iets)
    estimator.forward_censored_iets[m] = 1
    estimator.backward_censored_iets[m] = 1
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

