import netpython
from nets import *
import analysis_tools as at
import numpy as np
import plots
from scipy.stats import rankdata
import powerlaw

run_path = 'run/canarias_m_simple/'
kaplan = 'km'
dic_path='../data/mobile_network/canarias/canarias_reorder.p'
# 0) Preprocessing, where is data coming from?
# 1) Obtain basic network statistics
try:
    net = read_edgelist(run_path + 'net.edg')
except:
    net = total_calls()
    utils.write_edges(net, run_path + 'net.edg')

## create dictionary with times
times = dict_elements()
#utils.write_dic(times, run_path + 'times_dic.txt')

#del times

#at.write_sorted_edges(net, run_path + 'calls_perc_larg.edg') #percolation
#at.write_sorted_edges(net, run_path + 'calls_perc_small.edg', reverse=False) #percolation

#weights = list(net.weights)
#degrees = [net.deg(n) for n in net]
#fig, ax, p_wgh = plots.powerlaw(weights, 'Weight Distribution', r'$w$', r'$P(w)$', label='Weight (# calls)', factor=1.15)
#fig.savefig(run_path + '1.png')
#fig, ax, p_deg = plots.powerlaw(degrees, 'Degree Distribution', r'$k$', r'$P(k)$', label='Degree', factor=1.15)
#fig.savefig(run_path + '2.png')

# 1.a) Plot total network time
#net_tot = total_time()
#tot_times = list(net_tot.weights)
#at.write_sorted_edges(net_tot, run_path + 'tottime_perc_larg.edg') #percolation
#at.write_sorted_edges(net_tot, run_path + 'tottime_perc_small.edg', reverse=False) #percolation

#del net_tot

#fig, ax, p_tm = plots.powerlaw(tot_times, 'Total time call distribution', r'$t$' + ' (minutes)', r'$P(t)$', label='Total call time', fit=False)
#fig.savefig(run_path + '3.png')
overlap = at.get_weight_overlap(net)
#at.write_sorted_edges_from_dic(overlap, run_path + 'overl_perc_larg.edg') #percolation
#at.write_sorted_edges_from_dic(overlap, run_path + 'overl_perc_small.edg', reverse=False) #percolation
calls = at.weights_to_dic(net)

list_ov, list_st = at.overlap_list(overlap, net)
ind = np.array(list_st) < 110

list_ov = np.array(list_ov)[ind]
list_st = np.array(list_st)[ind]

#fig, ax = plots.log_bin_plot(list_st, list_ov, xlabel=r'$w$' + ' (# calls)', ylabel=r'$\langle O | w\rangle$', title='Average Overlap as a function of number of calls')
#fig.savefig(run_path + '4.png')

# 2) Deal with mean waiting time and burstiness
# 2.a) Create file
net_burstiness(output_path=run_path + 'burstiness.edg', kaplan=kaplan)
net_residual_times(output_path=run_path + 'mean_residual_times.edg', kaplan=kaplan)
# 2.b) Read file / check scale (why integers?)

net_res = read_edgelist(run_path + 'mean_residual_times.edg')
res_times = list(net_res.weights)

fig, ax, p_tm = plots.powerlaw(res_times, 'Mean inter-event time distribution', r'$\bar{\tau}$', r'$P(\bar{\tau})$', label='Inter-event times', fit=False)
fig.savefig(run_path + '5.png')

list_ov, list_rt = at.overlap_list(overlap, net_res)
list_st, _ = at.overlap_list(calls, net_res)
ind = (np.array(list_rt) > 0) & (np.array(list_st) < 101)
list_rt = np.array(list_rt)[ind]/(24*3600)
list_ov = np.array(list_ov)[ind]
list_st = np.array(list_st)[ind]
fig, ax = plots.lin_bin_plot(list_rt, list_ov, 10, r'$\tau$', r'$\langle O | \tau \rangle$', 'Overlap as a function of mean Inter-event time')
fig.savefig(run_path + '6.png')
fig, ax = plots.loglinheatmap(list_st, list_rt, list_ov, 1.65, 20)
fig.savefig(run_path + '7.png')
list_rt_rank = rankdata(list_rt)/len(list_rt)
fig, ax = plots.loglinheatmap(list_st, list_rt_rank, list_ov, 1.6, 20, xlabel=r'$w$', ylabel=r'$R_{\bar{\tau}}$' + ' (Rank of Inter-event times)', title='Overlap as a function of Calls and Rank of Inter-Event Times\n' + r'$\langle O | w, R_{\bar{\tau}}\rangle$')
fig.savefig(run_path + '8.png')
at.write_sorted_edges(net_res, run_path + 'waittimes_perc_larg.edg', reverse=False) #percolation
at.write_sorted_edges(net_res, run_path + 'waittimes_perc_small.edg') #percolation

del net_res
del res_times
net_brt = read_edgelist(run_path + 'burstiness.edg')
brs_time = list(net_brt.weights)

list_ov, list_br = at.overlap_list(overlap, net_brt)
list_st, _ = at.overlap_list(calls, net_brt)
ind = np.array(list_st) < 110
list_br = np.array(list_br)[ind]
list_ov = np.array(list_ov)[ind]
list_st = np.array(list_st)[ind]
fig, ax = plots.lin_bin_plot(list_br, list_ov)
fig.savefig(run_path + '9.png')
fig, ax = plots.loglinheatmap(list_st, list_br, list_ov, factor_x = 1.5, n_bins_y = 20)
fig.savefig(run_path + '10.png')

list_br_rank = rankdata(list_br)/float(len(list_br))

fig, ax = plots.loglinheatmap(list_st, list_br_rank, list_ov, factor_x = 1.5, n_bins_y = 20, ylabel='Burstiness Rank ' + r'$R_{B}$', title='Overlap as a function fo the number of calls \n and Burstiness Rank ' + r'$\langle O | w, R_B \rangle$')
fig.savefig(run_path + '10_rank.png')
at.write_sorted_edges(net_brt, run_path + 'burst_perc_larg.edg') #percolation
at.write_sorted_edges(net_brt, run_path + 'burst_perc_small.edg', reverse=False) #percolation

del net_brt

# Check time of call and overlap
net = read_edgelist(run_path + 'net.edg')
overlap = at.get_weight_overlap(net)
calls = at.weights_to_dic(net)
net_calltimes_mode(output_path=run_path + 'time_mode.edg')
net_mode = read_edgelist(run_path + 'time_mode.edg')
list_ov, list_tm = at.overlap_list(overlap, net_mode)
list_st, _ = at.overlap_list(calls, net_mode)
ind = (np.array(list_st) < 101)
list_tm = np.array(list_tm)[ind]
list_ov = np.array(list_ov)[ind]
list_st = np.array(list_st)[ind]

fig, ax = plots.loglinheatmap(list_st, list_tm, list_ov, ylabel='Call Mode (Time of day)', title='Overlap as a function of Number of Calls and\n Call Mode', n_bins_y=22, exp_f=25)
fig.savefig(run_path + '11.png')

at.write_sorted_edges_from_dic(overlap, run_path + 'overl_perc_larg.edg') #percolation
at.write_sorted_edges_from_dic(overlap, run_path + 'overl_perc_small.edg', reverse=False) #percolation

# Check age difference and overlap
age_diff = at.get_age_difference(net.edges, dic_path=dic_path)
list_ag, list_st = at.overlap_list(age_diff, net, all_net=True)
list_ov, list_st = at.overlap_list(overlap, net)
ind = (~np.isnan(np.array(list_ag))) & (np.array(list_st) < 110)
list_ag = np.array(list_ag)[ind]
list_ov = np.array(list_ov)[ind]
list_st = np.array(list_st)[ind]
list_ag_rank = rankdata(list_ag)/len(list_ag)

fig, ax = plots.lin_bin_plot(list_ag_rank, list_ov, 15, xlabel=r'$R_{AD}$', ylabel=r'$\langle O | R_{AD} \rangle$', title='Overlap as a function of Age Difference Rank')
fig.savefig(run_path + '13.png')

fig, ax = plots.loglinheatmap(list_st, list_ag_rank, list_ov, ylabel=r'$R_{AD}$', title = 'Overlap as a function of Number of Calls and \n Rank of Age Difference' + r'$\langle O | w, R_{AD} \rangle$')
fig.savefig(run_path + '14.png')

#3) Percolation
giant_cl, sus_cl = at.file_perc(run_path + 'calls_perc_larg.edg')
giant_cs, sus_cs = at.file_perc(run_path + 'calls_perc_small.edg')

giant_bl, sus_bl = at.file_perc(run_path + 'burst_perc_larg.edg')
giant_bs, sus_bs = at.file_perc(run_path + 'burst_perc_small.edg')

giant_wl, sus_wl = at.file_perc(run_path + 'waittimes_perc_larg.edg')
giant_ws, sus_ws = at.file_perc(run_path + 'waittimes_perc_small.edg')

fig, ax = at.plot_perc(giant_cl, label='Num-Calls L', line='r-')
fig, ax = at.plot_perc(giant_cs, label='', line='r--', fig=fig, ax=ax)
fig, ax = at.plot_perc(giant_bl, label='Busrtiness L', line='b-', fig=fig, ax=ax)
fig, ax = at.plot_perc(giant_bs, label='Burstiness S', line='b--', fig=fig, ax=ax)
fig, ax = at.plot_perc(giant_wl, label='Waiting-Time L', line='g-', fig=fig, ax=ax)
fig, ax = at.plot_perc(giant_ws, label='Waiting-Time S', line='g--', fig=fig, ax=ax)
fig.savefig(run_path + '12.png')
