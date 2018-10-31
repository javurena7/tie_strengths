# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from verkko.binner import bins as binner
from scipy.stats import binned_statistic, binned_statistic_2d, rankdata
from scipy.optimize import leastsq

def plot_powerlaw(values, main='', xlabel='', ylabel='', fig=None, ax=None, label=''):
    """
    Function for plotting power-log distribution. If many plots are to be added, iterate with fig and ax.
    (values): list or 1d array of values to plot distribution
    (main): plot title
    (xlabel): value
    (ylabel):
    (fig): pyplot figure if plotting various graphs
    (ax): pyplot subplot if plotting various graphs
    (label): name of the values to display
    """
    if not fig:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(main)
    n_val = len(values)
    if not isinstance(values[0], int):
        values = [int(v) for v in values]
    p = np.bincount(values)/float(n_val)
    x = np.linspace(0, max(values), len(p))
    ax.loglog(x, p, '.', label=label)
    ax.legend(loc=0)
    return fig, ax

def powerlaw(values, main='', xlabel='', ylabel='', label='', fig=None, ax=None, factor=1.2, fit=True):

    """
Plot powerlaw distribution by binning values into log-bins defined by factor. This version also fits a powerlaw distribution of the form:
    P(x) = a*exp(-x/xc)(x+x0)**(-alpha)

Parameters:
    (values): list or 1D array of values to obtain a density from
    (main): matplotlib title
    (xlabel): x-axis label
    (ylabel): y-axis label
    (label): label to appear on the graph
    (fig): matplotlib figure
    (ax): matplotlib axis
    (factor): factor > 1 to log-bin values using verkko.

Returns:
    (fig, ax): matplotlib figure and axis
    (p1): vector with fit coefficientes: [log(a), alpha, x0, xc]

    """
    if not fig:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(main)

    n_val = len(values)
    start = np.min(values)
    end = np.max(values)
    bins = binner.Bins(float, int(min(values)) + 1, max(values) + 1, 'log', factor)

    ind = bins.widths > 1
    counts, _,  _ = binned_statistic(values, values, statistic='count', bins=bins.bin_limits)
    counts = counts[ind]/float(n_val)
    bins_c = bins.centers[ind]
    ax.loglog(bins_c, counts, '.',  label=label)

    try:
        if fit:
            p1, succ = powerlaw_coeff(bins_c, counts)
            y_fit = 10**(powerlaw_func(p1, bins_c))
            ax.loglog(bins_c, y_fit, label= 'Fit: ' + r'$\gamma = $' + str(round(p1[1], 2)), color='blue')
        else:
            p1 = None
    except:
        p1 = None

    ax.legend(loc=0)
    return fig, ax, p1


powerlaw_func = lambda p, x: p[0] + -p[1]*np.log(x + p[2]) -x/p[3]
errfunc = lambda p, x, y: (powerlaw_func(p, x) - y)**2

def powerlaw_coeff(x, y):

    p0 = [0.9, 1.6, .5, 10000]
    y_log = np.log10(y)
    p1, success = leastsq(errfunc, p0, args=(x, y_log))
    return p1, success

def logmeans(bins):
    return [np.mean([bins[i], bins[i+1]]) for i in range(len(bins)-1)]


def log_bin_plot(x, y, factor=1.82, col=None, fig=None, ax=None, xlabel=r'$\bar{\tau}$' + ' (days)', ylabel=r'$\langle O | \bar{\tau} \rangle$', title='Overlap as a Function of Mean Inter-Event time', label=None):
    """
    Used for mean waiting time and weight vs Overlap
    """
    bins = binner.Bins(float, min(x)+1, max(x), 'log', factor)
    bin_means, _, _ = binned_statistic(x, y, bins=bins.bin_limits)
    if not fig:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ax.semilogx(bins.centers, bin_means, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if col:
        cols, _, _ = binned_statistic(x, col, bins=bins.bin_limits)
        ax.scatter(bins.centers, bin_means, c=cols)

    return fig, ax


def lin_bin_plot(x, y, x_bins=20, xlabel=r'$B$', ylabel=r'$\langle O | B \rangle$', title='Overlap as a function of Burstiness', fig=None, ax=None, label=None, arg='-'):
    """
    Used for burstiness VS overlap
    """
    bins = binner.Bins(float, int(np.floor(min(x))), max(x), 'lin', x_bins)
    bin_means, _, _ = binned_statistic(x, y, bins=bins.bin_limits)
    if not fig:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ax.plot(bins.centers, bin_means, arg, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return fig, ax

def plot_overlap_weight_difference(weight_1, overlap_1, weight_2, overlap_2):
    fig, ax = plots.log_bin_plot(weight_1, overlap_2, xlabel=r'$W$', ylabel=r'$\langle O | W \rangle$', title="Overlap as a fucnction of Number \n of calls", label='Company')
    fig, ax = plots.log_bin_plot(weight_2, overlap_2, fig=fig, ax=ax, label='All')


def plot_overlap_difference(simple, extended, bins=35):
    """
    used for plotting overlap calculated with the company set (set 2) vs company plus other users (set 5)
    """
    fig, ax = lin_bin_plot(simple, extended, bins, ylabel=r'$\langle O_{all} | O_{company} \rangle$', title='Overlap for all users VS overlap \n of company users', xlabel=r'$ O_{company}$')
    ax.plot(simple, extended, '.c', alpha=0.03)
    return fig, ax



def loglogheatmap(x, y, z, factor_x=1.5, factor_y=1.45, stat='mean', xlabel=r'$w$ (calls)', ylabel=r'$\bar{\tau}$ (days)', title='Overlap as a function of and Number of Calls and Inter-event Time\n' + r'$\langle O | w, \bar{\tau} \rangle$'):
    x = x+1
    y = y+1
    bins_x = binner.Bins(float, min(x), max(x), 'log', factor_x)
    bins_y = binner.Bins(float, min(y), max(y), 'log', factor_y)
    bin_means, _, _, _ = binned_statistic_2d(x, y, z, statistic=stat, bins=[bins_x.bin_limits, bins_y.bin_limits])
    print(bin_means.shape)
    bin_means = np.nan_to_num(bin_means.T)
    extent = [bins_x.bin_limits[0], bins_x.bin_limits[-1], bins_y.bin_limits[0], bins_y.bin_limits[-1]]
    fig, ax = plt.subplots(1)
    cax = ax.imshow(bin_means, origin="lower") #extent=extent
    #x_ticks = np.linspace(bins_x.bin_limits[0], bins_x.bin_limits[-1],
    #        len(bins_x.bin_limits))
    #ax.set_xticks(x_ticks)
    ax.set_xticklabels(np.int16(bins_x.bin_limits))
    #y_ticks = np.linspace(bins_y.bin_limits[0], bins_y.bin_limits[-1],
    #        len(bins_y.bin_limits))
    #ax.set_yticks(y_ticks)
    ax.set_yticklabels(np.round(bins_y.bin_limits, 1))
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.colorbar(cax)

    return fig, ax

def loglinheatmap(x, y, z, factor_x=1.5, n_bins_y=20, stat='mean', xlabel=r'$w$ (calls)', ylabel=r'$B$', title='Overlap as a function of Number of Calls and Burstiness\n' + r'$\langle O | w, B \rangle$', exp_f=1):
    """
    exp_f contains an "expansion factor". The linear part is ploted [0-1], but we use this expansion factor to depict different things (for instance, =25 for 24 hours)
    """

    y = np.array(y)
    y_alt = max(x)*y/exp_f
    bins_x = binner.Bins(float, min(x), max(x), 'log', factor_x)
    bins_y = binner.Bins(float, min(y_alt), max(y_alt), 'lin', n_bins_y)
    bin_means, _, _, _ = binned_statistic_2d(x, y_alt, z, statistic=stat, bins=[bins_x.bin_limits, bins_y.bin_limits])
    bin_means = np.nan_to_num(bin_means.T)
    extent = [bins_x.bin_limits[0], bins_x.bin_limits[-1], bins_y.bin_limits[0], bins_y.bin_limits[-1]]
    fig, ax = plt.subplots(1)
    cax = ax.imshow(bin_means, extent=extent, origin="lower")
    x_ticks = np.linspace(bins_x.bin_limits[0], bins_x.bin_limits[-1],
            len(bins_x.bin_limits))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(np.int16(bins_x.bin_limits))
    y_ticks = np.linspace(bins_y.bin_limits[0], bins_y.bin_limits[-1],
            len(bins_y.bin_limits))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([round(yb, 2) for yb in bins_y.bin_limits])
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.colorbar(cax)

    return fig, ax

def linlinheatmap(x, y, z, n_bins_x=30, n_bins_y=30, stat='mean', xlabel=r'$w$ (calls)', ylabel=r'$B$', title='Overlap as a function of Number of Calls and Burstiness\n' + r'$\langle O | w, B \rangle$', exp_f=1):
    """
    exp_f contains an "expansion factor". The linear part is ploted [0-1], but we use this expansion factor to depict different things (for instance, =25 for 24 hours)
    """

    bins_x = binner.Bins(float, min(x), max(x), 'lin', n_bins_x)
    bins_y = binner.Bins(float, min(y), max(y), 'lin', n_bins_y)
    bin_means, _, _, _ = binned_statistic_2d(x, y, z, statistic=stat, bins=[bins_x.bin_limits, bins_y.bin_limits])
    bin_means = np.nan_to_num(bin_means.T)
    extent = [bins_x.bin_limits[0], bins_x.bin_limits[-1], bins_y.bin_limits[0], bins_y.bin_limits[-1]]
    fig, ax = plt.subplots(1)
    cax = ax.imshow(bin_means, extent=extent, origin="lower")
    x_ticks = np.linspace(bins_x.bin_limits[0], bins_x.bin_limits[-1],
            len(bins_x.bin_limits))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([round(xb, 2) for xb in bins_x.bin_limits])
    y_ticks = np.linspace(bins_y.bin_limits[0], bins_y.bin_limits[-1],
            len(bins_y.bin_limits))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([round(yb, 2) for yb in bins_y.bin_limits])
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.colorbar(cax)
    return fig, ax


def cumulative_distribution(x, y, label='', size=100, fig=None, ax=None, xlabel='', ylabel='', title='', arg='.'):
    if not fig:
        fig, ax = plt.subplots(1)
    p_cum, y_mean, sum_x = [], [], 0
    n = float(len(x))
    x = rankdata(x)/n
    x_vals = np.linspace(0, 1 + 1/size, size)
    for v0, v1 in zip(x_vals[:-1], x_vals[1:]):
        ind_x = x < v1
        ind_y = (x >= v0) & ind_x
        y_h = y[ind_y]
        sum_x += len(y_h)
        p_cum.append(sum_x/n)
        y_mean.append(np.mean(y_h))
    ax.plot(p_cum, y_mean, arg, label=label)
    return fig, ax
