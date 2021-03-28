#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import plotly.graph_objects as go
import os

#Get lower and upper limits for plots
def getlowerupper(pred, true):
    lower = min(min(pred), min(true))
    upper = max(max(pred), max(true))
    
    return lower, upper

#Histogram with ratio subplot
def histo(modelname, true, pred, varname, lower, upper, bins):
    bins = np.histogram(true, bins = bins, range = [lower, upper])[1]
    bincountsx, binedgesx, binnumx = stats.binned_statistic(pred, pred, statistic = 'count', bins = bins, range = [lower, upper])
    bincountsy, binedgesy, binnumy = stats.binned_statistic(true, true, statistic = 'count', bins = bins, range = [lower, upper])
    fig,axs = plt.subplots(2, gridspec_kw = {'height_ratios':[3, 1]})
    fig.suptitle(varname)
    axs[0].hist(pred, bins = bins, range = [lower, upper], density = 1, histtype = 'step', label = "Model Output")
    axs[0].hist(true, bins = bins, range = [lower, upper], density = 1, histtype = 'step', label = "Analytical Solution")
    axs[0].scatter([], [], marker = 'x', color = 'red', label = 'Missing Analytical')
    axs[0].scatter([], [], marker = 'x', color = 'blue', label = 'Missing Model')
    axs[0].set_ylabel("Density=1")
    axs[0].legend()
    axs[0].set_xlim([lower, upper])
    binratio = bincountsx / bincountsy
    bincenters = (bins[1:] + bins[:-1]) / 2
    bincentersfinal = bincenters[binratio!=0]
    biniszero = binratio == 0
    binratio = binratio[binratio != 0]
    binisnan = np.isnan(binratio)
    binnonan =~ binisnan
    axs[1].scatter(bincentersfinal[binnonan], binratio[binnonan])
    axs[1].scatter(bincentersfinal[binisnan], np.ones(len(bincentersfinal[binisnan])), marker = 'x', color = 'red')
    axs[1].scatter(bincenters[biniszero], np.ones(len(bincenters[biniszero])), marker = 'x', color = 'blue')
    axs[1].axhline(y = 1)
    axs[1].set_xlim([lower, upper])
    axs[1].set_ylim([0.5, 1.5])
    _, pval = stats.ks_2samp(true, pred)
    axs[1].set_xlabel("KS-Test P-value: %.2e" % pval)
    fig.savefig("./results/Plots/%s/%s_histo" % (modelname, varname), bbox_inches = 'tight')
    plt.close()

#Cut items from an array outside of a given value range
def cuts(datain, datacheck, minval, maxval):
    dataout = datain[np.logical_and(datacheck > minval, datacheck < maxval)]
    
    return(dataout)

#Calculate the standard error of the mean
def sem(values):
    sem = stats.sem(values)
    
    return(sem)

#Plot gaussians from an array of residuals (y_true - y_pred)
def plotgaussian(modelname, true, pred, varname, lower, upper, bins):
    resmeans = []
    lowerbounds = []
    upperbounds = []
    lowerranges = []
    upperranges = []
    samples = []
    stddevs = []
    
    width = (upper - lower) / bins
    resids = true - pred
    
    for i in range(bins):
        lowertemp = lower + width * i
        uppertemp = lower + width * (i + 1)
        residscut = cuts(resids, true, lowertemp, uppertemp)
        varnametemp = "%s Residuals Distribution (%.2f to %.2f)" % (varname, lowertemp, uppertemp)
        residscut.sort()
        resmeantemp = np.mean(residscut)
        resstdtemp = np.std(residscut)
        samplestemp = len(residscut)
        stddevtemp = np.std(residscut)
        h = stddevtemp * 1.96
        lowerboundtemp = resmeantemp - h
        upperboundtemp = resmeantemp + h
        pdf = stats.norm.pdf(residscut, resmeantemp, resstdtemp)
        
        plt.hist(residscut, bins = 30, histtype = 'step', color = 'blue', density = 1, label = 'Residuals')
        plt.plot(residscut, pdf, label = "Normal Curve", color = 'black')
        plt.title(varnametemp)
        plt.axvline(resmeantemp, label = "Mean: %.2f" % resmeantemp, color = 'red')
        plt.xlabel("95%% Conf. Int.: [%.2f, %.2f] (samples: %.d)" % (lowerboundtemp, upperboundtemp, samplestemp))
        plt.axvspan(lowerboundtemp, upperboundtemp, facecolor = 'g', alpha = .3, label = '95% Conf. Int.')
        plt.legend()
        plt.savefig("./results/Plots/%s/%s Resids(%.2f to %.2f).png" % 
                    (modelname, varnametemp, lowertemp, uppertemp) , bbox_inches = 'tight')
        plt.close()
        
        resmeans.append(resmeantemp)
        lowerbounds.append(lowerboundtemp)
        upperbounds.append(upperboundtemp)
        samples.append(samplestemp)
        lowerranges.append(lowertemp)
        upperranges.append(uppertemp)
        stddevs.append(stddevtemp)
    
    resmean = np.mean(resids)
    resmeans.append(resmean)
    stddev = np.std(resids)
    h = stddev * 1.96
    stddevs.append(stddev)
    lowerbounds.append(resmean - h)
    upperbounds.append(resmean + h)
    samples.append(len(resids))
    lowerranges.append(lower)
    upperranges.append(upper)
    
    return stddevs, resmeans, lowerranges, upperranges, lowerbounds, upperbounds, samples

#Root mean squared error plot for a single set of data
def rmsplot(modelname, true, pred, varname, lower, upper, bins):
    width = (upper - lower) / bins
    dif = true - pred
#     xcenters = np.arange(lower + .5 * width, upper + .5 * width, width)
    xcenters = np.linspace(lower + .5 * width, upper + .5 * width, bins, endpoint = False)
    binstdy, binedgesy, binnumy = stats.binned_statistic(true, dif, statistic = 'std', bins = bins, range = [lower, upper])
    binsemy, _, _ = stats.binned_statistic(true, dif, statistic = sem, bins = bins, range = [lower, upper])
    fig, axs = plt.subplots(1)
    axs.scatter(xcenters, binstdy)
    axs.set_xlim(lower, upper)
    axs.errorbar(xcenters, binstdy, yerr = binsemy, xerr = width * .5, ls = 'none')
    axs.set_title("RMSE vs %s" % varname)
    axs.set_ylabel("RMS (true - pred)")
    axs.set_xlabel("%s True" % varname)
    fig.savefig("./results/Plots/%s/%s_RMSE Plot" % (modelname, varname), bbox_inches = 'tight')
    plt.close()
    
#Heatmap/2D Histogram of y_pred vs y_true
def heatmap(modelname, true, pred, varname, lower, upper, bins):
    heatmap, xedges, yedges = np.histogram2d(true, pred, bins = bins, range = [[lower, upper], [lower, upper]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.imshow(heatmap.T, extent = extent, origin = 'lower')
    plt.plot([lower, upper], [lower, upper], color = 'blue')
    fig = plt.gcf()
    plt.set_cmap('gist_heat_r')
    plt.xlabel("%s True" % varname)
    plt.ylabel("%s Pred" % varname)
    plt.title("Frequency Heatmap")
    plt.xlim(lower, upper)
    plt.ylim(lower, upper)
    plt.colorbar()
    fig.savefig("./results/Plots/%s/%s Heatmap" % (modelname, varname))
    plt.close()
                
#Create a table of confidence intervals
def confinttable(modelname, varname, stddevs, resmeans, lowerranges, upperranges, lowerbounds, upperbounds, samples):
    for i in range(len(resmeans)):
        center = (upperranges[i] + lowerranges[i]) / 2
        lowconf1 = center - stddevs[i] * 1.96 / 1
        uppconf1 = center + stddevs[i] * 1.96 / 1
        lowconf2 = center - stddevs[i] * 1.96 / 2
        uppconf2 = center + stddevs[i] * 1.96 / 2
        lowconf3 = center - stddevs[i] * 1.96 / 3
        uppconf3 = center + stddevs[i] * 1.96 / 3
        fig = go.Figure(data = [go.Table(header = dict(values = [varname, "Value range: %.2f - %.2f" % (lowerranges[i], upperranges[i])]),
                                         cells = dict(values = [['Number of Samples', 'Residuals Mean', 'Standard Deviation', 
                                                                 '95% Confidence Interval 1 Image', '95% Confidence Interval 2 Images',
                                                                 '95% Confidence Interval 3 Images'], [samples[i], "%.4f" % resmeans[i], 
                                                                 '%.3f' % stddevs[i], '(%.3f, %.3f)' % (lowconf1, uppconf1), 
                                                                 '(%.3f, %.3f)' % (lowconf2, uppconf2), 
                                                                 '(%.3f, %.3f)' % (lowconf3, uppconf3)]]))
                               ])
        fig.write_image("./results/Plots/%s/%s Table (%.2f - %.2f).png" % (modelname, varname, lowerranges[i], upperranges[i]))
