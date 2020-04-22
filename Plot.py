import matplotlib.pyplot as plt
from matplotlib import ticker
from Util import *
import pandas
from Model import *
from Simulate import *
from more_itertools import collapse
import pdb

def gather(T, series, variances, indices):
    outputSeries = [sum(x[index] for index in indices) for x in series]
    outputVariances = [x[indices, :][:, indices].sum() for x in variances]
    outputVariances = [np.sqrt(x) for x in outputVariances]
    return np.array(outputSeries), np.array(outputVariances)

def statePlot (series, variances, state, beginDate, step, groundTruth) : 
    T = len(series)
    compartments = {k: [3*i, 3*i + 1, 3*i + 2] for i, k in enumerate(['S', 'E', 'A', 'I', 'Xs', 'Xe', 'Xa', 'Xi', 'P', 'R'])}
    '''
    bins = ['0-20', '20-60', '60+']
    series = series.T.reshape((10, -1, T))
    std = np.sqrt(variances.T.reshape((10, -1, T)))
    fig, ax = plt.subplots(nrows=5, ncols=2, sharex=True, figsize=(20, 15))
    fig.suptitle(state)
    tickLabels = list(DateIter(beginDate, beginDate + T))[::step]
    tickLabels = [d.date for d in tickLabels]
    days = np.array([np.arange(T) for _ in range(3)])
    for a, var, st, c in zip(collapse(ax), series, std, compartments): 
        labels = [c + '_' + b for b in bins]
        lines = a.plot(days.T, var.T)
        a.legend(lines, labels)
        for v, s, l in zip(var, st, lines) : 
            a.fill_between(np.arange(T), np.maximum(v-s, 0), v+s, facecolor=l.get_c(), alpha=0.2)
        a.set_xlabel('Time / days')
        a.set_ylabel('Number of people')
        a.set_yscale('log')
        a.xaxis.set_major_locator(ticker.MultipleLocator(step))
        a.set_xticklabels(tickLabels, rotation = 'vertical')
    fig.savefig('./Plots/' + state)
    plt.close(fig)
    '''
    colors = ['b', 'g', 'r']
    p, p_std = gather(T, series, variances, compartments['P'])
    symptomatics, symptomatics_std = gather(T, series, variances, compartments['P'] + compartments['I'] + compartments['Xi'] + compartments['A'] + compartments['Xa'])
    
    #Plotting Standard Deviations for each state

    tickLabels = list(DateIter(beginDate, beginDate + T + 30))[::step]
    tickLabels = [d.date for d in tickLabels]
    tickLabels = ['', *tickLabels]

    fig_std, ax3 = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(20, 10))
    fig_std.suptitle(state, fontsize=25)

    ax3.plot(np.arange(T), p_std, color = colors[0], label = "Standard Deviation: Tested Positive")
    ax3.plot(np.arange(T), symptomatics_std, color = colors[1], label = "Standard Deviation: Infected")
    
    ax3.legend(fontsize = 20, loc="upper left")
    ax3.set_xlabel('Time / days', fontsize=25)
    ax3.set_ylabel('Number of people', fontsize=25)
    # ax1.set_yscale('log')
    ax3.xaxis.set_major_locator(ticker.MultipleLocator(step))
    ax3.set_xticklabels(tickLabels, rotation = 'vertical')
    ax3.tick_params(axis='both', which='major', labelsize=20)

    fig_std.savefig('./Plots/' + state + '_STDDEV')
    plt.close(fig_std)
    plt.clf()

    #Plotting Actual State Predictions

    fig, ax1 = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(20, 10))
    fig.suptitle(state, fontsize=25)
    
    ax1.plot(np.arange(T), p, color = colors[0], label = "Tested Positive")
    ax1.fill_between(np.arange(T), np.maximum(p - p_std, 0), p + p_std, facecolor = colors[0], alpha=0.2)
    
    ax1.plot(np.arange(T), symptomatics, color = colors[1], label = "Infected")
    ax1.fill_between(np.arange(T), np.maximum(symptomatics - symptomatics_std, 0), symptomatics + symptomatics_std, facecolor = colors[1], alpha=0.2)

    # groundTruthPositive = (groundTruth['Total Cases'] - groundTruth['Total Recovered'] - groundTruth['Total Dead']).to_numpy()
    # ax1.scatter(np.arange(len(groundTruthPositive)), groundTruthPositive, c= colors[2], label = "Reported Positive")
    ax1.scatter(np.arange(0), [], c= colors[2], label = "Reported Positive")
    
    ax1.legend(fontsize = 20, loc="upper left")
    ax1.set_xlabel('Time / days', fontsize=25)
    ax1.set_ylabel('Number of people', fontsize=25)
    # ax1.set_yscale('log')
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(step))
    ax1.set_xticklabels(tickLabels, rotation = 'vertical')
    ax1.tick_params(axis='both', which='major', labelsize=20)


    #### INSET GRAPH
    left, bottom, width, height = [0.17, 0.37, 0.35, 0.35]
    ax2 = fig.add_axes([left, bottom, width, height])
    T2 = Date('14 Apr') - beginDate
    
    p = p[:T2]
    p_std = p_std[:T2]
    symptomatics = symptomatics[:T2]
    symptomatics_std = symptomatics_std[:T2]
    ax2.plot(np.arange(T2), p, color = colors[0], label = "Tested Positive")
    ax2.fill_between(np.arange(T2), np.maximum(p - p_std, 0), p + p_std, facecolor = colors[0], alpha=0.2)
    
    ax2.plot(np.arange(T2), symptomatics, color = colors[1], label = "Infected")
    ax2.fill_between(np.arange(T2), np.maximum(symptomatics - symptomatics_std, 0), symptomatics + symptomatics_std, facecolor = colors[1], alpha=0.2)

    groundTruthPositive = (groundTruth['Total Cases'] - groundTruth['Total Recovered'] - groundTruth['Total Dead']).to_numpy()
    dataDate = groundTruth['Date'].iloc[0].split('-')
    dataDate = Date(f'{dataDate[0]} {dataDate[1]}')
    if (dataDate - beginDate) >= 0:
        ax2.scatter(np.arange(dataDate - beginDate, dataDate - beginDate + len(groundTruthPositive)), groundTruthPositive, c= colors[2], label = "Reported Positive")
    else:
        ax2.scatter(np.arange(len(groundTruthPositive[beginDate - dataDate:])), groundTruthPositive[beginDate - dataDate:], c= colors[2], label = "Reported Positive")
    
    # ax2.legend(fontsize = 20)
    # ax2.set_xlabel('Time / days', fontsize=25)
    # ax2.set_ylabel('Number of people', fontsize=25)
    # ax1.set_yscale('log')
    tickLabels = list(DateIter(beginDate, beginDate + T + 30))[::7]
    tickLabels = [d.date for d in tickLabels]
    tickLabels = ['', *tickLabels]
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(7))
    ax2.set_xticklabels(tickLabels, rotation = 'vertical')
    ax2.tick_params(axis='both', which='major', labelsize=18)

    plt.gcf().subplots_adjust(bottom=0.2)
    fig.savefig('./Plots/' + state)
    plt.close(fig)
    plt.clf()