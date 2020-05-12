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

def statePlot (series, variances, state, beginDate, step, groundTruth,population = None, threshold = None) : 
    T = len(series)
    compartments = {k: [3*i, 3*i + 1, 3*i + 2] for i, k in enumerate(['S', 'E', 'A', 'I', 'Xs', 'Xe', 'Xa', 'Xi', 'P', 'R'])}
    # Define a closure function to register as a callback
    
    def convert_fraction_to_number(axis):
        y1, y2 = axis.get_ylim()
        rightAxis.set_ylim(population * float(y1) / 100., population * float(y2) / 100.)
        rightAxis.figure.canvas.draw()

    def convert_fraction_to_number2(axis):
        y1, y2 = axis.get_ylim()
        rightAxis2.set_ylim(population * float(y1) / 100., population * float(y2) / 100.)
        rightAxis2.figure.canvas.draw()

    def displayNumbers(x, pos):
        if  x >= 1e7: return '%1.1fM' % (x * 1e-6)
        elif  x > 1e5: return '%1.2fM' % (x * 1e-6)
        elif x > 1e4: return '%1.0fk' % (x * 1e-3)
        elif x > 1e3: return '%1.1fk' % (x * 1e-3)
        else: return str(int(x))
    formatter = ticker.FuncFormatter(displayNumbers)

    def displayDate(y, pos):
        return (beginDate + y).date
    formatter_date = ticker.FuncFormatter(displayDate)

    colors = ['b', 'g', 'r']
    p, p_std = gather(T, series, variances, compartments['P'])
    symptomatics, symptomatics_std = gather(T, series, variances, compartments['P'] + compartments['I'] + compartments['Xi'] + compartments['A'] + compartments['Xa'])
    
    #Plotting Standard Deviations for each state

    # tickLabels = list(DateIter(beginDate, beginDate + T + 30))[::step]
    # tickLabels = [d.date for d in tickLabels]
    # tickLabels = ['', *tickLabels]

    fig_std, ax3 = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(20, 10))
    ax3.yaxis.set_major_formatter(formatter)
    ax3.xaxis.set_major_formatter(formatter_date)
    fig_std.suptitle(state, fontsize=25)

    ax3.plot(np.arange(T), p_std, color = colors[0], label = "Standard Deviation: Tested Positive")
    ax3.plot(np.arange(T), symptomatics_std, color = colors[1], label = "Standard Deviation: Infected")
    
    ax3.legend(fontsize = 20, loc="upper left")
    ax3.set_xlabel('Time / days', fontsize=25)
    ax3.set_ylabel('Number of people', fontsize=25)
    # ax1.set_yscale('log')
    ax3.xaxis.set_major_locator(ticker.MultipleLocator(step))
    plt.xticks(rotation = 'vertical')
    ax3.tick_params(axis='both', which='major', labelsize=20)

    fig_std.savefig('./Plots/' + state + '_STDDEV')
    plt.close(fig_std)
    plt.clf()

    #Plotting Actual State Predictions

    fig, ax1 = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(20, 10))
    plt.xticks(rotation = 'vertical')
    ax1.xaxis.set_major_formatter(formatter_date)
    if population is not None:
        rightAxis = ax1.twinx()
        rightAxis.yaxis.set_major_formatter(formatter)
        ax1.callbacks.connect("ylim_changed", convert_fraction_to_number)

    fig.suptitle(state, fontsize=25)
    
    ax1.plot(np.arange(T), p * 100. / population, color = colors[0], label = "Tested Positive")
    ax1.fill_between(np.arange(T), np.maximum(p - p_std, 0) * 100. / population, (p + p_std) * 100. / population, facecolor = colors[0], alpha=0.2)
    
    ax1.plot(np.arange(T), symptomatics * 100. / population, color = colors[1], label = "Infected")
    ax1.fill_between(np.arange(T), np.maximum(symptomatics - symptomatics_std, 0) * 100. / population, (symptomatics + symptomatics_std) * 100. / population , facecolor = colors[1], alpha=0.2)
    ## Find when we go over 2% active cases and shade
    if threshold is not None:
        index = Date('3 May') - beginDate
        while index < len(p) and p[index] < threshold * population:
            index += 1
        if index < len(p):
        	# Find when it ends (if at all) - Note that right now I am not handling the case where you shut down and then
        	# you start again...
        	shutIndex = index
        	while shutIndex < len(p) and p[shutIndex] > threshold * population:
        		shutIndex += 1
            # Shade
            ax1.fill_between(np.arange(index, shutIndex), 0, max(symptomatics + symptomatics_std) * 1.1 * 100./population, facecolor = colors[2], alpha=0.1)

    ax1.scatter(np.arange(0), [], c= colors[2], label = "Reported Positive")

    ax1.legend(fontsize = 20, loc="upper left")
    ax1.set_xlabel('Time / days', fontsize=25)
    rightAxis.set_ylabel('Number of people', fontsize=25)
    ax1.set_ylabel('Percentage of Total Population', fontsize=25)
    # ax1.set_yscale('log')
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(step))
    ax1.tick_params(axis='both', which='major', labelsize=20)
    if population is not None:
        rightAxis.tick_params(axis='both', which='major', labelsize=20)

    #### INSET GRAPH
    left, bottom, width, height = [0.18, 0.37, 0.35, 0.35]
    ax2 = fig.add_axes([left, bottom, width, height])
    
    if population is not None:
        rightAxis2 = ax2.twinx()
        rightAxis2.yaxis.set_major_formatter(formatter)
        ax2.callbacks.connect("ylim_changed", convert_fraction_to_number2)
        rightAxis2.tick_params(axis='both', which='major', labelsize=20)

    T2 = Date('4 May') - beginDate
    
    p = p[:T2]
    p_std = p_std[:T2]
    symptomatics = symptomatics[:T2]
    symptomatics_std = symptomatics_std[:T2]
    ax2.plot(np.arange(T2), p * 100. / population, color = colors[0], label = "Tested Positive")
    ax2.fill_between(np.arange(T2), np.maximum(p - p_std, 0) * 100. / population, (p + p_std) * 100. / population, facecolor = colors[0], alpha=0.2)
    
    ax2.plot(np.arange(T2), symptomatics * 100. / population, color = colors[1], label = "Infected")
    ax2.fill_between(np.arange(T2), np.maximum(symptomatics - symptomatics_std, 0) * 100. / population, (symptomatics + symptomatics_std) * 100. / population, facecolor = colors[1], alpha=0.2)

    groundTruthPositive = (groundTruth['Total Cases'] - groundTruth['Total Recovered'] - groundTruth['Total Dead']).to_numpy()
    dataDate = groundTruth['Date'].iloc[0].split('-')
    dataDate = Date(f'{dataDate[0]} {dataDate[1]}')
    if (dataDate - beginDate) >= 0:
        ax2.scatter(np.arange(dataDate - beginDate, dataDate - beginDate + len(groundTruthPositive)), groundTruthPositive * 100. / population, c= colors[2], label = "Reported Positive")
    else:
        ax2.scatter(np.arange(len(groundTruthPositive[beginDate - dataDate:])), groundTruthPositive[beginDate - dataDate:] * 100. / population, c= colors[2], label = "Reported Positive")
    
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