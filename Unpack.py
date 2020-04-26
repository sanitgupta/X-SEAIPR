import matplotlib.pyplot as plt
from matplotlib import ticker
import pickle
import pdb
import numpy as np

def gather(T, series, variances, indices):
    outputSeries = [sum(x[index] for index in indices) for x in series]
    outputVariances = [x[indices, :][:, indices].sum() for x in variances]
    outputVariances = [np.sqrt(x) for x in outputVariances]
    return np.array(outputSeries), np.array(outputVariances)

if __name__ == "__main__" :

    with open('series.pkl', 'rb') as fd : 
        seriesOfSeries = pickle.load(fd)

    with open('var.pkl', 'rb') as fd : 
        seriesOfVariances = pickle.load(fd)

    states = ['ANDAMAN&NICOBAR','ANDHRAPRADESH','ARUNACHALPRADESH',
        'ASSAM','BIHAR','CHANDIGARH',
        'CHHATTISGARH','DADRA&NAGARHAVELI','DAMAN&DIU',
        'GOA','GUJARAT','HARYANA',
        'HIMACHALPRADESH','JAMMU&KASHMIR','JHARKHAND',
        'KARNATAKA','KERALA','LADAK',
        'LAKSHADWEEP','MADHYAPRADESH','MAHARASHTRA',
        'MANIPUR','MEGHALAYA','MIZORAM',
        'NCTOFDELHI','NAGALAND','ODISHA',
        'PUDUCHERRY','PUNJAB','RAJASTHAN',
        'SIKKIM','TAMILNADU','TELANGANA',
        'TRIPURA','UTTARPRADESH','UTTARAKHAND','WESTBENGAL']

    for i in range(0,len(states)):
        series = seriesOfSeries[i]
        variances = seriesOfVariances[i]

        T = len(series)
        compartments = {k: [3*i, 3*i + 1, 3*i + 2] for i, k in enumerate(['S', 'E', 'A', 'I', 'Xs', 'Xe', 'Xa', 'Xi', 'P', 'R'])}

        colors = ['b', 'g', 'r', 'c', 'm']
        p, p_std = gather(T, series, variances, compartments['P'])
        infc, infc_std = gather(T, series, variances, compartments['I'])
        infx, infx_std = gather(T, series, variances, compartments['Xi'])
        asy, asy_std = gather(T,series, variances, compartments['A'])
        asyx, asyx_std = gather(T,series, variances, compartments['Xa'])

        fig, ax1 = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(20, 10))
        fig.suptitle(states[i], fontsize=25)
    
        ax1.plot(np.arange(T), p, color = colors[0], label = "Tested Positive")
        ax1.fill_between(np.arange(T), np.maximum(p - p_std, 0), p + p_std, facecolor = colors[0], alpha=0.2)
    
        ax1.plot(np.arange(T), infc, color = colors[1], label = "Infected")
        ax1.fill_between(np.arange(T), np.maximum(infc - infc_std, 0), infc + infc_std, facecolor = colors[1], alpha=0.2)

        ax1.plot(np.arange(T), infx, color = colors[2], label = "Infected (in lockdown)")
        ax1.fill_between(np.arange(T), np.maximum(infx - infx_std, 0), infx + infx_std, facecolor = colors[2], alpha=0.2)
    
        ax1.plot(np.arange(T), asy, color = colors[3], label = "Asymptomatic")
        ax1.fill_between(np.arange(T), np.maximum(asy - asy_std, 0), asy + asy_std, facecolor = colors[1], alpha=0.2)

        ax1.plot(np.arange(T), asyx, color = colors[4], label = "Asymptomatic (in lockdown)")
        ax1.fill_between(np.arange(T), np.maximum(asyx - asyx, 0), asyx + asyx_std, facecolor = colors[4], alpha=0.2)

        ax1.legend(fontsize = 20, loc="upper left")
        ax1.set_xlabel('Time / days', fontsize=25)
        ax1.set_ylabel('Number of people', fontsize=25)

        plt.gcf().subplots_adjust(bottom=0.2)
        fig.savefig('./Plots_Unpacked/' + states[i])
        plt.close(fig)
        plt.clf()

