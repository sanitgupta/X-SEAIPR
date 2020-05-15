from Util import *
import pickle
import Plot
from EKF import *
import json
import Model
from scipy.integrate import odeint
from functools import partial
import numpy as np 
import pandas as pd
import pdb
import os
import matplotlib.pyplot as plt
from matplotlib import ticker

def odeSimulator (model, x0, T) :
    dx = partial(model.dx, module=np)
    result = odeint(dx, x0, T)
    return result

class KalmanSimulator () :

    def __init__ (self, data, model, x0) : 
        self.x0 = x0

        self.data = data
        self.model = model
        self.dates = data['Date'].map(self.splitDates)

        self.firstCases = Date(self.dates.iloc[0])
        self.dataEndDate = Date(self.dates.iloc[-1])

        self.peopleDied = self.dates[data['Total Dead'] > 0].size > 0
        if self.peopleDied : 
            self.firstDeath = Date(self.dates[data['Total Dead'] > 0].iloc[0])
            self.startDate = self.firstDeath - 17
            self.deaths = self.data['Daily Dead'][data['Total Dead'] > 0].to_numpy()
        else : 
            self.startDate = self.firstCases

        self.P = (data['Total Cases'] - data['Total Recovered'] - data['Total Dead']).to_numpy()
        
        self.h1, self.h2 = [0] * 30, [0] * 30 
        self.h1[9:12] = model.mortality.tolist() # Setting mortality

        self.h1[21:24] = model.mortality.tolist() # Setting mortality

        self.h1[24:27] = model.mortality.tolist() # Setting mortality
        self.h2[-6:-3] = [1,1,1] # Setting P

        self.setP0() 
        self.setQ()

    def setP0(self) : 
        self.P0 = np.eye(30)

    def setQ (self) :
        self.Q = np.eye(30)
    
    def splitDates (self, date) : 
        d, m, _ = date.split('-')
        d = int(d)
        return f'{d} {m}'

    def H (self, date) : 
        if self.peopleDied : 
            if date < self.firstCases : 
                return np.array([self.h1])
            elif self.firstCases <= date <= self.dataEndDate - 17 :
                return np.array([self.h1, self.h2])
            elif self.dataEndDate - 17 < date <= self.dataEndDate : 
                return np.array([self.h2])
            else :
                return np.array([])
        else :
            if date <= self.dataEndDate : 
                return np.array([self.h2])
            else :
                return np.array([])

    def Z (self, date): 
        if self.peopleDied : 
            if date < self.firstCases : 
                m = self.deaths[date - self.startDate]
                return np.array([m])
            elif self.firstCases <= date <= self.dataEndDate - 17 :
                m = self.deaths[date - self.startDate]
                p = self.P[date - self.firstCases]
                return np.array([m, p])
            elif self.dataEndDate - 17 < date <= self.dataEndDate : 
                p = self.P[date - self.firstCases]
                return np.array([p])
            else :
                return np.array([])
        else : 
            if date <= self.dataEndDate : 
                p = self.P[date - self.firstCases]
                return np.array([p])
            else : 
                return np.array([])

    def R (self, date): 
        if self.peopleDied : 
            if date < self.firstCases : 
                return np.array([1])
            elif self.firstCases <= date <= self.dataEndDate - 17 :
                return np.eye(2)
            elif self.dataEndDate - 17 < date <= self.dataEndDate : 
                return np.array([1])
            else :
                return np.array([])
        else : 
            if date <= self.dataEndDate : 
                return np.array([1])
            else : 
                return np.array([])

    def __call__ (self, T) : 
        endDate = self.startDate + T
        series, variances = extendedKalmanFilter(
                self.model.dx, self.x0, self.P0, 
                self.Q, self.H, self.R, self.Z, 
                self.startDate, endDate)
        return series, variances

def getDeaths(dataDir, plot_start_date = Date('14 Mar')): 
    data_end_date = None
    with open('./Data/beta.json') as fd : 
        betas = json.load(fd)
    transportMatrix = np.loadtxt('./Data/transportMatrix.csv', delimiter=',')
    statePop  = [getStatePop(s) for s in Model.STATES]
    mortality = [0.01 * getAgeMortality(s) for s in Model.STATES]
    data = [getData(s) for s in Model.STATES]
    model = Model.IndiaModel(transportMatrix, betas, statePop, mortality, data) 
    seriesOfSeries = []
    lastSeries = []
    seriesOfVariances = []
    lastVariance = []
    x0 = [0] * 15
    tEnd = Date('15 Aug')

    with open(os.path.join(dataDir, 'series.pkl'), 'rb') as fd : 
        seriesOfSeries = pickle.load(fd)

    with open(os.path.join(dataDir, 'var.pkl'), 'rb') as fd : 
        seriesOfVariances = pickle.load(fd)

    state_id = 1
    total_population = 0
    for m, datum, series, variance ,state, population in zip(model.models, data, seriesOfSeries, seriesOfVariances, Model.STATES, statePop) : 
        ks = KalmanSimulator(datum, m, x0)
        total_population += population.sum()
        
        # recovered_total = np.sum(series[:, 27:30], axis = 1)
        recovered_total = series[:, 27:30]
        recovered_daily = np.zeros_like(recovered_total)
        recovered_daily[0, :] = recovered_total[0, :]
        for i in range(1, len(recovered_total)):
            recovered_daily[i, :] = recovered_total[i, :] - recovered_total[i - 1, :]
        
        # also has E + XE for now because they go into recovered too
        # infected_active = np.sum(series[:, 3:6] + series[:, 15:18] + series[:, 6:9] + series[:, 9:12] + series[:, 18:21] + series[:, 21:24] + series[:, 24:27], axis = 1)
        infected_active = series[:, 3:6] + series[:, 15:18] + series[:, 6:9] + series[:, 9:12] + series[:, 18:21] + series[:, 21:24] + series[:, 24:27]
        
        infected_daily = np.zeros_like(infected_active)
        infected_daily[0, :] = infected_active[0, :]
        for i in range(1, len(infected_daily)):
            infected_daily[i, :] = infected_active[i, :] - infected_active[i - 1, :]
        infected_daily = infected_daily + recovered_daily

        deads_daily = np.sum(getAgeMortality(state) * 0.01 * infected_daily, axis = 1)
        # deads_daily = deads_daily[:-17]
        deads_daily = np.concatenate([np.zeros(17), deads_daily])
        deads_daily = np.cumsum(deads_daily)

        if data_end_date is None:
            data_end_date = ks.startDate + deads_daily.shape[0]
            total_deaths = np.zeros((data_end_date - plot_start_date))
            total_deaths += deads_daily[len(deads_daily) -len(total_deaths):]
        else:
            assert data_end_date.date == (ks.startDate + deads_daily.shape[0]).date, "Inconsistency in the data - all simulations not ending at the same date"
            if len(deads_daily) < len(total_deaths):
                deads_daily = np.concatenate([np.zeros(- len(deads_daily) + len(total_deaths)), deads_daily])
            total_deaths += deads_daily[len(deads_daily)-len(total_deaths):]

        state_id = state_id + 1
    return total_deaths, total_population

def gather(T, series, variances, indices):
    outputSeries = [sum(x[index] for index in indices) for x in series]
    outputVariances = [x[indices, :][:, indices].sum() for x in variances]
    outputVariances = [np.sqrt(x) for x in outputVariances]
    return np.array(outputSeries), np.array(outputVariances)

def plot (base_deaths, intervention1_deaths, intervention2_deaths, beginDate, step,population = None, state = "India") : 
    T = len(base_deaths)
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
    #Plotting Actual State Predictions

    fig, ax1 = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(20, 10))
    plt.xticks(rotation = 'vertical')
    ax1.xaxis.set_major_formatter(formatter_date)
    if population is not None:
        rightAxis = ax1.twinx()
        rightAxis.yaxis.set_major_formatter(formatter)
        ax1.callbacks.connect("ylim_changed", convert_fraction_to_number)

    fig.suptitle(state + ": Predicted Deaths", fontsize=25)
    
    ax1.plot(np.arange(T), base_deaths * 100. / population, color = colors[0], label = "No Intervention")
    # ax1.fill_between(np.arange(T), np.maximum(p - p_std, 0) * 100. / population, (p + p_std) * 100. / population, facecolor = colors[0], alpha=0.2)
    
    ax1.plot(np.arange(T), intervention1_deaths * 100. / population, color = colors[1], label = "Intervention 1")
    # ax1.fill_between(np.arange(T), np.maximum(symptomatics - symptomatics_std, 0) * 100. / population, (symptomatics + symptomatics_std) * 100. / population , facecolor = colors[1], alpha=0.2)

    ax1.plot(np.arange(T), intervention2_deaths * 100. / population, color = colors[2], label = "Intervention 2")
    # ax1.fill_between(np.arange(T), np.maximum(symptomatics - symptomatics_std, 0) * 100. / population, (symptomatics + symptomatics_std) * 100. / population , facecolor = colors[1], alpha=0.2)
    
    # ax1.scatter(np.arange(0), [], c= colors[2], label = "Reported Positive")

    ax1.legend(fontsize = 20, loc="upper left")
    ax1.set_xlabel('Time / days', fontsize=25)
    rightAxis.set_ylabel('Number of people', fontsize=25)
    ax1.set_ylabel('Percentage of Total Population', fontsize=25)
    # ax1.set_yscale('log')
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(step))
    ax1.tick_params(axis='both', which='major', labelsize=20)
    if population is not None:
        rightAxis.tick_params(axis='both', which='major', labelsize=20)

    # #### INSET GRAPH
    # left, bottom, width, height = [0.18, 0.37, 0.35, 0.35]
    # ax2 = fig.add_axes([left, bottom, width, height])
    
    # if population is not None:
    #     rightAxis2 = ax2.twinx()
    #     rightAxis2.yaxis.set_major_formatter(formatter)
    #     ax2.callbacks.connect("ylim_changed", convert_fraction_to_number2)
    #     rightAxis2.tick_params(axis='both', which='major', labelsize=20)

    # T2 = Date('1 Jun') - beginDate
    
    # base_deaths = base_deaths[:T2]
    # intervention1_deaths = intervention1_deaths[:T2]
    # intervention2_deaths = intervention2_deaths[:T2]
    
    # ax2.plot(np.arange(T2), base_deaths * 100. / population, color = colors[0], label = "No Intervention")
    # # ax2.fill_between(np.arange(T2), np.maximum(p - p_std, 0) * 100. / population, (p + p_std) * 100. / population, facecolor = colors[0], alpha=0.2)
    
    # ax2.plot(np.arange(T2), intervention1_deaths * 100. / population, color = colors[1], label = "Intervention 1")
    # # ax2.fill_between(np.arange(T2), np.maximum(symptomatics - symptomatics_std, 0) * 100. / population, (symptomatics + symptomatics_std) * 100. / population, facecolor = colors[1], alpha=0.2)

    # ax2.plot(np.arange(T2), intervention2_deaths * 100. / population, color = colors[2], label = "Intervention 2")
    # # ax2.fill_between(np.arange(T2), np.maximum(symptomatics - symptomatics_std, 0) * 100. / population, (symptomatics + symptomatics_std) * 100. / population, facecolor = colors[1], alpha=0.2)

    # tickLabels = list(DateIter(beginDate, beginDate + T + 30))[::7]
    # tickLabels = [d.date for d in tickLabels]
    # tickLabels = ['', *tickLabels]
    # ax2.xaxis.set_major_locator(ticker.MultipleLocator(7))
    # ax2.set_xticklabels(tickLabels, rotation = 'vertical')
    # ax2.tick_params(axis='both', which='major', labelsize=18)

    plt.gcf().subplots_adjust(bottom=0.2)
    fig.savefig('temp')
    plt.close(fig)
    plt.clf()

if __name__ == "__main__":
    base_deaths, total_population = getDeaths('/Users/sahil/Desktop/sem8/covid/blossomRuns/base/', plot_start_date = Date('1 Apr'))
    intervention1_deaths, _ = getDeaths('/Users/sahil/Desktop/sem8/covid/blossomRuns/intervention1/', plot_start_date = Date('1 Apr'))
    intervention2_deaths, _ = getDeaths('/Users/sahil/Desktop/sem8/covid/blossomRuns/intervention2/', plot_start_date = Date("1 Apr"))
    plot(
        base_deaths = base_deaths,
        intervention1_deaths = intervention1_deaths,
        intervention2_deaths = intervention2_deaths,
        beginDate = Date('1 Apr'),
        step = 7,
        population = total_population
    )



