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

if __name__ == "__main__" : 
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
    for datum, m, nbar,state in zip(data, model.models, statePop, Model.STATES) : 
        E0 = [0, 10, 0]
        A0 = [0, 10, 0]
        I0 = [0, 10, 0]
        nbar[1] -= 30
        x0 = np.array([*(nbar.tolist()), *E0, *A0, *I0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        ks = KalmanSimulator(datum, m, x0)
        series, variances = ks(model.lockdownEnd - ks.startDate)
        #pdb.set_trace()
        seriesOfSeries.append(series[0:-1])
        lastSeries.append(series[-1])
        seriesOfVariances.append(variances[0:-1])
        lastVariance.append(variances[-1])
        Plot.statePlot(series, variances, state, ks.startDate, 7, datum, population = nbar.sum())

    x0 = np.hstack(lastSeries)
    n = x0.size
    P0 = np.zeros((n, n))
    for i, _ in enumerate(Model.STATES):
        P0[30*i:30*(i+1), 30*i: 30*(i+1)] = lastVariance[i]
    #pdb.set_trace()   
 
    Q = 0.1 * np.eye(n)
    H = lambda t : np.array([])
    R = lambda t : np.array([])
    Z = lambda t : np.array([])
    tStart = model.lockdownEnd
    tEnd = Date('15 Aug')

    newSeries, newVariances = extendedKalmanFilter(model.dx, x0, P0, Q, H, R, Z, tStart, tEnd)

    newVariances = [[v[30*i:30*(i+1), 30*i: 30*(i+1)] for i, _ in enumerate(Model.STATES)] for v in newVariances]
    newVariances = [[row[i] for row in newVariances] for i in range(len(newVariances[0]))] 

    newSeries = newSeries.T.reshape((len(Model.STATES), 30, -1))
    for i, _ in enumerate(Model.STATES) : 
        seriesOfSeries[i] = np.vstack((seriesOfSeries[i], newSeries[i].T))
        seriesOfVariances[i].extend(newVariances[i])

    with open('series.pkl', 'wb') as fd : 
        pickle.dump(seriesOfSeries, fd)

    with open('var.pkl', 'wb') as fd : 
        pickle.dump(seriesOfVariances, fd)

    state_id = 1
    for m, datum, series, variance ,state, population in zip(model.models, data, seriesOfSeries, seriesOfVariances, Model.STATES, statePop) : 
        ks = KalmanSimulator(datum, m, x0)
        Plot.statePlot(series, variance, state, ks.startDate, 7, datum, population = population.sum())

        # outputting into the csv
        # need to estimate daily values from the timeseries of all the compartments

        deads_daily = np.sum(getAgeMortality(state) * 0.01 * (series[:, 9:12] + series[:, 21:24] + series[:, 24:27]), axis = 1)
        deads_daily = deads_daily[:-17]
        deads_daily = np.concatenate([np.zeros(17), deads_daily])
        deads_total = np.cumsum(deads_daily)

        recovered_total = np.sum(series[:, 27:30], axis = 1)
        recovered_daily = np.insert(np.diff(recovered_total), 0 , recovered_total[0])
        recovered_daily = recovered_daily - deads_daily
        recovered_total = np.cumsum(recovered_daily)

        
        # also has E + XE for now because they go into recovered too
        infected_active = np.sum(series[:, 3:6] + series[:, 15:18] + series[:, 6:9] + series[:, 9:12] + series[:, 18:21] + series[:, 21:24] + series[:, 24:27], axis = 1)
        
        # if excluding E,Xe, can't compute infected_daily perfectly must settle with a 0.8 factor
        # infected_active = np.sum(series[:, 6:9] + series[:, 9:12] + series[:, 18:21] + series[:, 21:24] + series[:, 24:27], axis = 1)
        infected_daily = np.insert(np.diff(infected_active), 0 , infected_active[0])
        infected_daily = infected_daily + recovered_daily + deads_daily

        #print(deads_daily.shape, recovered_total.shape, recovered_daily.shape, infected_daily.shape)
        


        #Can get some negative terms clipping them to zero
        infected_daily = infected_daily.clip(min = 0)
        deads_daily = deads_daily.clip(min = 0)
        recovered_daily = recovered_daily.clip(min = 0)
       
        infected_active = infected_active.clip(min = 0)
        deads_total =  deads_total.clip(min = 0)
        recovered_total = recovered_total.clip(min = 0)

        state_ids = np.ones(tEnd - ks.startDate, dtype = int) * int(state_id)
        df = pd.DataFrame(data = [state_ids, infected_daily, deads_daily, recovered_daily], index = ["State Id", "Number of infected (new)", "Number of Death (New)", "Number of Recovery (New)"])
        df = df.T
        
        df2 = pd.DataFrame(data = [state_ids, infected_active, deads_total, recovered_total], index = ["State id", "Simulated total infected", "Simulated total death", "Simulated total recovery"])
        df2 = df2.T


        datelist = [f'{date.day}/{date.month}/2020' for date in DateIter(ks.startDate, tEnd + 1)]
        #print(len(datelist), len(infected_active))
        
        #print(len(datelist))

        #pdb.set_trace()
        df['Date'] = datelist

        df2['Date'] = datelist


        df = df[["State Id", "Date", "Number of infected (new)", "Number of Death (New)", "Number of Recovery (New)"]]
        df2 = df2[["State id", "Date", "Simulated total infected", "Simulated total death", "Simulated total recovery"]]

        if state_id == 1:
            DF = df
            DF2 = df2
        else:
            DF = pd.concat([DF, df], ignore_index = True)
            DF2 = pd.concat([DF2, df2], ignore_index = True)


        DF.to_csv('sheet2.csv', index = False) 
        DF2.to_csv('sheet3.csv', index = False) 



        state_id = state_id + 1
