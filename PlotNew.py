import pickle
import sys
import numpy as np
import Model
import json
from Util import *
import Plot
import os
from Simulate import KalmanSimulator

if __name__ == "__main__" : 
    assert len(sys.argv) == 2, "Usage: python PlotNew.py <path to directory which contains series.pkl and var.pkl>"
    with open('./Data/beta.json') as fd : 
        betas = json.load(fd)
    transportMatrix = np.loadtxt('./Data/transportMatrix.csv', delimiter=',')
    statePop  = [getStatePop(s) for s in Model.STATES]
    mortality = [0.01 * getAgeMortality(s) for s in Model.STATES]
    data = [getData(s) for s in Model.STATES]
    model = Model.IndiaModel(transportMatrix, betas, statePop, mortality, data) 
    
    seriesOfSeries = []
    seriesOfVariances = []
    
    with open(os.path.join( sys.argv[1], 'series.pkl'), 'rb') as fd : 
        seriesOfSeries = pickle.load(fd)

    with open(os.path.join( sys.argv[1], 'var.pkl'), 'rb') as fd : 
        seriesOfVariances = pickle.load(fd)

    x0 = np.hstack([series[-1] for series in seriesOfSeries])
    state_id = 1
    for m, datum, series, variance ,state in zip(model.models, data, seriesOfSeries, seriesOfVariances, Model.STATES) : 
        print("Plotting", state)
        ks = KalmanSimulator(datum, m, x0)
        Plot.statePlot(series, variance, state, ks.startDate, 7, datum)

        