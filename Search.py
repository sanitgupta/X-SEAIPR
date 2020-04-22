from itertools import product
from functools import partial
import time
import pandas
from Simulate import *
from Model import * 
from Losses import *
from Search import *
from multiprocessing import Process, Pipe, cpu_count
import math
from EKF import *

startDate = Date('29 Feb')
firstCases = Date('14 Mar')
firstDeath = Date('17 Mar')
endDate = Date('7 Apr')

def H (date) : 
    h1    = [0,0,0,.02,0,0,0,0,.02,0]
    h2    = [0,0,0,0.0,0,0,0,0,1.0,0]
    zeros = [0,0,0,0.0,0,0,0,0,0.0,0]
    if date < firstCases : 
        return np.array([h1, zeros])
    elif date >= firstCases and date < startDate + (endDate - firstDeath) :
        return np.array([h1, h2])
    else : 
        return np.array([zeros, h2])

def gridSearch (ivRanges, paramRanges, groundTruth, lossFunction) :
    def getVars (idx) : 
        return np.array([np.diag(P)[idx] for P in Ps])

    def getCov (i, j) :
        return np.array([P[i, j] for P in Ps])

    T = endDate - startDate

    deaths = getDailyDeaths(groundTruth, firstCases, firstDeath)
    deaths = np.pad(deaths, ((0, T - deaths.size)))

    P = getActive(groundTruth)
    P = np.pad(P, ((T - P.size, 0)))

    zs = np.stack([deaths, P]).T

    R = np.diag([5, 5])
    P0 = np.diag([1e3, 1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2])

    minLoss = math.inf
    minx0, minParams = None, None

    for x0, params in product(product(*ivRanges), dictProduct(paramRanges)) :
        model = Spaxire(params)  

        xs, Ps = extendedKalmanFilter(model.timeUpdate, np.array(x0), P0, 
                H, R, zs, startDate, endDate)
        i, xi, p = xs[:, 3], xs[:, 7], xs[:, 8]

        m = 0.02 * (i + xi + p) # Mortality

        iVar, xiVar, pVar = getVars(3), getVars(7), getVars(8)
        ipCov, pxiCov, ixiCov = getCov(3,8), getCov(7,8), getCov(3,7)

        mVar = ((0.02)**2)*(iVar + pVar + xiVar + ipCov + pxiCov + ixiCov) 
        loss = lossFunction(deaths, m, mVar) + lossFunction(P, p, pVar)

        if loss < minLoss : 
            minx0 = x0
            minParams = params
            minLoss = loss

    return minx0, minParams

def worker (conn, groundTruth, lossFunction, T) :

    # Do pre-processing of the deaths.
    lo, hi = T.min(), T.max()
    samples = 5
    T_ = np.linspace(lo, hi, (hi - lo) * samples)
    startIdx = groundTruth[groundTruth['Date'] == '20 Mar'].index[0] 
    deaths = groundTruth['New Deaths'][startIdx:].to_numpy()
    nDays = deaths.size

    while True : 
        item = conn.recv() 
        if item == 'e' : 
            conn.send('e')
            break
        else : 
            point, minLoss = item
            x0, params = point
            N = params['N']
            x0 = [N - sum(x0), *x0]
            model = Sixer(x0, params)
            result = simulator(model, T_)
            infections = result[:, 2][::samples]
            tested = result[:, -1][::samples]
            deathEstimate = 0.02 * (infections + tested)
            deathEstimate = deathEstimate[:nDays]
            loss = lossFunction(deaths, deathEstimate)
            if loss < minLoss: 
                conn.send((point, loss))
            else : 
                conn.send('n')

def parallelGridSearch (ivRanges, paramRanges, groundTruth, lossFunction, T) :
    totalChildren = cpu_count()

    # Setup duplex connections between parent and child.
    connections = [Pipe() for _ in range(totalChildren)]

    # Setup Child Processes
    processes = [
        Process(target=worker, args=(conn, groundTruth, lossFunction, T))
        for _, conn in connections
    ]  

    # Start Child Processes
    for p in processes : 
        p.start() 

    doneChildren =  0

    # Grid point generator
    points = product(product(*ivRanges), dictProduct(paramRanges)) 

    # Best grid point
    minLoss = math.inf
    minX0, minParams = None, None

    # Send initial data
    for p, _ in connections : 
        p.send((next(points), minLoss))

    nProcessed = 0
    power = 1
    st = time.time()

    while doneChildren < totalChildren : 

        # Poll all connections and 
        # send stuff if they are available.
        for conn, _ in connections : 
            hasSomething = conn.poll()
            if hasSomething : 
                obj = conn.recv() 
                if obj == 'e' :
                    # This signals that the child
                    # is done.
                    doneChildren += 1
                elif obj != 'n' : 
                    # This signals that the child
                    # found an improving grid point.
                    point, loss = obj
                    if loss < minLoss : 
                        minX0, minParams = point
                        minLoss = loss

                # Sentinel to guard against the 
                # possibility of querying an empty generator.
                point = next(points, False)

                nProcessed += 1
                if nProcessed % (10 ** power) == 0 : 
                    print(nProcessed, time.time() - st)
                    power += 1

                if point : 
                    conn.send((point, minLoss))
                else : 
                    # If no more points are left, signal
                    # to the child.
                    conn.send('e')

    # Join all child processes.
    for p in processes : 
        p.join()

    return minX0, minParams, minLoss

def main () :
    lockdownBegin = Date('24 Mar') - startDate
    lockdownEnd = Date('14 Apr') - startDate
    paramRanges = {
        'tl'    : [lockdownBegin], 
        'te'    : [lockdownEnd],
        'k0'    : [1/7], 
        'kt'    : [0.075],
        'mu'    : [1/7],
        'sigma' : [1/5],
        'gamma1': [1/21],
        'gamma2': [1/21],
        'gamma3': [1/17],
        'N'     : [1.1e8],
        'beta'  : np.arange(0, 1, 0.05),
        'beta1' : np.arange(0, 2, 0.10),
        'beta2' : [0.1],
        'f'     : [0.1]
    }
    E0, A0, I0 = 25, 25, 25
    N = 1.1e8
    ivRanges = [[N-E0-A0-I0],[E0],[A0],[I0],[0],[0],[0],[0],[0],[0]]
    data = pandas.read_csv('./Data/maha_data7apr.csv')
    print(gridSearch(ivRanges, paramRanges, data, heteroscedasticLoss))

if __name__ == "__main__" : 
    main()
