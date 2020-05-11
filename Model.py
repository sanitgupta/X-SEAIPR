from Util import *
import json
import random
import math
from more_itertools import collapse
from itertools import product
from functools import partial
import torch
import matplotlib.pyplot as plt
import numpy as np
from Simulate import *
from copy import deepcopy
from Plot import * 
import pdb

cat = {np : np.hstack, torch : torch.cat}

STATES = ['ANDAMAN&NICOBAR','ANDHRAPRADESH','ARUNACHALPRADESH',
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

# STATES = getAllPlaces()
# STATES.sort()

class IndiaModel () : 

    def __init__ (self, transportMatrix, betas, statePop, mortality, data) : 
        self.transportMatrix = transportMatrix
        self.betas = betas
        self.statePop = statePop
        self.bins = 3
        self.states = len(STATES)
        self.mortality = mortality
        self.data = data
        self.setStateModels()

    def splitDates (self, date) : 
        d, m, _ = date.split('-')
        d = int(d)
        return f'{d} {m}'

    def setTestingFractions(self, newTestingFractions):
        for i, place in enumerate(STATES):
            self.models[i].setTestingFractions(newTestingFractions[place][0], newTestingFractions[place][1], newTestingFractions[place][2])

    def dx (self, x, delta_t, startDate, module=np) : 

        #pdb.set_trace()
        xs = x.reshape((self.states, -1))
        derivatives = [m.dx(x, delta_t, startDate, module) for x, m in zip(xs, self.models)]
        
        #pdb.set_trace()
        for m in self.models : 
            m.send()

        #pdb.set_trace()

        for i in range(self.states) : 
            _, outChannel = self.links[i]
            data = outChannel.pop()
            for j in range(self.states) :
                data_ = dict()
                for key, val in data.items() : 
                    data_[key] = val * self.transportMatrix[j, i]
                inChannel, _ = self.links[j]
                inChannel.append(data_)

        #pdb.set_trace()

        for m in self.models : 
            m.receive()
        
        #pdb.set_trace()
        derivatives = [m.addCrossTerms(dx, module) for m, dx in zip(self.models, derivatives)]
        dx = cat[module](derivatives)
        return dx

    def timeUpdate (self, x, t, module=np) : 
        dx = self.dx(x, t, module)
        return x + dx

    def setStateModels (self):
        self.models = []
        self.links = []
        self.lockdownEnd = Date('3 May')
        for idx, state in enumerate(STATES) : 
            datum = self.data[idx]

            dates = datum['Date'].map(self.splitDates)

            firstCases = Date(dates.iloc[0])
            dataEndDate = Date(dates.iloc[-1])
            peopleDied = dates[datum['Total Dead'] > 0].size > 0
            if peopleDied : 
                firstDeath = Date(dates[datum['Total Dead'] > 0].iloc[0])
                startDate = firstDeath - 17
            else : 
                startDate = firstCases

            lockdownBegin = Date('24 Mar')
            lockdownEnd = self.lockdownEnd

            contactHome = np.loadtxt('./Data/home.csv', delimiter=',')
            contactSchool = np.loadtxt('./Data/school.csv', delimiter=',')
            contactWork = np.loadtxt('./Data/work.csv', delimiter=',')
            contactOther = np.loadtxt('./Data/home.csv', delimiter=',')
            contactTotal = np.loadtxt('./Data/total.csv', delimiter=',')

            changeContactStart = Date('10 Nov')
            changeContactEnd   = Date('11 Nov')

            changeKt = Date('10 Nov')
            deltaKt  = 10

            beta, lockdownLeakiness, tf1, tf2, tf3  = self.betas[state]

            params = {
                'tl'                : lockdownBegin, 
                'te'                : lockdownEnd,
                'k0'                : partial(bumpFn, ti=lockdownBegin, tf=lockdownEnd, x1=0, x2=1/7),
                'kt'                : partial(stepFn, t0=changeKt, x1=0.5, x2=1.0),
                'mu'                : partial(stepFn, t0=lockdownEnd, x1=0, x2=1/7),
                'sigma'             : 1/5,
                'gamma1'            : 1/19,
                'gamma2'            : 1/22,
                'gamma3'            : 1/22,
                'N'                 : self.statePop[idx],
                'beta'              : beta,
                'beta2'             : 0.1,
                'f'                 : 0.2,
                'lockdownLeakiness' : lockdownLeakiness,
                'contactHome'       : partial(bumpFn, ti=changeContactStart, tf=changeContactEnd, x1=contactHome, x2=0.5*contactHome),
                'contactTotal'      : partial(bumpFn, ti=changeContactStart, tf=changeContactEnd, x1=contactTotal, x2=0.5*contactTotal),
                'contactSchool'     : contactSchool,
                'bins'              : 3,
                'adultBins'         : [1],
                'testingFraction1'  : partial(climbFn, ti=changeKt, tf=changeKt+deltaKt, xi=tf1, xf=0.8),
                'testingFraction2'  : partial(climbFn, ti=changeKt, tf=changeKt+deltaKt, xi=tf2, xf=0.5),
                'testingFraction3'  : partial(climbFn, ti=changeKt, tf=changeKt+deltaKt, xi=tf3, xf=0.5),
                'totalOut'          : self.transportMatrix[:, idx].sum(),
                'Nbar'              : self.statePop[idx],
                'mortality'         : self.mortality[idx]
            }

            inChannel, outChannel = [], []
            self.links.append((inChannel, outChannel))

            self.models.append(SpaxireAgeStratified(params, inChannel, outChannel))

class SpaxireAgeStratified () : 
    """
    Current ODE Model class. 
    
    The constructor takes a dictionary
    of parameters and initializes the model.
    """
    def __init__ (self, params, inChannel=None, outChannel=None) :
        """
        ODE has a lot of parameters.
        These are present in a dictionary from
        which the model is initialized.

        Parameters
        ----------
        params : dictionary of parameters
            Many of the parameters are easy to
            fix because they are determined by
            the COVID situation in India. For 
            example kt is the testing rate. 
            Other parameters such as beta/beta1
            which are related to how the disease
            spreads aren't so easy to specify.
        """

        self.inChannel = inChannel
        self.outChannel = outChannel

        self.tl = params['tl']
        self.te = params['te']

        self.k0  = params['k0']
        self.kt  = params['kt']
        self.mu  = params['mu']

        self.sigma  = params['sigma']
        self.gamma1 = params['gamma1']
        self.gamma2 = params['gamma2']
        self.gamma3 = params['gamma3']

        self.N = params['N']

        self.beta  = params['beta']
        self.beta2 = params['beta2']

        self.f = params['f']
        self.lockdownLeakiness = params['lockdownLeakiness']

        self.contactHome = params['contactHome']
        self.contactTotal = params['contactTotal']
        self.contactSchool = params['contactSchool']
    
        self.bins = params['bins'] # Age bins
        self.Nbar = params['Nbar']
        self.adultBins = params['adultBins']

        self.testingFraction1 = params['testingFraction1']
        self.testingFraction2 = params['testingFraction2']
        self.testingFraction3 = params['testingFraction3']

        self.mortality = params['mortality']
        self.totalOut = params['totalOut']

        names = ['S', 'E', 'A', 'I', 'Xs', 'Xe', 'Xa', 'Xi', 'P', 'R']
        self.names = [[n + str(i) for i in range(1, self.bins + 1)] for n in names]
        self.names = list(collapse(self.names))
        
        r = [random.random() for _ in range(30)]
        g = [random.random() for _ in range(30)]
        b = [random.random() for _ in range(30)]

        self.colors = list(zip(r,g,b))


    def setTestingFractions(self, tf1, tf2, tf3):
        if type(tf1) not in [int, float]:
            self.testingFraction1 = tf1
        else:
            self.testingFraction1 = lambda t : tf1
        
        if type(tf2) not in [int, float]:
            self.testingFraction1 = tf2
        else:
            self.testingFraction1 = lambda t : tf2
        
        if type(tf3) not in [int, float]:
            self.testingFraction1 = tf3
        else:
            self.testingFraction1 = lambda t : tf3

    def send (self) : 
        # Q = self.s + self.e + self.a + self.i + self.r

        sOut = self.s[1] / self.N[1]
        eOut = self.e[1] / self.N[1] 
        aOut = self.a[1] / self.N[1]
        iOut = self.i[1] / self.N[1]
        rOut = self.r[1] / self.N[1] 

        data = {'s': sOut, 'e': eOut, 'a' : aOut, 'i' : iOut, 'r' : rOut} 
        self.outChannel.append(data)

        self.sOut = self.totalOut * sOut
        self.eOut = self.totalOut * eOut
        self.aOut = self.totalOut * aOut
        self.iOut = self.totalOut * iOut
        self.rOut = self.totalOut * rOut

    def receive (self) : 
        self.sIn = sum([data['s'] for data in self.inChannel])
        self.eIn = sum([data['e'] for data in self.inChannel])
        self.aIn = sum([data['a'] for data in self.inChannel])
        self.iIn = sum([data['i'] for data in self.inChannel])
        self.rIn = sum([data['r'] for data in self.inChannel])
        self.inChannel.clear()

    def dx (self, x, delta_t, startDate, module=np) : 
        """
        This gives the derivative wrt time
        of the state vector. 

        This function can be directly plugged
        into scipy's odeint with the initial 
        values to simulate the model.

        Parameters
        ----------
        x : state vector
        t : time step 
        module : whether to use torch or numpy
        """
        t = startDate + int(delta_t)
        #print("Model Date: "+str(t.date))

        s, e, a, i, xs, xe, xa, xi, p, r = x.reshape((-1, self.bins))

        # convert depending on usage of this function
        if module == torch : 
            ct   = torch.from_numpy(self.contactTotal(t))
            ch   = torch.from_numpy(self.contactHome(t))
            cs   = torch.from_numpy(self.contactSchool)
        else : 
            ct = self.contactTotal(t)
            ch = self.contactHome(t)
            cs = self.contactSchool

        self.Nbar = s + e + a + i + xs + xe + xa + xi + p + r

        b3 = 0.002 * self.lockdownLeakiness

        cl  = (ct - cs) *  self.lockdownLeakiness     + ch * (1.0 - self.lockdownLeakiness)
        cl2 = (ct - cs) * (self.lockdownLeakiness**2) + ch * (1.0 - self.lockdownLeakiness**2) 

        # lambda for non-lockdown
        current = ct * (i + a + self.beta2*e) / self.Nbar
        current += cl * (xi + xa + self.beta2*xe) / self.Nbar
        current[self.adultBins] += ct[self.adultBins, :] * b3 * p / self.Nbar[self.adultBins]
        lambdaNormal = module.sum(self.beta * current, axis=1)

        # lambda for lockdown
        current = cl * (i + a + self.beta2*e) / self.Nbar
        current += cl2 * (xi + xa + self.beta2*xe) / self.Nbar
        current[self.adultBins] += cl[self.adultBins, :] * b3 * p / self.Nbar[self.adultBins]
        lambdaLockdown = module.sum(self.beta * current, axis=1)

        # testing rates for presymptomatics, symptomatics and asymptomatics respectively
        testFrac1 = 3 * self.testingFraction1(t) / 8
        testFrac2 = 5 * self.testingFraction1(t) / (8 - 3 * self.testingFraction1(t))
        testFrac3 = self.testingFraction3(t)


        ds = -s * (lambdaNormal + self.k0(t)) + self.mu(t) * xs 
        de = self.f * lambdaNormal * s \
                - e * (self.k0(t) \
                    + self.gamma1 \
                    + testFrac3) \
                + self.mu(t) * xe 
        da = (1 - self.f) * lambdaNormal * s \
                - a * (self.k0(t) \
                    + self.sigma \
                    + testFrac1) \
                + self.mu(t) * xa 
        di = self.sigma * a \
                - i * (self.k0(t) \
                    + testFrac2 \
                    + self.gamma2) \
                + self.mu(t) * xi 
        dxs = - xs * (lambdaLockdown + self.mu(t)) \
                + self.k0(t) * s
        dxe = self.f * lambdaLockdown * xs \
                + self.k0(t) * e \
                - xe * (self.mu(t) \
                    + self.gamma1 \
                    + testFrac3)
        dxa = (1 - self.f) * lambdaLockdown * xs \
                - xa * (self.mu(t) \
                    + self.sigma \
                    + testFrac1) \
                + self.k0(t) * a 
        dxi = self.sigma * xa \
                + self.k0(t) * i \
                - xi * (self.mu(t) \
                    + testFrac2 \
                    + self.gamma2)
        dp = testFrac2 * (i + xi) \
                + testFrac1 * (a + xa) \
                + testFrac3* (e + xe) \
                - self.gamma3 * p
        dr = self.gamma3 * p \
                + self.gamma2 * (i + xi) \
                + self.gamma1 * (e + xe) 

        
        self.setStates (s, e, a, i, xs, xe, xa, xi, p, r)
        return cat[module]((ds, de, da, di, dxs, dxe, dxa, dxi, dp, dr))

    def setStates (self, s, e, a, i, xs, xe, xa, xi, p, r) : 
        self.s  = s
        self.e  = e
        self.a  = a
        self.i  = i
        self.xs = xs
        self.xe = xe
        self.xa = xa
        self.xi = xi
        self.p  = p
        self.r  = r

    def addCrossTerms (self, dx, module=np) : 
        ds, de, da, di, dxs, dxe, dxa, dxi, dp, dr = dx.reshape((-1, self.bins))

        ds[1]  += (self.sIn  - self.sOut)
        de[1]  += (self.eIn  - self.eOut)
        da[1]  += (self.aIn  - self.aOut)
        di[1]  += (self.iIn  - self.iOut)
        dr[1]  += (self.rIn  - self.rOut)

        #pdb.set_trace()
        return cat[module]((ds, de, da, di, dxs, dxe, dxa, dxi, dp, dr))

    def timeUpdate (self, x, t, module=np) : 
        dx = self.dx(x, t, module)
        return x + dx

def linearApprox (fn, x0, T) : 
    out = [x0]
    x = x0
    for t in range(T) : 
        dx = fn(x, t)
        x = x + fn(x, t)
        out.append(x)
    return np.array(out)

if __name__ == "__main__" :
    with open('./Data/beta.json') as fd : 
        betas = json.load(fd)
    transportMatrix = np.loadtxt('./Data/transportMatrix.csv', delimiter=',')
    mortality = [getAgeMortality(s) for s in STATES]
    statePop  = [getStatePop(s) for s in STATES]
    data = [getData(s) for s in STATES]
    model = IndiaModel(transportMatrix, betas, statePop, mortality, data) 
    x0 = []
    for Nbar in statePop : 
        N_ = deepcopy(Nbar)
        E0 = [0, 10, 0]
        A0 = [0, 10, 0]
        I0 = [0, 10, 0]
        ZE = [0, 0, 0]
        N_[1] -= 30
        x = [*N_, *E0, *A0, *I0, *ZE, *ZE, *ZE, *ZE, *ZE, *ZE]
        x0.extend(x)
    x0 = np.array(x0)
    results = linearApprox(model.dx, x0, 50)
    # results = results.T.reshape((len(STATES), 30, -1))
    # for r, s in zip(results, STATES) : 
    #     statePlot(r.T, s, Date('29 Feb'), 3)
