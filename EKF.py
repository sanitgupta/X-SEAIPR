import torch
from tqdm import tqdm
from torch.autograd import Variable
from functools import partial
from Util import *
from Model import *
from Simulate import *
import numpy as np
#import matplotlib.pyplot as plt
import scipy.linalg
import scipy.integrate
import pdb
import numdifftools as nd
import multiprocessing as mp

class Grad () : 
    
    def __init__ (self, f, x): 
        self.f = f
        self.x = x
        self.eta = 1e-2
        self.n = np.size(x)

    def __call__ (self, i) :
        dx = np.zeros(self.n)
        dx[i] = self.eta
        df = (self.f(self.x + dx) - self.f(self.x)) / self.eta
        return df

def getProcessJacobians(f, x):
    n = np.size(x)
    g = Grad(f, x)
    with mp.Pool() as p : 
        grads = p.map(g, range(n))
    jac = np.vstack(grads)
    return jac.T

def sin(x) :
    if torch.is_tensor(x) : 
        a, b = x
        return torch.sin(torch.stack([a+b, a-b]))
    else :
        return np.sin(x)

def step(x) :
    A = np.array([[1., 3.], [-1., 2.]])
    if torch.is_tensor(x) : 
        return torch.from_numpy(A) @ x
    else :
        return A @ x

def extendedKalmanFilter (updateStep, x0, P0, Q, H, R, Z, tStart, tEnd) :
    """
    All H, R, z are functions of the date. 
    """
    xPrev = x0
    PPrev = P0
    xs = [x0]
    Ps = [P0]

    d1 = Date('27 Apr')
    #print(tEnd.date)
        
    for date in tqdm(DateIter(tStart, tEnd)) :
        print(np.sum(xPrev))
        # Time update
        i = date - tStart
        xtMinus = scipy.integrate.odeint(updateStep,xPrev,[i,i+1],args=(tStart,))
        xtMinus = xtMinus[1]
        A = getProcessJacobians(partial(updateStep, delta_t=i, startDate=tStart), xPrev)
        m = np.shape(A)
        #A1 = nd.Jacobian(partial(updateStep, delta_t=i, startDate=tStart))(xPrev)
        phi = scipy.linalg.expm(A)
        #phi1 = scipy.linalg.expm(A1)
        PMinus = phi @ PPrev @ phi.T + Q

        # Measurement update
        h = H(date+1)
        r = R(date+1)
        z = Z(date+1)

        if h.size > 0 : 
            K = PMinus @ h.T @ np.linalg.inv(h @ PMinus @ h.T + r)
            xt = xtMinus + K @ (z - h @ xtMinus)
            Pt = (np.eye(PPrev.shape[0]) - K @ h) @ PMinus

            xt[xt < 0] = np.maximum(xt[xt < 0], np.maximum(0, xPrev[xt < 0])) # Shameless Hack

        else : 

            xt = xtMinus
            Pt = PMinus

        #if date - Date('16 Apr') == 0:
            #pdb.set_trace()

        xPrev = xt
        PPrev = Pt

        xs.append(xt)
        Ps.append(Pt)

        #print("EKF Date: "+str(date.date))

    return np.stack(xs), Ps

if __name__ == "__main__" : 
    print(getJacobian(sin, torch.tensor([0., 1.])))
