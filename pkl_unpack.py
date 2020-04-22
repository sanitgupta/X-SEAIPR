import pickle
import pdb
import numpy as np

with open('series_3a.pkl', 'rb') as fd : 
    seriesOfSeries = pickle.load(fd)

with open('var_3a.pkl', 'rb') as fd : 
    seriesOfVariances = pickle.load(fd)

pdb.set_trace()



