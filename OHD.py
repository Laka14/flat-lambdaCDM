import numpy as np
import pandas as pd
import emcee
import corner
import math
from matplotlib import pyplot as plt

data = np.genfromtxt("ohd.csv", delimiter = ',', skip_header= 1)

z = data[:,0]
hz = data[:,1]
hz_err = data[:,2]
#plt.scatter(z, hz, color = 'r')
#plt.errorbar(z, hz, hz_err, color = 'b', fmt = 'o')
#plt.show()


def log_prior(val):
    h0, ohmg = val
    if 60<=h0<=80 and 0<=ohmg<=1 :
        return 0
    else :
        return -np.inf
    
def log_likelihood(val):
    h0, ohmg = val
    a = (h0**2)*(1-ohmg)
    b = (h0**2)*(ohmg)
    fterm = np.sum(np.log(hz_err**2))
    model = np.sqrt(a+b*(1+z)**3)
   # improvemodel = model[np.logical_not(np.isnan(model))]
   # print(improvemodel, hz)
   # print()
   # remove NAN
    diff = hz - model
    sterm = (diff/hz_err)**2
    sterm = np.sum(sterm)

    return -0.5*(fterm + sterm)
    
def posterior(val):
#    print(log_likelihood(val))
    lp = log_prior(val)
    if not np.isfinite(lp):
        return -np.inf
    return log_prior(val) + log_likelihood(val)

#now sampling the factors for the plot
walkers = 50
dim = 2
h0i = np.random.uniform(60, 80, walkers)
ohmgi = np.random.uniform(0, 1, walkers)

sampler = emcee.EnsembleSampler(walkers,dim,posterior)

steps = 2000
burnin = 1000
theta = np.array([h0i, ohmgi]).T

sampler.run_mcmc(theta,steps + burnin)
lst = sampler.get_chain(flat = True,discard = burnin)
#plot
corner.corner(lst,show_titles = 1,labels = [r"$H_0$",r"$\Omega_m$"])
plt.show()
plt.savefig('plot.png')
