import numpy as np
import emcee
import corner
import math
import matplotlib.pyplot as plt

data = np.genfromtxt("OHD Data.csv",delimiter = ',',skip_header = 1)

z = data[:,0]
H = data[:,1]
H_err = data[:,2]
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
    model =np.sqrt(a+b*(1+z)**3)
   # model[int i: model[i] < 0] = 0
   # print(model, hz)
  #   print()
    diff = hz - model
    sterm = (diff/hz_err)**2
    sterm = np.sum(sterm)

    return -0.5*(fterm + sterm)
    
def posterior(val):
#    print(log_likelihood(val))
    
    return log_prior(val) + log_likelihood(val)

walker = 100
