import numpy as np
import corner
import matplotlib.pyplot as plt
import emcee
from scipy.integrate import quad

Ob = 0.05
Om = 0.3
Ol = 0.7
eta_f = 0.39     #Assumed in Mantz's Paper
eta_m = 1.065
K0 = 0.956      #Taken from Applegate 2016 paper on Weak Lensing
K0_err = 0.082
#Data is for LCDM Model
data = np.genfromtxt('Updated data.csv',delimiter = ',',skip_header = True)

z = data[:,1]
fg = data[:,6]
fg_err = data[:,7]
M25 = data[:,8]/3
M25_err = data[:,9]/3

integral = np.empty(len(z))
E = lambda z : 1/np.sqrt(Om*(1 + z)**3 + Ol)

for i in range(len(z)):
    tmp = quad(E,0,z[i])
    integral[i] = tmp[0]

da_ratio = np.log(1 + z)/(integral)
h_ratio = (1 + z)/(np.sqrt(Om*(1 + z)**3 + Ol))
ratio = (h_ratio**eta_f) * (da_ratio)**(eta_f - 1.5)


M25_rh = M25*(h_ratio**eta_m) * ((da_ratio)**(eta_m - 1))

fg_rh = fg*ratio
fg_rh_err = fg_rh*np.sqrt((K0_err/K0)**2 + (fg_err/fg)**2)

plt.scatter(z,fg_rh)
plt.errorbar(z,fg_rh,yerr = fg_rh_err,fmt = 'o')
plt.scatter(z,fg)
plt.errorbar(z,fg,yerr = fg_err,fmt = 'o')
plt.legend([r"$R_h = ct$",r"$LCDM$"])
plt.xlabel('z')
plt.ylabel(r"$f_{gas}$")
plt.title(r"Gas Mass fraction plot for $R_h = ct$ universe")
plt.show()

#Using Mantz et al model
def log_prior(val):
    g0,g1,alpha,scat = val
    
    if 0 <= g0 <= 1 and -0.2 <= g1 <= 0.1 and -1 <= alpha <= 1 and -5 <= scat <= 0:
        return 0

    else:
        return -np.inf

def log_like(val):
    g0,g1,alpha,scat = val

    scat = 10**(scat)

    err = fg_rh_err**2 + scat**2
    ft = np.sum(np.log(err))

    model = K0*g0*(1 +g1*z)*((M25_rh)**alpha)*(Ob/Om)

    diff = fg_rh - model
    st = (diff/err)**2
    st = np.sum(st)

    return -0.5*(ft + st)

def log_post(val):
    return log_prior(val) + log_like(val)

#MCMC Starts
walkers = 50
dim = 4
g0i = np.random.uniform(0.7,1,walkers)
g1i = np.random.uniform(-0.1,0.1,walkers)
#k0i = np.random.uniform(0,2,walkers)
#k1i = np.random.uniform(-0.1,0.1,walkers)
alphai = np.random.uniform(-1,1,walkers)
scati = np.random.uniform(-5,0,walkers)

sampler = emcee.EnsembleSampler(walkers,dim,log_post)

steps = 2000
burnin = 1000
theta = np.array([g0i,g1i,alphai,scati]).T

sampler.run_mcmc(theta,steps + burnin)
lst = sampler.get_chain(flat = True,discard = burnin)

corner.corner(lst,show_titles = 1,labels = [r"$\gamma_0$",r"$\gamma_1$",r"$\alpha$",r"$log(\sigma)$"])
plt.show()

#k0 = 1.27
#k1 = -0.07
# Luru was sleeping at this point
g0 = 0.79
g1 = -0.11
alpha = -0.01

best_model = g0*(1 +g1*z)*((M25_rh)**alpha)*(Ob/Om)
plt.scatter(z,fg)
plt.errorbar(z,fg,yerr = fg_err,fmt = 'o')
plt.plot(z,best_model,color = 'r')
plt.xlabel('z')
plt.ylabel(r'$f_{gas}$')
plt.legend(['observations','best fit curve'])
plt.title(r'Result for updated data assuming $LCDM$ model')
plt.show()