import numpy as np
import torch
import sys
sys.path.append("../utils")
import utils_kernels as KT
import matplotlib.pyplot as plt 
import time
import ppm as ppm
import pandas as pd

dtype = torch.float64

torch.set_default_dtype(torch.float64)




mu_ex = lambda x : 2*(torch.exp(-30*(x - 0.25)**2)+torch.sin(np.pi*x**2))
sigma_ex = lambda x : torch.exp((torch.sin(2*np.pi*x)))


sigma_ex_2 = lambda x : 10*torch.exp(-(1/(x**2).clamp(min = 1e-15)).clamp(max = 30))


def doubleGenerator(mu,sigma,n,xsample='uniform',nbumps =1,r = 0.5,plot = True):
    if xsample== 'uniform':
        x = torch.rand(n)
        density= lambda x : ((x>0)*(x<1)).double()
    if xsample == 'gaussian':
        sigmas =0.2
        mus = torch.tensor([0.5])
        x = torch.randn(n)
        x *= sigmas
        x+= mus
        def density(x):
            return  torch.exp(-((x-mus)**2)/(2*sigmas**2))/torch.sqrt(2*torch.tensor(np.pi)*sigmas**2)
    if xsample == 'square':
        x = torch.rand(n)
        v = torch.linspace(0,1,nbumps*2+2)
        l = v[1]-v[0]
        def density(x):
            y = torch.floor(((2*nbumps+1)*x))%2
            return (2*nbumps+1)*((y == 1)*(x>0)*(x<1)).double()/nbumps
        def sample(np):
            i = 2*torch.randint(nbumps,(np,))+1
            rs = l*torch.rand(np)
            return v[i]+rs
        x = sample(n)
    if xsample == 'hole_center':
        xx = torch.rand(n)
        t = 2/3 + r*(1-2/3)
        a1 = 2/(3*t)
        b1 = 0
        b11 = 1 - 2/(3*t)
        a2 = 1/(3*(1-t))
        b2 = (2-3*t)/(6-6*t)
        x = (a1*xx+b1)*(xx< (t/2)) + (a2*xx+b2)*(xx> (t/2))*(xx < (1-t/2)) + (a1*xx+b11)*(xx> (1-t/2))
        def density(x):
            return (t/2)*(x< (1/3)).double() + (3*(1-t))*(x> (1/3))*(x < (2/3)).double() + (t/2)*(x> 2/3).double()
    if xsample == 'weird':
        xx = torch.rand(n)
        x = xx.clamp(min = 0.5)
        def density(xp):
            return None
    if xsample == 'weird2':
        xx = torch.rand(n)
        x = (xx < 0.25)*xx.clamp(max = 0) + (xx > 0.25)*xx.clamp(min = 0.5)
        def density(xp):
            return None
    if xsample == 'weird3':
        xx = torch.rand(n)
        x = (xx < r)*(0.25*xx/r) + (xx > r)*((0.5 + r)/(1-r)*xx + 1-(0.5 + r)/(1-r)).clamp(min = 0.5)
        def density(xp):
            return None
    if xsample == 'doubleweird':
        xx = torch.rand(n)
        x = (xx > 0.5)*((2/3+2*r)*xx + 1 - (2/3+2*r)).clamp(min = 2/3) + (xx < 0.5)*((2/3+2*r)*xx).clamp(max = 1/3)
        def density(xp):
            return None
    if xsample == 'uniformhalf':
        xx = torch.rand(n)
        x = 0.5+0.5*xx
        def density(xp):
            return None
        
    m,s = mu(x),sigma(x)
    leps = torch.ones((n,))
    for i in range(n):
        leps[i] = s[i]*np.random.randn()
    y = m + leps
    if plot:
        plt.figure()
        plt.scatter(x,y,marker = '+')
        xp = torch.linspace(0,1,100)
        plt.plot(xp,mu(xp),'r')
        plt.show()
        
    return(x.view(n,1),y,density)
    
        


def generateDataGaussian(mu,sigma,n,plot =True,x=None):
    if isinstance(x,type(None)):
        x = torch.linspace(0,1,n)
    m,s = mu(x),sigma(x)
    leps = torch.ones((n,))
    for i in range(n):
        leps[i] = s[i]*np.random.randn()
    y = m + leps
    if plot:
        plt.figure()
        plt.scatter(x,y,marker = '+')
        xp = torch.linspace(0,1,100)
        plt.plot(xp,mu(xp),'r')
        plt.show()
        
    return(x.view(n,1),y)

def approximateGaussianQuantile(tau_l):
    s = np.random.randn(100000)
    res = np.quantile(s,tau_l)
    return res
    
def get_quantiles_HS(mu,sigma,xp,tau_l):
    q_vals = approximateGaussianQuantile(tau_l)
    res = torch.zeros((len(tau_l),xp.size(0)))
    m,s = mu(xp),sigma(xp)
    for t in range(len(tau_l)):
        res[t,:] = m+q_vals[t]*s
    return res

def plt_compare(xp,f_p,f_0,tau_l,title = "",xtrain=None,ytrain=None):
    T = f_p.size(0)
    plt.figure()
    for t in range(T):
        c = np.random.rand(3)
        plt.plot(xp,f_p[t,:],c = c , label = "{}".format(tau_l[t]))
        plt.plot(xp,f_0[t,:],c=c, linestyle = 'dashed')
    if not(isinstance(xtrain,type(None))):
        plt.scatter(xtrain,ytrain,color='r',marker='+',lw=2)
    plt.title("Quantile regression")
    plt.legend()
    plt.savefig(title+".png")
    plt.show()
    
    
    
        