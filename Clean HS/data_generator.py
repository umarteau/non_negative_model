import numpy as np
import torch
import matplotlib.pyplot as plt
import time

mu_ex = lambda x : 2*(torch.exp(-30*(x - 0.25)**2)+torch.sin(np.pi*x**2))
sigma_ex = lambda x : torch.exp((torch.sin(2*np.pi*x))/2)

def sigma_aux(n):
    def aux(x):
        return ((n*x - torch.floor(n*x)) > 0.5).double()
    return aux

#Plotting the function
def plt_mu_sigma(mu,sigma):
    n=100
    x = torch.linspace(0,1,n)
    plt.plot(x,mu(x),'r',label = "mu")
    plt.plot(x,mu(x)+1.96*sigma(x),'g',linestyle ='--')
    plt.plot(x,mu(x)-1.96*sigma(x),'g',linestyle = '--')
    plt.legend()
    plt.show()
    return()

###################
#Going to and from natural parameters
###################

def ms_to_nat(mu,sigma):
    return(mu/sigma**2,1/sigma**2)

def nat_to_ms(eta,la):
    return(eta/la,1/np.sqrt(la))


#Plotting natural parameters 
def plt_nat(mu,sigma):
    n=100
    x = torch.linspace(0,1,n)
    m,s = mu(x),sigma(x)
    eta,la = ms_to_nat(m,s)
    plt.figure()
    plt.plot(x,eta,'b',label = "eta")
    plt.legend()
    plt.show()
    plt.figure()
    plt.plot(x,la,'r',label = "lambda")
    plt.legend()
    plt.show()
    return()


def generateData(mu,sigma,n,plot =True):
    x = torch.linspace(0,1,n)
    m,s = mu(x),sigma(x)
    leps = np.ones((n,))
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

