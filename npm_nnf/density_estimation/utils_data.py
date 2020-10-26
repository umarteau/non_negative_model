import torch
import numpy as np
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.NOTSET,format='%(asctime)s  %(name)s %(levelname)s %(message)s')
import pickle

torch.set_default_dtype(torch.float64)



class Dataset(object):
    def __init__(self,X = None,y=None):
        self.X = X
        self.y = y

class DatasetDensity(Dataset):
    def __init__(self,X=None,y=None,density = None,sampler = None,params = {}):
        self.X = X
        self.y = y
        self.density = density
        self.sampler = sampler
        self.params = params

    def apply_density(self,x):
        return self.density(x=x,**self.params)


    def sample_from_density(self,n):
        return self.sampler(n=n,**self.params)




def density_gaussian_mixture(x = None,mu = 0.,sigma = 1.,n_mixture = 1,d = 1,p=None):
    if isinstance(x,type(None)):
        return None
    elif len(x) == 0:
        return torch.Tensor([])

    if not(isinstance(d,int)):
        if x.ndim == 1:
            d =1
        else:
            d = x.size(1)
    e1 = torch.zeros(d)
    e1[0] = 1

    if isinstance(mu,float):
        mu = mu*e1
    if isinstance(mu,list) and isinstance(mu[0],float):
        mu = [m*e1 for m in mu]

    if isinstance(mu,list) and isinstance(sigma,list):
        if len(mu) != len(sigma):
            raise Exception('mu and sigma are not of the same size')
        n_mixture = len(mu)

    elif isinstance(mu,list):
        n_mixture = len(mu)
        sigma = [sigma for k in range(n_mixture)]

    elif isinstance(sigma,list):
        n_mixture = len(sigma)
        if n_mixture > 2:
            raise Exception(f'sigma is of length {n_mixture} while only one value of mu specified')
        elif n_mixture == 1:
            mu = [mu]
        elif n_mixture == 2:
            mu = [mu,-mu]
        else:
            raise Exception('sigma is a list of no length')

    elif not(isinstance(sigma,list)) and not(isinstance(mu,list)):
        if n_mixture == 1:
            sigma = [sigma]
            mu = [mu]

        elif n_mixture == 2:
            sigma = [sigma,sigma]
            mu = [mu,-mu]

        else:
            raise Exception('n_mixture should be 1 or 2')
    if isinstance(p,type(None)):
        p = torch.ones(n_mixture)/n_mixture
    else:
        if not(isinstance(p,torch.Tensor)):
            p = torch.Tensor(p)
        p = p/p.sum()
    n_x = len(x)
    res = torch.zeros(n_x)
    if x.ndim == 1:
        x = x.unsqueeze(1)

    for k in range(n_mixture):
        sk = sigma[k]
        mk = mu[k]
        pk = p[k]
        valk = pk*torch.exp(-((x-mk.unsqueeze(0))**2).sum(1)/(2*sk**2) - d*np.log(2*np.pi*sk**2)/2)
        res += valk
    return res

def sampler_gaussian_mixture(n=0,mu = 0.,sigma = 1.,n_mixture = 1,d = 1,p=None):

    e1 = torch.zeros(d)
    e1[0] = 1

    if isinstance(mu,float):
        mu = mu*e1
    if isinstance(mu,list) and isinstance(mu[0],float):
        mu = [m*e1 for m in mu]

    if isinstance(mu,list) and isinstance(sigma,list):
        if len(mu) != len(sigma):
            raise Exception('mu and sigma are not of the same size')
        n_mixture = len(mu)

    elif isinstance(mu,list):
        n_mixture = len(mu)
        sigma = [sigma for k in range(n_mixture)]

    elif isinstance(sigma,list):
        n_mixture = len(sigma)
        if n_mixture > 2:
            raise Exception(f'sigma is of length {n_mixture} while only one value of mu specified')
        elif n_mixture == 1:
            mu = [mu]
        elif n_mixture == 2:
            mu = [mu,-mu]
        else:
            raise Exception('sigma is a list of no length')

    elif not(isinstance(sigma,list)) and not(isinstance(mu,list)):
        if n_mixture == 1:
            sigma = [sigma]
            mu = [mu]

        elif n_mixture == 2:
            sigma = [sigma,sigma]
            mu = [mu,-mu]

        else:
            raise Exception('n_mixture should be 1 or 2')
    if isinstance(p,type(None)):
        p = torch.ones(n_mixture)/n_mixture
    else:
        if not(isinstance(p,torch.Tensor)):
            p = torch.Tensor(p)
        p = p/p.sum()

    rs = torch.randn((n,d))

    mask = np.random.choice(list(range(n_mixture)),size = (n,),p=p.numpy())
    sigmas = torch.tensor([sigma[k] for k in mask])
    mus = torch.zeros((n,d))
    for i in range(n):
        mus[i,:] = mu[mask[i]]

    res = mus + sigmas.unsqueeze(1)*rs

    return res


def createDataSet(n=0,**params):
    x = sampler_gaussian_mixture(n=n,**params)
    y = density_gaussian_mixture(x=x,**params)
    res = DatasetDensity(X=x,y=y,density = density_gaussian_mixture,sampler=sampler_gaussian_mixture,params= params)
    return res


##############
#Old, normally can be removed
###############



def generateGaussian(mu,sigma,n):
    def density(x):
        return  torch.exp(-((x-mu)**2)/(2*sigma**2))/torch.sqrt(2*torch.tensor(np.pi)*sigma**2)
    def sample(np):
        rs = torch.randn(np)
        return mu+sigma*rs
    xplot = torch.linspace(mu-3*sigma,mu+3*sigma,200)
    plt.plot(xplot,density(xplot))
    res = sample(n)
    plt.scatter(res,density(res),marker = '+',color = 'r')
    return res,density


def generateGaussianD(mu,sigma,n,d):
    def density(x):
        return  torch.exp(-((x-mu.unsqueeze(0))**2).sum(1)/(2*sigma**2) - d*np.log(2*np.pi*sigma**2)/2)
    def sample(np):
        rs = torch.randn((np,d))
        return mu.unsqueeze(0)+sigma*rs
    #xplot = torch.linspace(mu-3*sigma,mu+3*sigma,200)
    #plt.plot(xplot,density(xplot))
    res = sample(n)
    #plt.scatter(res,density(res),marker = '+',color = 'r')
    return res,density



def generate2GaussianD(r,sigma,n,d):
    mu1,mu2 = torch.zeros(d),torch.zeros(d)
    mu1[0]+=r 
    mu2[0] -= r
    def density(x):
        r1 = torch.exp(-((x-mu1.unsqueeze(0))**2).sum(1)/(2*sigma**2) - d*np.log(2*np.pi*sigma**2)/2)
        r2 = torch.exp(-((x-mu2.unsqueeze(0))**2).sum(1)/(2*sigma**2) - d*np.log(2*np.pi*sigma**2)/2)
        return  0.5*(r1 + r2)
    def sample(np):
        mask = torch.randint(2,(np,))
        mask = mask.unsqueeze(1)
        rs = torch.randn((np,d))
        v1 = mu1.unsqueeze(0)+sigma*rs
        v2 = mu2.unsqueeze(0)+sigma*rs
        return mask*v1 + (1-mask)*v2
    #xplot = torch.linspace(mu-3*sigma,mu+3*sigma,200)
    #plt.plot(xplot,density(xplot))
    res = sample(n)
    nplot = 200
    xplot = torch.zeros((nplot,d))
    xplot[:,0] =  torch.linspace(-r-3*sigma,r + 3*sigma,nplot)
    plt.figure()
    plt.plot(xplot[:,0],density(xplot))
    plt.show()

    #plt.scatter(res,density(res),marker = '+',color = 'r')
    return res,density


def generate3GaussianD(r,sigma1,sigma2,n,d):
    mu1,mu2 = torch.zeros(d),torch.zeros(d)
    mu1[0]+=r 
    mu2[0] -= r
    def density(x):
        r1 = torch.exp(-((x-mu1.unsqueeze(0))**2).sum(1)/(2*sigma1**2) - d*np.log(2*np.pi*sigma1**2)/2)
        r2 = torch.exp(-((x-mu2.unsqueeze(0))**2).sum(1)/(2*sigma2**2) - d*np.log(2*np.pi*sigma2**2)/2)
        return  0.5*(r1 + r2)
    def sample(np):
        mask = torch.randint(2,(np,))
        mask = mask.unsqueeze(1)
        rs = torch.randn((np,d))
        v1 = mu1.unsqueeze(0)+sigma1*rs
        v2 = mu2.unsqueeze(0)+sigma2*rs
        return mask*v1 + (1-mask)*v2
    #xplot = torch.linspace(mu-3*sigma,mu+3*sigma,200)
    #plt.plot(xplot,density(xplot))
    res = sample(n)
    nplot = 200
    xplot = torch.zeros((nplot,d))
    xplot[:,0] =  torch.linspace(min(-r-3*sigma2,r-3*sigma1),max(r + 3*sigma1,-r+3*sigma2),nplot)
    plt.figure()
    plt.plot(xplot[:,0],density(xplot))
    plt.show()

    #plt.scatter(res,density(res),marker = '+',color = 'r')
    return res,density




def generateUniform(a,b,n):
    def density(x):
        return  (x>a).double()*(x<b).double()/(b-a)
    def sample(np):
        rs = torch.rand(np)
        return a+(b-a)*rs
    xplot = torch.linspace(a- (b-a)/2,b+ (b-a)/2,200)
    plt.plot(xplot,density(xplot))
    res = sample(n)
    plt.scatter(res,density(res),marker = '+',color = 'r')
    return res,density

def generateMultiple(nbumps,n):
    v = torch.linspace(0,1,nbumps*2+2)
    l = v[1]-v[0]
    def density(x):
        y = torch.floor(((2*nbumps+1)*x))%2
        return (2*nbumps+1)*((y == 1)*(x>0)*(x<1)).double()/nbumps
    def sample(np):
        i = 2*torch.randint(nbumps,(np,))+1
        rs = l*torch.rand(np)
        return v[i]+rs
    xplot = torch.linspace(-0.25,1.25,200)
    plt.plot(xplot,density(xplot))
    res = sample(n)
    plt.scatter(res,density(res),marker = '+',color = 'r')
    return res,density
    
    
    
def generateGaussianMixture(mu_l,sigma_l,n):
    M = len(mu_l)
    def density(x):
        res = torch.zeros(x.size(0))
        for m in range(M):
            sigma = sigma_l[m]
            mu = mu_l[m]
            res += 1/M*torch.exp(-((x-mu)**2)/(2*sigma**2))/torch.sqrt(2*torch.tensor(np.pi)*sigma**2)
        return res
    def sample(np):
        res = torch.randn(np)
        index = torch.LongTensor(np).random_(0, M)
        return torch.tensor(mu_l)[index] + torch.tensor(sigma_l)[index]*res
    xplot = torch.linspace(min(mu_l)-3*max(sigma_l),max(mu_l)+3*max(sigma_l),1000)
    plt.plot(xplot,density(xplot))
    res = sample(n)
    plt.scatter(res,density(res),marker = '+',color = 'r')
    return res,density
            
        
        
    