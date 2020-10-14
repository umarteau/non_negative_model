import torch
import numpy as np
import scipy.special as scp

torch.set_default_dtype(torch.float64)






#################################################################################################
#Kernels
#################################################################################################

#Gaussina kernel


def gaussianKernel(sigma):
    def aux(sigma,x1,x2):
        if x1.ndim == 1:
            x1 = x1.view((x1.size(0),1))
        if x2.ndim == 1:
            x2 = x2.view((x2.size(0),1))
        x1_norm =  (x1*x1).sum(1)
        x2_norm = (x2*x2).sum(1)
        try:
            dist = x1 @ x2.t()
        except RuntimeError:
            torch.cuda.empty_cache()
            dist = x1 @ x2.t()
        
        del x2
        del x1
        dist *= -2
        dist += x1_norm.unsqueeze_(1).expand_as(dist)
        del x1_norm
        dist += x2_norm.unsqueeze_(0).expand_as(dist)
        del x2_norm
        dist *= -1/(2*sigma**2)
        dist.clamp_(min = -30,max = 0)
        dist.exp_()
        return dist

    def auxaux(x1,x2):
        return aux(sigma,x1,x2)
    return auxaux

    
    

def integrateGuaussianVector(sigma,base = '1',mu_base = None,eta_base = None):
    def aux1(sigma,x):
        if x.ndim == 1:
            d=1
        else:
            d = x.size(1)
        n=x.size(0)
        c = np.exp(d*np.log(2*np.pi*sigma**2)/2)
        res = torch.ones((n,1),dtype = x.dtype,device = x.device)
        return c*res
    def aux2(sigma,mu,eta,x):
        if x.ndim == 1:
            d=1 
            norm = x*x
        else:
            d = x.size(1)
            norm = (x*x).sum(1)
        c1 = 0.5*d*np.log(sigma**2/(sigma**2 + eta**2))
        c2 = 0.5/(sigma**2 + eta**2)
        norm *= -c2
        norm  += c1 
        norm.clamp_(min = -30)
        norm.exp_()
        return norm
    if base == '1':
        def aux11(x):
            return aux1(sigma,x)
        return aux11
    elif base == 'gaussian':
        def aux22(x):
            return aux2(sigma,mu_base,eta_base,x)
        return aux22
    else:
        raise NameError("base unknown")
        
        
def integrateGuaussianMatrix(sigma,base = '1',mu_base = None,eta_base = None):
    def aux1(sigma,x1,x2):
        if x1.ndim == 1:
            d = 1
            x1 = x1.view((x1.size(0),1))
        if x2.ndim == 1:
            x2 = x2.view((x2.size(0),1))
        d = x1.size(1)
        c = d*np.log(np.pi*sigma**2)/2
        x1_norm =  (x1*x1).sum(1)
        x2_norm = (x2*x2).sum(1)
        try:
            dist = x1 @ x2.t()
        except RuntimeError:
            torch.cuda.empty_cache()
            dist = x1 @ x2.t()
        
        del x2
        del x1
        dist *= -2
        dist += x1_norm.unsqueeze_(1).expand_as(dist)
        del x1_norm
        dist += x2_norm.unsqueeze_(0).expand_as(dist)
        del x2_norm
        dist *= -1/(4*sigma**2)
        dist.clamp_(min = -30,max = 0)
        dist+= c
        dist.exp_()
        return dist
        
    def aux2(sigma,mu,eta,x1,x2):
        if x1.ndim == 1:
            x1 = x1.view((x1.size(0),1))
        if x2.ndim == 1:
            x2 = x2.view((x2.size(0),1))
        d = x1.size(1)
        c1 = 0.5*d*np.log(sigma**2/(sigma**2 + 2*eta**2))
        c2 = 1/sigma**2
        c3 = 2*eta**2/(sigma**2 + 2*eta**2)
        if type(mu) != torch.Tensor:
            mu = torch.tensor(mu).view(d)
        else:
            mu = mu.view(d)
        mug = mu.to(x1.device)
        x1 -= mug.unsqueeze(0).expand_as(x1)
        x2 -= mug.unsqueeze(0).expand_as(x2)
        x1_norm =  (x1*x1).sum(1)
        x2_norm = (x2*x2).sum(1)
        try:
            dist = x1 @ x2.t()
        except RuntimeError:
            torch.cuda.empty_cache()
            dist = x1 @ x2.t()
        
        del x2
        del x1
    
        dist *= -0.5*c3
        dist += 0.5*(1-0.5*c3)*x1_norm.unsqueeze_(1).expand_as(dist)
        del x1_norm
        dist += 0.5*(1-0.5*c3)*x2_norm.unsqueeze_(0).expand_as(dist)
        del x2_norm
        dist *= - c2
        dist += c1
        dist.clamp_(min = -30,max = 0)
        dist.exp_()
        return dist
    if base == '1':
        def aux11(x1,x2):
            return aux1(sigma,x1,x2)
        return aux11
    elif base == 'gaussian':
        def aux22(x1,x2):
            return aux2(sigma,mu_base,eta_base,x1,x2)
        return aux22
    else:
        raise NameError("base Unknown")






    

##################################################
#Exponetial 
########


def expoKernel(sigma):
    def aux(sigma,x1,x2):
        if x1.ndim == 1:
            x1 = x1.view((x1.size(0),1))
        if x2.ndim == 1:
            x2 = x2.view((x2.size(0),1))
        x1_norm =  (x1*x1).sum(1)
        x2_norm = (x2*x2).sum(1)
        try:
            dist = x1 @ x2.t()
        except RuntimeError:
            torch.cuda.empty_cache()
            dist = x1 @ x2.t()
        
        del x2
        del x1
        dist *= -2
        dist += x1_norm.unsqueeze_(1).expand_as(dist)
        del x1_norm
        dist += x2_norm.unsqueeze_(0).expand_as(dist)
        del x2_norm
        dist.sqrt_()
        dist *= -1/(sigma)
        dist.clamp_(min = -30,max = 0)
        dist.exp_()
        return dist
    def auxaux(x1,x2):
        return aux(sigma,x1,x2)
    return auxaux

    
    

def integrateExpoVector(sigma,base = '1',mu_base = None,eta_base = None):
    def aux1(sigma,x):
        if x.ndim == 1:
            d=1
        else:
            d = x.size(1)
        n=x.size(0)
        c = 2*np.exp(d*np.log(np.pi)/2 + d*np.log(sigma) +scp.gammaln(d) - scp.gammaln(d/2))
        res = torch.ones((n,1),dtype = x.dtype,device = x.device)
        return c*res
    def aux2(sigma,mu,eta,x):
        def aux2aux(sig,xx):
            t1 = 1/(4*sig**2) -xx/sig  + torch.log(torch.erfc(1/(2*sig) - xx))
            t2 = 1/(4*sig**2) +xx/sig  + torch.log(torch.erfc(1/(2*sig) + xx))
            res = torch.exp(t1)+torch.exp(t2)
            res *= 0.5
            return res
        if x.ndim == 1:
            x = x.view((x.size(0),1))
        d = x.size(1)
        if d > 1:
            raise NameError("error, d > 1 in integration of exponential kernel")
        if type(mu) != torch.Tensor:
            mu = torch.tensor(mu).view(d)
        else:
            mu = mu.view(d)
        sig = sigma/(np.sqrt(2)*eta)
        xx =(x - mu.unsqueeze(0))/(np.sqrt(2)*eta)
        return aux2aux(sig,xx)
    if base == '1':
        def aux11(x):
            return aux1(sigma,x)
        return aux11
    elif base == 'gaussian':
        def aux22(x):
            return aux2(sigma,mu_base,eta_base,x)
        return aux22
    else:
        raise NameError("base unknown")
        
def integrateExpoMatrix(sigma,base = '1',mu_base = None,eta_base = None):
    def aux1(sigma,x1,x2):
        if x1.ndim == 1:
            x1 = x1.view((x1.size(0),1))
        
        d = x1.dim(1)
        if d>1:
            raise NameError("error : integration for dimension d > 1 in exponential kernel")
        if x2.ndim == 1:
            x2 = x2.view((x2.size(0),1))
        x1_norm =  (x1*x1).sum(1)
        x2_norm = (x2*x2).sum(1)
        try:
            dist = x1 @ x2.t()
        except RuntimeError:
            torch.cuda.empty_cache()
            dist = x1 @ x2.t()
        
        del x2
        del x1
        dist *= -2
        dist += x1_norm.unsqueeze_(1).expand_as(dist)
        del x1_norm
        dist += x2_norm.unsqueeze_(0).expand_as(dist)
        del x2_norm
        dist.sqrt_()
        termexp = -dist/sigma
        termexp.clamp_(min = -30,max = 0)
        termexp.exp_()
        return (sigma + dist)*termexp
    
   
    def aux2(sigma,mu,eta,x1,x2):
        def aux2aux(sigma,x,y):
            x,y = torch.min(x,y),torch.max(x,y)
            t1 = -(x+y)/sigma + 1/sigma**2 +torch.log(torch.erfc(1/sigma-x))
            t2 = (x-y)/sigma + torch.log(torch.erf(y)-torch.erf(x))
            t3 = (x+y)/sigma + 1/sigma**2 + torch.log(torch.erfc(1/sigma+y))
            res = torch.exp(t1) + torch.exp(t2) + torch.exp(t3)
            res*= 0.5
            return res
        n = x1.size(0)
        m = x2.size(0)
        if x1.ndim > 1 and x1.size(1) > 1:
            raise NameError("error : d > 1 in integration of exponential kernel ")
        if type(mu) != torch.Tensor:
            mu = torch.tensor(mu).view(1)
        else:
            mu = mu.view(1)
        if x1.ndim != 1:
            x1 = x1.view((n,))
        if x2.ndim != 1:
            x2 = x2.view((m,))
        x11 = (x1 - mu)/(np.sqrt(2)*eta)
        x22 = (x2 - mu)/(np.sqrt(2)*eta)
    
        x11 = x11.unsqueeze_(1).expand((n,m))
        x22 = x22.unsqueeze_(0).expand((n,m))
    
        sig = sigma/(np.sqrt(2)*eta)
        mask = (x11<x22)
    
        return mask*aux2aux(sig,x11,x22) + (~mask)*aux2aux(sig,x22,x11)
        
    
    if base == '1':
        def aux11(x1,x2):
            return aux1(sigma,x1,x2)
        return aux11
    elif base == 'gaussian':
        def aux22(x1,x2):
            return aux2(sigma,mu_base,eta_base,x1,x2)
        return aux22
    else:
        raise NameError("unknown base")



##################################################################################################
#going to GPU 
##################################################################################################
    
    
    
    
    

def produceDU(useGPU = False):
    if useGPU:
        def aux(x):
            return x.to('cpu')
        def aux2(x):
            return x.to('cuda')

        return (aux,aux2)
    else:
        def aux(x):
            return x
        return (aux,aux)

 
    
        
def blockKernComp(A, B, kern, useGPU = False, nmax_gpu = None):
    
    
    if isinstance(B,type(None)):
        return(blockKernCompGPUSymmetric(A, kern, useGPU = useGPU, nmax_gpu = nmax_gpu))
    if A.ndim == 1:
        A = A.view((A.size(0),1))
    if B.ndim == 1:
        B = B.view((B.size(0),1))

    if useGPU:
        if isinstance(nmax_gpu,type(None)):
            nmax_gpu = min(A.size(0),B.size(0))
    
        if nmax_gpu > A.size(0) and A.size(0) <= B.size(0):
            nmaxA = A.size(0);
            nmaxB = nmax_gpu
        elif nmax_gpu > B.size(0):
            nmaxB = B.size(0)
            nmaxA = nmax_gpu
        else:
            nmaxA =nmax_gpu
            nmaxB = nmax_gpu

    else:
        nmaxA = A.size(0)
        nmaxB = B.size(0)


    download, upload = produceDU(useGPU)

    blkA = int(np.ceil(A.size(0)/nmaxA))
    As = np.ceil(np.linspace(0, A.size(0), blkA + 1)).astype(int)
    
    blkB = int(np.ceil(B.size(0)/nmaxB))
    Bs = np.ceil(np.linspace(0, B.size(0), blkB + 1)).astype(int)

    if blkA == 1 and blkB == 1:
        M = download(kern(upload(A), upload(B)))
    else:

        M = torch.zeros(A.size(0), B.size(0),dtype = torch.float64)

        for i in range(blkA):
            C1 = upload(A[As[i]:As[i+1],:])
            for j in range(blkB):
                C2 = upload(B[Bs[j]:Bs[j+1], :])
                M[As[i]:As[i+1], Bs[j]:Bs[j+1]] = download(kern(C1,C2))
                del C2
            del C1
    return M


def blockKernCompGPUSymmetric(A, kern, useGPU = False, nmax_gpu = None):
    if A.ndim == 1:
        A = A.view((A.size(0),1))
    if useGPU:
        if isinstance(nmax_gpu,type(None)):
            nmax =A.size(0)
        else:
            nmax = nmax_gpu
    else:
        nmax = A.size(0)
    
    download, upload = produceDU(useGPU)
    
    blkA = int(np.ceil(A.size(0)/nmax))
    As = np.ceil(np.linspace(0, A.size(0), blkA + 1)).astype(int)
    
    if blkA == 1:
        uA = upload(A)
        Mg = kern(uA, uA)
        M = download(Mg)
        del Mg
    else:

        M = torch.zeros(A.size(0), A.size(0),dtype = torch.float64);
        for i in range(blkA):
            C1 = upload(A[As[i]:As[i+1],:])
            M[As[i]:As[i+1], As[i]:As[i+1]] = download(kern(C1,C1))
            for j in range(i+1,blkA):
                C2 = upload(A[As[j]:As[j+1], :])
                Kr = kern(C1,C2)
                M[As[i]:As[i+1], As[j]:As[j+1]] = download(Kr)
                del Kr
        del C1
        del C2
        for i in range(blkA):
            for j in range(i+1,blkA):
                M[As[j]:As[j+1], As[i]:As[i+1]] = M[As[i]:As[i+1], As[j]:As[j+1]].t()
    return M