import numpy as np
import torch

torch.set_default_dtype(torch.float64)


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




def randomApproach(A,l,q=6,mu = 0,useGPU = False):
    download, upload = produceDU(useGPU)
    n = A.size(0)
    Omega = upload(torch.randn((n,l)))
    Ag = upload(A)
    Q,_ = torch.qr(Ag@Omega,some = True)
    for i in range(q):
        Q,_ = torch.qr(Ag@Q,some = True)
    s,V = torch.symeig(Q.T @ Ag @Q,eigenvectors = True)
    Qt = Q@V
    del Q 
    del V
    del Ag
    del Omega
    return (download(Qt@ torch.diag(s.clamp(min = mu+ 1e-15)-mu) @Qt.T))

def sureApproach(A,mu = 0,useGPU = False):
    download, upload = produceDU(useGPU)
    s,V = torch.symeig(upload(A),eigenvectors = True)
    return (download(V@ torch.diag(s.clamp(min = mu+ 1e-15)-mu) @V.T))
    
    
def mixedApproach(A,l,q=6,mu = 0,useGPU = False):
    n = A.size(0)
    if n <= l:
        return sureApproach(A,mu = mu,useGPU = useGPU)
    else:
        return randomApproach(A,l,q=q,mu = mu,useGPU = useGPU)
        

def topEV(A):
    s,_ = torch.symeig(A@A.T)
    return torch.sqrt(torch.max(s))


def softMaxRandom(A,l,q=6,mu = 0,useGPU = False):
    download, upload = produceDU(useGPU)
    n = A.size(0)
    Omega = upload(torch.randn((n,l)))
    Ag = upload(A)
    Q,_ = torch.qr(Ag@Omega,some = True)
    for i in range(q):
        Q,_ = torch.qr(Ag@Q,some = True)
    s,V = torch.symeig(Q.T @ Ag @Q,eigenvectors = True)
    Qt = Q@V
    del Q 
    del V
    del Ag
    del Omega
    return (download(Qt@ torch.diag(s.clamp(min = mu)+ s.clamp(max = -mu )) @Qt.T))

def softMaxSure(A,mu = 0,useGPU = False):
    download, upload = produceDU(useGPU)
    s,V = torch.symeig(upload(A),eigenvectors = True)
    return (download(V@ torch.diag(s.clamp(min = mu)+ s.clamp(max = -mu )) @V.T))

def softMaxMixed(A,l,q=6,mu = 0,useGPU = False):
    if mu == 0:
        return A
    n = A.size(0)
    if n <= l:
        return softMaxRandom(A,l,q=q,mu = mu,useGPU = useGPU)
    else:
        return softMaxSure(A,mu = mu,useGPU = useGPU)
    
def traceNormRandom(A,l,q=6,useGPU = False):
    download, upload = produceDU(useGPU)
    n = A.size(0)
    Omega = upload(torch.randn((n,l)))
    Ag = upload(A)
    Q,_ = torch.qr(Ag@Omega,some = True)
    for i in range(q):
        Q,_ = torch.qr(Ag@Q,some = True)
    s,V = torch.symeig(Q.T @ Ag @Q,eigenvectors = True)
    return torch.abs(download(s)).clamp(min = 1e-10).sum()


def traceNormSure(A,useGPU = False):
    download, upload = produceDU(useGPU)
    s,V = torch.symeig(upload(A),eigenvectors = True)
    return torch.abs(download(s)).clamp(min = 1e-10).sum()

def traceNormMixed(A,l,q=6,useGPU = False,isPositive = False):
    if isPositive:
        return torch.trace(A)
    n = A.size(0)
    if n <= l:
        return traceNormSure(A,useGPU = useGPU)
    else:
        return traceNormRandom(A,l,q=q,useGPU = useGPU)