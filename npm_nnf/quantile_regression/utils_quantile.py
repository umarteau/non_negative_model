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


def add_l(a,b,c):
    if b == None:
        return None
    if a == None:
        if type(b) == torch.Tensor:
            return c*b
        else:
            try:
                return [add_l(None,bb,c) for bb in b]
            except:
                return c*b
    if type(b) == torch.Tensor:
        return a + c*b
    try:
        return [add_l(aa,b[i],c) for i,aa in enumerate(a)]
    except:
        return a+c*b

def scal_prod(a,b):
    if b == None:
        return 0
    if type(b) == torch.Tensor:
        return torch.sum(a*b)
    try:
        res = 0
        for i,aa in enumerate(a):
            res += scal_prod(aa,b[i]) 
        return res
    except:
        return torch.sum(a*b)
        
def minus_l(b):
    return add_l(None,b,-1)

def clone_l(a):
    if a == None:
        return None
    if type(a) == torch.Tensor:
        return a.clone()
    try:
        return [clone_l(aa) for aa in a]
    except:
        return a.clone()


############################################################################################
#Cholesky decompositions
############################################################################################

def produceDU(useGPU = False):
    if useGPU:
        return (lambda x : x.to('cpu'),lambda x : x.to('cuda')) 
    else:
        return (lambda x : x,lambda x : x)


def chol(M,useGPU = False):
    m = M.size(0)
    if useGPU:
        Mg = M.to('cuda')
        Tg = Mg.cholesky(upper = True)
        T = Tg.to('cpu')
        del Mg
        del Tg
        return T
    else:
        T = M.cholesky(upper = True)
        return T    
    
    
    
def createT(kern,C,useGPU = False):
    K = kern(C,None)
    m = K.size(0)
    eps = 1e-15*m
    K[range(m),range(m)]+= eps
    T = chol(K,useGPU = useGPU)
    del K
    return T

    
############################################################################################
#Solving triangular systems
############################################################################################

def tr_solve(x,T,useGPU = False,transpose = False):
    m = T.size(0)
    if useGPU:
        download,upload = produceDU(useGPU = useGPU)
        xg = upload(x)
        Tg = upload(T)
        resg = torch.triangular_solve(xg,Tg,upper \
                        = True,transpose = transpose)
        res = download(resg[0])
        del xg
        del Tg
        del resg
        torch.cuda.empty_cache()
        return res
    else:
        return torch.triangular_solve(x,T,upper \
                                                   = True,transpose = transpose)[0]
        
###############################################################################
#Linear models
###############################################################################



        

class LMK(object): 
    def __init__(self,sigma,x,la,kernel = 'gaussian',useGPU = False,nmax_gpu = None,centered = False,c = 0,positive = False):
        n = x.size(0)
        self.n = x.size(0)
        self.x = x
        self.smoothness = 1/la
        self.la = la
        if x.ndim == 1:
            d = 1
        else:
            d = x.size(1)
        self.d = d
        self.useGPU = useGPU
        
        if kernel == 'expo':
            kernel_fun = KT.expoKernel(sigma)
        elif kernel == 'gaussian':
            kernel_fun = KT.gaussianKernel(sigma)
        
        kern_aux = lambda A,B :  KT.blockKernComp(A, B, kernel_fun, useGPU = useGPU, nmax_gpu = nmax_gpu)
        if centered == False:
            kern = lambda A : kern_aux(x,A)+c
        else:
            K_0 = kern_aux(x,None)
            K_0m = K_0.mean(1)
            K_0mm = K_0m.mean()
            def kern(A):
                K_a = kern_aux(x,A)
                K_am = K_a.mean(0)
                K_a -= K_am.unsqueeze(0).expand_as(K_a)
                K_a -= K_0m.unsqueeze(1).expand_as(K_a)
                K_a += K_0mm +c
                return K_a
            
        self.kern = kern
        
        K =  kern(None)
        self.ka = torch.sqrt((K[range(n),range(n)]).sum()/n)
        K[range(n),range(n)]+= 1e-12*n
        self.V = chol(K,useGPU = useGPU)
        self.positive = positive
        
        def dz():
            if positive:
                return [torch.zeros((n,1)),torch.zeros((n,1))]
            else:
                return torch.zeros((n,1))
                
        self.dz = dz
        
        if positive:
            self.renorm = np.sqrt(2)*self.ka/np.sqrt(la)
        else:
            self.renorm =self.ka/np.sqrt(la)
        self.norm = 1
        
        
        return None
    
    
    def R(self,a):
        n = self.n
        vals = (self.V).T @ a/(np.sqrt(n)*self.renorm)
        if self.positive:
            return [vals,vals]
        else:
            return vals
            
    def Rt(self,dv):
        n = self.n
        if self.positive:
            t1 = self.V@ (dv[0]+dv[1])/(self.renorm*np.sqrt(n))
        else:
            t1 = self.V@ dv/(self.renorm*np.sqrt(n))
        return t1 
        

        
    def Rx(self,a,xtest):
        n = self.n
        Ktt = self.kern(xtest)
        bid = tr_solve(Ktt,self.V,useGPU = self.useGPU,transpose = True)
        return (bid.T @ a).view(xtest.size(0)) 
    
    def Omega(self,a):
        return 0.5*self.la*((a**2).sum())
    
    def Omegas(self):    
        def fun_only(a):
            return 0.5*self.smoothness*(a**2).sum()
        def fun_grad(a):
            return (0.5*self.smoothness*(a**2).sum(),self.smoothness*a)
        return fun_only,fun_grad
    
    def recoverPrimal(self,a):
        return self.smoothness*a
    
    
    
    
    

class QKM(object): 
    def __init__(self,sigma,x,la,mu = 0,kernel = 'gaussian',useGPU = False,nmax_gpu = None,centered = False,c = 0,positive = False):
        n = x.size(0)
        self.n = n
        self.x = x
        
        if kernel == 'expo':
            kernel_fun = KT.expoKernel(sigma)
        elif kernel == 'gaussian':
            kernel_fun = KT.gaussianKernel(sigma)
        
        
        kern_aux = lambda A,B :  KT.blockKernComp(A, B, kernel_fun, useGPU = useGPU, nmax_gpu = nmax_gpu)
        if centered == False:
            kern = lambda A : kern_aux(x,A)+c
        else:
            K_0 = kern_aux(x,None)
            K_0m = K_0.mean(1)
            K_0mm = K_0m.mean()
            def kern(A):
                K_a = kern_aux(x,A)
                K_am = K_a.mean(0)
                K_a -= K_am.unsqueeze(0).expand_as(K_a)
                K_a -= K_0m.unsqueeze(1).expand_as(K_a)
                K_a += K_0mm +c
                return K_a
        self.kern = kern
        K = self.kern(None)
        self.ka = torch.sqrt((K[range(n),range(n)]**2).sum()/n)
        print(self.ka)
        K[range(n),range(n)]+= 1e-12*n
        self.V = chol(K,useGPU = useGPU)
        self.useGPU = useGPU
        
        def dz():
            return torch.zeros((n,1))
        self.dz = dz
        
        self.positive = positive
        self.la = la
        self.smoothness = 1/la
        self.mu = mu
        
        def aux(A):
            #return ppm.sureApproach(A,mu=mu,useGPU = useGPU)
            if positive == False:
                return ppm.softMaxMixed(A,150,q=6,mu = mu,useGPU = useGPU)
            else:
                return ppm.mixedApproach(A,150,q=6,mu = mu,useGPU = useGPU)
        self.sm = aux
        def aux2(A):
            return ppm.traceNormMixed(A,150,q=6,useGPU = useGPU,isPositive = positive)
        self.tn = aux2
        
        self.renorm = self.ka/np.sqrt(la)
        self.norm = 1
        
        return None
        
      
    
    def R(self,f):
        n = self.n
        D,U = produceDU(useGPU = self.useGPU)
        Vg = U(self.V)
        fg = U(f)
        res = D((Vg * (fg @ Vg)).sum(0))/(self.renorm*np.sqrt(n))
            
        return res.view((n,1))
        
    def Rt(self,dv):
        n = self.n
        alpha = dv.view(n,)
        D,U = produceDU(useGPU = self.useGPU)
        Vg = U(self.V)
        return D(Vg @ (U(alpha) * Vg).T)/(self.renorm*np.sqrt(n))

        
        
    def Rx(self,f,xtest):
        Ktt = self.kern(xtest)
        bid = tr_solve(Ktt,self.V,useGPU = self.useGPU,transpose = True)
        return ((bid * (f@bid)).sum(0)).view(xtest.size(0))
    
    def Omega(self,f):
        return self.mu * self.tn(f) + 0.5*self.la*((f**2).sum())
    
    def Omegas(self):    
        def fun_only(f):
            fpos = self.sm(f)
            return 0.5*self.smoothness*(fpos**2).sum()
        def fun_grad(f):
            fpos =self.sm(f)
            return (0.5*self.smoothness*(fpos**2).sum(),self.smoothness*fpos)
        return fun_only,fun_grad
    
    def recoverPrimal(self,f):
        fpos =self.sm(f)
        return (self.smoothness*fpos)
        





   
    
    
    
###############################################################################
#Quantile models 
###############################################################################
    
    
    
class quantileLinearModel(object):
    def __init__(self,T,LMM,LMD):
        T_m = (T-1)//2
        A = torch.zeros((T,T))
        J = torch.zeros((T_m,T_m))
        for i in range(T_m):
            for j in range(i,T_m):
                J[i,j] =1
        A[:T_m,:T_m] = -J
        A[T_m+1:,T_m+1:] = J.T
        A[:,T_m] = torch.ones((T,))
        n = LMM.n
        
        diag_op = LMD.renorm*torch.ones((T,))
        diag_op[T_m] = LMM.renorm
        
        self.A = A
        self.Abis = A@torch.diag(diag_op)
        self.norm = max(LMM.norm,LMD.norm)*ppm.topEV(self.Abis)
        self.T=T
        self.Tm = T_m
        self.n = n
        
        self.LMM = LMM
        self.LMD = LMD

        def dz():
            res = torch.zeros((T,n))
            res2= []
            for t in range(T):
                if t == T_m:
                    v = self.LMM.dz()
                else:
                    v = self.LMD.dz()
                if type(v) == torch.Tensor:
                    res2.append(None)
                else:
                    res2.append(v[1])
            return [res,res2]
        self.dz = dz
                
        
        
    def R(self,D):
        vsimple = torch.zeros(self.T,self.n)
        lconstraints = []
        for t in range(self.T):
            if t == self.Tm:
                rt = self.LMM.R(D[t])
            else:
                rt = self.LMD.R(D[t])
            if type(rt) == torch.Tensor:
                lconstraints.append(None)
                vsimple[t,:] = rt.view(self.n)
            else:
                lconstraints.append(rt[1])
                vsimple[t,:] = rt[0].view(self.n)   
                
        return [self.Abis@vsimple,lconstraints]
            
    
    def Rt(self,vcomplex):
        n = self.n
        vsimpleb,constraints = vcomplex[0],vcomplex[1]
        vsimple = (self.Abis).T@vsimpleb
        res = []
        for t in range(self.T):
            if t == self.Tm:
                Rtt = self.LMM.Rt
            else:
                Rtt = self.LMD.Rt
            if isinstance(constraints[t],type(None)):
                res.append(Rtt(vsimple[t,:].view(n,1)))
            else:
                res.append(Rtt([vsimple[t,:].view(n,1),constraints[t]]))
        return res
    
    def Rx(self,D,xtest):
        bbet = torch.zeros((self.T,xtest.size(0)))
        for t in range(self.T):
            if t == self.Tm:
                bbet[t,:] = self.LMM.Rx(D[t],xtest)
            else:
                bbet[t,:] = self.LMD.Rx(D[t],xtest)
        return self.A@bbet
    
    def Omega(self,D):
        res = 0
        for t in range(self.T):
            if t == self.Tm:
                omega = self.LMM.Omega
            else:
                omega = self.LMD.Omega
            res += omega(D[t])
        return res
        
    
    def Omegas(self):
        def fun_only(D):
            res = 0
            for t in range(self.T):
                if t == self.Tm:
                    omegas = self.LMM.Omegas
                else:
                    omegas = self.LMD.Omegas
                res += omegas()[0](D[t])
            return res
        def fun_grad(D):
            fun = 0
            grad = []
            for t in range(self.T):
                if t == self.Tm:
                    omegas = self.LMM.Omegas
                else:
                    omegas = self.LMD.Omegas
                funt, gradt = omegas()[1](D[t])
                fun += funt 
                grad.append(gradt)
            return fun,grad
        return fun_only,fun_grad
    
    def recoverPrimal(self,D):
        res = []
        for t in range(self.T):
            if t == self.Tm:
                rp = self.LMM.recoverPrimal
            else:
                rp = self.LMD.recoverPrimal
            res.append(rp(D[t]))
        return res
    
    


########################################
#Loss function 
#######################################



        


 
def psi(tau):
    return lambda x : torch.max(tau*x,(tau-1)*x) 

def psi_prox(tau,c,x):
    return (x-c*tau).clamp(min = 0) + (x+c*(1-tau)).clamp(max = 0)

def psi_star(tau):
    def aux(s):
        if s.ndim == 0:
            if tau-1 <= s and s <= tau:
                return 0
            else:
                return torch.tensor(np.inf)
        elif s.ndim == 1:
            n = s.size(0)
            res = torch.zeros((n,))
            for i in range(n):
                if tau-1 <= s[i] and s[i] <= tau:
                    res[i]=0
                else:
                    res[i] = torch.tensor(np.inf)
            return res
    return aux

def psi_star_prox(tau,c,x):
    err = 1e-12
    return torch.clamp(torch.clamp(x,min = tau-1+err),max = tau-err)




    

class quantileLoss(object):
    
    def __init__(self,y,tau_l):
        self.tau_l = tau_l
        self.T = len(tau_l)
        self.y = y
        return None
        
        
    def L(self,dv):
        alpha = dv[0]
        n12 = np.sqrt(alpha.size(1))
        res = 0
        for t in range(self.T):
            res += torch.mean(psi(self.tau_l[t])(self.y-n12*alpha[t,:]))
        return res
    
    def Ls(self,dv):
        alpha = dv[0]
        n12 = np.sqrt(alpha.size(1))
        res = 0
        for t in range(self.T):
            if not(isinstance(dv[1][t],type(None))) and not((dv[1][t] > 0).sum() ==0):
                return torch.tensor(np.inf)
            else:
                tau = self.tau_l[t]
                res += torch.mean(psi_star(tau)(-n12*alpha[t,:]) + n12*self.y*alpha[t,:])
        return res
        
    def Lsprox(self,c,dv):
        alpha = dv[0]
        beta = dv[1]
        res = torch.zeros((self.T,alpha.size(1)))
        resb = []
        n12 = np.sqrt(alpha.size(1))
        for t in range(self.T):
            tau = self.tau_l[t]
            res[t,:] = -(1/n12)*(psi_star_prox(tau,c,c*self.y - n12*alpha[t,:]))
            if beta[t] == None:
                resb.append(None)
            else:
                resb.append(beta[t].clamp(max = 0))
        return res,resb
    
    
    
    
class quantileModel(object):
    def __init__(self,lmodel,y,tau_l):
        self.loss = quantileLoss(y,tau_l)
        self.lmodel = lmodel
        self.smoothness = self.lmodel.norm**2 
    
    def F_primal(self,B):
        return self.loss.L(self.lmodel.R(B)) + self.lmodel.Omega(B)
    
    def F_dual(self,alpha):
        return -(self.loss.Ls(alpha) + self.lmodel.Omegas()[0](minus_l(self.lmodel.Rt(alpha))))
    
    def pfd(self,alpha):
        return self.lmodel.recoverPrimal(minus_l(self.lmodel.Rt(alpha)))
    
    def F_primald(self,alpha):
        return self.F_primal(self.pfd(alpha))
        
        
    def Rx_dual(self,alpha,xtest):
        B = self.pfd(alpha)
        return self.lmodel.Rx(B,xtest)
    
    
    def cb_prox(cobj,al):
        return None
    
    def cbcboj_pd(self,freq,plot= False):
        cobj = {}
        cobj['it'] = 0
        cobj['primal'] = []
        cobj['dual'] = []
        cobj['itfreq']=[]
        def cb(cobj,al):
            if cobj['it']%freq == 0:
                print("---iteration: {}---".format(cobj['it']+1))
                cobj['primal'].append(self.F_primald(al))
                cobj['dual'].append(self.F_dual(al))
                cobj['itfreq'].append(cobj['it'])
                if plot:
                    plt.semilogy(cobj['itfreq'],np.array(cobj['primal']) - np.array(cobj['dual']))
                    plt.xlabel("iterations")
                    plt.ylabel("dual gap")
                    plt.show()
                cobj['it'] +=1
            else:
                cobj['it'] +=1
        return cb,cobj
    

    
    def prox_method(self,Niter,cb = cb_prox,cobj = {},freq = None):
        if isinstance(freq,type(None)):
            freq = Niter
    
        O_fun,O_fungrad = self.lmodel.Omegas()
    
        def Oms_dual(dv):
            x = minus_l(self.lmodel.Rt(dv))
            f_alpha,g_x = O_fungrad(x)
            g_alpha =  minus_l(self.lmodel.R(g_x))
            def Gl_dual(Lf,talpha):
                d = add_l(talpha,dv,-1)
                f_approx = f_alpha + scal_prod(g_alpha,d) + 0.5*Lf*scal_prod(d,d)
                f_alphat = O_fun(minus_l(self.lmodel.Rt(talpha)))
                return f_approx-f_alphat
            return f_alpha,g_alpha,Gl_dual
    


        al = self.lmodel.dz()
        al2 = clone_l(al)
                
    
        tk = 1
    
        loss_iter = []
    
        Lf = 0.05*self.smoothness
        eta_Lf = 1.1
        
        #Lmax = ka
        for i in range(Niter):
            if i % freq == 0:
                print("iteration {} out of {}".format(i+1,Niter))
            
            Oval,Ograd,Gl_dual = Oms_dual(al2)
            if i > 0:
                loss_iter.append(-self.loss.Ls(al) - O_fun(minus_l(self.lmodel.Rt(al))))
            while True:
                c = 1/Lf
                al1 = self.loss.Lsprox(c,add_l(al2,Ograd,-c))
                if Gl_dual(Lf,al1) >=0:
                    break
                else:
                    Lf *= eta_Lf 
            tk1 = (1 + np.sqrt(1+4*tk**2))/2
            th = (tk - 1)/(tk1)
            al2 = add_l(al1,add_l(al1,al,-1),th)
            al = al1
            tk = tk1
            cb(cobj,al)
    
        plt.plot(list(range(len(loss_iter))),(loss_iter))
        return(al)
    


 
        

############################################################################################################
#
##########################################################################################################





class QlossNW(object):
    def __init__(self,y,tau_l,eps=0.01):
        
        n = y.size(0)
        T = len(tau_l)
        
        def L(x):
            res = 0
            for t in range(T):
                tau = tau_l[t]
                res += torch.sum(psi(tau)(y - x[t,:]))
            return res
        
        def proxL(c,x):
            res = torch.zeros((T,n))
            for t in range(T):
                tau = tau_l[t]
                res[t,:] = y - psi_prox(tau,c,y-x[t,:])
            return res
        
        self.eps = eps
        self.L = L
        self.proxL = proxL
        return None
    
    def Leps(self,alpha):
        eps = self.eps
        yy = self.proxL(eps,alpha)
        d = add_l(alpha,yy,-1)
        return self.L(yy) + scal_prod(d,d)/(2*eps)
    
    def gradLeps(self,alpha):
        eps = self.eps
        return (add_l(add_l(None,alpha,1/eps) , self.proxL(eps,alpha),-1/eps))
    


class kernelModel(object):
    def __init__(self,sigma,x,kernel = 'gaussian',centered = False,c=0,useGPU = False, nmax_gpu = None,positive = False):
        n = x.size(0)
        self.n = n
        if x.ndim == 1:
            d =1
        else:
            d = x.size(1)
        self.d = d
        
        if kernel == 'expo':
            kernel_fun = KT.expoKernel(sigma)
        elif kernel == 'gaussian':
            kernel_fun = KT.gaussianKernel(sigma)
        kern_aux = lambda A,B :  KT.blockKernComp(A, B, kernel_fun, useGPU = useGPU, nmax_gpu = nmax_gpu)
        if centered == False:
            kern = lambda A : kern_aux(x,A)+c
        else:
            if positive == True:
                print('error positive centered')
            K_0 = kern_aux(x,None)
            K_0m = K_0.mean(1)
            K_0mm = K_0m.mean()
            def kern(A):
                K_a = kern_aux(x,A)
                K_am = K_a.mean(0)
                K_a -= K_am.unsqueeze(0).expand_as(K_a)
                K_a -= K_0m.unsqueeze(1).expand_as(K_a)
                K_a += K_0mm +c
                return K_a
        self.kern = kern
        
        K =  kern(None)
        self.K = K
        self.ka = (K[range(n),range(n)]).sum()/n
        #K[range(n),range(n)]+= 1e-12*n
        
        
        def dz():
            return torch.zeros((n,1))
        self.dz = dz
        
    def Rx(self,a,xtest):
        n = self.n
        Ktt = self.kern(xtest)
        return (Ktt.T @ a).view(xtest.size()) 

    
    
class QLinearModelNW(object):
    
    
    def __init__(self,LMM,LMD,T):
        n = LMM.n
        self.n = n
        
        self.LMM = LMM
        self.LMD = LMD
        
        T_m = (T-1)//2
        self.Tm = T_m
        self.T = T
        
        A = torch.zeros((T,T))
        J = torch.zeros((T_m,T_m))
        for i in range(T_m):
            for j in range(i,T_m):
                J[i,j] =1
        A[:T_m,:T_m] = -J
        A[T_m+1:,T_m+1:] = J.T
        A[:,T_m] = torch.ones((T,))
        
        self.A = A
        self.ka = max(LMM.ka,LMD.ka)*ppm.topEV(A)
        
        def dz():
            return torch.zeros((T,n))
        self.dz =dz
        
        
    def Kfun(self,var):
        n =self.n
        res = self.dz()
        T = self.T
        Tm = self.Tm
        for t in range(self.T):
            if t != Tm:
                res[t,:] = (self.LMD.K@ var[t,:].view(n,1)).view(n)
            else:
                res[t,:] = (self.LMM.K@ var[t,:].view(n,1)).view(n)
        return self.A@res,res
    
    def Kt(self,var):
        n = self.n
        res = (self.A).T @ var
        T = self.T
        Tm = self.Tm
        for t in range(self.T):
            if t != Tm:
                res[t,:] = (self.LMD.K@ res[t,:].view(n,1)).view(n)
            else:
                res[t,:] = (self.LMM.K@ res[t,:].view(n,1)).view(n)
        return res
        
        
            
    
    
    def Rx(self,var,xtest):
        n = self.n
        T = self.T
        Tm = self.Tm
        ntest = xtest.size(0)
        res = torch.zeros(T,ntest)
        
        for t in range(self.T):
            if t != Tm:
                res[t,:] = (self.LMD.Rx(var[t,:].view(n,1),xtest)).view(ntest)
            else:
                res[t,:] = (self.LMM.Rx(var[t,:].view(n,1),xtest)).view(ntest)
        return (self.A@res)

    
    
    
class QModelNW(object):
    def __init__(self,lossmodel,lam,lad,kmodel):
        self.kmodel = kmodel
        n = kmodel.n
        self.n = n
        self.lossmodel = lossmodel
        self.lam = lam
        self.lad = lad
        
        self.T = kmodel.T
        self.Tm = kmodel.Tm
        
        self.cm = lam*n
        self.cd = lad*n
        
        kaM = self.kmodel.LMM.ka
        kaD = self.kmodel.LMD.ka
        
    
        self.smoothness = n**2 *kmodel.ka**2/lossmodel.eps + n**2 * max(kaM*lam,kaD*lad)
        
    def proj(self,x):
        res = self.kmodel.dz()
        T = self.T
        Tm = self.Tm
        for t in range(T):
            if t != Tm:
                res[t,:] = torch.clamp(x[t,:],min = 0)
            else:
                res[t,:] = x[t,:]
        return res
    
    def Loss_tot(self,x):
        K = self.kmodel.Kfun
        fv,kv = K(x)
        kv *= 0.5*self.cd*x
        kv[self.Tm,:] *= self.cm/self.cd
        
        return kv.sum() + self.lossmodel.Leps(fv)
    
    def grad(self,x):
        K = self.kmodel.Kfun
        fv,kv = K(x)
        kv *= self.cd
        kv[self.Tm,:] *= self.cm/self.cd
        grad = kv + self.kmodel.Kt(self.lossmodel.gradLeps(fv))
        return grad
    
    def cb_prox(cobj,al):
        return None
    
    def FISTA(self,Niter,cb = cb_prox,cobj = {},freq = None):
        if isinstance(freq,type(None)):
            freq = Niter

        def Gl(alpha):
            f_alpha = self.Loss_tot(alpha)
            g_alpha = self.grad(alpha)
            def aux(Lf,talpha):
                dd = add_l(talpha, alpha,-1)
                f_approx = f_alpha + scal_prod(g_alpha, dd) + 0.5*Lf*scal_prod(dd,dd)
                f_alphat = self.Loss_tot(talpha)
                return f_approx-f_alphat
            return f_alpha,g_alpha,aux

        
        al = self.kmodel.dz()
        al2 = self.kmodel.dz()
    
        tk = 1
    
        loss_iter = []
    
        Lf = 0.01*self.smoothness
        eta_Lf = 1.1
        
        #Lmax = ka
        for i in range(Niter):
            if i % freq == 0:
                print("iteration {} out of {}".format(i+1,Niter))
             
            fun,grad,GL = Gl(al2)
            if i > 0:
                loss_iter.append(self.Loss_tot(al))
            while True:
                gamma = 1/Lf
                al1 = self.proj(add_l(al2,grad,-gamma))
                if GL(Lf,al1) >=0 :
                    #or Lf >= self.smoothness
                    break
                else:
                    Lf *= eta_Lf 
            #print('Lf/Lmax = {}'.format(Lf/self.smoothness))
            tk1 = (1 + np.sqrt(1+4*tk**2))/2
            th = (tk - 1)/(tk1)
            al2 = add_l(al1 ,add_l(al1,al,-1),th)
            al = al1
            tk = tk1
            cb(cobj,al)
    
        plt.plot(list(range(len(loss_iter))),(loss_iter))
        return(al)  
    
    
##########################################################
#
###########################################################


class kernelModelExpo(object):
    def __init__(self,sigma,x,kernel = 'gaussian',c=0,useGPU = False, nmax_gpu = None,positive = 'False'):
        n = x.size(0)
        self.n = n
        if x.ndim == 1:
            d =1
        else:
            d = x.size(1)
        self.d = d
        
        if kernel == 'expo':
            kernel_fun = KT.expoKernel(sigma)
        elif kernel == 'gaussian':
            kernel_fun = KT.gaussianKernel(sigma)
        kern_aux = lambda A,B :  KT.blockKernComp(A, B, kernel_fun, useGPU = useGPU, nmax_gpu = nmax_gpu)
        kern = lambda A : kern_aux(x,A)+c
        self.kern = kern
        self.positive = positive
        
        K =  kern(None)
        self.K = K
        self.ka = np.sqrt((K[range(n),range(n)]).sum())
        K[range(n),range(n)]+= 1e-12*n
        self.V = chol(K,useGPU = useGPU)
        self.useGPU = useGPU
        self.nmax_gpu = nmax_gpu
        
        
        def dz():
            return torch.randn((n,1))
        self.dz = dz
        
    def R(self,a):
        return self.V.T@a
        
    def Rx(self,a,xtest):
        n = self.n
        Ktt = self.kern(xtest)
        bid = tr_solve(Ktt,self.V,useGPU = self.useGPU,transpose = True)
        return (bid.T @ a).view(xtest.size(0))
    
    def pdata(self,a):
        rr = self.R(a)
        if self.positive:
            return torch.exp(rr)
        else:
            return rr
        

    def px(self,a,xtest):
        rx = self.Rx(a,xtest)
        if self.positive:
            return(torch.exp(rx))
        else:
            return rx
    
    def Jacobian(self,a):
        V = self.V
        n = self.n
        if self.positive:
            pdata = torch.exp(a.T@V ).view(n)
            return V *pdata[None,:],pdata.view(n,1)
        else:
            return V,self.R(a)
      

        
    





    
    
class QLinearModelKernelExpo(object):
    
    
    def __init__(self,LMM,LMD,T):
        n = LMM.n
        self.n = n
        
        self.LMM = LMM
        self.LMD = LMD
        
        T_m = (T-1)//2
        self.Tm = T_m
        self.T = T
        
        A = torch.zeros((T,T))
        J = torch.zeros((T_m,T_m))
        for i in range(T_m):
            for j in range(i,T_m):
                J[i,j] =1
        A[:T_m,:T_m] = -J
        A[T_m+1:,T_m+1:] = J.T
        A[:,T_m] = torch.ones((T,))
        
        self.A = A
        self.ka = max(LMM.ka,LMD.ka)*ppm.topEV(A)
        
        def dz():
            return torch.randn((T,n))
        self.dz =dz
        
    def pdata(self,var):
        n =self.n
        T = self.T
        res = torch.zeros((T,n))
        Tm = self.Tm
        for t in range(self.T):
            if t != Tm:
                res[t,:] = (self.LMD.pdata(var[t,:].view(n,1))).view(n)
            else:
                res[t,:] = (self.LMM.pdata(var[t,:].view(n,1))).view(n)
        return self.A@res,res
    
    def px(self,var,xtest):
        ntest = xtest.size(0)
        n =self.n
        T = self.T
        res = torch.zeros((T,ntest))
        Tm = self.Tm
        for t in range(self.T):
            if t != Tm:
                res[t,:] = (self.LMD.px(var[t,:].view(n,1),xtest)).view(ntest)
            else:
                res[t,:] = (self.LMM.px(var[t,:].view(n,1),xtest)).view(ntest)
        return self.A@res
    
    def Jacobian(self,var):
        n =self.n
        T = self.T
        res = torch.zeros((T,n))
        Tm = self.Tm
        def aux(tvar):
            res = torch.zeros((T,n))
            for t in range(T):
                if t == Tm:
                    J,p = self.LMM.Jacobian(var[t,:].view(n,1))
                    res[t,:] = (J@tvar[t,:].view(n,1) ).view(n)
                else:
                    J,p = self.LMD.Jacobian(var[t,:].view(n,1))
                    res[t,:] = (J@tvar[t,:].view(n,1) ).view(n)
            res = self.A@res 
            return res
        return aux
    
        
    
    
    
    
class QModelExpo(object):
    def __init__(self,lossmodel,lam,lad,kmodel):
        
        
        self.kmodel = kmodel
        n = kmodel.n
        self.n = n
        self.lossmodel = lossmodel
        self.lam = lam
        self.lad = lad
        
        self.T = kmodel.T
        self.Tm = kmodel.Tm
        
        self.cm = lam*n
        self.cd = lad*n
        
        kam = self.kmodel.LMM.ka
        kad = self.kmodel.LMD.ka
        
    
        self.smoothness =  max(kam,kad)**2/lossmodel.eps + n * max(lam,lad)
        
    
    def Loss_tot(self,x):
        pdata = self.kmodel.pdata
        fv,kv = pdata(x)
        kv *= 0.5*self.cd*x
        kv[self.Tm,:] *= self.cm/self.cd
        return kv.sum() + self.lossmodel.Leps(fv)
    
    def grad(self,x):
        pdata =self.kmodel.pdata
        fv,kv = pdata(x)
        kv *= self.cd
        kv[self.Tm,:] *= self.cm/self.cd
        
        J = self.kmodel.Jacobian(x)
        
        grad = add_l(J(self.lossmodel.gradLeps(fv)),kv,1)
        return grad
    
    
    def cb_prox(cobj,al):
        return None
    
    def GD(self,Niter,cb = cb_prox,cobj = {},gamma_min= 0.01,freq = None):
        if isinstance(freq,type(None)):
            freq = Niter
        
            
    

        def Gl(alpha):
            f_alpha = self.Loss_tot(alpha)
            g_alpha = self.grad(alpha)
            def aux(Lf,talpha):
                dd = add_l(talpha, alpha,-1)
                f_approx = f_alpha + scal_prod(g_alpha,dd) + 0.5*Lf*scal_prod(dd,dd)
                f_alphat = self.Loss_tot(talpha)
                return f_approx-f_alphat
            return f_alpha,g_alpha,aux

        
        al = self.kmodel.dz()
        al2 = self.kmodel.dz()
    
        tk = 1
    
        loss_iter = []
    
        Lf = 0.01*self.smoothness
        eta_Lf = 1.1
        
        #Lmax = ka
        for i in range(Niter):
            if i % freq == 0:
                print("iteration {} out of {}".format(i+1,Niter))
            fun,grad,GL = Gl(al2)
            if i > 0:
                loss_iter.append(self.Loss_tot(al))
            while True:
                gamma = 1/Lf
                al1 = add_l(al2,grad,-gamma)
                if GL(Lf,al1) >=0:
                    #or Lf >= self.smoothness
                    break
                else:
                    Lf *= eta_Lf 
            #print('Lf/Lmax = {}'.format(Lf/self.smoothness))
            tk1 = (1 + np.sqrt(1+4*tk**2))/2
            th = (tk - 1)/(tk1)
            al2 = add_l(al1,add_l(al1,al,-1),th)
            al = al1
            tk = tk1
            cb(cobj,al)
    
        plt.plot(list(range(len(loss_iter))),(loss_iter))
        return(al)
        
    