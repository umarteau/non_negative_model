import numpy as np
import torch
import npm_nnf.utils.utils_kernels as KT
import matplotlib.pyplot as plt
import npm_nnf.utils.ppm as ppm

torch.set_default_dtype(torch.float64)

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
        
        
def add_l(a,b,c):
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
    


def clone_l(a):
    if type(a) == torch.Tensor:
        return a.clone()
    try:
        return [clone_l(aa) for aa in a]
    except:
        return a.clone()
    
    
def minus_l(b):
    return add_l(None,b,-1)

def scal_prod(a,b):
    if type(b) == torch.Tensor:
        return torch.sum(a*b)
    try:
        res = 0
        for i,aa in enumerate(a):
            res += scal_prod(aa,b[i]) 
        return res
    except:
        return torch.sum(a*b)

########################################
#Linear Model
#######################################


    
    
class LMK(object): 
    def __init__(self,sigma,x,la,kernel = 'gaussian',useGPU = False,nmax_gpu = None,centered = False,c = 0):
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
        
        def dz():
            return torch.zeros((n,1))
        self.dz = dz
        
        self.renorm = self.ka/np.sqrt(la)
        self.norm = 1
        

    
    
    def R(self,a):
        n = self.n
        vals = (self.V).T @ a/(np.sqrt(n)*self.renorm)
        return vals
            
    def Rt(self,dv):
        n = self.n
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
                return ppm.softMaxMixed(A, 150, q=6, mu = mu, useGPU = useGPU)
            else:
                return ppm.mixedApproach(A, 150, q=6, mu = mu, useGPU = useGPU)
        self.sm = aux
        def aux2(A):
            return ppm.traceNormMixed(A, 150, q=6, useGPU = useGPU, isPositive = positive)
        self.tn = aux2
        
        self.renorm = self.ka/np.sqrt(la)
        self.norm = 1
        

        
      
    
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
    

class HSLinearModel(object):
    def __init__(self,LME,LML):
       
        self.norm = max(LME.norm,LML.norm)
        self.n = LME.n
        self.LME = LME
        self.LML = LML
        self.renorm = [LME.renorm,LML.renorm]
        def dz():
            return[LME.dz(),LML.dz()]
        self.dz =dz
        
        
    def R(self,var):
        return [self.LME.R(var[0]),self.LML.R(var[1])]
        
            
    
    def Rt(self,var):
        return [self.LME.Rt(var[0]),self.LML.Rt(var[1])]
    
    def Rx(self,var,xtest):
        return [self.LME.Rx(var[0],xtest),self.LML.Rx(var[1],xtest)]
    
    def Omega(self,var):
        return self.LME.Omega(var[0]) + self.LML.Omega(var[1])
    
    def Omegas(self): 
        
        def fun_only(var):
            fun = self.LME.Omegas()[0](var[0]) + self.LML.Omegas()[0](var[1])
            return fun
        def fun_grad(var):
            fun0,grad0 = self.LME.Omegas()[1](var[0]) 
            fun1,grad1 =  self.LML.Omegas()[1](var[1])
            return (fun0+fun1,[grad0,grad1])
        return fun_only,fun_grad
    
    def recoverPrimal(self,var):
        return [self.LME.recoverPrimal(var[0]),self.LML.recoverPrimal(var[1])]
    

        
########################################
#Loss function 
#######################################
g = lambda eta,la : 0.5*eta**2/la - 0.5*torch.log(la) + 0.5*torch.log(torch.tensor(2*np.pi))
gstar = lambda al,ga : -0.5*torch.log(-(4*np.pi*np.e)*(ga + 0.5*al**2))

      
def gprox_a(a1,a2):
    ra = a1**2/a2
    def poly(eta0,la0,c):
        c1 = 2*ra*c - la0 
        c2 = ra**2 * c**2 - 2*la0*ra*c - 0.5*c
        c3 = la0*ra**2*c**2 + 0.5*eta0**2*ra*c + ra*c**2
        c3 *= -1
        c4 = - 0.5*c**3*ra**2
        return [1,c1,c2,c3,c4]
    def aux(c,eta0,la0):
        v = np.roots(poly(eta0,la0,c))
        m= max(np.real(v[np.isreal(v)]))
        if m < 0:
            print("error ")
        return(eta0*m/(m+ra*c),m)
        
    def p(c,eta,la):
        n = eta.size(0)
        res1,res2 = torch.zeros((n,1)),torch.zeros((n,1))
        for i in range(n):
            res1[i,0],res2[i,0] = aux(c,eta[i,0],la[i,0])
        return res1,res2
    return p

########################################################
#Computing L
########################################################




    
class LossHS(object):
    def __init__(self,y,renorm = [1,1]):
        n = y.size(0)
        self.n = n
        self.T = [y.view(n,1),-y.view(n,1)**2/2]
        self.a1= renorm[0]
        self.a2 = renorm[1]
    def L(self,var):
        n12 = np.sqrt(self.n)
        a1,a2 = self.a1,self.a2
        eta,la = var[0],var[1]
        return torch.mean(g(n12*a1*eta,n12*a2*la) - n12*a1*eta*self.T[0] - n12*a2*la*self.T[1])
    def Ls(self,var):
        n12 = np.sqrt(self.n)
        a1,a2 = self.a1,self.a2
        eta,la = var[0],var[1]
        return torch.mean(gstar((n12/a1)*eta + self.T[0],(n12/a2)*la+self.T[1]))
    def Lsprox(self,c,var):
        n12 = np.sqrt(self.n)
        a1,a2 = self.a1,self.a2
        eta,la = var[0],var[1]
        pg = gprox_a(a1,a2)
        var00,var10 = pg(1/c,(n12*eta + a1*self.T[0])/c,(n12*la + a2*self.T[1])/c)
        var00,var10 = eta - (c/n12)*var00,la - (c/n12)*var10
        return [var00,var10]
        
        
    
    



    
    


        
########################################
#Model
#######################################


class HSModel(object):
    def __init__(self,lmodel,y):
        self.loss = LossHS(y,renorm = lmodel.renorm)
        self.lmodel = lmodel
        self.smoothness = lmodel.norm
    
    def F_primal(self,B):
        return self.loss.L(self.lmodel.R(B)) + self.lmodel.Omega(B)
    
    def F_dual(self,alpha):
        return -(self.loss.Ls(alpha) + self.lmodel.Omegas()[0](self.lmodel.Rt(minus_l(alpha))))
    
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
    
    def cbTest(self,data_test,freq,name = "",print_it = False):
        xtest,ytest = data_test
        ntest = ytest.size(0)
        def tLoss(alpha):
            T = [ytest.view(ntest,),-ytest.view(ntest,)**2/2]
            haha = self.Rx_dual(alpha,xtest)
            eta,la = haha[0],haha[1]
            return  torch.mean(g(eta,la) - eta*T[0] -la*T[1])
        def cb(cobj,al):
            if not(name in cobj.keys()):
                if not(print_it):
                    print("starting "+ name)
                cobj['it'] = 0
                cobj['itfreq'] = []
                cobj[name] = []
            if cobj['it']%freq == 0:
                if print_it:
                    print(name + "---iteration: {}---".format(cobj['it']+1))
                cobj[name].append(tLoss(al))
                cobj['itfreq'].append(cobj['it'])
                cobj['it'] +=1
            else:
                cobj['it'] +=1
        return cb
        
    
    def prox_method(self,Niter,cb = cb_prox,cobj = {}):
    
        O_fun,O_fungrad = self.lmodel.Omegas()
    
        def Oms_dual(alpha):
            x = minus_l(self.lmodel.Rt(alpha))
            f_alpha,g_x = O_fungrad(x)
            g_alpha =  minus_l(self.lmodel.R(g_x))
            def Gl_dual(Lf,talpha):
                dd = add_l(talpha,alpha,-1)
                f_approx = f_alpha + scal_prod(g_alpha,dd) + 0.5*Lf*scal_prod(dd,dd)
                f_alphat = O_fun(self.lmodel.Rt(minus_l(talpha)))
                return f_approx-f_alphat
            return f_alpha,g_alpha,Gl_dual

        
        al = self.lmodel.dz()
        al2 = self.lmodel.dz()
    
        tk = 1
    
        loss_iter = []
    
        Lf = 0.05*self.smoothness
        eta_Lf = 1.1
        
        #Lmax = ka
        for i in range(Niter):
            Oval,Ograd,Gl_dual = Oms_dual(al2)
            if i > 0:
                loss_iter.append(-self.loss.Ls(al) - O_fun(self.lmodel.Rt(minus_l(al))))
            while True:
                c = 1/Lf
                al1 = self.loss.Lsprox(c,add_l(al2,Ograd,-c))
                if Gl_dual(Lf,al1) >= 0:
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

class LossNW(object):
    def __init__(self,L,proxL,eps):
        self.eps = eps
        self.L = L
        self.proxL = proxL
        return None
    
    def Leps(self,alpha):
        eps = self.eps
        y = self.proxL(eps,alpha)
        d = add_l(alpha,y,-1)
        return self.L(self,y) + scal_prod(d,d)/(2*eps)
    
    def gradLeps(self,alpha):
        eps = self.eps
        return (add_l(add_l(None,alpha,1/eps) , self.proxL(eps,alpha),-1/eps))
    


gNW = lambda x : (0.5*x[0]**2/x[1] - 0.5*torch.log(x[1]) + 0.5*torch.log(torch.tensor(2*np.pi))).sum()

"""
def gNWprox(c,x):
    def aux(cc,x0):
        err = 1e-14
        l = max(0,x0[1])
        r = max(1,x0[1] + cc*(x0[0]**2+1)/2)
        aux_fun = lambda la : 0.5*cc*(x0[0]**2/(la+cc)**2 + 1/la)
        while r-l > err:
            m = (r+l)/2
            if aux_fun(m)>m-x0[1]:
                l=m
            else:
                r = m
        return(x0[0]*m/(m+cc),m)
    n = x[0].size(0)
    res1,res2 = torch.zeros((n,1)),torch.zeros((n,1))
    for i in range(n):
        res1[i,0],res2[i,0] = aux(c,[x[0][i,0],x[1][i,0]])
    return [res1,res2] """


def gNWprox(c,x):
    ra = 1
    def poly(eta0,la0,c):
        c1 = 2*ra*c - la0 
        c2 = ra**2 * c**2 - 2*la0*ra*c - 0.5*c
        c3 = la0*ra**2*c**2 + 0.5*eta0**2*ra*c + ra*c**2
        c3 *= -1
        c4 = - 0.5*c**3*ra**2
        return [1,c1,c2,c3,c4]
    def aux(c,eta0,la0):
        v = np.roots(poly(eta0,la0,c))
        m= max(np.real(v[np.isreal(v)]))
        if m < 0:
            print("error ")
        return(eta0*m/(m+ra*c),m)
        
    n = x[0].size(0)
    res1,res2 = torch.zeros((n,1)),torch.zeros((n,1))
    for i in range(n):
        res1[i,0],res2[i,0] = aux(c,x[0][i,0],x[1][i,0])
    return [res1,res2]


class HSlossNW(object):
    def __init__(self,y,eps):
        n = y.size(0)
        T = [y.view(n,1),-y.view(n,1)**2/2]
        self.eps = eps
        self.L = lambda x : gNW(x) - (x[0]*T[0]).sum() - (x[1]*T[1]).sum()
        def proxL(c,x):
            return gNWprox(c,add_l(x,T,c))
        self.proxL = proxL
        return None
    
    def Leps(self,alpha):
        eps = self.eps
        y = self.proxL(eps,alpha)
        d = add_l(alpha,y,-1)
        return self.L(y) + scal_prod(d,d)/(2*eps)
    
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

    
    
class HSLinearModelNW(object):
    def __init__(self,LME,LML):
        self.ka = max(LME.ka,LML.ka)
        self.n = LME.n
        self.LME = LME
        self.LML = LML
        
        def dz():
            return[LME.dz(),LML.dz()]
        self.dz =dz
        
        
    def Kfun(self,var):
        return [self.LME.K @ var[0],self.LML.K@ var[1]]
        
            
    
    
    def Rx(self,var,xtest):
        return [self.LME.Rx(var[0],xtest),self.LML.Rx(var[1],xtest)]

    
    
    
class HSModelNW(object):
    def __init__(self,kmodel,lae,lal,y,eps):
        self.kmodel = kmodel
        n = kmodel.n
        self.n = n
        self.lossmodel = HSlossNW(y,eps)
        self.lae = lae
        self.lal = lal
        
        self.ce = lae*n
        self.cl = lal*n
        
        kae = self.kmodel.LME.ka
        kal = self.kmodel.LML.ka
        
    
        self.smoothness = n**2 * max(kae,kal)**2/eps + n**2 * max(kae*lae,kal*lal)
        
    def proj(self,x):
        return [x[0],torch.clamp(x[1],min = 0)]
    
    def Loss_tot(self,x):
        K = self.kmodel.Kfun
        fv = K(x)
        return self.ce*(x[0]*fv[0]).sum()/2 + self.cl*(x[1]*fv[1]).sum()/2 + self.lossmodel.Leps(fv)
    
    def grad(self,x):
        K = self.kmodel.Kfun
        fv = K(x)
        grad = add_l([self.ce*fv[0],self.cl*fv[1]],K(self.lossmodel.gradLeps(K(x))),1)
        return grad
    
    def cb_prox(cobj,al):
        return None
    
    def FISTA(self,Niter,cb = cb_prox,cobj = {}):
    

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
        
        
############################################################
#
#############################################################




class kernelExpoModel(object):
    def __init__(self,sigma,x,kernel = 'gaussian',c=0,centered = False,useGPU = False, nmax_gpu = None,positive = 'False'):
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
      

        
    



    
    
class HSLinearModelExpo(object):
    def __init__(self,LME,LML):
        self.ka = max(LME.ka,LML.ka)
        self.n = LME.n
        self.LME = LME
        self.LML = LML
        
        def dz():
            return[LME.dz(),LML.dz()]
        self.dz =dz
        
        
    def pdata(self,var):
        return [self.LME.pdata(var[0]),self.LML.pdata(var[1])]
        
            
    def px(self,var,xtest):
        return [self.LME.px(var[0],xtest),self.LML.px(var[1],xtest)]
    
    def Jacobian(self,var):
        J0,p0 = self.LME.Jacobian(var[0])
        J1,p1 = self.LML.Jacobian(var[1])
        def aux(tvar):
            return[J0@ tvar[0],J1 @ tvar[1]]
        return aux,[p0,p1]


class HSModelExpo(object):
    def __init__(self,kmodel,lae,lal,y,eps):
        self.kmodel = kmodel
        n = kmodel.n
        self.n = n
        self.lossmodel = HSlossNW(y,eps)
        self.lae = lae
        self.lal = lal
        
        self.ce = lae*n
        self.cl = lal*n
        
        kae = self.kmodel.LME.ka
        kal = self.kmodel.LML.ka
        
    
        self.smoothness =  max(kae,kal)**2/eps + n * max(lae,lal)
        
    
    def Loss_tot(self,x):
        return self.ce*(x[0]**2).sum()/2 + self.cl*(x[1]**2).sum()/2 + self.lossmodel.Leps(self.kmodel.pdata(x))
    
    def grad(self,x):
        J,pdata = self.kmodel.Jacobian(x)
        grad = add_l(J(self.lossmodel.gradLeps(pdata)),[self.ce*x[0],self.cl*x[1]],1)
        return grad
    
    def cb_prox(cobj,al):
        return None
    
    def GD(self,Niter,cb = cb_prox,cobj = {},gamma_min= 0.01):
    

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
        al2 = clone_l(al)
    
        tk = 1
    
        loss_iter = []
    
        Lf = 0.01*self.smoothness
        eta_Lf = 1.1
        
        #Lmax = ka
        for i in range(Niter):
            fun,grad,GL = Gl(al2)
            if i > 0:
                loss_iter.append(self.Loss_tot(al))
            while True:
                gamma = 1/Lf
                al1 = add_l(al2,grad,-gamma)
                if GL(Lf,al1) >=0 :
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
    
    def GDr(self,Niter,cb = cb_prox,cobj = {},Nrestarts=1,acceleration = False,eps = 0.1):
    

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
        al_final = None
        
        
        loss_val = torch.tensor(np.inf)
        loss_iter = []
        
        for j in range(Nrestarts):
    
            al = add_l(al,self.kmodel.dz(),eps)
        
            if acceleration:
                tk = 1
                al2 = clone_l(al)
    
            Lf = 0.01*self.smoothness
            eta_Lf = 1.1
        
            #Lmax = ka
            for i in range(Niter):
                if acceleration:
                    fun,grad,GL = Gl(al2)
                else:
                    fun,grad,GL = Gl(al)
                if i > 0:
                    loss_iter.append(self.Loss_tot(al))
                while True:
                    gamma = 1/Lf
                    if acceleration:
                        al1 = add_l(al2,grad,-gamma)
                    else:
                         al1 = add_l(al,grad,-gamma)
                    if GL(Lf,al1) >=0 :
                        #or Lf >= self.smoothness
                        break
                    else:
                        Lf *= eta_Lf 
                #print('Lf/Lmax = {}'.format(Lf/self.smoothness))
                
                if acceleration:
                    tk1 = (1 + np.sqrt(1+4*tk**2))/2
                    th = (tk - 1)/(tk1)
                    al2 = add_l(al1,add_l(al1,al,-1),th)
                    tk = tk1
                al = al1
                
                cb(cobj,al)
            loss_valj = self.Loss_tot(al)
            if loss_valj < loss_val:
                loss_val = loss_valj
                al_final = clone_l(al)
             
    
        plt.plot(list(range(len(loss_iter))),(loss_iter))
        
        
        return(al_final)
        
    
    

            
    

    