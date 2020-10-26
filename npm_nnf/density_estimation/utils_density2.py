import numpy as np
import torch
import npm_nnf.utils.utils_kernels as KT
import matplotlib.pyplot as plt
import npm_nnf.utils.ppm as ppm
import pickle
import logging

logging.basicConfig(level=logging.NOTSET,format='%(asctime)s  %(name)s %(levelname)s %(message)s')
logger = logging.getLogger('ROOT')
logger.setLevel(logging.INFO)

torch.set_default_dtype(torch.float64)

############################################################################################
#Cholesky decompositions
############################################################################################

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

class Error(Exception):
    pass

class OptimError(Error):
    def __init__(self,m = ""):
        self.message = m

class FitError(Error):
    def __init__(self,m=""):
        self.message = m
        pass

class stopper(object):
    def __init__(self,stype = 'auto',Niter = None,tol_loss = 1e-2,tol_int = 1e-2,tol_dg = 1e-2,d = 6,maximize = True):
        self.losses = []
        self.integrals = []
        self.iter = 0
        self.count_int= 0
        self.count_int1 = 0
        self.max_loss = -np.inf
        self.min_loss = + np.inf
        self.stype = stype
        self.Niter = Niter
        self.tol_loss = tol_loss
        self.tol_int = tol_int
        self.d = d
        self.ldg = []
        self.tol_dg = tol_dg
        self.gap = 0
        self.maximize = maximize

    def update(self,loss,integral = None,dg = None):
        if self.maximize == False:
            loss = -loss
        self.losses.append(loss)
        self.iter += 1

        if not(isinstance(integral,type(None))):
            self.count_int += 1
            self.integrals.append(integral)
            if integral < 1 + self.tol_int and integral > 1 - self.tol_int:
                self.count_int1 += 1
            else:
                self.count_int1 = 0
        if loss > self.max_loss:
            self.max_loss = loss
            self.gap = self.max_loss - self.min_loss
        if loss < self.min_loss and self.iter >=2:
            self.min_loss = loss
            self.gap = self.max_loss - self.min_loss
        if not(isinstance(dg,type(None))):
            self.ldg.append(dg)
        pass

    def stop(self):
        if self.iter < 10:
            return False

        if len(self.ldg) > 0 and self.ldg[-1] < self.tol_dg and self.ldg[-1] >= 0:
            logging.info(f'optimization stopped after dual gap reached {self.tol_dg}, iteration : {self.iter}')
            return True
        elif len(self.ldg) > 0 and self.ldg[-1] < -0.1:
            logging.warning(f'problem : dual gap < 0')
            raise OptimError('problem : dual gap < 0')
            return True

        if self.stype == 'fixed':
            if self.iter >= self.Niter:
                logging.info(f'optimization stopped after {self.iter} iterations')
                return True
        elif self.stype == 'dg':
            return False
        if self.iter > 100 and self.max_loss - self.losses[-1] > 0.1*self.gap :
            logging.warning('training loss is decreasing too much')
            raise OptimError('training loss is decreasing too much')
        if self.stype == 'auto':
            if self.iter >= self.Niter:
                logging.warning(f'optimization reached max number of iterations {self.iter}')
                return True


            if self.count_int == 0:
                res = True
            elif self.iter > self.d and self.count_int1 > 0 and self.count_int // self.count_int1 >= self.d:
                    res = True
            else:
                res = False

            k = self.iter // self.d
            r = (self.losses[self.iter - 1] - self.losses[k * (self.d - 1)])/self.gap
            res = (res and (r >= 0) and (r <= self.tol_loss))
            return res
        return False


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

class LMK2(object):
    def __init__(self, sigma, x, kernel='gaussian', centered=False, c=0, base='1'
                 , mu_base=None, eta_base=None, useGPU=False, nmax_gpu=None, target_norm=1):
        n = x.size(0)
        if x.ndim == 1:
            d = 1
        else:
            d = x.size(1)

        self.n = n
        self.x = x
        self.d = d
        self.target_norm = target_norm
        self.useGPU = useGPU

        if kernel == 'expo':
            kernel_fun = KT.expoKernel(sigma)
        elif kernel == 'gaussian':
            kernel_fun = KT.gaussianKernel(sigma)

        def kern_aux(A, B):
            return KT.blockKernComp(A, B, kernel_fun, useGPU=useGPU, nmax_gpu=nmax_gpu)

        if centered == False:
            def kern(A):
                return kern_aux(x, A) + c
        else:
            K_0 = kern_aux(x, None)
            K_0m = K_0.mean(1)
            K_0mm = K_0m.mean()

            def kern(A):
                K_a = kern_aux(x, A)
                K_am = K_a.mean(0)
                K_a -= K_am.unsqueeze(0).expand_as(K_a)
                K_a -= K_0m.unsqueeze(1).expand_as(K_a)
                K_a += K_0mm + c
                return K_a

        self.kern = kern

        K = kern(None)
        K[range(n), range(n)] += 1e-12 * n
        self.renorm = [1, 1]
        self.renorm[0] = torch.sqrt( (K[range(n), range(n)]).sum() / n)
        self.V = chol(K, useGPU=useGPU)

        if kernel == 'gaussian':
            iv = KT.integrateGuaussianVector(sigma, base=base, mu_base=mu_base, eta_base=eta_base)
        elif kernel == 'expo':
            iv = KT.integrateExpoVector(sigma, base=base, mu_base=mu_base, eta_base=eta_base)

        o = torch.ones((n, 1))
        kt = iv(x).view(n, 1)

        if c > 0 and base == '1':
            raise NameError("Model not integrable, c > 0 and base is lebesgue")

        if centered:
            ko = kern_aux(x, None) @ o
            coef = -kt.T @ o / n + c + ko.T @ o / n ** 2
            Sig = kt - ko / n + coef * o
        else:
            Sig = kt + c * o

        Sig = tr_solve(Sig, self.V, useGPU=useGPU, transpose=True)


        self.bigRenorm = torch.sqrt((Sig ** 2).sum())
        self.constraint = Sig.view((1,n))/self.bigRenorm
        self.renorm[1] = np.sqrt(2 )
        self.Sig = Sig.view((1, n))

        if base == '1':
            def nu(x):
                return torch.zeros(x.size(0)) + 1
        elif base == 'gaussian':
            def nu(x):
                if type(mu_base) != torch.Tensor:
                    mu_b = torch.tensor(mu_base).view(d)
                else:
                    mu_b = mu_base.view(d)
                if x.ndim > 1:
                    res = torch.exp(-((x - mu_b.unsqueeze(0)) ** 2).sum(1) / (2 * eta_base ** 2) - d * np.log(
                        2 * np.pi * eta_base ** 2) / 2)
                else:
                    res = torch.exp(-(x - mu_b) ** 2 / (2 * eta_base ** 2) - d * np.log(2 * np.pi * eta_base ** 2) / 2)
                return res.view(x.size(0))

        self.nu = nu

        def dz():
            return [torch.zeros((n, 1)), torch.zeros((1, 1))]

        self.dz = dz

    def R(self, a):
        n = self.n
        vals = (self.V).T @ a / (np.sqrt(n) * self.renorm[0])
        equality = self.constraint @ a / self.renorm[1]
        return [vals, equality]

    def Rt(self, dv):
        n = dv[0].size(0)
        t1 = self.V @ dv[0] / (np.sqrt(n) * self.renorm[0])
        t2 = dv[1] * self.constraint.T / self.renorm[1]
        return t1 + t2

    def integral(self, a):
        return self.constraint @ a

    def Rx(self, a, xtest):
        Ktt = self.kern(xtest)
        bid = tr_solve(Ktt, self.V, useGPU=self.useGPU, transpose=True)
        return (bid.T @ a).view(xtest.size(0))/self.bigRenorm

    def px(self, a, xtest):
        return self.Rx(a, xtest) * (self.nu(xtest).view(xtest.size(0)))


class Sreg(object):
    def __init__(self,la):
        self.smoothness = 1/la
        self.la = la
        pass
        
    def Omega(self,f):
        return 0.5*self.la*((f**2).sum())
    
    def Omegas(self):
        def fun_only(f,ispos = True):
            return 0.5*self.smoothness*(f**2).sum()
        def fun_grad(f):
            return (0.5*self.smoothness*(f**2).sum(),self.smoothness*f)
        return fun_only,fun_grad
    
    def recoverPrimal(self,f):
        return self.smoothness*f




class LinearEstimator(object):
    def __init__(self,la = 1,sigma = 1,Niter = None,stype = 'dg',score_param = 'normal',kernel = 'gaussian',
                 centered = False,c = 0,
                 base = 'gaussian',mu_base = None,eta_base = None,target_norm = 1,is_plot = False,
                 x_train = None, y_train = None,
                 al = None, B = None, fitted = False, tol_loss = 1e-2, tol_int = 1e-2, tol_dg = 1e-2):


        self.la = la
        self.sigma = sigma
        self.Niter = Niter
        self.score_param = score_param
        self.kernel = kernel
        self.centered = centered
        self.c = c
        self.base = base
        self.mu_base = mu_base
        self.eta_base = eta_base
        self.target_norm = target_norm
        self.stype = stype
        self.is_plot = is_plot
        self.x_train = x_train
        self.y_train = y_train
        self.al = al
        self.B = B
        self.fitted = fitted
        self.tol_loss = tol_loss
        self.tol_int = tol_int
        self. tol_dg = tol_dg

    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"la": self.la, "sigma": self.sigma,"Niter" : self.Niter,
                "score_param": self.score_param,
                "kernel" : self.kernel,
                "centered" : self.centered,
                "c" : self.c,
                "base" : self.base,
                "mu_base" : self.mu_base,
                "eta_base" : self.eta_base,
                "target_norm" : self.target_norm,
                "stype" : self.stype,
                "is_plot":self.is_plot,
                "x_train":self.x_train,
                "y_train":self.y_train,
                "al":self.al,
                "B":self.B ,
                "fitted":self.fitted,
                "tol_loss":self.tol_loss ,
                "tol_int":self.tol_int ,
                "tol_dg":self.tol_dg
                }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self,X,y = None):
        self.Xtrain = X
        self.regmodel = Sreg(self.la)
        self.lmodel = LMK2(self.sigma,X,kernel = self.kernel,centered = self.centered,c = self.c,base = self.base
                 ,mu_base = self.mu_base,eta_base = self.eta_base,target_norm = self.target_norm)
        self.dModel = densityModel(self.regmodel, self.lmodel,is_plot=self.is_plot,tol_int=self.tol_int)

        self.x_train = X
        self.y_train = y

        try:
            res = self.dModel.prox_method(Niter=self.Niter, stype=self.stype, tol_dg=self.tol_dg,
                                          tol_loss=self.tol_loss)
            self.al = res[0]
            self.B = res[1]
            self.fitted = True
        except OptimError:
            self.fitted = False




    def predict(self,X,y = None):
        if self.fitted:
            return self.lmodel.px(self.B, X)
        else:
            raise FitError('model not fitted')
        pass


    def score(self,X,y = None):
        if self.fitted:
            p = self.lmodel.px(self.B, X)
            integral = self.lmodel.integral(self.B)
            if integral > 1.1 or integral < 0.9:
                logging.warning(f'integral = {integral}; score set to np.nan')
                return np.nan
            if self.score_param == 'normal':
                if (p <= 0).sum() > 0:
                    logging.warning(f'prediction contains negative probabilities; score set to np.nan')
                    return np.nan
                else:
                    return (torch.log(p)).mean()
        else:
            raise FitError('model not fitted')


    def save(self,filename):
        res = {"la": self.la, "sigma": self.sigma, "Niter": self.Niter, "score_param": self.score_param,
         "kernel": self.kernel,
         "centered": self.centered,
         "c": self.c,
         "base": self.base,
        "target_norm" : self.target_norm,
         "mu_base": self.mu_base,
         "eta_base": self.eta_base,
         "is_plot": self.is_plot,
         "al": self.al,
         "B" : self.B,
         "x_train": self.x_train,
         "y_train": self.y_train,
        "stype" : self.stype,
        "fitted" : self.fitted,
        "tol_loss" : self.tol_loss,
        "tol_int" : self.tol_int,
        "tol_dg" : self.tol_dg
         }
        pickle.dump(res,open(filename,"wb"))
    def load(self,filename = None):
        if not(isinstance(filename,type(None))):
            p = pickle.load(open(filename, "rb"))
            for parameter, value in p.items():
                setattr(self, parameter, value)

        self.lmodel = LMK2(self.sigma, self.x_train, kernel=self.kernel, centered=self.centered,
                           c=self.c, base=self.base, mu_base=self.mu_base,
                           eta_base=self.eta_base, target_norm=self.target_norm)
        self.regmodel = Sreg(self.la)
        self.dModel = densityModel(self.regmodel, self.lmodel,is_plot = self.is_plot,tol_int = self.tol_int)



class QKM2(object):
    def __init__(self, sigma, x, kernel='gaussian', centered=False, c=0, base='1', mu_base=None, eta_base=None,
                 useGPU=False, nmax_gpu=None, target_norm=1):
        if c > 0 and base == '1':
            raise NameError("Model not integrable, c > 0 and base is lebesgue")

        n = x.size(0)
        if x.ndim == 1:
            d = 1
        else:
            d = x.size(1)
        self.d = d
        self.n = n
        self.x = x
        self.target_norm = target_norm
        self.useGPU = useGPU

        if kernel == 'expo':
            kernel_fun = KT.expoKernel(sigma)
        elif kernel == 'gaussian':
            kernel_fun = KT.gaussianKernel(sigma)

        def kern_aux(A, B):
            return KT.blockKernComp(A, B, kernel_fun, useGPU=useGPU, nmax_gpu=nmax_gpu)

        if centered == False:
            def kern(A):
                return kern_aux(x, A) + c
        else:
            K_0 = kern_aux(x, None)
            K_0m = K_0.mean(1)
            K_0mm = K_0m.mean()

            def kern(A):
                K_a = kern_aux(x, A)
                K_am = K_a.mean(0)
                K_a -= K_am.unsqueeze(0).expand_as(K_a)
                K_a -= K_0m.unsqueeze(1).expand_as(K_a)
                K_a += K_0mm + c
                return K_a

        self.kern = kern

        K = kern(None)
        K[range(n), range(n)] += 1e-12 * n

        self.renorm = [1, 1]
        self.renorm[0] = torch.sqrt(2 * (K[range(n), range(n)] ** 2).sum() / n) / target_norm
        self.V = chol(K, useGPU=useGPU)

        if base == '1' and c > 0:
            raise NameError("Model not integrable, c > 0 and base is lebesgue")

        if kernel == 'gaussian':
            k1_fun = KT.integrateGuaussianVector(sigma, base=base, mu_base=mu_base, eta_base=eta_base)
            k2_fun = KT.integrateGuaussianMatrix(sigma, base=base, mu_base=mu_base, eta_base=eta_base)
        if kernel == 'expo':
            k1_fun = KT.integrateExpoVector(sigma, base=base, mu_base=mu_base, eta_base=eta_base)
            k2_fun = KT.integrateExpoMatrix(sigma, base=base, mu_base=mu_base, eta_base=eta_base)

        K2 = KT.blockKernComp(x, None, k2_fun, useGPU=useGPU, nmax_gpu=nmax_gpu)

        if not(centered):
            Sig = K2 + c
        else:
            vv = torch.ones((n, 1)) / n
            K0 = kern_aux(x, None)
            vK1 = k1_fun(x).view(n, 1)
            vK0 = K0 @ vv
            vK2 = K2 @ vv
            cK0 = vv.T @ vK0
            cK1 = vv.T @ vK1
            cK2 = vv.T @ vK2

            Mbig = K2 - vK1 @ vK0.T - vK0 @ vK1.T + vK0 @ vK0.T
            vbig = (cK1 * vK0 + cK0 * vK1 - cK0 * vK0 - vK2).expand((n, n))
            cbig = cK2 - 2 * cK1 * cK0 + cK0 * cK0

            Sig = Mbig + vbig + vbig.T + cbig

        Sigg = tr_solve(Sig, self.V, useGPU=useGPU, transpose=True)
        Sig = tr_solve(Sigg.T, self.V, useGPU=useGPU, transpose=True).T

        self.renorm[1] = np.sqrt(2) / target_norm
        self.Sig = Sig
        self.bigRenorm = torch.sqrt((Sig**2).sum())
        self.constraint = self.Sig/self.bigRenorm
        if base == '1':
            def nu(x):
                return 0 * x + 1

        elif base == 'gaussian':
            def nu(x):
                if type(mu_base) != torch.Tensor:
                    mu_b = torch.tensor(mu_base).view(d)
                else:
                    mu_b = mu_base.view(d)
                if x.ndim > 1:
                    res = (torch.exp(-((x - mu_b.unsqueeze(0)) ** 2).sum(1) / (2 * eta_base ** 2) - d * np.log(
                        2 * np.pi * eta_base ** 2) / 2)).view(x.size(0))
                else:
                    res = (torch.exp(
                        -(x - mu_b) ** 2 / (2 * eta_base ** 2) - d * np.log(2 * np.pi * eta_base ** 2) / 2)).view(
                        x.size(0))
                return res

        self.nu = nu

        def dz():
            return [torch.zeros((n, 1)), torch.zeros((1, 1))]

        self.dz = dz

        pass

    def R(self, f):
        res = self.dz()
        n = self.n
        D, U = produceDU(useGPU=self.useGPU)
        Vg = U(self.V)
        fg = U(f)
        res[0] = D((Vg * (fg @ Vg)).sum(0)).view(n, 1) / (np.sqrt(n) * self.renorm[0])
        res[1] = ((fg * U(self.constraint)).sum()).view(1, 1) / (self.renorm[1])

        return res

    def Rt(self, alpha):
        n = self.n
        D, U = produceDU(useGPU=self.useGPU)
        Vg = U(self.V)
        return (D(Vg @ (U(alpha[0].view(n)) * Vg).T) / (np.sqrt(self.n) * self.renorm[0]) + alpha[1] * self.constraint /
                self.renorm[1])

    def integral(self, f):
        return (self.constraint * f).sum()

    def Rx(self, f, xtest):
        Ktt = self.kern(xtest)
        bid = tr_solve(Ktt, self.V, useGPU=self.useGPU, transpose=True)
        return ((bid * (f @ bid)).sum(0)).view(xtest.size(0))/self.bigRenorm

    def px(self, a, xtest):
        return self.Rx(a, xtest) * (self.nu(xtest).view(xtest.size(0)))


class ENreg(object):
    def __init__(self,mu,la,useGPU = False):
        self.smoothness = 1/la
        self.mu = mu
        self.la = la
        def aux(A):
            #return ppm.sureApproach(A,mu=mu,useGPU = useGPU)
            return ppm.mixedApproach(A, 150, q=6, mu = mu, useGPU = useGPU)
        self.pp = aux
        
    def Omega(self,f):
        return self.mu * torch.trace(f) + 0.5*self.la*((f**2).sum())
    
    def Omegas(self):    
        def fun_only(f,ispos = False):
            if not(ispos):
                fpos = self.pp(f)
            else:
                fpos =f
            return 0.5*self.smoothness*(fpos**2).sum()
        def fun_grad(f):
            fpos =self.pp(f)
            return (0.5*self.smoothness*(fpos**2).sum(),self.smoothness*fpos)
        return fun_only,fun_grad
    
    def recoverPrimal(self,f):
        fpos =self.pp(f)
        return (self.smoothness*fpos)
 

    
class logLikelihoodConstrained(object):
    def __init__(self,renorm,tol_constraint):
        self.ka = renorm[0]
        self.eq = 1/renorm[1]
        self.tol_constraint = tol_constraint
        pass

    def L(self,alpha):
        ka = self.ka
        eq = self.eq
        tol = self.tol_constraint
        n = alpha[0].size(0)
        if (alpha[1] /eq > 1+tol) or (alpha[1] / eq < 1- tol) or (alpha[0]<=0).sum() > 0 :
            return torch.tensor(np.inf)
        else:
            return torch.mean(-torch.log(np.sqrt(n)*ka*alpha[0]))
    
    def Ls(self,alpha):
        ka = self.ka
        eq = self.eq
        n = alpha[0].size(0)
        return (-1 + alpha[1]*eq  + torch.mean(-torch.log(-np.sqrt(n)*alpha[0]/ka)))
    
    def Lsprox(self,c,alpha):
        ka = self.ka
        eq = self.eq
        n = alpha[0].size(0)
        def aux_prox(x):
            return 0.5*(x + torch.sqrt(x**2 + 4 * c/ka**2))
        res = []
        res.append( (-ka/(np.sqrt(n)))*aux_prox(-np.sqrt(n)*alpha[0]/ka))
        res.append(alpha[1]-c*eq)
        return res    
    
    
class densityModel(object):
    def __init__(self,reg,lmodel,tol_int = 1e-2,is_plot = True):
        self.reg = reg
        self.lmodel = lmodel
        self.loss = logLikelihoodConstrained(lmodel.renorm,tol_constraint= tol_int)
        self.smoothness = self.reg.smoothness*self.lmodel.target_norm**2
        self.is_plot = is_plot
        self.tol_int = tol_int

    
    def pfd(self,alpha):
        return self.reg.recoverPrimal(minus_l(self.lmodel.Rt(alpha)))

    def primal_loss(self,alpha,B):
        return self.loss.L(self.lmodel.R(B)) + self.reg.Omega(B)

    def dual_loss(self,alpha,B):
        return -(self.loss.Ls(alpha) + self.reg.Omegas()[0](B / self.reg.smoothness, ispos=True))


    
    def prox_method(self,stype = 'dg',Niter = None,tol_loss = 1e-2,tol_dg=1e-2,d=5):

        is_plot = self.is_plot


        if isinstance(Niter,type(None)):
            Niter = int(1000 + 10*np.sqrt(1/self.reg.la))

        stop = stopper(Niter = Niter,stype = stype,tol_int = self.tol_int,tol_dg=tol_dg,tol_loss=tol_loss,d=d)


        O_fun,O_fungrad = self.reg.Omegas()
    
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
    
        Lf = 0.05*self.smoothness
        eta_Lf = 1.1
        
        #Lmax = ka
        while stop.stop() == False:


            Oval,Ograd,Gl_dual = Oms_dual(al2)

            while True:
                c = 1/Lf
                al1 = self.loss.Lsprox(c,add_l(al2,Ograd,-c))
                if Gl_dual(Lf,al1) >=0 or Lf >= 2*eta_Lf*self.smoothness:
                    break
                else:
                    Lf *= eta_Lf

            tk1 = (1 + np.sqrt(1+4*tk**2))/2
            th = (tk - 1)/(tk1)
            al2 = add_l(al1,add_l(al1,al,-1),th)
            al = al1
            tk = tk1
            B = self.pfd(al)
            primal = self.primal_loss(al,B)
            dual = self.dual_loss(al,B)
            integral = self.lmodel.integral(B)
            stop.update(dual,integral = integral,dg = primal-dual)
        logger.info(f'finished optimization with dual gap : {primal-dual}, loss : {dual}, integral : {integral}')

        if is_plot:
            plt.figure()
            plt.plot(list(range(len(stop.losses))),stop.losses)
            plt.show()

        return (al,B)




class QuadraticEstimator(object):
    def __init__(self,la = 1,sigma = 1,stype = 'auto',Niter = None,score_param = 'normal',
                 mu = None,kernel = 'gaussian',centered = False,c = 0,
                 base = 'gaussian',mu_base = None,eta_base = None,is_plot = False,x_train = None,y_train = None,
                 al = None,B = None,fitted = False,tol_loss = 1e-2,tol_int = 1e-2,tol_dg = 1e-2):
        self.la = la
        self.sigma = sigma
        self.Niter = Niter
        self.stype = stype
        self.score_param = score_param
        self.mu = mu
        self.kernel = kernel
        self.centered = centered
        self. c = c
        self.base = base
        self.mu_base = mu_base
        self.eta_base = eta_base
        self.is_plot = is_plot
        self.x_train = x_train
        self.y_train = y_train
        self.al = al
        self.B = B
        self.fitted = fitted
        self.tol_int = tol_int
        self.tol_loss = tol_loss
        self.tol_dg = tol_dg





    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"la": self.la, "sigma": self.sigma,"Niter" : self.Niter,"score_param": self.score_param,
                "mu" : self.mu,
                "kernel" : self.kernel,
                "centered" : self.centered,
                "c" : self.c,
                "base" : self.base,
                "mu_base" : self.mu_base,
                "eta_base" : self.eta_base,
                "is_plot" : self.is_plot,
                "al" : self.al,
                "B" : self.B,
                "x_train" : self.x_train,
                "y_train" : self.y_train,
                "stype" : self.stype,
                "fitted" : self.fitted,
                "tol_loss" : self.tol_loss,
                "tol_dg" : self.tol_dg,
                "tol_int" : self.tol_int
                }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self,X,y = None):

        if isinstance(self.mu,type(None)):
            self.mu = self.la*0.01


        self.target_norm = np.sqrt(self.mu)
        self.lmodel = QKM2(self.sigma, X, kernel=self.kernel, centered=self.centered,
                           c=self.c, base=self.base, mu_base=self.mu_base,
                           eta_base=self.eta_base, target_norm=self.target_norm)
        self.regmodel = ENreg(self.la, self.mu)
        self.dModel = densityModel(self.regmodel, self.lmodel,is_plot = self.is_plot,tol_int = self.tol_int)
        self.x_train = X
        self.y_train = y
        try:
            res = self.dModel.prox_method(Niter = self.Niter,stype = self.stype,tol_dg=self.tol_dg,tol_loss=self.tol_loss)
            self.al = res[0]
            self.B = res[1]
            self.fitted = True
        except OptimError:
            self.fitted = False




    def predict(self,X,y = None):
        if self.fitted:
            return self.lmodel.px(self.B, X)
        else:
            raise FitError('model not fitted')
        pass


    def score(self,X,y = None):
        if self.fitted:
            p = self.lmodel.px(self.B, X)
            integral = self.lmodel.integral(self.B)
            if integral > 1.1 or integral < 0.9:
                logging.warning(f'integral = {integral}; score set to np.nan')
                return np.nan
            if self.score_param == 'normal':
                if (p <= 0).sum() > 0:
                    logging.warning(f'prediction contains negative probabilities; score set to np.nan')
                    return np.nan
                else:
                    return (torch.log(p)).mean()
        else:
            raise FitError('model not fitted')


    def save(self,filename):
        res = {"la": self.la, "sigma": self.sigma, "Niter": self.Niter, "score_param": self.score_param,
         "mu": self.mu,
         "kernel": self.kernel,
         "centered": self.centered,
         "c": self.c,
         "base": self.base,
         "mu_base": self.mu_base,
         "eta_base": self.eta_base,
         "is_plot": self.is_plot,
         "al": self.al,
         "B" : self.B,
         "x_train": self.x_train,
         "y_train": self.y_train,
        "stype" : self.stype,
        "fitted" : self.fitted,
        "tol_loss" : self.tol_loss,
        "tol_int" : self.tol_int,
        "tol_dg" : self.tol_dg
         }
        pickle.dump(res,open(filename,"wb"))


    def load(self,filename = None):
        if not(isinstance(filename,type(None))):
            p = pickle.load(open(filename, "rb"))
            for parameter, value in p.items():
                setattr(self, parameter, value)
        if isinstance(self.mu,type(None)):
            self.mu = self.la*0.01
        self.target_norm = np.sqrt(self.mu)
        self.lmodel = QKM2(self.sigma, self.x_train, kernel=self.kernel, centered=self.centered,
                           c=self.c, base=self.base, mu_base=self.mu_base,
                           eta_base=self.eta_base, target_norm=self.target_norm)
        self.regmodel = ENreg(self.la, self.mu)
        self.dModel = densityModel(self.regmodel, self.lmodel,is_plot = self.is_plot,tol_int = self.tol_int)



        

###########################################################
#NW method
############################################################
            

            

def logloss(x):
    return -(torch.log(x)).sum()

def logloss_prox(eps,x):
    return (x+ torch.sqrt(x**2 + 4*eps))/2

class loglossNW(object):
    def __init__(self,eps):
        self.eps = eps
        self.L = logloss
        self.proxL = logloss_prox
        return None
    
    def Leps(self,alpha):
        eps = self.eps
        y = self.proxL(eps,alpha)
        return self.L(y) + ((alpha-y)**2).sum()/(2*eps)
    
    def gradLeps(self,alpha):
        eps = self.eps
        return (alpha - self.proxL(eps,alpha))/eps

class kernelModel(object):
    def __init__(self,sigma,x,kernel = 'gaussian',c = 0,base = '1',mu_base = None,eta_base = None,useGPU = False,nmax_gpu = None):
        n = x.size(0)
        if x.ndim == 1:
            d = 1
        else:
            d = x.size(1)
    
        self.n = n
        self.x = x
        self.d = d
        self.useGPU = useGPU
        
        if kernel == 'expo':
            kernel_fun = KT.expoKernel(sigma)
        elif kernel == 'gaussian':
            kernel_fun = KT.gaussianKernel(sigma)
            
        def kern_aux(A,B):
            return KT.blockKernComp(A, B, kernel_fun, useGPU = useGPU, nmax_gpu = nmax_gpu)
        def kern(A):
            return kern_aux(x,A)+c
        self.kern = kern
        
        
        K =  kern(None)
        #K[range(n), range(n)] += 1e-12 * n
        self.K = K
        self.ka = (K[range(n),range(n)]).sum()/n

        
        if base == '1':
            def nu(x):
                return 0*x +1
        elif base == 'gaussian':
            def nu(x):
                if type(mu_base) != torch.Tensor:
                    mu_b = torch.tensor(mu_base).view(d)
                else:
                    mu_b = mu_base.view(d)
                if x.ndim > 1:
                    res = torch.exp(-((x-mu_b.unsqueeze(0))**2).sum(1)/(2*eta_base**2)-d*np.log(2*np.pi*eta_base**2)/2).view(x.size(0))
                else:
                    res = torch.exp(-(x-mu_b)**2/(2*eta_base**2)-d*np.log(2*np.pi*eta_base**2)/2).view(x.size(0))
                return res

        self.nu = nu
        
        if kernel == 'gaussian':
            iv = KT.integrateGuaussianVector(sigma,base = base,mu_base = mu_base,eta_base = eta_base)
        elif kernel == 'expo':
            iv = KT.integrateExpoVector(sigma,base = base,mu_base = mu_base,eta_base = eta_base)
        o = torch.ones((n,1))
        kt = iv(x).view(n,1)
        if c > 0 and base == '1':
            raise NameError("Model not integrable, c > 0 and base is lebesgue")

        self.Sig = kt +c*o
        self.renorm = torch.sqrt((self.Sig**2).sum())

        
        def dz():
            return torch.zeros((n,1))
        self.dz = dz
    def integral(self,a):
        return self.Sig.T @ a/self.renorm
        
    def Rx(self,a,xtest):
        Ktt = self.kern(xtest)
        return (Ktt.T @ a/self.renorm).view(xtest.size(0))
    
    def px(self,a,xtest):
        return (self.Rx(a,xtest)*(self.nu(xtest).view(xtest.size(0)))).view(xtest.size(0))

class densityModelNW(object):
    def __init__(self,kmodel,la,eps = 0.001,is_plot = True):
        self.kmodel = kmodel
        n = kmodel.n
        self.lossmodel = loglossNW(eps)
        self.la = la
        
        self.c = la*kmodel.n
        self.constraint = kmodel.Sig/kmodel.renorm
        self.smoothness = n**2 * kmodel.ka**2/eps + n**2 * kmodel.ka*la
        self.is_plot = is_plot
        
    def proj(self,x):
        a = self.constraint
        n = x.size(0)
        u = x/a
        us,ind = torch.sort(u,dim = 0,descending = True)
        a2s = (a**2)[ind.view(n),:]
        uasc = (us*a2s).cumsum(0)
        a2sc = a2s.cumsum(0)
    
        A = uasc-a2sc*us
    
        k = 0
    
        while k< n-1  and A[k+1,0]<1  :
            k+=1

        la = (uasc[k,0] -1)/a2sc[k,0]

        return torch.clamp(x-la*a,min = 0)
    
    def Loss_tot(self,x):
        K = self.kmodel.K
        return self.c*x.T @ K @ x /2 + self.lossmodel.Leps(K@x)
    
    def grad(self,x):
        K = self.kmodel.K
        grad = self.c*K@x + K@self.lossmodel.gradLeps(K@x)
        return grad

    
    def FISTA(self,stype = 'auto',Niter = None,tol_loss = 1e-2,d=5):
        is_plot = self.is_plot

        if isinstance(Niter, type(None)):
            Niter = int(100 + 10*np.sqrt(self.smoothness))

        stop = stopper(Niter=Niter, stype=stype, tol_int=None, tol_dg=None, tol_loss=tol_loss, d=d,maximize = False)

        def Gl(alpha):
            f_alpha = self.Loss_tot(alpha)
            g_alpha = self.grad(alpha)
            def aux(Lf,talpha):
                dd = talpha- alpha
                f_approx = f_alpha + g_alpha.T @ dd + 0.5*Lf*(dd**2).sum()
                f_alphat = self.Loss_tot(talpha)
                return f_approx-f_alphat
            return f_alpha,g_alpha,aux

        al = self.kmodel.dz()
        al2 = self.kmodel.dz()
    
        tk = 1
    
        loss_iter = []
    
        Lf = 0.01*self.smoothness
        eta_Lf = 1.1

        while stop.stop()==False:
            fun,grad,GL = Gl(al2)

            while True:
                gamma = 1/Lf
                al1 = self.proj(al2-gamma*grad)
                if GL(Lf,al1) >=0:
                    break
                else:
                    Lf *= eta_Lf
            tk1 = (1 + np.sqrt(1+4*tk**2))/2
            th = (tk - 1)/(tk1)
            al2 = al1 + th*(al1-al)
            al = al1
            tk = tk1

            loss_val = self.Loss_tot(al)
            stop.update(loss_val)

        logger.info(f'finished optimization after {stop.iter} iterations with loss : {loss_val}, integral : {self.kmodel.integral(al)}')

        if is_plot:
            plt.plot(list(range(len(loss_iter))),(loss_iter))
            plt.show()
        return(al)

class NadarayaWatsonEstimator(object):
    def __init__(self,**params):
        default_params = {
            "Niter" : None,
            "score_param" : "normal",
            "kernel" : "gaussian",
            "c" : 0,
            "base" : "1",
            "mu_base" : None,
            "eta_base" : None,
            "eps" : 0.001,
             "is_plot" : False,
            "al" : None,
            "x_train" : None,
            "y_train" : None,
            "stype" : "auto",
            "fitted" : False,
            "tol_loss" : 1e-2
        }
        for parameter, value in default_params.items():
            setattr(self, parameter, value)

        for parameter, value in params.items():
            setattr(self, parameter, value)
        pass


    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        list_to_get = ["la","sigma","Niter","score_param","kernel","c","base","mu_base","eta_base","eps",
                       "is_plot","al","x_train","y_train","stype","fitted","tol_loss"]
        res = {}
        for key in list_to_get:
            res[key] = getattr(self,key)
        return res

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self,X,y = None):

        self.kmodel = kernelModel(self.sigma, X, kernel=self.kernel, c=self.c, base=self.base, mu_base=self.mu_base,
                                  eta_base=self.eta_base)
        self.densityModel = densityModelNW(self.kmodel, self.la, eps= self.eps,is_plot=self.is_plot)
        self.x_train = X
        self.y_train = y

        try:
            al = self.densityModel.FISTA(Niter = self.Niter,stype = self.stype,tol_loss=self.tol_loss)
            self.al = al
            self.fitted = True
        except OptimError:
            self.fitted = False



    def predict(self,X,y = None):
        if self.fitted:
            return self.kmodel.px(self.al, X)
        else:
            raise FitError('model not fitted')
        pass


    def score(self,X,y = None):
        if self.fitted:
            p = self.kmodel.px(self.al, X)
            if self.score_param == 'normal':
                if (p <= 0).sum() > 0:
                    logging.warning(f'prediction contains negative probabilities; score set to np.nan')
                    return np.nan
                else:
                    return (torch.log(p)).mean()
        else:
            raise FitError('model not fitted')

    def save(self, filename):
        list_to_save = ["la", "sigma", "Niter", "score_param", "kernel", "c", "base", "mu_base", "eta_base", "eps",
                       "is_plot", "al", "x_train", "y_train", "stype", "fitted", "tol_loss"]
        res = {}
        for key in list_to_save:
            res[key] = getattr(self, key)
        pickle.dump(res, open(filename, "wb"))

    def load(self, filename=None):
        if not (isinstance(filename, type(None))):
            p = pickle.load(open(filename, "rb"))
            for parameter, value in p.items():
                setattr(self, parameter, value)

        self.kmodel = kernelModel(self.sigma, self.x_train, kernel=self.kernel, c=self.c, base=self.base, mu_base=self.mu_base,
                                  eta_base=self.eta_base)
        self.densityModel = densityModelNW(self.kmodel, self.la, eps=self.eps, is_plot=self.is_plot)


########################################################
#
#########################################################

class kernelExpoModel(object):
    def __init__(self,sigma,x,rgrid,ngrid,cgrid = None,kernel = 'gaussian',centered = False,c = 0,base = '1',mu_base = None,eta_base = None,useGPU = False,nmax_gpu = None):
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
            
        def kern_aux(A,B):
            return KT.blockKernComp(A, B, kernel_fun, useGPU = useGPU, nmax_gpu = nmax_gpu)
        if centered == False:
            def kern(A):
                return kern_aux(x,A)+c
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
        self.K = K
        self.ka = np.sqrt((K[range(n),range(n)]).sum())
        K[range(n),range(n)]+= 1e-12*n
        self.V = chol(K,useGPU = useGPU)
        self.useGPU = useGPU
        self.nmax_gpu = nmax_gpu
        
        if cgrid == 0 or isinstance(cgrid,type(None)):
            cgrid = torch.zeros((ngrid,d))
        else:
            cgrid = cgrid.view((1,d))
            cgrid = cgrid.expand((ngrid,d))
        xgrid = 2*rgrid*torch.rand((ngrid,d))-rgrid + cgrid
        self.ngrid = ngrid
        self.xgrid = xgrid
        #self.lgrid = xgrid[1]-xgrid[0]
        self.lgrid = (2*rgrid)**d/ngrid
        Kgrid =  kern(xgrid)
        self.Vgrid = tr_solve(Kgrid,self.V,useGPU = self.useGPU,transpose = True)
        def dz():
            return torch.zeros((n,1))
        self.dz = dz
        
    def R(self,a):
        return self.V.T@a
        
    def Rx(self,a,xtest):
        n = self.n
        Ktt = self.kern(xtest)
        bid = tr_solve(Ktt,self.V,useGPU = self.useGPU,transpose = True)
        return (bid.T @ a).view(xtest.size(0))
    
    def pdata(self,a):
        V = self.V
        n = self.n
        ggrid = a.T@self.Vgrid
        mggrid = ggrid.max()
        ggrid -= mggrid
        num_p = torch.exp(ggrid)
        int_estimate = self.lgrid*num_p.sum()
        pdata = torch.exp(a.T@V - mggrid).view(n,1)/int_estimate
        return pdata
    
    def Jacobian(self,a):
        V = self.V
        n = self.n
        ggrid = a.T@self.Vgrid
        mggrid = ggrid.max()
        ggrid -= mggrid
        num_p = torch.exp(ggrid)
        int_estimate = self.lgrid*num_p.sum()
        expectation_estimate = ((self.lgrid*num_p.view(self.ngrid)*self.Vgrid).sum(1)).view(n)/int_estimate
        pdata = torch.exp(a.T@V - mggrid).view(n)/int_estimate
        return (V - expectation_estimate[:,None])*pdata[None,:],pdata.view(n,1)
        
        
    
    def px(self,a,xtest):
        V = self.V
        n = self.n
        ggrid = a.T@self.Vgrid
        mggrid = ggrid.max()
        ggrid -= mggrid
        num_p = torch.exp(ggrid)
        int_estimate = self.lgrid*num_p.sum()
        rx = self.Rx(a,xtest)-mggrid
        return(torch.exp(rx)/int_estimate)

class kernelExpoModelTer(object):
    def __init__(self,sigma,x,ngrid,kernel = 'gaussian',centered = False,c = 0,useGPU = False,nmax_gpu = None,base = 'gaussian',mu_base = 0,eta_base = 1):
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
            
        def kern_aux(A,B):
            return KT.blockKernComp(A, B, kernel_fun, useGPU = useGPU, nmax_gpu = nmax_gpu)
        if centered == False:
            def kern(A):
                return kern_aux(x,A)+c
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
        self.K = K
        self.ka = np.sqrt((K[range(n),range(n)]).sum())
        K[range(n),range(n)]+= 1e-12*n
        self.V = chol(K,useGPU = useGPU)
        self.useGPU = useGPU
        self.nmax_gpu = nmax_gpu
        
        
        if type(mu_base) != torch.Tensor:
            mu_b = torch.tensor(mu_base).expand(d)
        else:
            mu_b = mu_base.view(d)
        
        if base == '1':

            volume = (2*eta_base)**d
            def nu(x):
                return (1 - ((x - mu_b.unsqueeze(0)) > eta_base).sum(1).double() - ((x - mu_b.unsqueeze(0)) < -eta_base).sum(1).double()).clamp(min=0)/volume
            self.nu = nu
            xgrid = mu_b.unsqueeze(0) + eta_base*(2*torch.rand((ngrid,d))-1)
        elif base == 'gaussian':
            def nu(x):
                if x.ndim > 1:
                    res = torch.exp(-((x-mu_b.unsqueeze(0))**2).sum(1)/(2*eta_base**2)-d*np.log(2*np.pi*eta_base**2)/2).view(x.size(0))
                else:
                    res = torch.exp(-(x-mu_b)**2/(2*eta_base**2)-d*np.log(2*np.pi*eta_base**2)/2).view(x.size(0))
                return res
            self.nu = nu
            eps = torch.randn((ngrid,d))
            xgrid = mu_b.unsqueeze(0)+eta_base * eps

        self.nudata = (self.nu(x)).view(n)
        self.ngrid = ngrid
        self.xgrid = xgrid
        #self.lgrid = xgrid[1]-xgrid[0]
        Kgrid =  kern(xgrid)
        self.Vgrid = tr_solve(Kgrid,self.V,useGPU = self.useGPU,transpose = True)
        def dz():
            return torch.zeros((n,1))
        self.dz = dz
        
    def R(self,a):
        return self.V.T@a
        
    def Rx(self,a,xtest):
        n = self.n
        Ktt = self.kern(xtest)
        bid = tr_solve(Ktt,self.V,useGPU = self.useGPU,transpose = True)
        return (bid.T @ a).view(xtest.size(0))
    
    def pdata(self,a):
        V = self.V
        n = self.n
        ggrid = a.T@self.Vgrid
        mggrid = ggrid.max()
        ggrid -= mggrid
        num_p = torch.exp(ggrid)
        int_estimate = num_p.mean()
        pdata = torch.exp(a.T@V - mggrid).view(n,1)*self.nudata.view(n,1)/int_estimate
        return pdata
    
    def Jacobian(self,a):
        V = self.V
        n = self.n
        ggrid = a.T@self.Vgrid
        mggrid = ggrid.max()
        ggrid -= mggrid
        num_p = torch.exp(ggrid)
        int_estimate = num_p.mean()
        expectation_estimate = ((num_p.view(self.ngrid)*self.Vgrid).mean(1)).view(n)/int_estimate
        pdata = torch.exp(a.T@V - mggrid).view(n)*self.nudata/int_estimate
        return (V - expectation_estimate[:,None])*pdata[None,:],pdata.view(n,1)
        
        
    
    def px(self,a,xtest):
        V = self.V
        n = self.n
        ggrid = a.T@self.Vgrid
        mggrid = ggrid.max()
        ggrid -= mggrid
        num_p = torch.exp(ggrid)
        int_estimate = num_p.mean()
        rx = self.Rx(a,xtest)-mggrid
        nux = self.nu(xtest)
        return(torch.exp(rx)*nux/int_estimate)

class kernelExpoModelBis(object):
    def __init__(self,sigma,x,ngrid,base_sampler,base_density,kernel = 'gaussian',centered = False,c = 0,useGPU = False,nmax_gpu = None):
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
            
        def kern_aux(A,B):
            return KT.blockKernComp(A, B, kernel_fun, useGPU = useGPU, nmax_gpu = nmax_gpu)
        if centered == False:
            def kern(A):
                return kern_aux(x,A)+c
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
        self.K = K
        self.ka = np.sqrt((K[range(n),range(n)]).sum())
        K[range(n),range(n)]+= 1e-12*n
        self.V = chol(K,useGPU = useGPU)
        self.useGPU = useGPU
        self.nmax_gpu = nmax_gpu
        
        
        xgrid = base_sampler(ngrid)
        self.nu= base_density
        self.nudata = base_density(x).view(n)
        self.ngrid = ngrid
        self.xgrid = xgrid
        #self.lgrid = xgrid[1]-xgrid[0]
        Kgrid =  kern(xgrid)
        self.Vgrid = tr_solve(Kgrid,self.V,useGPU = self.useGPU,transpose = True)
        def dz():
            return torch.zeros((n,1))
        self.dz = dz
        
    def R(self,a):
        return self.V.T@a
        
    def Rx(self,a,xtest):
        n = self.n
        Ktt = self.kern(xtest)
        bid = tr_solve(Ktt,self.V,useGPU = self.useGPU,transpose = True)
        return (bid.T @ a).view(xtest.size(0))
    
    def pdata(self,a):
        V = self.V
        n = self.n
        ggrid = a.T@self.Vgrid
        mggrid = ggrid.max()
        ggrid -= mggrid
        num_p = torch.exp(ggrid)
        int_estimate = num_p.mean()
        pdata = torch.exp(a.T@V - mggrid).view(n,1)*self.nudata.view(n,1)/int_estimate
        return pdata
    
    def Jacobian(self,a):
        V = self.V
        n = self.n
        ggrid = a.T@self.Vgrid
        mggrid = ggrid.max()
        ggrid -= mggrid
        num_p = torch.exp(ggrid)
        int_estimate = num_p.mean()
        expectation_estimate = ((num_p.view(self.ngrid)*self.Vgrid).mean(1)).view(n)/int_estimate
        pdata = torch.exp(a.T@V - mggrid).view(n)*self.nudata/int_estimate
        return (V - expectation_estimate[:,None])*pdata[None,:],pdata.view(n,1)
        
        
    
    def px(self,a,xtest):
        V = self.V
        n = self.n
        ggrid = a.T@self.Vgrid
        mggrid = ggrid.max()
        ggrid -= mggrid
        num_p = torch.exp(ggrid)
        int_estimate = num_p.mean()
        rx = self.Rx(a,xtest)-mggrid
        nux = self.nu(xtest)
        return(torch.exp(rx)*nux/int_estimate)

class densityModelExpo(object):
    def __init__(self,kmodel,la,eps = 0.001,is_plot = False):
        self.kmodel = kmodel
        n = kmodel.n
        self.lossmodel = loglossNW(eps)
        self.la = la
        
        self.c = la*kmodel.n
        self.smoothness =  kmodel.ka**2/eps + n*la
        self.is_plot = is_plot
        
    
    def Loss_tot(self,x):
        return self.c*(x**2).sum()/2 + self.lossmodel.Leps(self.kmodel.pdata(x))
    
    def grad(self,x):
        J,pdata = self.kmodel.Jacobian(x)
        grad = self.c*x + J@self.lossmodel.gradLeps(pdata)
        return grad
    

    
    def GD(self,stype = 'auto',Niter = None,N_restarts = 3,tol_loss = 1e-2,d=5):
        is_plot = self.is_plot
        l_restarts = []

        for kk in range(N_restarts):
            pert = 0.1
            al = self.kmodel.dz()
            al = al + pert*torch.randn(al.size())

            if isinstance(Niter, type(None)):
                Niter = int(100 + 10 * np.sqrt(self.smoothness))

            stop = stopper(Niter=Niter, stype=stype, tol_int=None, tol_dg=None, tol_loss=tol_loss, d=d, maximize=False)

            def Gl(alpha):
                f_alpha = self.Loss_tot(alpha)
                g_alpha = self.grad(alpha)
                def aux(Lf,talpha):
                    dd = talpha- alpha
                    f_approx = f_alpha + g_alpha.T @ dd + 0.5*Lf*(dd**2).sum()
                    f_alphat = self.Loss_tot(talpha)
                    return f_approx-f_alphat
                return f_alpha,g_alpha,aux

            al2 = al.clone()
            tk = 1
            Lf = 0.000001*self.smoothness
            eta_Lf = 1.1
            while stop.stop()== False:
                fun,grad,GL = Gl(al2)
                while True:
                    gamma = 1/Lf
                    al1 = al2-gamma*grad
                    if GL(Lf,al1) >=0 :
                        #or Lf >= self.smoothness
                        break
                    else:
                        Lf *= eta_Lf
                #print('Lf/Lmax = {}'.format(Lf/self.smoothness))
                tk1 = (1 + np.sqrt(1+4*tk**2))/2
                th = (tk - 1)/(tk1)
                al2 = al1 + th*(al1-al)
                al = al1
                tk = tk1

                loss_val = self.Loss_tot(al)
                stop.update(loss_val)
            logger.info(f'restart number {kk} -- finished optimization after {stop.iter} iterations with loss : {loss_val}')
            if is_plot:
                plt.plot(list(range(len(stop.losses))), stop.losses)
                plt.show()
            l_restarts.append((al.clone(), stop.losses[-1]))
        return l_restarts[min(range(len(l_restarts)),key = lambda i : l_restarts[i][1])][0]



class ExpoEstimator(object):
    def __init__(self,**params):
        default_params = {
            "la" : 1,
            "sigma" : 1,
            "rgrid" : 4,
            "ngrid" : 100,
            "cgrid" : None,
            "Niter" : None,
            "N_restarts" : 3,
            "score_param" : "normal",
            "kernel" : "gaussian",
            "c" : 0,
            "centered" : False,
            "base" : "1",
            "mu_base" : None,
            "eta_base" : None,
            "eps" : 0.001,
             "is_plot" : False,
            "al" : None,
            "x_train" : None,
            "y_train" : None,
            "stype" : "auto",
            "fitted" : False,
            "tol_loss" : 1e-2
        }
        for parameter, value in default_params.items():
            setattr(self, parameter, value)

        for parameter, value in params.items():
            setattr(self, parameter, value)
        pass


    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        list_to_get = ["la","sigma","Niter","score_param","kernel","c","centered","base","mu_base","eta_base","eps",
                       "is_plot","al","x_train","y_train","stype","fitted","tol_loss","rgrid","ngrid","cgrid","N_restarts"]
        res = {}
        for key in list_to_get:
            res[key] = getattr(self,key)
        return res

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self,X,y = None):
        self.kmodel = kernelExpoModel(self.sigma,X,self.rgrid,self.ngrid,cgrid = self.cgrid,
                                            kernel = self.kernel ,centered = self.centered,c = self.c,
                                      base = self.base,mu_base = self.mu_base,eta_base = self.eta_base)

        self.densityModel = densityModelExpo(self.kmodel,self.la,eps=self.eps)
        self.x_train = X
        self.y_train = y

        try:
            al = self.densityModel.GD(Niter = self.Niter,stype = self.stype,tol_loss=self.tol_loss)
            self.al = al
            self.fitted = True
        except OptimError:
            self.fitted = False

    def predict(self,X,y = None):
        if self.fitted:
            return self.kmodel.px(self.al, X)
        else:
            raise FitError('model not fitted')
        pass


    def score(self,X,y = None):
        if self.fitted:
            p = self.kmodel.px(self.al, X)
            if self.score_param == 'normal':
                if (p <= 0).sum() > 0:
                    logging.warning(f'prediction contains negative probabilities; score set to np.nan')
                    return np.nan
                else:
                    return (torch.log(p)).mean()
        else:
            raise FitError('model not fitted')

    def save(self, filename):
        list_to_save = ["la","sigma","Niter","score_param","kernel","c","centered","base","mu_base","eta_base","eps",
                       "is_plot","al","x_train","y_train","stype","fitted","tol_loss","rgrid","ngrid","cgrid","N_restarts"]
        res = {}
        for key in list_to_save:
            res[key] = getattr(self, key)
        pickle.dump(res, open(filename, "wb"))

    def load(self, filename=None):
        if not (isinstance(filename, type(None))):
            p = pickle.load(open(filename, "rb"))
            for parameter, value in p.items():
                setattr(self, parameter, value)

        self.kmodel = kernelExpoModel(self.sigma, self.x_train, self.rgrid, self.ngrid, cgrid=self.cgrid,
                                      kernel=self.kernel, centered=self.centered, c=self.c,
                                      base=self.base, mu_base=self.mu_base, eta_base=self.eta_base)

        self.densityModel = densityModelExpo(self.kmodel, self.la, eps=self.eps)










