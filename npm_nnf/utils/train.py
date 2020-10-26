import numpy as np
import torch
import pandas as pd
import os
import pickle
from copy import deepcopy


from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

from scipy.stats import randint
from scipy.stats import uniform
from scipy.stats import loguniform

import importlib

import optuna
optuna.logging.set_verbosity(0)


import warnings
warnings.filterwarnings('ignore')

import sys
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')



import npm_nnf.utils.utils_kernels as KT
import npm_nnf.density_estimation.utils_density as utils
import npm_nnf.density_estimation.utils_data as generators
import npm_nnf.utils.utils_train as utils_train

import sys, getopt
import json

torch.set_default_dtype(torch.float64)


def main(argv):
   n_jobs = None
   X = None
   try:
      opts, args = getopt.getopt(argv,"hc:nd",["config=","njobs=","dataset="])
   except getopt.GetoptError:
      print('optimize_parameters.py -c <config path>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print('optimize_parameters.py -c <config path>')
         sys.exit()
      elif opt in ("-c", "--config"):
         with open(arg) as config_file:
            data = json.load(config_file)
      elif opt in ("-n","--njobs"):
         n_jobs = arg
      elif opt in ("-d","--dataset"):
         X = torch.load(arg)
   if isinstance(X,type(None)):
      X = torch.load(os.path.join(data['data_set_path'],data['data_set_file']))
   d = X.size(1)
   n = X.size(0)
   if isinstance(n_jobs,type(None)):
      n_jobs = data["n_jobs"]
   version = data["version"]
   eta = data["eta"]
   cv = data["cv"]
   save_path  = data["save_path"]
   iii = importlib.import_module(data['model'][0])
   model = getattr(iii,data['model'][1])

   params = data['parameters']
   if "mu_base" in fixed_params.keys():
      if isinstance(fixed_params['mu_base'],int):
         fixed_params['mu_base'] = torch.zeros((d,))


   utils_train.perform_study(model, X, fixed_params=fixed_params, variable_params=variable_params, cv=cv,
                                   prune=prune,
                                   n_trials=n_trials, save_path=save_path, version=version, eta=eta, n_jobs=n_jobs)



if __name__ == "__main__":
   main(sys.argv[1:])