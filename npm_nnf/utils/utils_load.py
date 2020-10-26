
import torch

import os
import pickle


import importlib

import optuna
optuna.logging.set_verbosity(0)


import warnings
warnings.filterwarnings('ignore')




import json

torch.set_default_dtype(torch.float64)

def load(config_file_path):
    with open(config_file_path) as config_file:
        data = json.load(config_file)
    return load_data(data)

def load_data(data):
    version = data["version"]
    file_path = data['save_path']
    file_name = data['save_name']
    model_name = data['model'][1]
    ds = pickle.load(open(os.path.join(data['data_set_path'], data['data_set_file']),'rb'))
    n = ds.X.size(0)
    d = ds.X.size(1)
    file_name = f'{file_name}_{model_name}_dimension{d}_datasetsize{n}'
    if isinstance(version, type(None)):
        version = 0
        while os.path.isfile(os.path.join(file_path, f'{file_name}_{version + 1}.pickle')):
            version += 1
    file_name = f'{file_name}_{version}.pickle'
    file = os.path.join(file_path, file_name)
    model_version = pickle.load(open(file,'rb'))



    iii = importlib.import_module(data['model'][0])
    model = getattr(iii, model_name)
    clf = model()
    if "best_parameters" in model_version:
        clf.set_params(**model_version['best_parameters'])
    clf.load()
    return (clf,ds)

