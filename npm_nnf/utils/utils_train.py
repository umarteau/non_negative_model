
import os
import pickle
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

from scipy.stats import randint
from scipy.stats import uniform
from scipy.stats import loguniform
import optuna
optuna.logging.set_verbosity(0)

import warnings
warnings.filterwarnings('ignore')

import sys
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')


import npm_nnf.utils.utils_kernels as KT
import npm_nnf.density_estimation.utils_density as utils
import npm_nnf.density_estimation.utils_data_generator as generators

from sklearn.model_selection import KFold, StratifiedKFold, ParameterGrid, ParameterSampler
from sklearn import metrics
import numpy
import torch
import pandas


class PrunedCV_altered:

    """PrunedCV applied pruning to cross-validation. Based on scores
    from initial splits (folds) is decides whether it's worth to
    continue the cross-validation. If not it stops the process and returns
    estimated final score.

    If the trial is worth checking (the initial scores are
    better than the best till the time or withing tolerance border) it's equivalent
    to standard cross-validation. Otherwise the trial is pruned.


    Args:
        cv:
            Number of folds to be created for cross-validation
        tolerance:
            Default = 0.1.
            The value creates boundary
            around the best score.
            If ongoing scores are outside the boundary,
            the trial is pruned.
        splits_to_start_pruning:
            Default = 2
            The fold at which pruning may be first applied.
        minimize:
            Default = True
            The direction of the optimization.

    Usage example:

        from lightgbm import LGBMRegressor
        from sklearn.datasets import fetch_california_housing
        from prunedcv import PrunedCV
        import numpy as np

        data = fetch_california_housing()
        x = data['data']
        y = data['target']

        pruner = PrunedCV(cv=8, tolerance=.1)

        model1 = LGBMRegressor(max_depth=25)
        model2 = LGBMRegressor(max_depth=10)
        model3 = LGBMRegressor(max_depth=2)

        pruner.cross_val_score(model1, x, y)
        pruner.cross_val_score(model2, x, y)
        pruner.cross_val_score(model3, x, y)

        print('best score: ', round(sum(pruner.best_splits_list_) / len(pruner.best_splits_list_),4))
            """

    def __init__(self,
                 cv,
                 tolerance=0.1,
                 splits_to_start_pruning=2,
                 minimize=True):

        if not isinstance(cv, int):
            raise TypeError
        if cv < 2:
            raise ValueError

        self.cv = cv
        self.set_tolerance(tolerance)
        self.splits_to_start_pruning = splits_to_start_pruning
        self.minimize = minimize
        self.prune = False
        self.cross_val_score_value = None
        self.current_splits_list_ = []

        self.best_splits_list_ = []
        self.first_run_ = True

    def set_tolerance(self,
                      tolerance):
        """Set tolerance value

        Args:
            tolerance:
            The value creates boundary
            around the best score.
            If ongoing scores are outside the boundary,
            the trial is pruned.
        """

        if not isinstance(tolerance, float):
            raise TypeError
        if tolerance < 0:
            raise ValueError

        self.tolerance = tolerance

    def cross_val_score(self,
                        model,
                        x,
                        y = None,
                        metric = None,
                        shuffle=False,
                        random_state=None):

        """Calculates pruned scores

        Args:
            model:
                An estimator to calculate cross-validated score
            x:
                numpy ndarray or pandas DataFrame
            y:
                numpy ndarray or pandas Series
            sample_weight:
                Default = None
                None or numpy ndarray or pandas Series
            metric:
                Default = 'mse'
                Metric from scikit-learn metrics to be optimized.
            shuffle:
                Default = False
                If True, shuffle the data before splitting them into folds.
            random_state:
                Default = None
                If any integer value, creates a seed for random number generation.

        Usage example:

            Check PrunedCV use example.
        """
        if not isinstance(x, (numpy.ndarray, pandas.core.frame.DataFrame,torch.Tensor)):
            raise TypeError("Parameter x has wrong type")

        if not isinstance(y, (numpy.ndarray, pandas.core.series.Series,torch.Tensor,type(None))):
            raise TypeError

        if isinstance(metric,type(None)):
            kf = KFold(n_splits=self.cv,
                       shuffle=shuffle,
                       random_state=random_state)


        elif metric in ['mse',
                      'mae']:
            kf = KFold(n_splits=self.cv,
                       shuffle=shuffle,
                       random_state=random_state)

        elif metric in ['accuracy',
                        'auc']:

            kf = StratifiedKFold(n_splits=self.cv,
                                 shuffle=shuffle,
                                 random_state=random_state)

        else:
            raise ValueError

        for train_idx, test_idx in kf.split(x, y = y):
            if not self.prune:

                if isinstance(x, numpy.ndarray):
                    x_train = x[train_idx]
                    x_test = x[test_idx]
                elif isinstance(x, torch.Tensor):
                    if x.ndim == 1:
                        x_train = x[train_idx]
                        x_test = x[test_idx]
                    elif x.ndim == 2:
                        x_train = x[train_idx,:]
                        x_test = x[test_idx,:]
                else:
                    x_train = x.iloc[train_idx, :]
                    x_test = x.iloc[test_idx, :]
                if isinstance(y, numpy.ndarray):
                    y_train = y[train_idx]
                    y_test = y[test_idx]
                elif isinstance(y,type(None)):
                    y_train = None
                    y_test = None
                elif isinstance(y,torch.Tensor):
                    if y.ndim == 1:
                        y_train = y[train_idx]
                        y_test = y[test_idx]
                    elif y.ndim == 2:
                        y_train = y[train_idx,:]
                        y_test = y[test_idx,:]


                else:
                    y_train = y.iloc[train_idx]
                    y_test = y.iloc[test_idx]



                model.fit(x_train, y = y_train)

                if metric == 'mse':
                    y_test_teor = model.predict(x_test)

                    self._add_split_value_and_prun(metrics.mean_squared_error(y_test,
                                                                              y_test_teor))

                elif metric == 'mae':
                    y_test_teor = model.predict(x_test)

                    self._add_split_value_and_prun(metrics.mean_absolute_error(y_test,
                                                                               y_test_teor,
                                                                               ))

                elif metric == 'accuracy':
                    y_test_teor = model.predict(x_test)

                    self._add_split_value_and_prun(metrics.accuracy_score(y_test,
                                                                          y_test_teor,
                                                                          ))

                elif metric == 'auc':
                    y_test_teor = model.predict_proba(x_test)[:, 1]

                    self._add_split_value_and_prun(metrics.roc_auc_score(y_test,
                                                                         y_test_teor,
                                                                         ))
                else:
                    self._add_split_value_and_prun( -model.score(x_test,y = y_test))


        self.prune_prec = self.prune
        self.prune = False
        return self.cross_val_score_value

    def _add_split_value_and_prun(self,
                                  value):

        if not isinstance(value, float):
            try:
                value = value.item()
            except:
                raise TypeError

        if len(self.current_splits_list_) == 0:
            self.prune = False

        if self.minimize:
            self.current_splits_list_.append(value)
        else:
            self.current_splits_list_.append(-value)

        if self.first_run_:
            self._populate_best_splits_list_at_first_run(value)
        else:
            self._decide_prune()

        if len(self.current_splits_list_) == self.cv:
            self._serve_last_split()

    def _populate_best_splits_list_at_first_run(self,
                                                value):

        if self.minimize:
            self.best_splits_list_.append(value)
        else:
            self.best_splits_list_.append(-value)

        if len(self.best_splits_list_) == self.cv:
            self.first_run_ = False

    def _decide_prune(self):

        split_num = len(self.current_splits_list_)
        mean_best_splits = sum(self.best_splits_list_[:split_num]) / split_num
        mean_curr_splits = sum(self.current_splits_list_) / split_num

        if self.cv > split_num >= self.splits_to_start_pruning:

            self.prune = self._significantly_higher_value(mean_best_splits,
                                                          mean_curr_splits,
                                                          self.minimize,
                                                          self.tolerance)

            if self.prune:
                self.cross_val_score_value = self._predict_pruned_score(mean_curr_splits,
                                                                        mean_best_splits)
                self.current_splits_def = numpy.array(self.current_splits_list_)
                self.current_splits_list_ = []

    @staticmethod
    def _significantly_higher_value(mean_best_splits,
                                    mean_curr_splits,
                                    minimize,
                                    tolerance):
        tol_scaler = 1+tolerance
        if minimize:
            return mean_best_splits*tol_scaler < mean_curr_splits
        else:
            return mean_curr_splits*tol_scaler < mean_best_splits


    def _predict_pruned_score(self,
                              mean_curr_splits,
                              mean_best_splits):
        return (mean_curr_splits / mean_best_splits) * (sum(self.best_splits_list_) / self.cv)

    def _serve_last_split(self):

        if sum(self.best_splits_list_) > sum(self.current_splits_list_):
            self.best_splits_list_ = self.current_splits_list_

        self.cross_val_score_value = sum(self.current_splits_list_) / self.cv
        self.current_splits_def = numpy.array(self.current_splits_list_)
        print("servelastsplit")
        print(self.cross_val_score_value)
        print(self.current_splits_def)
        print("endservelastsplit")
        self.current_splits_list_ = []

def find_last_version(path,model = None,extension='pickle'):
    version = 0
    if isinstance(model,type(None)):
        path_model = f'{path}'
    else:
        path_model = f'{path}_{model.__name__}'
    while os.path.isfile(f'{path_model}_{version+1}.{extension}'):
        version +=1
    return version

def save_version(info,path,model = None,version = None,extension = 'pickle'):
    if isinstance(model,type(None)):
        path_model = f'{path}'
    else:
        path_model = f'{path}_{model.__name__}'

    if isinstance(version,type(None)):
        version = find_last_version(path,model = model,extension = extension)
        version +=1

    filename = f'{path_model}_{version}.{extension}'
    pickle.dump(info, open(filename, 'wb'))
    return None

def perform_study(model, X,fixed_params = {}, variable_params = {} , y = None,cv= 5, prune = False,
                  n_trials = 1,save_path = None,version = None,eta = 0,n_jobs = 1,gs_algo = 'optuna'):
    if prune == True:
        prun = PrunedCV_altered(cv,0.05,minimize = True)
    if gs_algo == 'optuna':

        def objective(trial):
            param = {}
            for key in fixed_params.keys():
                param[key] = fixed_params[key]
            for key in variable_params.keys():
                val = variable_params[key]
                if val[0] == 'loguniform':
                    param[key] = trial.suggest_loguniform(key,val[1],val[2])
                elif val[0] == 'uniform' :
                    param[key] = trial.suggest_uniform(key,val[1],val[2])
                else:
                    raise Exception("not done yet : set new possibilities for keys")
            res = model(**param)
            if prune == True:
                score = prun.cross_val_score(res, X,y=y)
                print("here we are ")
                print(score)
                print(prun.current_splits_def)
                scores = prun.current_splits_def


            else:
                scores = -cross_val_score(res,X,y=y,cv = cv)
                score = scores.mean()
            std = scores.std()
            trial.set_user_attr('accuracy',scores.mean())
            trial.set_user_attr('std',scores.std())
            trial.set_user_attr('scores',scores)
            return score+eta*std
        study = optuna.create_study(direction = "minimize")
        study.optimize(objective,
                   n_trials=n_trials, show_progress_bar=True, n_jobs=n_jobs)
        best_parameters = study.best_params
        for key in fixed_params.keys():
            best_parameters[key] = fixed_params[key]
        result = {'best_parameters' : best_parameters, 'trials_df' : study.trials_dataframe()}

    elif gs_algo == 'gs_sklearn':
        raise Exception("not done yet")

    save_version(result,save_path,model = model,version =version)
    return study

def get_results(path,model = None,version = None,extension = 'pickle'):
    if isinstance(model, type(None)):
        path_model = f'{path}'
    else:
        path_model = f'{path}_{model.__name__}'

    if isinstance(version, type(None)):
        version = find_last_version(path, model=model, extension=extension)

    filename = f'{path_model}_{version}.{extension}'
    loaded_model = pickle.load(open(filename, 'rb'))
    df = loaded_model['trials_df']
    best_params = loaded_model['best_parameters']
    def aux(eta):
        res = best_params.copy()
        row = df.loc[df['user_attrs_scores'].apply(lambda x : x.mean() + eta*x.std()).sort_values().index[0]]
        for k in row.index:
            if k.startswith('params'):
                param_name = k[7:]
                res[param_name] = row[k]
        return res

    return df,best_params,aux
