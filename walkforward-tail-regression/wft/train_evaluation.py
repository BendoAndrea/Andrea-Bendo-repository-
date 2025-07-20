#%%
import numpy as np
import pandas as pd
from typing import Tuple, Union, Callable
class Evaluation_model:
    def __init__(self, metric: str, lookahead: int):
        """
        metric: str, evaluation metric to be used, e.g. 'mse', 'mae', 'tails_sr'
       
        lookahead: int, number of periods to look ahead
        """
        try:
            self.eval_crit = getattr(self, metric)
        except AttributeError:
            raise ValueError(f'Unknown metric: {metric}, choose: "mse", "mae", "tails_sharpe_ratio"')
        self.lookahead = lookahead


    def eval_model(self, model: Callable,
                    X_train: Union[pd.DataFrame, np.ndarray], y_train:Union[pd.Series, np.ndarray],
                    X_test: Union[pd.DataFrame, np.ndarray], y_test:Union[pd.Series, np.ndarray]
                    ) -> float:
        """ Evaluate model with given metric
        model: Function, model to be evaluated
        X_train: training inputs
        y_train: training outputs
        X_test:  testing inputs
        y_test:  testing outputs
        """
        preds = model.fit(X_train,y_train).predict(X_test)
        return self.eval_crit(preds, y_test)

    def data_splitter(self, 
                       X: Union[pd.DataFrame, np.ndarray], 
                       y: Union[pd.Series, np.ndarray],
                       train_indexes: Tuple[int, int],
                       test_indexes: Tuple[int, int]):
                      

        """ Function to evaluate current set of parameters """
        X = np.asarray(X)
        y = np.asarray(y)
        assert test_indexes[0]<test_indexes[1], 'Wrong testing indexes passed' # Should never happen
        X_train = np.concatenate([ X[:train_indexes[0]],X[train_indexes[1]:]])
        y_train = np.concatenate([ y[:train_indexes[0]],y[train_indexes[1]:]])
        X_test  = X[test_indexes[0]:test_indexes[1]]
        y_test  = y[test_indexes[0]:test_indexes[1]]   
        return X_train, y_train, X_test, y_test

    def get_train_test_splits_cv(self, n:int, n_obs:int, cv_n_fold) -> Union[Tuple[int,int], Tuple[int,int]]:  
        """Just helper function to get splits"""
        start_test = n*(n_obs//cv_n_fold)
        end_test   = start_test+ n_obs//cv_n_fold 
        train_untill = max(0,start_test - self.lookahead)  # add Buffers for lookahead bias
        train_from = min(n_obs, end_test +  self.lookahead)   # add Buffers for lookahead bias
        
        test_indexes = (start_test, end_test)
        train_indexes = (train_untill, train_from)
        return train_indexes, test_indexes


    @staticmethod
    def mse(pred, y)  -> float:
        return np.mean((pred - y)**2)
    
    @staticmethod
    def mae(pred, y)  -> float:
        return np.mean(np.abs(pred-y))
    
    @staticmethod
    def tails_sr(pred, y, alpha = .1)  -> float:
        """
        Computes the sharpe ratio of tail predictions
        Be aware if the set is too small this is unstable,
        consider increasing alpha
        """

        if alpha >.5:
            raise ValueError('alpha has to be smaller than 0.5, got: {alpha}')
        
        quantile_10, quantile_90 = np.quantile(pred, [alpha, 1-alpha])
        # Short if below quantile_10, long if above quantile_90
        r = np.where(pred<=quantile_10, -y, np.where(pred>=quantile_90,y, 0))
        tails_r = r[r!=0] 
        sharpe = tails_r.mean()/(tails_r.std() +1e-8)
        return -sharpe  # negative because other functions stepwise and crossvalidation minimize this
