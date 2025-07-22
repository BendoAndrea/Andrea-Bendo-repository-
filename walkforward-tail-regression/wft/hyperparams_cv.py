#%%
import numpy as np
import pandas as pd
from typing import Callable, Union
from .train_evaluation import Evaluation_model
class Hyperparams_tuner:
    
    def __init__(self, lookahead , params_kwargs: dict[str: str|int], eval_metric: str = 'mse',cv_folds: int = 10):
        """
        lookahead: int, number of periods to look ahead 
        params_kwargs: dict, hyperparameters to be tuned, e.g. {'pca':{'n_components':2}, 'ols': {'add_intercept':True }}
        eval_metric: str, evaluation metric to be used, e.g. 'mse', 'mae', 'tails_sr'
        cv_folds: int, number of cross-validation folds
        """
        self.params_kwargs = params_kwargs
        self.evaluator = Evaluation_model(eval_metric, lookahead=lookahead)
        self.cv_folds = cv_folds

        if cv_folds<=1:
            raise ValueError('cv_fold has to be >1, got : {cv_folds}')
        
    def cross_validation(self,
                         model: Callable, 
                         inputs: Union[pd.DataFrame, np.ndarray], 
                         outputs: Union[pd.Series, np.ndarray]) -> dict:
        """ Evaluation of model hyper parameters """ 
        X = np.asarray(inputs)
        y = np.asarray(outputs)        
        best_score  = np.inf
        best_params = None
        n_obs       = y.size
        if self.params_kwargs is None:
            raise ValueError('Hyperparameters grid for cross validation cannot be none')
        for params_ in self.params_kwargs:
            curr_score = np.zeros(self.cv_folds)
            model.set_params(**params_) # Set model params allocation is done here
            for i in range(self.cv_folds):  # Test for each fold
                train_indexes, test_indexes= self.evaluator.get_train_test_splits_cv(i, n_obs=n_obs, cv_n_fold=self.cv_folds) # Get indexes
                X_train, y_train, X_test, y_test = self.evaluator.data_splitter(X=X, y=y,
                                                                                train_indexes=train_indexes,
                                                                                test_indexes=test_indexes)  # Split data
                curr_score[i] = self.evaluator.eval_model(model = model, X_train =X_train, y_train = y_train,
                                                        X_test = X_test, y_test = y_test) # Eval model
            mean_score = np.mean(curr_score) 
            if mean_score<best_score:
                best_params = params_
                best_score = mean_score
        return best_params
# %%
