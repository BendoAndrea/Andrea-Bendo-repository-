#%%
import numpy as np
import pandas as pd
from typing import List
from wft import Evaluation_model, LinearRegression

class StepWiseSelection:
    def __init__(self, lookahead, eval_metric: str = 'mse' ,max_terms: int = 3, n_set_to_keep: int = 1, cv_folds: int =0):
        """
        Perform stepwise selection with crossv alidation with given model to find best predictors with linear regression,
        eval_metric:
            Loss function for evaluating the model,
            mean square error => "mse",
            mean absolute error => "mae",
            tail prediction sharpe ratio => "tails_sr"
        max_terms:
            Maximum number of predictors to be returned
        n_set_to_keep:
            for each iteration of the step wise selction, this many combination are kept
        cv_folds:
            perform crossvalidation if cv_folds>1, else evaluate on the full set passed with 80/20 train test split
        """

        self.evaluator = Evaluation_model(eval_metric, lookahead=lookahead)
        self.max_terms = max_terms
        self.n_set_to_keep = n_set_to_keep
        self.cv_folds = cv_folds  
        self.lookahead = lookahead
        
    def selection(self, 
                     inputs: pd.DataFrame,
                     target:  pd.Series | np.ndarray 
                 ) -> List[str]:
        
        """
        candidates:
            Predictors from which to select, pd.DataFrame
        target:
            Variable to be predicted, pd.Series or numpy array
        """

        n_set_to_keep = self.n_set_to_keep  
        cv_folds = self.cv_folds  
        max_terms = self.max_terms
        candidates = inputs.copy()
        
        name_set =  candidates.columns   # Name of the candidates
        vars_result = []   # here store the result of set of regressor
        n_obs = len(target)
        model = LinearRegression()
        assert inputs.shape[1]>max_terms,f'variable passed: {inputs.shape[1]} <= variable to keep:{max_terms}, Stepwise selection'
                
      
        for n in range(max_terms):   # go until we have enough terms
            if n ==0:   # first iteration we do just one pass for each candidate
                for name in name_set:
                    if self.cv_folds>1:
                        curr_score =  np.zeros(self.cv_folds)
                            # Get crossvalidation score
                        for i in range(cv_folds):  # Test for each fold
                                train_indexes, test_indexes= self.evaluator.get_train_test_splits_cv(i, n_obs=n_obs, cv_n_fold=self.cv_folds)
                                X_train, y_train, X_test, y_test = self.evaluator.data_splitter(candidates[name], target, train_indexes, test_indexes)
                                curr_score[i] = self.evaluator.eval_model(model = model ,X_train =X_train, y_train = y_train,
                                                                  X_test = X_test, y_test = y_test)
    
                        result = np.mean(curr_score)

                    else:
                        start_test_i = int(n_obs*.8)
                        train_indexes = (0, start_test_i-self.lookahead)
                        test_indexes  = (start_test_i, n_obs)
                        X_train, y_train, X_test, y_test = self.evaluator.data_splitter(candidates[name], target, train_indexes, test_indexes)
                        result = self.evaluator.eval_model(model = model ,X_train =X_train, y_train = y_train,
                                                                  X_test = X_test, y_test = y_test)                        
                    vars_result.append([name, result])

                sorted_vars_result = sorted(vars_result, key= lambda metric:metric[1])   # sort based on metric chosen
                best_sets = [best_vars  for best_vars ,_ in sorted_vars_result[:n_set_to_keep]]   # keep the top n regressor
                vars_result  = []   # reset result list
       
            else:
                # add a new predictor to the best sets of the last iteration and choose the best n_keep
                for comb in best_sets:         # loop trough the list of best candidates from the last iteration
                    if type(comb) != list:     # after the first iteration we have string instead of lists
                        comb = [comb]          # transform it in a list
                    for name in name_set:      # loop trough each candidate var
                        if name not in comb:   # check if it is already in the current list 
                            test_names = comb + [name]   # add the column name to the list
                        else:
                            continue
                            # Get crossvalidation score
                        if cv_folds>0:
                            curr_score = np.zeros(cv_folds)
                            for i in range(cv_folds):  # Test for each fold
                                train_indexes, test_indexes= self.evaluator.get_train_test_splits_cv(i, n_obs=n_obs, cv_n_fold=self.cv_folds)
                                X_train, y_train, X_test, y_test = self.evaluator.data_splitter(candidates[test_names], target, train_indexes, test_indexes)
                                curr_score[i] = self.evaluator.eval_model(model = model, X_train =X_train, y_train = y_train,
                                                                  X_test = X_test, y_test = y_test)

                            result = np.mean(curr_score)
                            
                        else:
                            start_test_i = int(n_obs*.8)
                            train_indexes = (0, start_test_i-self.lookahead)
                            test_indexes  = (start_test_i, n_obs)
                            result = self.evaluator.eval_model(model = model ,X_train =X_train, y_train = y_train,
                                                                  X_test = X_test, y_test = y_test)   
                        vars_result.append([test_names,result])   

                sorted_vars_result = sorted(vars_result, key=lambda metric:metric[1])  # sort based on metric chosen
                unique_vars_result = self.unique_inside_list(sorted_vars_result)   # Drop duplicates
                best_sets = [top_vars for top_vars in unique_vars_result[:n_set_to_keep]]  #best candidates for this iteration
                vars_result  = []  # reset result list
                
        return best_sets[0] # Best set
    
    @staticmethod
    def unique_inside_list(list_ : List) -> List:
        # This function takes a list of lists and returns a list of unique lists
        unique = []
        seen = set()
    
        for piece in list_:
            sub_list = piece[0] 
            key = frozenset(sub_list)
            if key not in seen:
                seen.add(key)
                unique.append(sub_list)
        return unique    