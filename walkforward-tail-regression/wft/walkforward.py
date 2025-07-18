#%%
import numpy as np
import pandas as pd
from typing import Union, Optional, List, Callable
from wft import ProfitMetrics, TrainClass, DataManager, Preprocessing
    
# %%
class WalkForward:
    def __init__(
            self,
            inputs: pd.DataFrame,
            targets: pd.Series,
            ohlc: pd.DataFrame, 
            max_lookahead: int,
            model_components: List[str],
            criterion: str,
            min_trades:  Union[int, float],
            model_kwargs: dict[dict] = None,
            min_crit_value: Optional[float] = 0, 
            stepwise_kwargs: dict = None,
            kwargs_cv: dict = None, 
            grid_hyperparams: List[dict] = None,
            interaction_terms: bool = False,
            side: Optional[str] =None,
            preprocess_variables: Optional[List[str]] = None,
            preprocess_type: Optional[List[str]] = None
        ):
        
        """
        Walk Forward Class:
            Perform a Walk forward with train and step size predifned by the user,
            
            inputs: 
                Dataframe with the predictors to be used, with datetime index
            targets: 
                pandas Series to be predicted, with datetime index
            ohlc:
                pandas DataFrame of original data with Close present, with datetime index
            max_lookahead:
                number of observations that the targets lookahead in the future:
                    eg. 6 if one is computing the return of the following 6 days
                    necesary to avoid lookahead bias
            model_components:
                list containing model components, order is respected,
                ex model: -> ["pca", "ols"] or just ['ols']
                choices: ["pca", "nn", "ols"]
                                        
            criterion: 
                Profit metrics function to be optimized
                
            min_trades: 
                float or int, respectively fraction or integer value of  mandatory trades  
                
            model_kwargs:
                Parameters of the model components,
                nested dictionary -> {"model_component": {"param_name":param_value}, ... }
                example_1 -> {"pca": {"n_components":3}, "nn": {"n_hidden":3}}
                                 
            min_crit_value:
                Minimum value of the Profit metrics function, under which the model do not act.
                If sample performance is below this threshold no position on the short or long side
                will be taken in the test set

            stepwise_kwargs:
                if passed perform stepswise selction:
                kwargs:
                eval_metric(str): to be optimized: "mse", "mae", "tail_sr"
                max_terms(int): Maximum number of term to be returned
                n_set_to_keep(int): for each iteration of the step wise selction, this many combination are kept
                cv_folds(int): perform crossvalidation if cv_folds>1, else evaluate on the full set passed with 80/20 train test split


            kwargs_cv:
                cv_folds(int): number of fold in cross validation
                eval_metric(str): to be optimized: "mse", "mae", "tail_sr"
                            
            grid_hyperparams:
                list[dict[dict]], list containing a nested dictionary of hyperparameters combination,
                dict like in model_kwargs:
                example_1 -> [{"nn": {"n_hidden":3}}, {"nn": {"n_hidden":4}}, ... ]
                example_2 -> [{"pca": {"n_components":3}, "nn": {"n_hidden":4}}, => 2 combination tested
                              {"pca": {"n_components":4}, "nn": {"n_hidden":2}},
                               ....]              

            min_crit_value:
                Minimum value of the performance metric, Default = 0, below which we do not act
                
            side:
                ['long', 'short' or None], wheter to optimize threshold on both side or just one
                            
            returns: Out-of-sample returns, long and short trades (OOS)
        """        
        
        try:
            self.crit_funct = getattr(ProfitMetrics, criterion)
        except AttributeError:
            raise ValueError(f'Unknown model or criterion: {criterion}')

        self.train_class     = TrainClass(
            model_components = model_components,
            criterion        = self.crit_funct,
            min_trades       = min_trades, 
            lookahead        = max_lookahead,
            model_kwargs     = model_kwargs,
            stepwise_kwargs  = stepwise_kwargs,
            kwargs_cv        = kwargs_cv,
            grid_hyperparams = grid_hyperparams,
            min_crit_value   = min_crit_value,
            side = side)  
    

        if interaction_terms:  # Compute cross products and interaction terms
            inputs =  self.train_class.compute_interaction_terms(inputs)  

        if stepwise_kwargs is not None:
            if len(stepwise_kwargs)==0:
                self.stepwise_cv = False
            else:
                self.stepwise_cv = True
        else:
            self.stepwise_cv = False  

              
        self.max_lookahead = max_lookahead
        self.side          = side

        self.data_manager = DataManager(inputs, targets, ohlc)
        self.inputs, self.targets, self.ohlc = self.data_manager.clean_data()

        target_references = {'y', 'target', 'targets', 'outs', 'all'}
        input_references =  {'X', 'input', 'inputs', 'ins', 'all'}
        if preprocess_variables is not None: # Check if user want to preprocess vars during walkforward
            if target_references & set(preprocess_variables): # Which variable?
                self.y_scaler = Preprocessing(preprocess_type) # initiate instance
            else:
                self.y_scaler = None
                
            if input_references & set(preprocess_variables):  # repeat for indipendent vars
                self.x_scaler = Preprocessing(preprocess_type)
            else:
                self.x_scaler = None
        else:
            self.y_scaler = None
            self.x_scaler = None




        self.predict_new_case = Predict_new_case(side = self.side)
        self.position_Manager = PositionManager(lookahead = self.max_lookahead)
        self.stats_collector  = StatsCollector(side = side, profit_metric=self.crit_funct)
        self.bars_returns = self.ohlc['Close'].diff(1).shift(-1).fillna(0)  # make sure log close is passed
        self.n_obs = len(self.inputs)
        self.oos_returns = pd.Series(np.zeros(len(self.ohlc)), index=self.ohlc.index )
        
    def run(self, start_train: int, train_size: int, step_size: int, commision: float = 0.0002):
        """
        Run the Walk Forward procedure
            start_train:
                Integer, index of the first observation of the first training period
            train_size:
                Integer, past data used to train the model (IF SET TO ZERO USE ALL PAST AVAILABLE DATA)
            step_size:
                Integer, how often to train the model 
            commision:
                Float, commision to be paid for each trade, default 0.0002 (0.02%)
        """
        next_train = start_train  # start training when specified by used
        n_fold = 0
        model_trained = False
        oos_positions = np.zeros(self.n_obs)

        for i in range(self.n_obs):
            # at each time we train on (i - size_train - max_lookahead: i - lookahead) 
            # and test on i : next_train - 1        
            if i == next_train: # check if it is training time
                X_train, y_train = self.data_manager.get_window_train(i, train_size, self.max_lookahead) # get train dataset
                # scale variables?
                if self.y_scaler is not None:
                    y_train = self.y_scaler.fit_transform(variable = y_train )
                if self.x_scaler is not None:
                    X_train = self.x_scaler.fit_transform(variable = X_train )

                # Train the model
                metrics_dict, model = self.train_class.train_model(inputs = X_train, out = y_train)  # Train model
                model_trained  = True  # Model is Train
                next_train += step_size   # Compute next train period
                start_test_i = i  # Start of the testing period      
            # is the model trained?
            if model_trained: # Test fold
                inputs_i = self.inputs.iloc[i] # current case
                # apply scaling?
                if self.x_scaler is not None:
                    inputs_i = self.x_scaler.transform(variable = inputs_i )
                # if stepwise selection is used, select the variables
                if self.stepwise_cv:
                    inputs_i = inputs_i[metrics_dict.get('selected_vars')]
                # Get prediction
                raw_position = self.predict_new_case.get_prediction(inputs_i, model , metrics_dict)
                # Compute adjusted position, this does nothing if lookahead == 1
                adjusted_pos = self.position_Manager.compute_rolling_avg_position(raw_position)
                # Compute returns
                self.oos_returns.iloc[i] = adjusted_pos*self.bars_returns.iloc[i]
                oos_positions[i] = adjusted_pos # save position
                if oos_positions[i]!=oos_positions[i-1]: # if position changed, pay commision
                    delta = np.abs(oos_positions[i] - oos_positions[i-1])
                    fee = delta * commision
                    self.oos_returns.iloc[i] -= fee
                # Check if we are at the end of the fold
                if i+1>= next_train or i +1>=self.n_obs:
                    # Collect fold results
                    slice_fold_pos = oos_positions[start_test_i:i+1]
                    slice_fold_rets = self.oos_returns.iloc[start_test_i:i+1]
                    self.stats_collector.update(metrics_dict, slice_fold_rets,
                                            slice_fold_pos, X_train.keys,
                                            self.ohlc.index[start_test_i], self.ohlc.index[i])
                    # Reset The model, OOS fold is finished, Retrain
                    n_fold +=1
                    model_trained = False
                if i +1>=self.n_obs: # complete loop
                    break

        longs_df, shorts_df = self.stats_collector.finalize()
        self.model = model
        return self.oos_returns, longs_df, shorts_df
 


class Predict_new_case:
    """
    Called from the walkforward to classify a new observation
    """
    def __init__(self, side):
        self.side = side

    def get_prediction(self, X_i: pd.DataFrame, model, thresholds: dict) -> float:
        new_pred = model.predict(X_i)
        prediction =   0.0
        if self.side != 'short':
            thresh_long = thresholds['long']['thresh']
            if new_pred >= thresh_long:
                prediction =  1.0

        if self.side != 'long':
            thresh_short = thresholds['short']['thresh']
            if new_pred <= thresh_short:
                prediction =  -1.0
            
        return prediction
    



        
class PositionManager:
    """ 
    This is optional but if the lookahead is > 1, positions may overlap, 
    making it tricky to keep track of position size and returns, instead if a moving average is used
    the maximum |position| open at any given time will be 1, if and only if the past n_lookahead predictions
    had  the same sign and a position a of |size| = 1/lookahead will be opened if the model fitted say so,
    note that if lookahead == 1 this does nothing.

    In short:
        compute moving average with (window=lookahead) of the position size based on past values of the position it's self
    """

    def __init__(self, lookahead: int):
        self.window = lookahead
        self.position_hist     = []
        self.adjusted_pos_hist = []

    def compute_rolling_avg_position(self, new_pred: float) -> float:
        self.position_hist.append(new_pred)
        if len(self.position_hist)<self.window:
            current_position = np.sum(self.position_hist)/len(self.position_hist)
            self.adjusted_pos_hist.append(current_position)
        else:
            position_window = self.position_hist[-self.window:]
            current_position = np.sum(position_window)/self.window
            self.adjusted_pos_hist.append(current_position)
        return current_position
    

class StatsCollector:
    """
    Collect some metrics for each fold of the walk forward
    """
    def __init__(self, side: str | None, profit_metric: Callable): 

        self.side = side
        self.long_stats  = []
        self.short_stats = []
        self.returns     = []
        self.profit_metric = profit_metric

    def update(self,
                metric_dict: dict, 
                slice_fold_rets: np.ndarray, 
                slice_fold_pos: np.ndarray,
                model_vars: List[str], 
                start_test_index: pd.DatetimeIndex,
                end_test_index: pd.DatetimeIndex):
        long_r_fold = slice_fold_rets[slice_fold_pos>0]
        short_r_fold = slice_fold_rets[slice_fold_pos<0]
        longs_dict = {}
        shorts_dict = {}
        
        if self.side != 'short':
            if len(long_r_fold)>0:
                metric_l = self.profit_metric(long_r_fold)
                sum_r_long = np.sum(long_r_fold)
            else:
                metric_l = 0.0
                sum_r_long = 0.0
            longs_dict.update({
                 "Evaluation_metric":metric_l, 
                 "Fold_return":sum_r_long,
                 "Model_vars": model_vars,
                 "Start_test": start_test_index,
                 "End_test": end_test_index,
                 "Percentile_threshold":metric_dict['long']['thresh_percentile %'],
                 "threshold": metric_dict['long']['thresh'],
                })
            if 'best_params_cv' in metric_dict:
                longs_dict.update({"best_params_cv":metric_dict['best_params_cv']})
            if 'selected_vars' in metric_dict:
                longs_dict.update({"best_params_cv":metric_dict['selected_vars']})
            self.long_stats.append(longs_dict)
        if self.side != 'long':
            if len(short_r_fold)>0:
                metric_s = self.profit_metric(short_r_fold)
                sum_r_short = np.sum(short_r_fold)   
            else:
                metric_s = 0.0
                sum_r_short = 0.0
            shorts_dict.update({
                "Evaluation_metric":metric_s, 
                "Fold_return":sum_r_short,
                "Model_vars": model_vars,
                "Start_test": start_test_index,
                "End_test": end_test_index,
                "Percentile_threshold":metric_dict['short']['thresh_percentile %'],
                "threshold": metric_dict['short']['thresh'],
                })
            
            if 'best_params_cv' in metric_dict:
                shorts_dict.update({"best_params_cv":metric_dict['best_params_cv']})
            if 'selected_vars' in metric_dict:
                shorts_dict.update({"best_params_cv":metric_dict['selected_vars']})

            self.short_stats.append(shorts_dict)
        return self
    def finalize(self) -> Union[pd.DataFrame, pd.DataFrame]:
        self.long_metrics_df = pd.DataFrame(self.long_stats)
        self.short_metrics_df = pd.DataFrame(self.short_stats)  
        return  self.long_metrics_df ,self.short_metrics_df
