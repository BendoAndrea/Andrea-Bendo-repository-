# %%
import numpy as np
import pandas as pd
from typing import Union, Optional, List, Callable
from .models import Pipeline
from .feature_selection import StepWiseSelection
from .hyperparams_cv import Hyperparams_tuner
# %%
class TrainClass:
    def __init__(self, 
                 model_components: List[str],
                 criterion: Callable,
                 min_trades: Union[int, float],
                 lookahead: int,
                 model_kwargs: dict = None,
                 stepwise_kwargs: dict = None,
                 kwargs_cv: dict = None,
                 grid_hyperparams: List[dict] = None,
                 min_crit_value: Optional[float] = 0.0,
                 side: Optional[str] = None,                 
                ):  

        """
        Main Train Class:
            model_components:
                list containing model components, order is respected,
                ex model: -> ["pca", "ols"]
                choices: ["pca", "nn", "ols"]
                                        
            criterion: 
                Profit metrics function to be optimized
                
            min_trades: 
                float or int, respectively fraction or integer of  mandatory trades  
                
            lookahead:
                int,
                Number of observation the target looks ahead used to avoid lookahead bias
            
            model_kwargs:
                Parameters of the model components,
                nested dictionary -> {"model_component": {"param_name":param_value}, ... }
                
            stepwise_kwargs:
                if passed perform stepswise selction:
                kwargs:
                eval_metric(str): to be optimized: "mse", "mae", "tails_sr"
                max_terms(int): Maximum number of term to be returned
                n_set_to_keep(int): for each iteration of the step wise selction, this many combination are kept
                cv_folds(int): perform crossvalidation if cv_folds>1, else evaluate on the full set passed with 80/20 train test split

            kwargs_cv:
                cv_folds(int): number of fold in cross validation
                eval_metric(str): to be optimized: "mse", "mae", "tails_sr"

            grid_hyperparams: list[dict[dict]]:
                 list containing a nested dictionaries of parameter combination,
                 model_kwargs ->  [{"nn": {"n_hidden":3}}, {"nn": {"n_hidden":4}}, ... ]
            
            min_crit_value (float):
                Minimum value of the performance metric, Default = 0, below which we do not act
                
            side:
                ['long', 'short' or None], wheter to optimize threshold on both side or just one
                            
            returns: Regression coefficients, Optimal Thresholds
        """
        
        self.crit = criterion
        self.model_kwargs = model_kwargs

        self.model = Pipeline(model_components) # Buid the model
        if model_kwargs is not None:
            self.model.set_params(**model_kwargs)  # set initial params if any provided
  
        self.min_trades     = min_trades
        self.side           = side
        self.lookahead      = lookahead
        if stepwise_kwargs is not None:
            if len(stepwise_kwargs)==0:
                self.stepwise_cv = False
            else:
                self.stepwise_cv = True
        else:
            self.stepwise_cv = False

        if kwargs_cv is not None:
            if len(kwargs_cv)==0:
                self.hyperparams_cv = False
            else:
                self.hyperparams_cv = True
        else:
            self.hyperparams_cv = False


        self.thresh_optim = ThresholdOptimizer(metric = criterion, min_trades=min_trades, min_crit_value =  min_crit_value,side = side)

        if self.stepwise_cv:
            self.stepwise = StepWiseSelection(lookahead=self.lookahead, **stepwise_kwargs)


        if self.hyperparams_cv:
            self.hyperparam_tuner = Hyperparams_tuner(lookahead= self.lookahead, params_kwargs=grid_hyperparams, **kwargs_cv)          

    def train_model(self, inputs:pd.DataFrame, out: pd.Series ) -> Union[dict, Callable]:  
        # Main train function where everything is processed       
        if self.stepwise_cv:
            best_vars_names = self.stepwise.selection(inputs = inputs, target = out )
            inputs  = inputs[best_vars_names]   # Set best variables
        if self.hyperparams_cv:
            best_params = self.hyperparam_tuner.cross_validation(model = self.model, inputs=inputs, outputs = out)
            self.model.set_params(**best_params)  # Set bets params
            
        in_sample_stats = self.thresh_optim.get_thresholds(inputs = inputs, target = out, model = self.model)  
       

        if self.stepwise_cv:   
            in_sample_stats['selected_vars'] = best_vars_names
        if self.hyperparams_cv:
            in_sample_stats['best_params_cv'] = best_params
        return in_sample_stats, self.model  
    
    @staticmethod
    def compute_interaction_terms(candidates: pd.DataFrame) -> pd.DataFrame:
        """Compute cross products and square terms of predictios""" 
        new_vars = candidates.copy()   
        squares_terms = candidates.copy()**2   # Square terms
        squares_terms = squares_terms.add_suffix('_squared')   # Change columns names of square terms
        new_vars = pd.concat([new_vars, squares_terms], axis = 1, join = 'inner')     
        dummy_df = candidates.copy()     #dummy df to drop columns after computing all interaction term for a variable
        for name in candidates.keys():   # multiplicative terms from original variables
            var = candidates[name]      
            df_ex_var = dummy_df.loc[:, dummy_df.columns!=name]   #  get all other variables
            for key, value in df_ex_var.items():    
                new_vars[f'{name}_{key}'] = value*var  
            dummy_df.drop(name,inplace=True,axis=1)   # drop the term of the first loop avoid duplicates
            
        return new_vars  # return new variables 
class ThresholdOptimizer:
    def __init__(self,
                metric: Callable,
                min_trades: int | float = 0.0,
                min_crit_value: float = 0.0,
                side: str = None):
    
        self.profit_metric = metric
        self.side = side
        self.min_trades = min_trades
        self.min_crit_value = min_crit_value
    def calculate_n_trades(self, targets: pd.Series | np.ndarray) -> None: 
        # Check if value is float or integer to compute number of mandatory trades
        if type(self.min_trades) is float:
            self.min_trades_percentile =  int(round(self.min_trades,2)*100) # bound between [0,100]
            self.min_trades_n = self.min_trades*len(targets) # Min number of trades as Integer
            if self.side is None:   # if both long and short, divide by 2 
                self.min_trades_percentile = int(self.min_trades_percentile/2) 
        else:
            self.min_trades_percentile = int(-(-self.min_trades*100/len(targets))) # bound between [0,100]
            self.min_trades_n = self.min_trades
            if self.side is None:   # if both long and short, divide by 2 
                self.min_trades_percentile = int(self.min_trades_percentile/2) 
                self.min_trades_n/=2
        return self
        
    def get_thresholds(self, 
                       inputs:  pd.DataFrame | np.ndarray,  
                       target: pd.Series| np.ndarray,
                       model: Callable) -> Union[dict, np.ndarray, np.ndarray]:  
        # Obtain optimal predictions bounds
        # Get Predictions of the model and side to optimized
        self.calculate_n_trades(targets = target)
        X = np.asarray(inputs)
        y = np.asarray(target)
  
        preds =  model.fit(X , y).predict(X)
        self.preds = preds
        
        side = self.side
        long_rets = 0
        short_rets = 0


        # Compute percetile of predictions from [0,100], will be used to optimize the threshold
        percentiles_preds = np.percentile(preds, np.arange(101))
        # Assuming preds below 50 percentile are negative and above are positve
        sigs = np.where(preds>percentiles_preds[50],1,-1)
        dummy_r = sigs * y
        #Allocate space to store performance metric of each percentile
        shorts_crits, longs_crits = np.zeros(percentiles_preds.size), np.zeros(percentiles_preds.size)
                
        if len(y) < max(1, self.min_trades_n): # Should never happen 
            raise ValueError(f"Not enough data points after cleaning {len(y)}<{self.min_trades_n}.")

        # Apply the profit/risk criterion and find the threshold that maximize the criterion
        if side != 'long': # Short Threshold computations
            if len(dummy_r[sigs==-1])>=self.min_trades_n:  # Are there enough trades?
                for i in range(self.min_trades_percentile , 50): # Compute performance
                    shorts_crits[i] = self.profit_metric(dummy_r[preds<=percentiles_preds[i]])

                # Find where the max is
                max_idx_short = np.argmax(shorts_crits)  # index of the max
                value_crit_short = shorts_crits[max_idx_short]  # criterion at the threshold
                thresh_crit_shrot =  percentiles_preds[max_idx_short]  # optimal threshold
                
                # Evaluate if performance is satisfactory
                if value_crit_short> self.min_crit_value:
                    short_t = {
                        'value_crit':value_crit_short,
                        'thresh': thresh_crit_shrot, 
                        'perc': max_idx_short}
                    # Returns shorts
                    short_rets = dummy_r[preds<=thresh_crit_shrot]
                else:  # performance not statisfactory 
                    short_t = {'value_crit': 0 , 'thresh': -1e6, 'perc': np.nan}  
            else: # If not enough trades
                short_t = {'value_crit': 0 , 'thresh': -1e6, 'perc': np.nan}  
            
        if side != 'short': # Long Threshold computations same as short
            if len(dummy_r[sigs==1])>=self.min_trades_n:
                for i in range(50, 101-self.min_trades_percentile):
                    longs_crits[i] = self.profit_metric(dummy_r[preds>=percentiles_preds[i]])
                # Find Where the max is
                max_idx_long = np.argmax(longs_crits)
                value_crit_long = longs_crits[max_idx_long]
                thresh_crit_long = percentiles_preds[max_idx_long]  
                # Evaluate if performance is satisfactory
                if value_crit_long> self.min_crit_value:
                    long_t = {
                        'value_crit':value_crit_long,
                        'thresh': thresh_crit_long,
                        'perc': max_idx_long}
                    # Returns long
                    long_rets = dummy_r[preds>=thresh_crit_long]
                else:   #If performance not statisfactory 
                    long_t = {'value_crit':  0 , 'thresh': 1e6, 'perc': np.nan}                
            else:  # If not enough trades 
                long_t = {'value_crit':  0, 'thresh': 1e6, 'perc': np.nan}            

        # return a dictionary with requested optimization
        if side is None:
           thresh_dict = {
               'short' : short_t,
               'long'  : long_t
               }
        elif side == 'long':
            thresh_dict = {
                'long'  : long_t
                }
        elif side == 'short':
            thresh_dict = {
                'short' : short_t,
                }            
        else: # Should never happen
            raise ValueError(f"side must be 'long', 'short' or None, not: {side!r}")
        
        stats = self.get_metrics_sample(long_rets, short_rets, thresh_dict)

        return  stats

    def get_metrics_sample(self, long_rets, short_rets, thresholds):
        # Collect data of the model
        stats = {}
        if self.side!= 'short':
            sum_l = np.sum(long_rets)
            long_stats = {
                'sample_r': sum_l , 'sample_perf': thresholds['long']['value_crit'],
                'thresh_percentile %': thresholds['long']['perc'], 
                'thresh':thresholds['long']['thresh']
            }
            stats.update({'long': long_stats})
        if self.side != 'long':
            sum_s = np.sum(short_rets)
            short_stats = {
                'sample_r': sum_s, 'sample_perf': thresholds['short']['value_crit'],
                'thresh_percentile %': thresholds['short']['perc'],
                'thresh':thresholds['short']['thresh']
            } 
            stats.update({'short': short_stats})

        return stats
# %%
