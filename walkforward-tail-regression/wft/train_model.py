# %%
import numpy as np
import pandas as pd
from typing import Union, Optional, List, Callable
from .models import Pipeline
from .feature_selection import StepWiseSelection
from .hyperparams_cv import Hyperparams_tuner
from .optim_threshold import ThresholdOptimizer
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

