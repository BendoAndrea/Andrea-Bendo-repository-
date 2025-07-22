from typing import Callable, Union
import pandas as pd
import numpy as np

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
            if self.min_trades>= 0.5:
                raise ValueError("min_trades must be <.5, got {self.min_trades}")
            
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
                for i in range(50, 100-self.min_trades_percentile + 1):
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
