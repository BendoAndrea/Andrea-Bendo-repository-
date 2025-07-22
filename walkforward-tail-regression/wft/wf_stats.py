from typing import Callable, List, Union
import numpy as np
import pandas as pd


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
                model: Callable, 
                model_vars: List[str], 
                start_test_index: pd.DatetimeIndex,
                end_test_index: pd.DatetimeIndex):
        long_r_fold = slice_fold_rets[slice_fold_pos>0]
        short_r_fold = slice_fold_rets[slice_fold_pos<0]
        longs_dict = {}
        shorts_dict = {}
        params = model.get_params()
        if self.side != 'short':
            if len(long_r_fold)>0:
                metric_l = self.profit_metric(long_r_fold)
                sum_r_long = np.sum(long_r_fold)
            else:
                metric_l = 0.0
                sum_r_long = 0.0
            current_threshold_long = np.nan if  metric_dict['long']['thresh'] >100 else metric_dict['long']['thresh']
            longs_dict.update({
                 "Evaluation_metric":metric_l, 
                 "Fold_return":sum_r_long,
                 "Model_vars": model_vars,
                 "Start_train": start_test_index,
                 "Start_test": end_test_index,
                 "Percentile_threshold":metric_dict['long']['thresh_percentile %'],
                 "threshold":current_threshold_long,
                 "params":params
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
            current_threshold_short = np.nan if  metric_dict['short']['thresh'] <-100 else metric_dict['short']['thresh']
            shorts_dict.update({
                "Evaluation_metric":metric_s, 
                "Fold_return":sum_r_short,
                "Model_vars": model_vars,
                "Start_train": start_test_index,
                "Start_test": end_test_index,
                "Percentile_threshold":metric_dict['short']['thresh_percentile %'],
                "threshold": current_threshold_short,
                "params":params
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
