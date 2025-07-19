# %%
import pandas as pd
import numpy as np
from typing import Callable, Tuple, Union, List
from scipy import special
from scipy import stats
from wft import ProfitMetrics as pm
from wft import LinearRegression 
#%% 
class assess_performance:
    """
    Compute Various statistics to asses validity of the model
    """
    @staticmethod
    def p_value_returns(returns: pd.Series | np.ndarray) -> Union[float, Tuple]:
        r = np.asarray(returns)
        r = r[r!=0]
        mu = np.mean(r)
        std_dev = np.std(r, ddof=1)
        t_stat = (np.sqrt(r.size) * mu)/std_dev
        p_value = 1 - stats.t.cdf(x = t_stat, df = r.size-1)

        lower_ci = mu - (std_dev/np.sqrt(r.size) * special.stdtrit(r.size -1 ,.95))
        upper_ci =  mu + (std_dev/np.sqrt(r.size) * special.stdtrit(r.size -1 ,.95))
        ci = [np.round(lower_ci*100,4), np.round(upper_ci*100,4)]
        return p_value ,ci

    @staticmethod
    def extrapolate_decision_model(X: pd.DataFrame, model_final: Callable, thresholds: dict, transform: str = None):
      
        if transform == 'standardize':
            transformed_inputs = (X - np.mean(X,axis=0))/np.std(X,axis=0) 
        elif transform =='normalize':
            transformed_inputs = (X - np.min(X,axis=0))/(np.max(X,axis=0)-np.min(X,axis=0)) 
        else:
            transformed_inputs = X
                
        predictions = model_final.predict(transformed_inputs)
        if not thresholds["long"].empty:
            long_threshold = thresholds["long"]['threshold'].iloc[-1]
            long_decision_i = np.where(predictions>=long_threshold)
        else:
            long_decision_i = np.zeros(1)
        if not thresholds["short"].empty:
            short_threshold = thresholds["short"]['threshold'].iloc[-1]
            short_decision_i = np.where(predictions<=short_threshold)
        else:
            short_decision_i = np.zeros(1)
            
        if short_decision_i[0].size>0:
            shorts_X = X.iloc[short_decision_i]
            mean_x_value_short = np.mean(shorts_X, axis = 0)
            std_dev_x_short = np.std(shorts_X, axis = 0)
        else:
            shorts_X = 0
            mean_x_value_short = 0
            std_dev_x_short = 0

        if long_decision_i[0].size>0:
            longs_X = X.iloc[long_decision_i]
            mean_x_value_long = np.mean(longs_X, axis = 0)
            std_dev_x_long = np.std(longs_X, axis = 0)
        else:
            longs_X = 0
            mean_x_value_long = 0
            std_dev_x_long = 0     


        indexes = X.keys()

        if not thresholds["short"].empty and not thresholds["long"].empty:
            cols = ["mu_long_over_t","std_long", "mu_short_over","std_short"]
            data =  np.c_[mean_x_value_long,std_dev_x_long ,mean_x_value_short, std_dev_x_short]
            df_summary = pd.DataFrame(data = data, columns = cols, index =indexes)
            df_summary['mu_long/std_long'] = df_summary['mu_long_over_t']/df_summary['std_long'] 
            df_summary['mu_short/std_short'] = df_summary['mu_short_over']/df_summary['std_short'] 
        elif not thresholds["short"].empty:
            cols = ["mu_short_over","std_short"]
            data =  np.c_[mean_x_value_short, std_dev_x_short]
            df_summary = pd.DataFrame(data = data, columns = cols, index =indexes)
            df_summary['mu_short/std_short'] = df_summary['mu_short_over']/df_summary['std_short'] 
        elif not thresholds["long"].empty:
            cols = ["mu_long_over_t","std_long"]
            data =  np.c_[mean_x_value_long, std_dev_x_long]
            df_summary = pd.DataFrame(data = data, columns = cols, index =indexes)
            df_summary['mu_long/std_long'] = df_summary['mu_long_over_t']/df_summary['std_long']
        else:
            return None
        
        

        if model_final.steps[0][0]=='ols':
            coefs = model_final.get_params()['ols']['coefficients'][:X.shape[1]] #last term is
            df_summary['params_ols'] = coefs

        return df_summary

    @staticmethod

    def performance_table(returns_model: pd.Series, asset_name: str = 'asset_0',
                           benchmark_returns:pd.Series = None, start:int = 0):
        # Compute performance table for a single model
        if benchmark_returns is not None:
            benchmark_returns = benchmark_returns.reindex(returns_model.index, method='ffill')
        else:
            benchmark_returns = pd.Series(np.zeros(returns_model.shape), index=returns_model.index)
        if returns_model.empty:
            return pd.DataFrame()
        
        returns = returns_model.dropna().copy()
        pf =  round(pm.profit_factor(returns),2)
        sr =  round(pm.sharpe_ratio(returns)*(252**.5),2)
        mu = round(pm.mean_return(returns)*100,2)
        total_return =  round(pm.total_return(returns)*100,2)
        annual_r = round(returns[252*2:].mean()*25200, 2)
        max_drawdown =  round(pm.drawdown(returns)*100,2)
   
        time_in_market = len(returns.iloc[start:][returns.iloc[start:]!=0])/len(returns.iloc[start:])*100
        time_in_market = round(time_in_market,2)
        if benchmark_returns is not None:
             SR_bechmark = round(pm.sharpe_ratio(benchmark_returns)*(252**.5),2)
        else:
            SR_bechmark = 'Nan'
        dict = {asset_name: {'Annual Return %':annual_r,'Sharpe Ratio Annualized': sr,
                'Mean Return %': mu, 'Total Return %': total_return, 'Max Drawdown %': max_drawdown,
                'Profit Factor': pf,  'Sharpe Ratio benchmarck': SR_bechmark,
                 'Time in the market %':time_in_market }}
                
        df = pd.DataFrame.from_dict(dict, orient='index')
        return  df
    
    @staticmethod
    def all_performance_table(returns: List[pd.Series], names: List[str], benchmark_returns:pd.Series = None ):
        # compute performance table for multiple Series
        if benchmark_returns is None:
            benchmark_returns = pd.Series(np.zeros(returns[0].shape), index=returns[0].index)
        else:
            benchmark_returns = benchmark_returns.reindex(returns[0].index, method='ffill')
        if len(returns) == 0:
            return pd.DataFrame()


        portfolio_returns = np.zeros(returns[0].shape)
        for i, asset_r in enumerate(returns):
            name = names[i]
            if i == 0:
                df = assess_performance.performance_table(asset_r, name, benchmark_returns)
                portfolio_returns+=asset_r
            else:
                df = pd.concat([df,assess_performance.performance_table(asset_r, name, benchmark_returns)])
                portfolio_returns+=asset_r

        portfolio_returns/=len(returns) # Equal Weigth
        df = pd.concat([df,assess_performance.performance_table(portfolio_returns, 'Model porfolio EW', benchmark_returns)])
        df = pd.concat([df,assess_performance.performance_table(benchmark_returns, 'Benchmark')])
        return df
# %%
