# %%
import numpy as np
import pandas as pd
class ProfitMetrics:
    """
    Performance Profit Based Metrics Dataset
    """
    @staticmethod
    def profit_factor(returns):
        r = returns[returns!=0]
        if len(r)<1:
            return 0
        denom = np.sum(np.abs(r[r<0]))
        pf = np.sum(r[r>0])/(denom+1e-6) #Avoid divisions by zeros
        return pf
    
    @staticmethod
    def drawdown(returns):
        if len(returns)<=1:
            return  0.0       
        r = pd.Series(returns)
        price = (r).cumsum()
        peak = price.cummax()
        max_drawdown = (price - peak).min()
        return -max_drawdown
    @staticmethod
    def martin_ratio(returns, risk_free_rate = 0.0):
        """
        Martin Ratio defined as:
            Martin Ratio = (mean(return – risk_free_rate)) / Ulcer Index
            Ulcer Index  = Volatility of drawdowns
        """  
        if len(returns)<=1:
            return  0.0
        # turn into a series in case an array is passed
        r = pd.Series(returns)
        # Cumlative sum of returns Assuming log returns are used
        price = (r).cumsum()
        
        # rolling peak to compute drawdowns in decimal terms
        peak = price.cummax()
        drawdown = (price - peak)
        # squared drawdowns
        sq_dd = drawdown ** 2
        # Ulcer Index  
        ulcer_index = np.sqrt(sq_dd.mean())
        avg_excess_return = (r - risk_free_rate).mean()
        # Martin Ratio
        Martin_ratio = avg_excess_return / (ulcer_index+1e-15)
        return Martin_ratio
        
    @staticmethod
    def sharpe_ratio(returns, risk_free_rate = 0.0):
        r = returns[returns!=0]
        if len(r)<1:
            return 0
        avg_excess_return = np.mean(np.asarray(r) - risk_free_rate)
        Volatility = np.std(r)
        Sharpe_ratio = avg_excess_return/(Volatility + 1e-15) #Avoid divisions by zeros
        
        return Sharpe_ratio
        
    @staticmethod
    def total_return(returns):
        return np.sum(returns)
        
    @staticmethod
    def mean_return(returns):
        returns = returns[returns!=0]
        if len(returns)<1:
            return 0
        return np.mean(returns) 
    
    @classmethod
    def compute(cls, metric:str, returns, **kwargs):   
        """
        Dispatch to the right metric. Valid metrics:
        profit_factor, mean_return, total_return,
        sharpe_ratio, martin_ratio
        """
        
        try:
            func = getattr(cls, metric)
        except AttributeError:
            raise ValueError(f'Unknown metric: {metric}')
            
        return func(returns, **kwargs)

# %%
