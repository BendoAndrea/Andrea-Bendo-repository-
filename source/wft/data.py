# %%
import pandas as pd
import numpy as np
import yfinance as yf
class DataManager:
    def __init__(self,
                inputs:pd.DataFrame,
                targets:pd.Series,
                ohlc_df:pd.DataFrame):
        self.inputs  = inputs.copy(deep  = True)
        self.targets = targets.copy(deep  = True)
        self.ohlc_df = ohlc_df.copy(deep  = True)

    def clean_data(self) ->  pd.DataFrame | pd.Series| pd.DataFrame:  
    
        # Drop Nans
        clean_inputs  = self.inputs.dropna()
        clean_targets = self.targets.dropna() 
        clean_ohlc    = self.ohlc_df.dropna() 
        #Get new indexes
        new_indexes   = clean_inputs.index.intersection(clean_targets.index)
        self.X  = clean_inputs.loc[new_indexes]
        self.y = clean_targets.loc[new_indexes]
        self.ohlc    = clean_ohlc.loc[new_indexes]
        if not np.all(self.X.index==self.y .index) or not np.all(self.ohlc.index==self.y .index):
            raise ValueError('Check your data, in DataManager.clean_data() something went wrong')

        return self.X, self.y, self.ohlc

    def get_window_train(self, current_i:int,
                     size_train:int,
                     lookahead: int
                     )  -> pd.DataFrame| pd.Series:  
        
        """ 
        return training window of X, y, while removing for lookahead bias
        """
        if size_train>0:
            back = current_i-size_train - lookahead + 1 
        else:
            back = 0

        back = back if back >=0 else 0 
        X_train = self.X[back:  current_i - lookahead +1] 
        y_train = self.y[back:  current_i - lookahead +1]
        return X_train, y_train
# %%
def fetch_yf_data(Symbol):
    ticker = yf.Ticker(Symbol)
    data = ticker.history(start='1995-01-01', auto_adjust=False)
    data= data[data['Volume']!=0]
    data = data.dropna()
    data =np.log(data[['Open','High','Low', 'Close','Volume']]) 
    data = data[~data.index.duplicated(keep='first')]
    return data