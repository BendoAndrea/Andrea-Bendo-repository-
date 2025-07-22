
# %%
import numpy as np
import pandas as pd
class Preprocessing:
    def __init__(self, preprocess_type):

        self.transformations = preprocess_type

    def fit(self, variable: pd.DataFrame | pd.Series) -> None:  

        if 'standardize' in self.transformations:
            self.means    = np.mean(variable, axis = 0)
            self.std_devs = np.std(variable, axis = 0)
             
        elif 'normalize' in self.transformations:
            self.maxs = np.amax(variable, axis = 0)
            self.mins = np.amin(variable, axis = 0)
        else:
            pass
        
        return self
    
    def transform(self, variable: pd.DataFrame | pd.Series) ->  pd.DataFrame | pd.Series:  
        var = variable.copy()
        if 'standardize' in self.transformations:
            return  (var - self.means)/self.std_devs
        elif 'normalize' in self.transformations:
            return  (var - self.mins)/(self.maxs - self.mins)
        else:
            return var
        
    def fit_transform(self, variable):
        return self.fit(variable).transform(variable)
    
    def cap_outliers(self, variable : pd.DataFrame | pd.Series, coeff = 1.5) ->  pd.DataFrame | pd.Series:
        
        if 'cap_outliers' in set(self.transformations):
            var = variable.copy()
            q1,q3 =  np.percentile(var, [25,75], axis = 0)
            iqr = q3 - q1
            coef = coeff*iqr
            lower_bound = q1 - coef
            upper_bound = q3 + coef
            if isinstance(var, pd.Series):
                var.loc[var<lower_bound] = lower_bound
                var.loc[var>upper_bound] = upper_bound
                return var
            elif isinstance(var, pd.DataFrame):
                for i, name in enumerate(var.columns):
                    var[name].loc[var[name]<lower_bound[i]] = lower_bound[i]
                    var[name].loc[var[name]>upper_bound[i]] = upper_bound[i]
                return var
            else:
                raise TypeError(f'Wrong Data type passed expected: "pd.DataFrame" or "pd.Series", got: {type(var)}')
        else:
            return variable
            
# %%
