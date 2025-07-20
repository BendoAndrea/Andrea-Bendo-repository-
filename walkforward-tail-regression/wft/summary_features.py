# %%
import pandas as pd
import numpy as np

class VarsAnalysis:
    """
    Descriptive Statistics for variable passed
    """
    @staticmethod
    def range_to_iqr(x):
        q75, q25 = np.percentile(x, [75 ,25])
        iqr = q75 - q25
        range_to_iqr = (x.max() - x.min())/iqr
        return range_to_iqr
    @staticmethod
    def vars_stats(x):
        mean_x = np.mean(x)
        std_x  = np.std(x)
        skew_x = np.mean(((x - mean_x)/std_x)**3)
        kurt_x = np.mean(((x - mean_x)/std_x)**4) - 3
        median_x = np.median(x)
        min_x, max_x = min(x), max(x) 
        summary = {
            'mean': mean_x,
            'standard_dev': std_x,
            'skew': skew_x,
            'kurtosis': kurt_x,
            'median': median_x,
            'min value': min_x,
            'max value': max_x
            }
        return summary
    @classmethod
    def analyze_variable(cls, variable: pd.Series | np.ndarray, name: str) -> pd.DataFrame:
        """
        Computes basic statistics for Variables passed
        """
        var = np.asarray(variable, dtype=float)
        if len(var) == 0:
            raise ValueError("Empty data passed to analyze_variable()")        
        
        iqr_ratio = cls.range_to_iqr(var)
        summary = cls.vars_stats(var)
        summary['Range/IQ'] = iqr_ratio
                  
        df = pd.DataFrame.from_dict(summary, orient="index", columns=["Values"])
        if name:
            df.index.name = name
        
        return df
# %%
