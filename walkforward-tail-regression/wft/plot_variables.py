# %%
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
from scipy import stats
import seaborn as sns
import pandas as pd 
import numpy as np
from typing import List, Tuple, Callable, Optional, Union
from .models import LinearRegression
class Visualization:
    """
    Visualization class for plotting various data and statistics
    """
    @staticmethod
    def plot_correlation(variables_df: pd.DataFrame):
        corr_matrix = variables_df.corr()
        plt.figure(figsize=(7,7))
        plt.title('Correlation matrix')
        sns.heatmap(corr_matrix, cmap = 'viridis', annot= True)
        plt.show()

    @staticmethod
    def plot_time_series(variables: [pd.DataFrame, pd.Series]): # type: ignore
        """
        Plot time series or dataframe data
        """
        if isinstance(variables,pd.DataFrame):
            n_series = variables.shape[1]
            column_names =  variables.columns
           
            fig,axes = plt.subplots(nrows = n_series, ncols = 1,figsize=(10, int(n_series*1.5)))
            for i,ax in enumerate(axes):
                ax.plot(variables.index, variables[column_names[i]], color="black", linewidth = .75)
                ax.set_title(f"{column_names[i].upper()}")
                mu = variables[column_names[i]].mean()
                std_2 = variables[column_names[i]].std()*2
                ax.axhline( mu, color='r', linestyle = '--' ,label='mean', linewidth = .75)
                ax.axhline( mu + std_2, color='grey', linewidth = .75,linestyle = '--' ,label=' μ +/- 2* σ')
                ax.axhline( mu - std_2, color='grey', linewidth = .75,linestyle = '--' )
                ax.set_ylabel('Range variable', fontsize = 12)
                ax.legend(loc='upper right', fontsize= 8)
            plt.tight_layout() 
            plt.show()
            
        else:
            n_series = variables.shape
            column_name = 'Variable'
            mu = variables[column_names[i]].mean()
            std_2 = variables[column_names[i]].std()*2
            plt.figure(figsize = (10,3))
            plt.title(f"{column_name}")
            plt.plot(variables.index, variables, color="black", linewidth = .75)
            plt.axhline( variables.mean(), color='r', linestyle = '--' ,label='mean')
            plt.axhline( mu + std_2, color='grey', linewidth = .75,linestyle = '--' ,label=' μ +/- 2*σ')
            plt.axhline( mu - std_2, color='grey', linewidth = .75,linestyle = '--' )
            plt.ylabel('Range variable', fontsize = 12)
            plt.tight_layout()  
            plt.legend(loc='upper right', fontsize= 8)
            plt.show()        
            
    @staticmethod
    def plot_histogram(variables: [pd.DataFrame, pd.Series], name_var: Optional[ str ] = 'Variable_1'): # type: ignore
        """
        Plot series or dataframe histogram
        """   
        def plot_hist(variable, name, ax, fontsize=12):
            df_ = variable.copy().dropna()
            ax.hist(df_, density = True, bins = len(df_)//40, color = 'blue', alpha = .5, edgecolor = 'black')
            ax.set_title(name, fontsize=fontsize)
            ax.set_xlabel('Range Variable', fontsize=fontsize)
            ax.set_ylabel('Density', fontsize=fontsize)
            mean_var = df_.mean()
            std_var = df_.std()
            x_grid, y_kde = Visualization.compute_kde(df_)
            ax.plot(x_grid, y_kde, color='darkred', label = 'kde')
            ax.axvline(x = mean_var, linestyle = '--', linewidth = 1, color= 'red' , label= 'mean')
            ax.axvline(x = mean_var + std_var, linestyle = '--', linewidth = 1, color= 'green' , label = '1_std')
            ax.axvline(x = mean_var + std_var*2, linestyle = '--', linewidth = 1, color= 'black' , label = '2_std')
            ax.axvline(x = mean_var - std_var, linestyle = '--', linewidth = 1, color= 'green')
            ax.axvline(x = mean_var - std_var*2, linestyle = '--', linewidth = 1, color= 'black')
            ax.legend(loc = 'upper right')
                      
          
        if isinstance(variables,pd.DataFrame):

            

            n_series = variables.shape[1]
            if n_series == 2 or n_series == 4:
                n_rows , n_cols = 2, 2
            else:
                n_rows = n_series//3
                n_cols = 3
            if n_series!= n_rows*n_cols:
                n_off = n_series % n_cols
                n_off = range(1, 1+n_cols)[-n_off]
            else:
                n_off = 1

            h = n_rows*3
            h = h if h <=12 else 12
            fig = plt.figure(figsize= (10,h))
            axs = []
            gs = gridspec.GridSpec(n_rows, n_cols*n_off)

            for i, name in enumerate(variables.columns):
                row = i// n_cols
                col = i % n_cols
                if row == n_rows -1 and n_off!=1:
                    off = int(n_off * (n_cols - n_series % n_cols) / 2)
                else:
                    off = 0
                ax = plt.subplot(gs[row, n_off*col + off: n_off*(col + 1) + off])  
                plot_hist(variables[name], name, ax, fontsize=10)  
                axs.append(ax)   

            plt.tight_layout()
            plt.show()
        else:
            fig = plt.figure(figsize = (8, 5))
            plot_hist(variables, 'Variable', plt, fontsize=10)
            plt.show()

    @staticmethod
    def compute_kde(var, method = 'silverman'):
        kde = stats.gaussian_kde(var ,bw_method='silverman')
        var_min, var_max = var.min(), var.max()
        x_grid = np.linspace(var_min, var_max, 500)
        y_kde = kde(x_grid)
        return x_grid, y_kde
    
    @staticmethod
    def plot_joint_distribution(
        dependent : Union[np.ndarray, pd.Series], 
        independent_var: Union[np.ndarray, pd.Series],
        percentiles: Tuple = [5, 95], tails:bool =False
    ):
        """
        Plot the joint distribution of two variables with regression line
        dependent:
            Dependent variable, pd.Series or numpy array
        independent_var:
            Independent variable, pd.Series or numpy array
        percentiles:
            Percentiles to be used for the tails, default is [5, 95]
        tails:
            If True, plot only the tails of the distribution
        """
        y = np.asarray(dependent )
        X = np.asarray(independent_var)

            
        p0 = np.min(percentiles)
        p1 = np.max(percentiles)
        perc_5, perc_95 = np.percentile(X, [p0,p1])
        if tails:
            y = y[(X>=perc_95)|(X<=perc_5)]
            X = X[(X>=perc_95)|(X<=perc_5)]
        # Get Regression
        x_min, x_max = X.min(), X.max()
        X_intc = np.stack([X, np.ones(X.size)]).T
        lr = LinearRegression().fit(X_intc, y)

        beta = lr.params[0]
        intercept = lr.params[1]
        x_line = np.array([x_min, x_max])
        y_line = intercept + beta*x_line
    
        # Lower Tail
        d1 = y[X<=perc_5] 
        x_grid1, y_kde1 = Visualization.compute_kde(d1)
        mean_d1 = np.mean(d1)
        #Upper tail
        d2= y[X>=perc_95] 
        x_grid2, y_kde2 = Visualization.compute_kde(d2)
        mean_d2 = np.mean(d2)
        
        fig= plt.figure(figsize=(11, 7))
        #gs = fig.add_gridspec(2,2 )
        
        ax =  plt.subplot(2, 1, 1)#fig.add_subplot(gs[0, 1])
        ax.scatter(x=X, y=y, c=('blue', 0.4))
        ax.plot(x_line,y_line, color= 'red', label = f'Regression Slope {beta:.4f}')
        plt.axhline(0, color= 'black', linestyle = '--', label = 'Zero line')
        ax.set_ylabel('Dipendent Variable', fontsize = 10)
        ax.set_xlabel('Indipendent Variable', fontsize = 10)
        ax.set_title('Joint Distribution with regression slope', fontsize = 10)
        ax.legend()
    
        # Conditional Distributions
        ax1 =  plt.subplot(223)
        ax1.plot(x_grid1, y_kde1, color='darkred')
        ax1.hist(d1, bins= 100, density=True, color = 'blue', alpha = .5)
        ax1.axvline(mean_d1, color = 'black', label= f'Mean = {mean_d1:.4f}')
        ax1.set_xlabel('Dipendent Variable', fontsize = 10)
        ax1.set_ylabel('PDF', fontsize = 10)
        ax1.set_title(f'PDF (Dependent variable | indepepndent var <= {perc_5:.4f})', fontsize = 10)
        ax1.legend()
    
        ax2 =  plt.subplot(224)
        ax2.plot(x_grid2, y_kde2, color='darkred')
        ax2.hist(d2, bins= 100, density=True, color = 'blue', alpha = .5)
        ax2.axvline(mean_d2, color = 'black', label= f'Mean = {mean_d2:.4f}')
        ax2.set_xlabel('Dipendent Variable', fontsize = 10)
        ax2.set_ylabel('PDF', fontsize = 10)
        ax2.set_title(f'PDF (Dependent variable | indepepndent var >= {perc_95:.4f})', fontsize = 10)
        ax2.legend()
        
        plt.tight_layout()
        
        plt.show()
    @staticmethod
    def plot_single_equity_curve(returns: pd.Series, model:Callable,
                                    X:pd.DataFrame,long_stats: pd.Series, 
                                    short_stats: pd.Series, bench_mark_returns: pd.Series= None, transform: str = None):

        """ Plot single equity curve with predictions and thresholds
        returns: pd.Series of returns
        model: Function of the model used for predictions (WF.model)
        X: pd.DataFrame of independent variables (WF.inputs)
        long_stats: pd.Series of long position statistics
        short_stats: pd.Series of short position statistics
        transform: str for transformation type
        """

        if transform == 'standardize':
            transformed_inputs = (X - np.mean(X,axis=0))/np.std(X,axis=0) 
        elif transform =='normalize':
            transformed_inputs = (X - np.min(X,axis=0))/(np.max(X,axis=0)-np.min(X,axis=0)) 
        else:
            transformed_inputs = X
        transformed_inputs = transformed_inputs.loc[returns.index]
        X =  X.loc[returns.index].copy()
        
        preds = model.predict(transformed_inputs)
        preds = pd.Series(preds, index = returns.index)
        cumsum_r = returns.cumsum().copy()
        df = pd.DataFrame(index = returns.index)  
    

        df["preds"] = preds
        df["Equity"] = cumsum_r
        if not long_stats.empty:
            long_stats=long_stats.set_index('Start_test' )
            thresh_longs = long_stats['threshold'].copy()
            thresh_longs = thresh_longs[~thresh_longs.index.duplicated(keep='first')]
            df['Upper_t'] = thresh_longs 
        else:
            thresh_longs = None
        if not short_stats.empty:
            short_stats =short_stats.set_index('Start_test' )
            thresh_shorts = short_stats['threshold'].copy()
            thresh_shorts = thresh_shorts[~thresh_shorts.index.duplicated(keep='first')]
            df['Lower_t'] = thresh_shorts 
        else:
            thresh_shorts = None
        
        first_index = df["Equity"][df["Equity"] !=0].index[0] # First index with non-zero equity
        last_index = df["Equity"].index[-1]  
        df.ffill(inplace=True) # Fill NaN values
        df = df.loc[first_index:last_index] # For fair comparison




        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14,8), gridspec_kw={'height_ratios':[3, 1]})
        ax1.set_ylabel('Log return')
        ax1.plot(df.index, df["Equity"], color='black', label = 'Model returns')

        if bench_mark_returns is not None:
            market_cumreturn = bench_mark_returns.loc[first_index:last_index].cumsum()
            ax1.plot(df.index, market_cumreturn, color='lightblue', label = 'Market returns')

        ax1.legend(loc = 'upper left', fontsize = 8)
        ax1.set_title("Equity Curve after commission fees", loc='center')

        
        ax2.plot(df.index, df["preds"], color='black', label ='prediction', linewidth = .75)
        ax2.set_ylabel('Predictions range')
        if thresh_longs is not None:
            ax2.plot(df.index, df["Upper_t"], color = 'green', label = 'long_threshold')
        if thresh_shorts is not None:
            ax2.plot(df.index, df["Lower_t"], color = 'red', label = 'short_threshold')
           
        ax2.set_title("Predictions angainst computed threshold", loc='center')
        ax2.legend(loc = 'upper left', fontsize = 8)

        plt.title("Equity Curve after commission fees", loc='center')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_multiple_equity_curve(returns: List, Symbol_name: List, bench_mark_r: pd.Series, start: int):
        plt.figure(figsize= (16,8))
        portfolio_returns = pd.Series(np.zeros(len(bench_mark_r)), index = bench_mark_r.index)
        for i,model_asset_r in enumerate(returns):
            plt.plot(model_asset_r.iloc[start:].cumsum(), label = Symbol_name[i], linewidth = .75)
            portfolio_returns.loc[model_asset_r.index]+=model_asset_r

        portfolio_returns/=len(returns) # Equal Weigth       
        plt.plot(portfolio_returns.iloc[start:].cumsum(), label = 'Equal Weigth portfolio Return', color = 'black', linewidth = 1.5) 
        plt.plot(bench_mark_r.iloc[start:].cumsum(), label = 'Market Return' ,linewidth = 1.5)
        plt.plot()
        plt.legend(loc = 'upper left', fontsize = 8)
        plt.show()  
# %%
