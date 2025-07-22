# %%
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


class PrincipalComponents:
    def __init__(self, n_components=3):
        self.n_components = n_components
        
    def fit(self, X):
        covariance = np.cov(X, rowvar = False)
        w, v = np.linalg.eigh(covariance)
        indexes = np.argsort(w)[::-1]
        self.v = v[:, indexes]
        new_X = np.dot(X, self.v[:self.n_components].T)
        return self
        
    def transform(self, X):
        return  np.dot(X, self.v[:self.n_components].T)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def set_params(self, n_components):
        self.n_components = n_components
        return self
    def get_params(self):
        params_ = {'n_componenets':self.n_components}
        return self.n_components
  
class BaseEstimator(ABC):
    @abstractmethod
    def fit(self, X, y):
        ...

    @abstractmethod
    def predict(self, X, y):
        ...

class LinearRegression(BaseEstimator):
    def __init__(self, add_intercept: bool = False):
        self.add_intercept = add_intercept
        
    def fit(self,X: [pd.DataFrame, np.ndarray], y:[pd.Series, np.ndarray]):   # type: ignore
        X = np.asarray(X)
        y = np.asarray(y)
    
        if np.ndim(X)==1:
            X = X.reshape(-1,1)
        if self.add_intercept:
            X = np.concatenate([X, np.ones(len(X)).reshape(-1,1)],axis= 1)     
        XtX = X.T @ X
        XtX_inv = np.linalg.inv(XtX)
        self.params = XtX_inv @ X.T @ y 
        return self   

    def predict(self,X: [pd.DataFrame, np.ndarray]):   # type: ignore
        X = np.array(X)
        if X.ndim == 0:
            # if scalar is passed
            if not self.add_intercept:
                return  self.params * X 
            else:
                return np.r_[X, 1] @ self.params

        if np.ndim(X) == 1:
            
            n_features = self.params.shape[0]  if np.ndim(self.params)>0 else 1
            if self.add_intercept:
                n_features-=1
            length = X.shape[0]
            if length == n_features:
                #X = X.reshape(1,-1)
                if self.add_intercept:
                    X = np.concatenate([X,  np.ones(1) ] )
            elif  n_features ==1:
                X = X.reshape(-1,1)
                if self.add_intercept:
                    X = np.c_[X, np.ones(len(X)).reshape(-1,1)]
                    
            elif length>1 and np.ndim(X)==1:
                X = X.reshape(-1,1)
                if self.add_intercept:
                    X = np.c_[X, np.ones(len(X)).reshape(-1,1)]
                    
        else:     
            if np.ndim(X)>=2:
                if self.add_intercept:
                    X = np.c_[X, np.ones(len(X)).reshape(-1,1)]
        out = X @ self.params
        return out
    def set_params(self, add_intercept: bool = False):
        self.add_intercept = add_intercept
    def get_params(self):
        params_ = {'coefficients': self.params}
        return params_
    
    def standard_errors(self, ins:pd.DataFrame, outs:pd.Series, print_table= True):
        X = np.asarray(ins)
        y = np.asarray(outs)
        self.fit(X,y)
        

            
        residuals = y - np.dot(X, self.params)
        # Variance Residuals
        n, p = X.shape
        variance_res = np.dot(residuals, residuals)/( n - p - 1)
        # Covariance matrix
        cov_beta = variance_res * np.linalg.inv(np.dot(X.T, X))
        # Standard Errors
        se = np.sqrt(np.diag(cov_beta))
        
        if print_table:
            t_scores = self.params/se
            dict_summuray = {'Beta coef':self.params, 'Standard errors':se, 'T-scores': t_scores}
            summuray_df = pd.DataFrame(dict_summuray, index = ins.columns).T
            print(summuray_df)
        return summuray_df
    def r_square(self, ins:pd.DataFrame, outs:pd.Series):
        TSS = np.sum((outs - outs.mean())**2)
        preds = self.fit(ins, outs).predict(ins)
        RSS = np.sum((outs - preds)**2)
        R2 = 1 - (RSS/TSS)
        return R2
        
class NeuralNetwork(BaseEstimator):
    def __init__(self,
                 n_hidden: int = 3,
                 activation_f: str = 'sigmoid',
                 bias: bool = True,
                 lr = 0.1,
                 n_iter = 200,
                 warm_start = False):
        """
        Standard Artificial Neural Network
        Minimization of MSE with backpropagation
        n_hidden:
            Number of neurons of the input layer and the first hidden layer
        activation_f:
            Activation function to be used after the input layer
        bias:
            If True, add bias to the model
        lr:
            Learning rate for the model
        n_iter:
            Number of iterations for the model
        warm_start:
            If True, the model will not reinitialize the weights and biases, but will continue training
        """
        
        if type(n_hidden) != int:
            raise ValueError(f'Incorrect n_hidden inputs expected tuple of int got: {n_hidden}')
        
        if not isinstance(activation_f, str):
            raise ValueError(f'Incorrect activation function expected str got: {type(activation_f)}')

        self.n_hidden = n_hidden
        self.activation_f = activation_f
        self.bias = bias
        self.lr = lr
        self.n_iter = n_iter
        self.warm_start = warm_start
        np.random.seed(1)
    def init_params(self, n_ins, n_outs):
        """ Initialize weights and biases of the model"""
        rng = np.random.default_rng(1)
        # Xavier/Glorot
        lim1 = np.sqrt(6 / (n_ins + self.n_hidden))
        lim2 = np.sqrt(6 / (self.n_hidden + n_outs))
        self.w1 = rng.uniform(-lim1, lim1, (n_ins, self.n_hidden))
        self.w2 = rng.uniform(-lim2, lim2, (self.n_hidden, n_outs))
        if self.bias:
            self.b1 = np.zeros((1, self.n_hidden))
            self.b2 = np.zeros((1, n_outs))

    def predict(self, ins):
        
        X = np.asarray(ins.copy())
        if np.ndim(X)<=1 and self.w1.shape[0]==1:
            X = X.reshape(-1,1)
        else:
            X = np.atleast_2d(X).astype(float)  
        X1 = X @ self.w1 
        if self.bias:
           X1 = X1 + self.b1
        A1 = self.act_f(X1)
        X2 = A1 @ self.w2 
        if self.bias:
           X2=  X2 + self.b2
    
        return X2.squeeze()

    
    def fit(self, ins, outs):
        """ Fit the model to the data, using gradient descent and Adagrad optimizer"""
        lr = self.lr
        n_iter = self.n_iter

        X = np.asarray(ins.copy())
        y = np.asarray(outs.copy())
        
        if np.ndim(y)==1:
            y = y.reshape(-1,1)
            
        if np.ndim(X)==1:
            X = X.reshape(-1,1)
            
        X = np.atleast_2d(X).astype(float)     

        cases, n_ins = X.shape
        n_outs = y.shape[1]
        if not self.warm_start or not hasattr(self, "w1"):
            self.init_params(n_ins, n_outs)

        try:
            self.act_f   = getattr(self, self.activation_f)
            self.d_act_f = getattr(self, self.activation_f + '_d')
        except:
            raise ValueError(f'choose between: "relu", "sigmoid", got: {self.activation_f}') # type: ignore
            
        self.current_loss = []
        
  
        w1_sq_sum = np.zeros_like(self.w1)
        w2_sq_sum = np.zeros_like(self.w2)
        if self.bias:
            b1_sq_sum = np.zeros_like(self.b1)
            b2_sq_sum = np.zeros_like(self.b2)

        for _ in range(n_iter):
            
            X1 = X @ self.w1 
            if self.bias:
                X1 = X1 + self.b1
            A1 = self.act_f(X1)
            X2 = A1 @ self.w2 
            if self.bias:
                X2 = X2 + self.b2
       
            
            dX2  =  2/cases * (X2 - y) 
            dA1  = self.d_act_f(X1)        
            delta_1 = (dX2 @ self.w2.T) * dA1
            
            gradient_w1 = X.T @ delta_1
            gradient_w2 = A1.T @ dX2
            
            w1_sq_sum+=gradient_w1**2
            w2_sq_sum+=gradient_w2**2
            lr_w1 = lr/(np.sqrt(w1_sq_sum) + 1e-8)
            lr_w2 = lr/(np.sqrt(w2_sq_sum) + 1e-8)    
            
            self.w1 -= lr_w1 * gradient_w1
            self.w2 -= lr_w2 * gradient_w2 
            
        
            
            if self.bias:
                gradient_b1 = np.sum(delta_1, axis=0, keepdims=True)
                gradient_b2 = np.sum(dX2, axis=0, keepdims=True)
                
                b1_sq_sum += gradient_b1**2 
                b2_sq_sum += gradient_b2**2
                lr_b1 = lr/(np.sqrt(b1_sq_sum) + 1e-8)
                lr_b2 = lr/(np.sqrt(b2_sq_sum) + 1e-8)
                
                self.b1 -= lr_b1 * gradient_b1
                self.b2 -= lr_b2 * gradient_b2

            self.current_loss.append(self.mean_square_error(X2, y))

        return self

        
    @staticmethod
    def mean_square_error(pred, y):
        return np.mean((pred - y)**2)
        
    @staticmethod
    def sigmoid(x):
        x_clipped = np.clip(x, -500, 500)
        return 1 / ( 1 + np.exp(-x_clipped))
        
    def sigmoid_d(self, x):
        sig = self.sigmoid(x)
        return  sig * (1- sig)
        
    @staticmethod
    def relu(x):
        return np.maximum(x,0)

    @staticmethod
    def relu_d(x):
        return np.where(x>0,1,0)

    def set_params(self, 
                 n_hidden: int = None,
                 activation_f: str = None,
                 bias: bool = None,
                 lr: int =None,
                 n_iter:int = None,
                 warm_start: bool = False):
        
        """ 
        Set parameters of the model
        if left to None use the current value
        """

        self.n_iter = n_iter if n_iter is not None else self.n_iter
        self.activation_f = activation_f if activation_f is not None else self.activation_f
        self.lr = lr if lr is not None else self.lr
        self.bias = bias if bias is not None else self.bias
        self.n_hidden = n_hidden if n_hidden is not None else self.n_hidden
        self.activation_f = activation_f if activation_f is not None else self.activation_f
        
        self.warm_start = warm_start
        return self
    
    def get_params(self):
        params_ = {}
        params_['weights_1'] = self.w1
        params_['weights_2'] = self.w2
        if self.bias:
            params_['bias_1'] = self.b1
            params_['bias_2'] = self.b2  
        return params_         
class Pipeline:
    """
    Pipeline for the model, it can contain PCA and a model as of now
    """
    def __init__(self, model_components):

        MODEL_REGISTRY = {
            'pca': ('pca' , PrincipalComponents()),
            "ols": ("ols" , LinearRegression()),
            "nn" : ("nn"  , NeuralNetwork())
        } 

        self.steps=  []
        for path in model_components:
            if path not in MODEL_REGISTRY.keys():
                raise ValueError(f'Model component not included, expected one of: {MODEL_REGISTRY.keys()}, got: {path}')          
            self.steps.append(MODEL_REGISTRY[path])
            ##except:
           #     raise ValueError(f'Model component not included, expected one of: {MODEL_REGISTRY.keys()}, got: {path}')
                                
    def fit(self, X, y):
        
        for step in self.steps:
            name, func = step
            if name =='pca': # PCA compute -> fit
                X = func.fit_transform(X)
            else:  #Model
                func.fit(X, y) 
        return self

    def predict(self, X):
        for step in self.steps:
            name, func = step
            if name =='pca': # PCA fit
                X = func.transform(X)
            else:  #Model
                return func.predict(X)  
                
    def set_params(self, **kwargs):
        # Modify params of passed model
        for name_model,val in kwargs.items():
            for name, func in self.steps:
                if name == name_model and val is not None:
                    func.set_params(**val)
    
        return self
    def get_params(self):
        params_ = {}
        for step in self.steps:
            name, func = step
            p = func.get_params()
            params_[name] = p
        return params_