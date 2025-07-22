from typing import Union
import numpy as np
class PositionManager:
    """ 
    This is optional but if the lookahead is > 1, positions may overlap, 
    making it tricky to keep track of position size and returns, instead if a moving average is used
    the maximum |position| open at any given time will be 1, if and only if the past n_lookahead predictions
    had  the same sign and a position a of |size| = 1/lookahead will be opened if the model fitted say so,
    note that if lookahead == 1 this does nothing.

    In short:
        compute moving average with (window=lookahead) of the position size based on past values of the position it's self
    """

    def __init__(self, lookahead: int):

        self.window = lookahead
        self.position_hist     = []
        self.adjusted_pos_hist = []
        self.last_position = 0
    def compute_rolling_avg_position(self, new_pred: float, commision_fee:float) ->  Union[float,float]:
        self.position_hist.append(new_pred)
        if len(self.position_hist)<self.window:
            current_position = np.sum(self.position_hist)/len(self.position_hist)
            self.adjusted_pos_hist.append(current_position)

            fee = self.compute_commision(self.last_position , current_position, commision_fee)
            self.last_position = current_position
        else:
            position_window = self.position_hist[-self.window:]
            current_position = np.sum(position_window)/self.window


            fee = self.compute_commision(self.last_position , current_position, commision_fee)
            self.last_position = current_position
            self.adjusted_pos_hist.append(current_position)
            
        return current_position, fee
    
    def compute_commision(self, last_position, current_position, commision_fee):
        return np.abs(current_position - last_position)*commision_fee
    
