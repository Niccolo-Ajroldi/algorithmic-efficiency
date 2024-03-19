
import torch
from collections import deque

class ListOfParams():
  def __init__(self, params) -> None:
    self._params = [p.detach().clone() for p in params]

  def update(self, params):
    self._params = [p.detach().clone() for p in params]
  
  def parameters(self):
    return self._params
    
  def state_dict(self):
    return {key: value for key, value in self.__dict__.items()}
  
  def load_state_dict(self, state_dict):
    self.__dict__.update(state_dict)


class LAWAEma:
  def __init__(self, beta) -> None:
    self.beta = beta
    self.ema = None
  
  def push(self, params):
    if self.ema is None:
      self.ema = [p.detach().clone(memory_format=torch.preserve_format) for p in params]      
      return
    
    beta = self.beta
    for p_ema, p in zip(self.ema, params):
      p_ema.mul_(beta).add_(p, alpha=1-beta)
  
  def get_avg(self):
    return self.ema
  
  def state_dict(self):
    return {key: value for key, value in self.__dict__.items()}
  
  def load_state_dict(self, state_dict):
    self.__dict__.update(state_dict)
