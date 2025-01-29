"""
Offline LAWA Queue

Hyperparameters:
  - lawa_burnin_pct
  - lawa_every_pct
  - lawa_queue_len

"""

import math
from typing import Dict, Iterator, List, Tuple

from collections import deque
from itertools import islice

from absl import logging

from algorithmic_efficiency import spec
from algorithmic_efficiency.pytorch_utils import pytorch_setup

USE_PYTORCH_DDP = pytorch_setup()[0]


class LAWA():
  def __init__(self, hyperparameters, workload) -> None:
    self.tmp_params = None
    self.beta = hyperparameters.lawa_beta
    assert self.beta >= 0 and self.beta <= 1, f"invalud value of lawa_beta: {self.beta}"
    self.ema = None
    
    self.start_step = math.ceil(workload.step_hint * hyperparameters.lawa_burnin_pct)
    
    has_pct = getattr(hyperparameters, "lawa_every_pct", None) is not None
    has_step = getattr(hyperparameters, "lawa_every_steps", None) is not None
    if not has_pct and not has_step:
      raise ValueError("Missing hyperparameter: lawa_every_steps or lawa_every_pct")
    if has_step and has_pct:
      raise ValueError("Both lawa_every_steps and lawa_every_pct are defined")

    if has_step:
      self.every_step = int(hyperparameters.lawa_every_steps)
    else:
      self.every_step = math.ceil(workload.step_hint * hyperparameters.lawa_every_pct)
    logging.info('=== Running LAWA with self.every_step = %d ===', self.every_step)

  def store_tmp_params(self, params):
    self.tmp_params = [p.detach().cpu() for p in params]

  def update_ema(self, params):
    if self.ema is None:
      self.ema = [p.detach().cpu() for p in params]
    else:
      beta = self.beta
      for p_ema, p in zip(self.ema, params):
        p_ema.mul_(beta).add_(p.detach().cpu(), alpha=1-beta)
  
  def state_dict(self):
    return {key: value for key, value in self.__dict__.items()}

  def load_state_dict(self, state_dict):
    self.__dict__.update(state_dict)


def init_optimizer_state(workload: spec.Workload,
                         model_params: spec.ParameterContainer,
                         model_state: spec.ModelAuxiliaryState,
                         hyperparameters: spec.Hyperparameters,
                         rng: spec.RandomState) -> spec.OptimizerState:
  """Creates a NAdamW optimizer and a learning rate schedule."""
  del model_params
  del model_state
  del rng

  optimizer_state = {}
  optimizer_state['lawa'] = LAWA(hyperparameters, workload)

  return optimizer_state


def update_params(workload: spec.Workload,
                  current_param_container: spec.ParameterContainer,
                  current_params_types: spec.ParameterTypeTree,
                  model_state: spec.ModelAuxiliaryState,
                  hyperparameters: spec.Hyperparameters,
                  batch: Dict[str, spec.Tensor],
                  loss_type: spec.LossType,
                  optimizer_state: spec.OptimizerState,
                  eval_results: List[Tuple[int, float]],
                  global_step: int,
                  rng: spec.RandomState) -> spec.UpdateReturn:
  """Return (updated_optimizer_state, updated_params, updated_model_state)."""
  del workload
  del current_params_types
  del hyperparameters
  del batch
  del loss_type
  del eval_results
  del rng

  lawa = optimizer_state['lawa']
  current_model = current_param_container

  # Update LAWA
  if global_step >= lawa.start_step and global_step % lawa.every_step == 0:
    lawa.update_ema(current_model.parameters())
    # logging.info(f"EMA --- Updated EMA")

  return (optimizer_state, current_param_container, model_state)


def prepare_for_eval(workload: spec.Workload,
                     current_param_container: spec.ParameterContainer,
                     current_params_types: spec.ParameterTypeTree,
                     model_state: spec.ModelAuxiliaryState,
                     hyperparameters: spec.Hyperparameters,
                     loss_type: spec.LossType,
                     optimizer_state: spec.OptimizerState,
                     eval_results: List[Tuple[int, float]],
                     global_step: int,
                     rng: spec.RandomState) -> spec.UpdateReturn:
  del workload
  del current_params_types
  del hyperparameters
  del loss_type
  del eval_results
  del rng

  lawa = optimizer_state['lawa']
  current_model = current_param_container

  if global_step < lawa.start_step:
    return (optimizer_state, current_model, model_state)

  # logging.info(f"EMA --- Loading avg into model")

  # Load ema into model
  if lawa.ema is not None:
    for p, p_avg in zip(current_model.parameters(), lawa.ema):
        p.data.copy_(p_avg.data)  # move avg to GPU

  return (optimizer_state, current_model, model_state)

