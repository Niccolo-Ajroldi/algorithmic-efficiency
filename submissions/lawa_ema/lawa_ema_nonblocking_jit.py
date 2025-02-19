"""
LAWA ema on NAdamW optimizer with warmup+cosine LR in PyTorch.
LAWA ema offloaded to cpu.

Hyperparameters:
  - lawa_burnin_pct
  - lawa_every_pct
  - lawa_beta

TODO: explore Tensor.copy_(orig, non_blocking=True)
"""

import math
from typing import Dict, Iterator, List, Tuple

from itertools import islice

from absl import logging
import torch
import torch.jit
from torch import Tensor
import torch.distributed.nn as dist_nn

from algorithmic_efficiency import spec
from algorithmic_efficiency.pytorch_utils import pytorch_setup

USE_PYTORCH_DDP = pytorch_setup()[0]


# Modified from github.com/pytorch/pytorch/blob/v1.12.1/torch/optim/adamw.py.
class NAdamW(torch.optim.Optimizer):
  r"""Implements NAdamW algorithm.

    See Table 1 in https://arxiv.org/abs/1910.05446 for the implementation of
    the NAdam algorithm (there is also a comment in the code which highlights
    the only difference of NAdamW and AdamW).
    For further details regarding the algorithm we refer to
    `Decoupled Weight Decay Regularization`_.

    Args:
      params (iterable): iterable of parameters to optimize or dicts defining
          parameter groups
      lr (float, optional): learning rate (default: 1e-3)
      betas (Tuple[float, float], optional): coefficients used for computing
          running averages of gradient and its square (default: (0.9, 0.999))
      eps (float, optional): term added to the denominator to improve
          numerical stability (default: 1e-8)
      weight_decay (float, optional): weight decay coefficient (default: 1e-2)
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
  """

  def __init__(self,
               params,
               lr=1e-3,
               betas=(0.9, 0.999),
               eps=1e-8,
               weight_decay=1e-2):
    if not 0.0 <= lr:
      raise ValueError(f'Invalid learning rate: {lr}')
    if not 0.0 <= eps:
      raise ValueError(f'Invalid epsilon value: {eps}')
    if not 0.0 <= betas[0] < 1.0:
      raise ValueError(f'Invalid beta parameter at index 0: {betas[0]}')
    if not 0.0 <= betas[1] < 1.0:
      raise ValueError(f'Invalid beta parameter at index 1: {betas[1]}')
    if not 0.0 <= weight_decay:
      raise ValueError(f'Invalid weight_decay value: {weight_decay}')
    defaults = {
        'lr': lr, 'betas': betas, 'eps': eps, 'weight_decay': weight_decay
    }
    super().__init__(params, defaults)

  def __setstate__(self, state):
    super().__setstate__(state)
    state_values = list(self.state.values())
    step_is_tensor = (len(state_values) != 0) and torch.is_tensor(
        state_values[0]['step'])
    if not step_is_tensor:
      for s in state_values:
        s['step'] = torch.tensor(float(s['step']))

  @torch.no_grad()
  def step(self, closure=None):
    """Performs a single optimization step.

        Args:
          closure (callable, optional): A closure that reevaluates the model
              and returns the loss.
    """
    self._cuda_graph_capture_health_check()

    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()

    for group in self.param_groups:
      params_with_grad = []
      grads = []
      exp_avgs = []
      exp_avg_sqs = []
      state_steps = []
      beta1, beta2 = group['betas']

      for p in group['params']:
        if p.grad is None:
          continue
        params_with_grad.append(p)
        if p.grad.is_sparse:
          raise RuntimeError('NAdamW does not support sparse gradients')
        grads.append(p.grad)

        state = self.state[p]

        # State initialization
        if len(state) == 0:
          state['step'] = torch.tensor(0.)
          # Exponential moving average of gradient values
          state['exp_avg'] = torch.zeros_like(
              p, memory_format=torch.preserve_format)
          # Exponential moving average of squared gradient values
          state['exp_avg_sq'] = torch.zeros_like(
              p, memory_format=torch.preserve_format)

        exp_avgs.append(state['exp_avg'])
        exp_avg_sqs.append(state['exp_avg_sq'])
        state_steps.append(state['step'])

      nadamw(
          params_with_grad,
          grads,
          exp_avgs,
          exp_avg_sqs,
          state_steps,
          beta1=beta1,
          beta2=beta2,
          lr=group['lr'],
          weight_decay=group['weight_decay'],
          eps=group['eps'])

    return loss


def nadamw(params: List[Tensor],
           grads: List[Tensor],
           exp_avgs: List[Tensor],
           exp_avg_sqs: List[Tensor],
           state_steps: List[Tensor],
           beta1: float,
           beta2: float,
           lr: float,
           weight_decay: float,
           eps: float) -> None:
  r"""Functional API that performs NAdamW algorithm computation.
    See NAdamW class for details.
  """

  if not all(isinstance(t, torch.Tensor) for t in state_steps):
    raise RuntimeError(
        'API has changed, `state_steps` argument must contain a list of' +
        ' singleton tensors')

  for i, param in enumerate(params):
    grad = grads[i]
    exp_avg = exp_avgs[i]
    exp_avg_sq = exp_avg_sqs[i]
    step_t = state_steps[i]

    # Update step.
    step_t += 1

    # Perform stepweight decay.
    param.mul_(1 - lr * weight_decay)

    # Decay the first and second moment running average coefficient.
    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

    # Only difference between NAdamW and AdamW in this implementation.
    # The official PyTorch implementation of NAdam uses a different algorithm.
    # We undo these ops later on, which could cause numerical issues but saves
    # us from having to make an extra copy of the gradients.
    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

    step = step_t.item()

    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step

    step_size = lr / bias_correction1

    bias_correction2_sqrt = math.sqrt(bias_correction2)
    denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

    param.addcdiv_(exp_avg, denom, value=-step_size)
    exp_avg.sub_(grad, alpha=1 - beta1).div_(beta1)


class WarmCosine(object):
  def __init__(self, optimizer, lr_min, lr_max, warmup_steps, T):
    self.optimizer = optimizer
    self.lr_min = lr_min
    self.lr_max = lr_max
    self.warmup_steps = warmup_steps
    self.T = T
    self.t = 0
    for group in self.optimizer.param_groups:
      group["lr"] = lr_min
    
  def schedule(self, t):
    if t <= self.warmup_steps:
      return self.lr_min + (self.lr_max-self.lr_min)/self.warmup_steps * t
    elif t <= self.T:
      return self.lr_min + 0.5 * (self.lr_max-self.lr_min) * (1 + math.cos((t-self.warmup_steps) * math.pi / (self.T-self.warmup_steps)))
    return self.lr_min

  def step(self):
    self.t += 1
    # get LR for this step
    lr = self.schedule(self.t)
    # set LR in optimizer
    for group in self.optimizer.param_groups:
      group["lr"] = lr

  def state_dict(self):
    return {key: value for key, value in self.__dict__.items() if key != "optimizer"}

  def load_state_dict(self, state_dict):
    self.__dict__.update(state_dict)


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

  @torch.no_grad()
  def store_tmp_params(self, params):
    """Stores parameters asynchronously using non-blocking transfers."""
    self.tmp_params = [p.detach().to("cpu", non_blocking=True).pin_memory() for p in params]

  @torch.no_grad()
  def update_ema(self, params):
    """Performs an asynchronous non-blocking EMA update using torch.jit.fork."""
    if self.ema is None:
      # Initialize EMA in pinned memory for efficient async CPU transfers
      self.ema = [p.detach().cpu().pin_memory() for p in params]
    else:
      beta = self.beta
      # Non-blocking transfer of model params to CPU
      params_cpu = [p.detach().to("cpu", non_blocking=True) for p in params]
      # Fork a background thread for EMA update
      torch.jit.fork(self._ema_update, self.ema, params_cpu, beta)

  @staticmethod
  def _ema_update(ema, params_cpu, beta):
    """Performs the EMA update asynchronously in a separate thread."""
    for p_ema, p_cpu in zip(ema, params_cpu):
      p_ema.mul_(beta).add_(p_cpu, alpha=1 - beta)
  
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
  del model_state
  del rng

  optimizer_state = {}

  optimizer_state['optimizer'] = NAdamW(
      model_params.parameters(),
      lr=hyperparameters.learning_rate,
      betas=(1.0 - hyperparameters.one_minus_beta1,
              hyperparameters.beta2),
      eps=1e-8,
      weight_decay=hyperparameters.weight_decay)
  
  optimizer_state['lawa'] = LAWA(hyperparameters, workload)
    
  optimizer_state['scheduler'] = WarmCosine(
      optimizer_state['optimizer'], 
      lr_min = 1e-10, 
      lr_max = hyperparameters.learning_rate, 
      warmup_steps = int(hyperparameters.warmup_factor * workload.step_hint), 
      T = workload.step_hint)

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
  del current_params_types
  del loss_type
  del eval_results

  lawa = optimizer_state['lawa']
  current_model = current_param_container

  # Update LAWA (async transfer starts early)
  #  (last condition avoids to store the temp params in the ema and
  #  ensures true async update, at the price of 
  #  skipping one EMA update when tmp_params is not None)
  if (global_step+1) >= lawa.start_step and (global_step+1) % lawa.every_step == 0 and \
      lawa.tmp_params is None:  
    lawa.update_ema(current_model.parameters())

  # Discard average and load previous params
  if lawa.tmp_params is not None:
    for p, p_old in zip(current_model.parameters(), lawa.tmp_params):
      p.data.copy_(p_old.data)
    lawa.tmp_params = None

  current_model.train()
  optimizer_state['optimizer'].zero_grad()

  logits_batch, new_model_state = workload.model_fn(
      params=current_model,
      augmented_and_preprocessed_input_batch=batch,
      model_state=model_state,
      mode=spec.ForwardPassMode.TRAIN,
      rng=rng,
      update_batch_norm=True)

  label_smoothing = (
      hyperparameters.label_smoothing if hasattr(hyperparameters,
                                                 'label_smoothing') else 0.0)
  if hasattr(hyperparameters, 'grad_clip'):
    grad_clip = hyperparameters.grad_clip
  else:
    grad_clip = None

  loss_dict = workload.loss_fn(
      label_batch=batch['targets'],
      logits_batch=logits_batch,
      mask_batch=batch.get('weights'),
      label_smoothing=label_smoothing)
  summed_loss = loss_dict['summed']
  n_valid_examples = loss_dict['n_valid_examples']
  if USE_PYTORCH_DDP:
    # Use dist_nn.all_reduce to ensure correct loss and gradient scaling.
    summed_loss = dist_nn.all_reduce(summed_loss)
    n_valid_examples = dist_nn.all_reduce(n_valid_examples)
  loss = summed_loss / n_valid_examples

  loss.backward()

  if grad_clip is not None:
    torch.nn.utils.clip_grad_norm_(
        current_model.parameters(), max_norm=grad_clip)
  optimizer_state['optimizer'].step()
  optimizer_state['scheduler'].step()

  return (optimizer_state, current_param_container, new_model_state)


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
  
  lawa = optimizer_state['lawa']
  current_model = current_param_container
  
  if global_step < lawa.start_step or lawa.ema is None:
    return (optimizer_state, current_model, model_state)

  # Save parameters for next step (asynchronous non-blocking transfer)
  lawa.store_tmp_params(current_model.parameters())

  # Load ema into model
  for p, p_avg in zip(current_model.parameters(), lawa.ema):
      p.data.copy_(p_avg.data)  # move avg to GPU

  return (optimizer_state, current_model, model_state)


def get_batch_size(workload_name):
  # Return the global batch size.
  if workload_name == 'criteo1tb':
    return 262_144
  elif workload_name == 'fastmri':
    return 32
  elif workload_name == 'imagenet_resnet':
    return 1024
  elif workload_name == 'imagenet_resnet_silu':
    return 512
  elif workload_name == 'imagenet_resnet_gelu':
    return 512
  elif workload_name == 'imagenet_vit':
    return 1024
  elif workload_name == 'librispeech_conformer':
    return 256
  elif workload_name == 'librispeech_deepspeech':
    return 256
  elif workload_name == 'ogbg':
    return 512
  elif workload_name == 'wmt':
    return 128
  elif workload_name == 'mnist':
    return 16
  else:
    raise ValueError(f'Unsupported workload name: {workload_name}.')


def data_selection(workload: spec.Workload,
                   input_queue: Iterator[Dict[str, spec.Tensor]],
                   optimizer_state: spec.OptimizerState,
                   current_param_container: spec.ParameterContainer,
                   model_state: spec.ModelAuxiliaryState,
                   hyperparameters: spec.Hyperparameters,
                   global_step: int,
                   rng: spec.RandomState) -> Dict[str, spec.Tensor]:
  """Select data from the infinitely repeating, pre-shuffled input queue.
  Each element of the queue is a batch of training examples and labels.
  """
  del workload
  del optimizer_state
  del current_param_container
  del model_state
  del hyperparameters
  del global_step
  del rng
  batch = next(input_queue)
  return batch
