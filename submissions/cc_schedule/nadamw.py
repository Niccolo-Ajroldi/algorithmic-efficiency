"""Submission file for an NAdamW optimizer with warmup+cosine LR in PyTorch."""

import math
from typing import Dict, Iterator, List, Tuple

from absl import logging
import torch
from torch import Tensor
import torch.distributed.nn as dist_nn
from torch.optim.lr_scheduler import LambdaLR

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


class CarvedCosine(object):
  """
    Implements cosine schedule with linear cooldowns.
    colldown = linear decay + linear warmup.
  """

  def __init__(self, optimizer, lr_min, lr_max, T, warmup_steps, cc_decay_factor, cc_warmup_factor):
    self.optimizer = optimizer
    self.lr_min = lr_min
    self.lr_max = lr_max
    self.T = T
    self.t = 0

    # first linear warmup
    self.warmup_steps = warmup_steps

    # linear cooldown = linear decay + warmup
    self.d1 = T * cc_decay_factor
    self.d2 = T * cc_warmup_factor

    # amplitude of cosine schedule
    self.a_1 = 0.5 * (self.lr_max-self.lr_min)
    self.a_2 = self.a_1 * self.g(0.25 * T) / self.g(0.25 * T + self.d1 + self.d2)
    self.a_3 = self.a_2 * self.g(2.0 * 0.25 * T) / self.g(2.0 * 0.25 * T + self.d1 + self.d2)
    self.a_4 = self.a_3 * self.g(3.0 * 0.25 * T) / self.g(3.0 * 0.25 * T + self.d1 + self.d2)

  def a(self, phi):
    # amplitude of cosine schedule, phi is the cycle index
    if phi==1: 
      return self.a_1
    elif phi==2:
      return self.a_2
    elif phi==3:
      return self.a_3
    elif phi==4:
      return self.a_4

  def g(self, t):
    # wrapper for cosine
    return 1 + math.cos(t * math.pi / self.T)

  def f(self, t, phi):
    # cosine(t) at cycle index phi
    return self.lr_min + self.a(phi) * self.g(t)

  def decay(self, t, phi):
    # decay(t) at cycle index phi
    return self.lr_min + (self.lr_min - self.f(phi*0.25*self.T, phi))/self.d1 * (t - phi*0.25*self.T - self.d1)

  def warmup(self, t, phi):
    # warmup(t) at cycle index phi
    return self.lr_min + (-self.lr_min + self.f(phi*0.25*self.T, phi))/self.d2 * (t - phi*0.25*self.T - self.d1)

  def schedule(self, t):
    # cosine with linear cooldowns
    if t<= self.warmup_steps:
      return self.lr_min + self.lr_max/self.warmup_steps * t
    for phi in range(1,4):
      if t <= phi * 0.25 * self.T:
        return self.f(t, phi)
      elif t <= phi * 0.25 * self.T + self.d1:
        return self.decay(t, phi)
      elif t <= phi * 0.25 * self.T + + self.d1 + self.d2:
        return self.warmup(t, phi)  
    if t <= self.T:
      return self.f(t, 4)
    return self.lr_min

  def step(self):
    self.t += 1
    # set LR in optimizer
    lr = self.schedule(self.t)
    for group in self.optimizer.param_groups:
      group["lr"] = lr

  def state_dict(self):
    return {key: value for key, value in self.__dict__.items() if key != "optimizer"}

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

  optimizer_state = {
      'optimizer':
          NAdamW(
              model_params.parameters(),
              lr=hyperparameters.learning_rate,
              betas=(1.0 - hyperparameters.one_minus_beta1,
                     hyperparameters.beta2),
              eps=1e-8,
              weight_decay=hyperparameters.weight_decay),
  }

  # Create learning rate schedule.
  optimizer_state['scheduler'] = CarvedCosine(
    optimizer = optimizer_state['optimizer'],
    lr_max = hyperparameters.learning_rate, 
    lr_min = 1e-10,
    T = workload.step_hint,
    warmup_steps = int(hyperparameters.warmup_factor * workload.step_hint),
    cc_decay_factor = hyperparameters.cc_decay_factor,
    cc_warmup_factor = hyperparameters.cc_warmup_factor
  )

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
  del global_step

  current_model = current_param_container
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

  optimizer_state['optimizer'].step()
  optimizer_state['scheduler'].step()

  return (optimizer_state, current_param_container, new_model_state)


def get_batch_size(workload_name):
  # Return the global batch size.
  if workload_name == 'criteo1tb':
    return 262_144
  elif workload_name == 'fastmri':
    return 32
  elif workload_name == 'imagenet_resnet':
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
