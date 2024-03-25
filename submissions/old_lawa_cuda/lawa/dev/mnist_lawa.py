"""Training algorithm track submission functions for MNIST."""

from typing import Dict, Iterator, List, Tuple
from collections import deque

import torch

import wandb

from algorithmic_efficiency import spec

from .lawa_utils import LAWAQueue, ListOfParams

def mynorm(params):
  return torch.norm(torch.stack([torch.norm(p.detach().clone(), 2) for p in params]), 2)
      
def get_batch_size(workload_name):
  # Return the global batch size.
  batch_sizes = {'mnist': 1024}
  return batch_sizes[workload_name]


def init_optimizer_state(workload: spec.Workload,
                         model_params: spec.ParameterContainer,
                         model_state: spec.ModelAuxiliaryState,
                         hyperparameters: spec.Hyperparameters,
                         rng: spec.RandomState) -> spec.OptimizerState:
  del model_state
  del workload
  del rng
  optimizer_state = {
      'optimizer':
          torch.optim.Adam(
              model_params.parameters(),
              lr=hyperparameters.learning_rate,
              betas=(1.0 - hyperparameters.one_minus_beta_1, 0.999),
              eps=hyperparameters.epsilon),
        'queue': LAWAQueue(maxlen=hyperparameters.k),
        'prev_model': ListOfParams(model_params.parameters()),
  }
  return optimizer_state


def update_params(workload: spec.Workload,
                  current_param_container: spec.ParameterContainer,
                  current_params_types: spec.ParameterTypeTree,
                  model_state: spec.ModelAuxiliaryState, # sempre None da workload.model_fn()
                  hyperparameters: spec.Hyperparameters,
                  batch: Dict[str, spec.Tensor],
                  loss_type: spec.LossType,
                  optimizer_state: spec.OptimizerState,
                  eval_results: List[Tuple[int, float]],
                  global_step: int,
                  rng: spec.RandomState) -> spec.UpdateReturn:
  """Return (updated_optimizer_state, updated_params)."""
  del loss_type
  del current_params_types
  del eval_results
  
  current_model = current_param_container
  prev_model = optimizer_state['prev_model']
  queue = optimizer_state['queue']
  lawa_start_step = hyperparameters.lawa_start_step
  lawa_interval = hyperparameters.lawa_interval
  
  ### log model norm before loading
  if wandb.run is not None:
    wandb.log({
        'w_step': global_step,
        'norm_1_at_start': mynorm(current_model.parameters())})
    
  # Discard average and load previous params
  # REMOVED that check on % lawa_interval == 0
  if global_step > lawa_start_step and queue.full():
    for p,p_old in zip(current_model.parameters(), prev_model.parameters()):
      p.data = p_old.clone()
  
  ### log model norm after loading
  if wandb.run is not None:
    wandb.log({
        'w_step': global_step,
        'norm_2_after_loading': mynorm(current_model.parameters())})
    
  current_model.train()
  for param in current_model.parameters():
    param.grad = None

  output, new_model_state = workload.model_fn(
      params=current_model,
      augmented_and_preprocessed_input_batch=batch,
      model_state=model_state,
      mode=spec.ForwardPassMode.TRAIN,
      rng=rng,
      update_batch_norm=True)

  loss_dict = workload.loss_fn(
      label_batch=batch['targets'], logits_batch=output)
  loss = loss_dict['summed'] / loss_dict['n_valid_examples']
  loss.backward()
  optimizer_state['optimizer'].step()
  
  # Save previous parameters
  # (this is redundant if c==1, just use queue[-1] in that case)
  # (or also if we return the avg only when a queue update occurs)
  if global_step >= lawa_start_step:
    prev_model.update(current_model.parameters())
  
  ### log model norm before averaging
  if wandb.run is not None:
    wandb.log({
        'w_step': global_step,
        'norm_3_PRE_AVG': mynorm(current_model.parameters())})

  # Update queue and avg
  if global_step >= lawa_start_step and \
      (global_step-lawa_start_step) % lawa_interval == 0:
    # Update queue
    queue.push(current_model.parameters())
    # Update avg
    if queue.full():
      queue.update_avg()
    ### Log
    if hyperparameters.wandb_log and wandb.run is not None:
      wandb.log({'my_step': global_step, 'is_avg_step': 1})
  
  # Load avg into model
  if queue.full():
    avg = queue.get_avg()
    for p, p_avg in zip(current_model.parameters(), avg):
      assert p.data.shape == p_avg.shape, "LAWA Shape mismatch"
      p.data = p_avg.clone()

  ### check logs before return
  if wandb.run is not None:
    if queue.full():
      wandb.log({
          'w_step': global_step,
          'norm_4_returned_model': mynorm(current_model.parameters()),
          'norm_5_avg': mynorm(queue.get_avg()),
          })
    else:
      wandb.log({
          'w_step': global_step,
          'norm_4_returned_model': mynorm(current_model.parameters())
          })
        
  return (optimizer_state, current_param_container, new_model_state)


# Not allowed to update the model parameters, hyperparameters, global step, or
# optimzier state.
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
  del optimizer_state
  del current_param_container
  del global_step
  del rng
  return next(input_queue)
