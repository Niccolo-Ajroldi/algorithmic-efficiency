r"""
Eval from a checkpoint folder, using LAWA, EMA or Polyak.


Example command:

# pylint: disable=line-too-long
python3 submission_runner.py \
    --workload=mnist \
    --framework=jax \
    --submission_path=reference_algorithms/development_algorithms/mnist/mnist_jax/submission.py \
    --tuning_ruleset=external \
    --tuning_search_space=reference_algorithms/development_algorithms/mnist/tuning_search_space.json \
    --num_tuning_trials=3 \
    --experiment_dir=/home/znado/experiment_dir \
    --experiment_name=baseline
"""


import datetime
import gc
import importlib
from inspect import signature
import itertools
import json
import os
import struct
import time
from types import MappingProxyType
from typing import Any, Dict, Optional, Tuple
import wandb

from absl import app
from absl import flags
from absl import logging
import jax
import torch
import torch.distributed as dist

from flax.training.checkpoints import latest_checkpoint

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Disables tensorRT, cuda warnings.
import tensorflow as tf

# Hide any GPUs form TensorFlow. Otherwise TF might reserve memory and make
# it unavailable to JAX.
tf.config.set_visible_devices([], 'GPU')

from algorithmic_efficiency import checkpoint_utils
from algorithmic_efficiency import halton
from algorithmic_efficiency import logger_utils
from algorithmic_efficiency import random_utils as prng
from algorithmic_efficiency import spec
from algorithmic_efficiency.profiler import PassThroughProfiler
from algorithmic_efficiency.profiler import Profiler
from algorithmic_efficiency.pytorch_utils import pytorch_init
from algorithmic_efficiency.pytorch_utils import pytorch_setup
from algorithmic_efficiency.pytorch_utils import sync_ddp_time
from algorithmic_efficiency.workloads import workloads
from algorithmic_efficiency import fixed_space

# disable only for deepspeech if it works fine for other workloads.
os.environ['XLA_FLAGS'] = '--xla_gpu_enable_triton_gemm=false'


# TODO(znado): make a nicer registry of workloads that lookup in.
BASE_WORKLOADS_DIR = workloads.BASE_WORKLOADS_DIR

# Workload_path will be appended by '_pytorch' or '_jax' automatically.
WORKLOADS = workloads.WORKLOADS

flags.DEFINE_string(
    'submission_path',
    None,
    'The relative path of the Python file containing submission functions. '
    'NOTE: the submission dir must have an __init__.py file!')
flags.DEFINE_string(
    'workload',
    None,
    help=f'The name of the workload to run.\n Choices: {list(WORKLOADS.keys())}'
)


flags.DEFINE_enum(
    'tuning_ruleset',
    'external',
    enum_values=['external', 'self'],
    help='Which tuning ruleset to use.')
flags.DEFINE_string(
    'tuning_search_space',
    None,
    'The path to the JSON file describing the external tuning search space.')
flags.DEFINE_integer('num_tuning_trials',
                     1,
                     'The number of external hyperparameter trials to run.')
flags.DEFINE_integer(
    'trial_index',
    None,
    'Only run trial trial_index/num_tuning_trials, '
    'should range from 1 to num_tuning_trials')  # (nico)
flags.DEFINE_boolean(
    'fixed_space',
    False,
    'Fixed space: no sampling from grid.')  # (nico)
flags.DEFINE_string('data_dir', '~/data', 'Dataset location.')
flags.DEFINE_string('imagenet_v2_data_dir',
                    None,
                    'Dataset location for ImageNet-v2.')
flags.DEFINE_string('librispeech_tokenizer_vocab_path',
                    '',
                    'Location to librispeech tokenizer.')

flags.DEFINE_enum(
    'framework',
    None,
    enum_values=['jax', 'pytorch'],
    help='Whether to use Jax or Pytorch for the submission. Controls among '
    'other things if the Jax or Numpy RNG library is used for RNG.')
flags.DEFINE_boolean(
    'torch_compile',
    True,
    'Whether to use `torch.compile` to JIT-compile PyTorch code. '
    'This will only take effect when `framework`==pytorch.')

flags.DEFINE_string(
    'experiment_dir',
    None,
    'The root directory to store all experiments. '
    'It is required and the directory should have '
    'an absolute path rather than a relative path.')
flags.DEFINE_string('experiment_name', None, 'Name of the experiment.')

flags.DEFINE_boolean(
    'save_checkpoints',
    True,
    'Whether or not to save checkpoints of the model and optimizer '
    'at every eval and after training.')
flags.DEFINE_boolean(
    'save_intermediate_checkpoints',
    True,
    'Whether to save any intermediate checkpoints. '
    'If False, it will only keep the latest checkpoint.')
flags.DEFINE_boolean(
    'resume_last_run',
    None,
    'Whether to resume the experiment from its last run.'
    'If resume_experiment_name is not specified, it resumes from '
    'experiment_dir/experiment_name/workload/trial.'
    'If resume_experiment_name is specified, it resumes from '
    'experiment_dir/resume_experiment_name.')
flags.DEFINE_string(
    'resume_experiment_name',
    None,
    'The name of the experiment from which resuming. '
    'It should be smth like resume_exp_name/workload/trial. '
    'See --resume_last_run for how to use it.')
flags.DEFINE_boolean(
    'append_timestamp',
    False,
    'If True, the current datetime will be appended to the experiment name. '
    'Useful for guaranteeing a unique experiment dir for new runs.')
flags.DEFINE_boolean('use_wandb',
                     False,
                     'Whether to use Weights & Biases logging.')
flags.DEFINE_boolean('profile', False, 'Whether to produce profiling output.')
flags.DEFINE_integer('max_global_steps',
                     None,
                     'Maximum number of update steps.')
flags.DEFINE_boolean(
    'overwrite',
    False,
    'Whether to overwrite the experiment with identical experiment_dir and'
    'experiment_name.')
flags.DEFINE_integer(
    'hparam_start_index',
    None,
    'Start index to slice set of hyperparameters in tuning search space.')
flags.DEFINE_integer(
    'hparam_end_index',
    None,
    'End index to slice set of hyperparameters in tuning search space.')
flags.DEFINE_integer(
    'rng_seed',
    None,
    'Value of rng seed. If None, a random seed will'
    'be generated from hardware.')
flags.DEFINE_boolean('set_pytorch_max_split_size',
                     False,
                     'If true, set pytorch max_split_size_mb to 256')
flags.DEFINE_integer(
    'pytorch_eval_num_workers',
    0,
    'Number of workers for ImageNet PyTorch evaluation data loaders.'
    'WARNING: Setting pytorch_eval_num_workers != 0, will result '
    'in incorrect evals currently, see issues/732.')
flags.DEFINE_float('max_pct_of_global_steps',
                   None,
                   'Maximum number of update steps.')
flags.DEFINE_boolean(
    'halve_CUDA_mem',
    False,
    'Halve the available VRAM.')  # (nico)
flags.DEFINE_boolean(
    'allow_tf32',
    False,
    'Allow TF32 on Ampere.')  # (nico)
flags.DEFINE_integer(
    'custom_eval_period_time_sec',
    None,
    '')  # (nico)
flags.DEFINE_integer(
    'cluster_id',
    None,
    'Cluster JOB ID.')  # (nico)
flags.DEFINE_integer(
    'process_id',
    None,
    'Process JOB ID.')  # (nico)
flags.DEFINE_boolean(
    'run_until_the_end',
    False,
    'Run the workload until global_step==step_hint.')  # (nico)
flags.DEFINE_integer(
    'eval_every_n_steps',
    None,
    'Eval every n steps, replaces timed eval.')  # (nico)
flags.DEFINE_boolean(
    'deterministic',
    False,
    'Deterministic mode for PyTorch.')  # (nico)
flags.DEFINE_boolean(
    'log_lr',
    True,
    'Log Learning Rate to wandb.')  # (nico)
flags.DEFINE_integer(
    'save_ckpt_freq',
    None,
    'Save checkpoint every n steps.')  # (nico)

flags.DEFINE_string(
    'baseline_ckpt_dir',
    None,
    'baseline_ckpt_dir')


FLAGS = flags.FLAGS
USE_PYTORCH_DDP, RANK, DEVICE, N_GPUS = pytorch_setup()

def _get_time():
  if torch.cuda.is_available():
    torch.cuda.synchronize()
  return time.time()


def _get_time_ddp():
  torch.cuda.synchronize()
  t = time.time()
  return sync_ddp_time(t, DEVICE)


if USE_PYTORCH_DDP:
  get_time = _get_time_ddp
else:
  get_time = _get_time

def _reset_cuda_mem():
  if FLAGS.framework == 'pytorch' and torch.cuda.is_available():
    torch._C._cuda_clearCublasWorkspaces()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()



def train_once(
    workload: spec.Workload,
    workload_name: str,
    global_batch_size: int,
    global_eval_batch_size: int,
    data_dir: str,
    imagenet_v2_data_dir: str,
    init_optimizer_state: spec.InitOptimizerFn,
    update_params: spec.UpdateParamsFn,
    data_selection: spec.DataSelectionFn,
    prepare_for_eval: Optional[spec.PrepareForEvalFn],
    hyperparameters: Optional[spec.Hyperparameters],
    rng_seed: int,
    rng: spec.RandomState,
    profiler: Profiler,
    max_global_steps: int = None,
    max_pct_of_global_steps: float = None,  # (nico)
    log_dir: Optional[str] = None,
    baseline_ckpt_dir: Optional[str] = None,  # (nico)
    save_checkpoints: Optional[bool] = True
) -> Tuple[spec.Timing, Dict[str, Any]]:
  _reset_cuda_mem()
  _, opt_init_rng, model_init_rng, rng = prng.split(rng, 4)

  # Workload setup.
  if hasattr(workload, '_eval_num_workers'):
    workload.eval_num_workers = FLAGS.pytorch_eval_num_workers
  logging.info('Initializing model.')
  with profiler.profile('Initializing model'):
    dropout_rate = None
    aux_dropout_rate = None
    if hasattr(hyperparameters, 'dropout_rate'):
      dropout_rate = hyperparameters.dropout_rate
    if hasattr(hyperparameters, 'aux_dropout_rate'):
      aux_dropout_rate = hyperparameters.aux_dropout_rate
    model_params, model_state = workload.init_model_fn(
        model_init_rng, dropout_rate, aux_dropout_rate)
    if FLAGS.framework == 'pytorch' and FLAGS.torch_compile:
      compile_error_workloads = [
          'librispeech_conformer',
          'ogbg',
          'criteo1tb',
          'imagenet_vit',
      ]
      eager_backend_workloads = ['librispeech_deepspeech']
      aot_eager_backend_workloads = []
      loss_compilation_workloads = [
          'fastmri', 'librispeech_deepspeech', 'ogbg', 'wmt'
      ]
      base_workload = workloads.get_base_workload_name(workload_name)
      if base_workload in compile_error_workloads:
        logging.warning(
            'These workloads cannot be fully compiled under current '
            'PyTorch version. Proceeding without `torch.compile`.')
      elif base_workload in eager_backend_workloads:
        logging.warning(
            'These workloads cannot be fully compiled under current '
            'PyTorch version. Proceeding with `backend=eager`.')
        model_params = torch.compile(model_params, backend='eager')
      elif base_workload in aot_eager_backend_workloads:
        logging.warning(
            'These workloads cannot be fully compiled under current '
            'PyTorch version. Proceeding with `backend=aot_eager`.')
        model_params = torch.compile(model_params, backend='aot_eager')
      else:
        logging.info('Performing `torch.compile`.')
        model_params = torch.compile(model_params)
      if base_workload in loss_compilation_workloads:
        workload.loss_fn = torch.compile(workload.loss_fn)
  logging.info('Initializing optimizer.')
  with profiler.profile('Initializing optimizer'):
    optimizer_state = init_optimizer_state(workload,
                                           model_params,
                                           model_state,
                                           hyperparameters,
                                           opt_init_rng)
  logging.info('Initializing metrics bundle.')

  # Check if 'train_state' is in the function signature
  needs_train_state = 'train_state' in signature(update_params).parameters

  # Bookkeeping.
  train_state = {
      'validation_goal_reached': False,
      'test_goal_reached': False,
      'is_time_remaining': True,
      'last_eval_time': 0,
      'training_complete': False,
      'accumulated_submission_time': 0,
      'accumulated_eval_time': 0,
      'accumulated_logging_time': 0,
      'last_step_end_time': None,
  }
  global_step = 0
  eval_results = []
  preemption_count = 0

  # Loggers and checkpoint setup.
  logging.info('Initializing checkpoint and logger.')
  if log_dir is not None:
    # (nico): skipping checkpoint restoring
    meta_file_name = os.path.join(log_dir, f'meta_data_{preemption_count}.json')
    logging.info(f'Saving meta data to {meta_file_name}.')
    meta_data = logger_utils.get_meta_data(workload, rng_seed)
    logger_utils.write_json(meta_file_name, meta_data)
    flag_file_name = os.path.join(log_dir, f'flags_{preemption_count}.json')
    logging.info(f'Saving flags to {flag_file_name}.')
    logger_utils.write_json(flag_file_name, flags.FLAGS.flag_values_dict())
    metrics_logger = None
    if RANK == 0:
      metrics_logger = logger_utils.set_up_loggers(log_dir,
                                                   flags.FLAGS,
                                                   hyperparameters)
      workload.attach_metrics_logger(metrics_logger)

  # What's the last ckpt?)
  last_ckpt_path = latest_checkpoint(baseline_ckpt_dir)
  last_ckpt_step = int(os.path.basename(last_ckpt_path).split('_')[-1])

  logging.info('Starting training loop.')
  goals_reached = train_state['validation_goal_reached']

  while global_step <= workload.step_hint and \
      not train_state['training_complete'] and \
      (FLAGS.run_until_the_end or not goals_reached):

    step_rng = prng.fold_in(rng, global_step)
    _, update_rng, prep_eval_rng, eval_rng = prng.split(step_rng, 4)

    # Assume that an update step has been done, load the ckpt into the model
    # so that model_params contains the latest params

    # append step to checkpoint base path
    ckpt_path_global_step = os.path.join(baseline_ckpt_dir, f"checkpoint_{global_step}")
    
    if os.path.isfile(ckpt_path_global_step):

      # Load only model params from checkpoint
      checkpoint_state = torch.load(ckpt_path_global_step, map_location=DEVICE)
      if isinstance(
        model_params,
        (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        model_params = model_params.module
      model_params.load_state_dict(checkpoint_state['model_params'])
      checkpoint_state['model_params'] = model_params
      # logging.info(f'Loaded checkpoint from {ckpt_path_global_step}.')

      try:
        with profiler.profile('Update parameters'):
          optimizer_state, model_params, model_state = update_params(
              workload=workload,
              current_param_container=model_params,
              current_params_types=workload.model_params_types,
              model_state=model_state,
              hyperparameters=hyperparameters,
              batch=None,
              loss_type=workload.loss_type,
              optimizer_state=optimizer_state,
              eval_results=eval_results,
              global_step=global_step,
              rng=update_rng,
              **({'train_state': MappingProxyType(train_state)}
                if needs_train_state else {}))
      except spec.TrainingCompleteError:
        train_state['training_complete'] = True

    global_step += 1
    # (nico): train for a fixed pct of step_hint
    if (max_pct_of_global_steps is not None) and \
        (max_pct_of_global_steps < 1.0) and \
        (global_step / workload.step_hint >= max_pct_of_global_steps):
      train_state['training_complete'] = True
    # (nico): eval again if we reached the last saved checkpoint
    if global_step == last_ckpt_step:
      train_state['training_complete'] = True

    # Check if submission is eligible for an untimed eval.  (nico): eval every n steps
    if (global_step % FLAGS.eval_every_n_steps == 0 or train_state['training_complete']):

      # Prepare for eval
      if prepare_for_eval is not None:

        with profiler.profile('Prepare for eval'):
          optimizer_state, model_params, model_state = prepare_for_eval(
              workload=workload,
              current_param_container=model_params,
              current_params_types=workload.model_params_types,
              model_state=model_state,
              hyperparameters=hyperparameters,
              loss_type=workload.loss_type,
              optimizer_state=optimizer_state,
              eval_results=eval_results,
              global_step=global_step,
              rng=prep_eval_rng)

      # Eval
      with profiler.profile('Evaluation'):
        _reset_cuda_mem()

        try:
          latest_eval_result = workload.eval_model(global_eval_batch_size,
                                                    model_params,
                                                    model_state,
                                                    eval_rng,
                                                    data_dir,
                                                    imagenet_v2_data_dir,
                                                    global_step)
          # Check if targets reached.
          train_state['validation_goal_reached'] = (
              workload.has_reached_validation_target(latest_eval_result) or
              train_state['validation_goal_reached'])
          train_state['test_goal_reached'] = False  # (nico): skip test eval
          goals_reached = train_state['validation_goal_reached']

          logging.info(f'\tStep: {global_step}, \t{latest_eval_result}')
          eval_results.append((global_step, latest_eval_result))

          if log_dir is not None and RANK == 0:
            metrics_logger.append_scalar_metrics(
                latest_eval_result,
                global_step=global_step,
                preemption_count=preemption_count,
                is_eval=True,
            )

          _reset_cuda_mem()

        except RuntimeError as e:
          logging.exception(f'Eval step {global_step} error.\n')
          if 'out of memory' in str(e):
            logging.warning(
                'Error: GPU out of memory during eval during step '
                f'{global_step}, error : {str(e)}.')
            _reset_cuda_mem()

  metrics = {'eval_results': eval_results, 'global_step': global_step}

  if log_dir is not None and RANK == 0:
    metrics_logger.append_scalar_metrics(
        {'score': train_state['accumulated_submission_time']},
        global_step=global_step,
        preemption_count=preemption_count)
    metrics_logger.finish()

  return train_state['accumulated_submission_time'], metrics


def score_submission_on_workload(workload: spec.Workload,
                                 workload_name: str,
                                 submission_path: str,
                                 data_dir: str,
                                 tuning_ruleset: str,
                                 profiler: Optional[Profiler] = None,
                                 max_global_steps: Optional[int] = None,
                                 max_pct_of_global_steps: Optional[float] = None,  # (nico)
                                 imagenet_v2_data_dir: Optional[str] = None,
                                 tuning_search_space: Optional[str] = None,
                                 num_tuning_trials: Optional[int] = None,
                                 trial_index: Optional[int] = None,
                                 log_dir: Optional[str] = None,
                                 baseline_ckpt_dir: Optional[str] = None,  # (nico)
                                 save_checkpoints: Optional[bool] = True,
                                 hparam_start_index: Optional[bool] = None,
                                 hparam_end_index: Optional[bool] = None,
                                 rng_seed: Optional[int] = None):
  # Expand paths because '~' may not be recognized
  data_dir = os.path.expanduser(data_dir)
  if imagenet_v2_data_dir:
    imagenet_v2_data_dir = os.path.expanduser(imagenet_v2_data_dir)

  # Remove the trailing '.py' and convert the filepath to a Python module.
  submission_module_path = workloads.convert_filepath_to_module(submission_path)
  submission_module = importlib.import_module(submission_module_path)

  init_optimizer_state = submission_module.init_optimizer_state
  update_params = submission_module.update_params
  prepare_for_eval = getattr(submission_module, 'prepare_for_eval', None)

  global_batch_size = None
  global_eval_batch_size = workload.eval_batch_size

  if tuning_ruleset == 'external':
    # If the submission runner is responsible for hyperparameter tuning, load in
    # the search space and generate a list of randomly selected hyperparameter
    # settings from it.
    if tuning_search_space is None:
      raise ValueError(
          'Must provide a tuning search space JSON file when using external '
          'tuning.')

    # (nico) halton.generate_search always produce the same list, but order may vary
    with open(tuning_search_space, 'r', encoding='UTF-8') as search_space_file:
      if not FLAGS.fixed_space:
        # (nico) original code
        tuning_search_space = halton.generate_search(
            json.load(search_space_file), num_tuning_trials)
      else:
        # (nico) my code for generating trials TODO: check before submission
        tuning_search_space = fixed_space.generate_search(
            json.load(search_space_file), num_tuning_trials)

    tuning_search_space.sort()

    all_timings = []  # (nico)
    all_metrics = []  # (nico)
    tuning_search_space_iter = itertools.islice(
        enumerate(tuning_search_space), hparam_start_index, hparam_end_index)
    for hi, hyperparameters in tuning_search_space_iter:

      if trial_index is not None and (hi + 1) != trial_index:
        continue

      # Generate a new seed from hardware sources of randomness for each trial.
      if not rng_seed:
        rng_seed = struct.unpack('I', os.urandom(4))[0]
      logging.info('Using RNG seed %d', rng_seed)
      rng = prng.PRNGKey(rng_seed)
      # Because we initialize the PRNGKey with only a single 32 bit int, in the
      # Jax implementation this means that rng[0] is all zeros, which means this
      # could lead to unintentionally reusing the same seed of only rng[0] were
      # ever used. By splitting the rng into 2, we mix the lower and upper 32
      # bit ints, ensuring we can safely use either rng[0] or rng[1] as a random
      # number.
      rng, _ = prng.split(rng, 2)
      logging.info(f'--- Tuning run {hi + 1}/{num_tuning_trials} ---')

      tuning_dir_name = None
      if log_dir is not None:
        tuning_dir_name = os.path.join(log_dir, f'trial_{hi + 1}')
        logging.info(f'Creating tuning directory at {tuning_dir_name}.')
        logger_utils.makedir(tuning_dir_name)

        # If existing hyperparameter exists, use saved
        # hyperparameters for consistency.
        hyperparameters = logger_utils.write_hparams(hyperparameters,
                                                     tuning_dir_name)
        tuning_search_space[hi] = hyperparameters

      with profiler.profile('Train'):
        timing, metrics = train_once(workload, workload_name,
                                     global_batch_size,
                                     global_eval_batch_size,
                                     data_dir, imagenet_v2_data_dir,
                                     init_optimizer_state,
                                     update_params, None,
                                     prepare_for_eval,
                                     hyperparameters,
                                     rng_seed,
                                     rng,
                                     profiler,
                                     max_global_steps,
                                     max_pct_of_global_steps,  # (nico)
                                     tuning_dir_name,
                                     baseline_ckpt_dir,  # (nico)
                                     save_checkpoints=save_checkpoints,)
      # (nico): modified
      all_timings.append(timing)
      all_metrics.append(metrics)
      
      logging.info(f'Tuning trial {hi + 1}/{num_tuning_trials}')
      logging.info(f'Hyperparameters: {tuning_search_space[hi]}')
      logging.info(f'Metrics: {metrics}')
      logging.info(f'Timing: {timing}')
      num_evals = len(metrics['eval_results'])
      logging.info(f'Total number of evals: {num_evals}')
      logging.info('=' * 20)
    score = min(all_timings)
  else:
    raise ValueError('self tuning not supported')
  return score


def main(_):

  if FLAGS.framework == 'pytorch':
    if FLAGS.halve_CUDA_mem:
      torch.cuda.set_per_process_memory_fraction(0.5, device=DEVICE)
    if FLAGS.allow_tf32:
      torch.backends.cuda.matmul.allow_tf32 = True
      torch.backends.cudnn.allow_tf32 = True
    else:
      torch.backends.cuda.matmul.allow_tf32 = False
      torch.backends.cudnn.allow_tf32 = False
    if FLAGS.deterministic:
      torch.use_deterministic_algorithms(True)
      os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
      torch.backends.cudnn.benchmark = False

  if FLAGS.profile:
    profiler = Profiler()
  else:
    profiler = PassThroughProfiler()

  if FLAGS.framework == 'pytorch':
    pytorch_init(USE_PYTORCH_DDP, RANK, profiler)

  # TODO: remove once issue resolved.
  if FLAGS.pytorch_eval_num_workers != 0:
    logging.warning(
        'WARNING: Setting pytorch_eval_num_workers != 0, will result '
        'in incorrect evals currently, see issues/732.')

  workload_metadata = WORKLOADS[FLAGS.workload]

  # Prevent OOM on librispeech conformer.
  base_workload = workloads.get_base_workload_name(FLAGS.workload)
  if base_workload == 'librispeech_conformer':
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.85'

  if FLAGS.set_pytorch_max_split_size:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'

  # Extend path according to framework.
  workload_metadata['workload_path'] = os.path.join(
      BASE_WORKLOADS_DIR,
      workload_metadata['workload_path'] + f'_{FLAGS.framework}',
      'workload.py')
  workload_init_kwargs = {}
  if FLAGS.librispeech_tokenizer_vocab_path:
    workload_init_kwargs['tokenizer_vocab_path'] = (
        FLAGS.librispeech_tokenizer_vocab_path)
  workload = workloads.import_workload(
      workload_path=workload_metadata['workload_path'],
      workload_class_name=workload_metadata['workload_class_name'],
      workload_init_kwargs=workload_init_kwargs)

  experiment_name = FLAGS.experiment_name
  if experiment_name and FLAGS.append_timestamp:
    experiment_name += datetime.datetime.now().strftime('-%Y-%m-%d-%H-%M-%S')
  logging_dir_path, _ = logger_utils.get_log_dir(
    FLAGS.experiment_dir,
    FLAGS.workload,
    FLAGS.framework,
    experiment_name,
    FLAGS.resume_last_run,
    resume_experiment_name=None,  # (nico)
    overwrite=FLAGS.overwrite)

  score = score_submission_on_workload(
      workload=workload,
      workload_name=FLAGS.workload,
      submission_path=FLAGS.submission_path,
      data_dir=FLAGS.data_dir,
      tuning_ruleset=FLAGS.tuning_ruleset,
      profiler=profiler,
      max_global_steps=FLAGS.max_global_steps,
      max_pct_of_global_steps=FLAGS.max_pct_of_global_steps,  # (nico)
      imagenet_v2_data_dir=FLAGS.imagenet_v2_data_dir,
      tuning_search_space=FLAGS.tuning_search_space,
      num_tuning_trials=FLAGS.num_tuning_trials,
      trial_index=FLAGS.trial_index,
      log_dir=logging_dir_path,
      baseline_ckpt_dir=FLAGS.baseline_ckpt_dir,  # (nico)
      save_checkpoints=FLAGS.save_checkpoints,
      hparam_start_index=FLAGS.hparam_start_index,
      hparam_end_index=FLAGS.hparam_end_index,
      rng_seed=FLAGS.rng_seed)
  logging.info(f'Final {FLAGS.workload} score: {score}')

  if FLAGS.profile:
    logging.info(profiler.summary())

  if USE_PYTORCH_DDP:
    # Cleanup.
    dist.destroy_process_group()


if __name__ == '__main__':
  flags.mark_flag_as_required('workload')
  flags.mark_flag_as_required('framework')
  flags.mark_flag_as_required('submission_path')
  flags.mark_flag_as_required('experiment_dir')
  flags.mark_flag_as_required('experiment_name')
  app.run(main)
