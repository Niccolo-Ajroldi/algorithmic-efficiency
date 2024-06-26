"""This script can
1. Summarize the raw submission times for each workload run in a set of studies and trials.
2. Produce the performance profiles and scores of a group of submissions. 
Note that for performance profiles and final scores are computed w.r.t. a group of submissions.
If you only have logs for one submission you may group it with some reference submission
to compare the performance.

Example usage:
python3 score_submissions.py \
  --submission_directory $HOME/algorithmic-efficiency/prize_qualification_baselines/logs \
  --strict True
  --compute_performance_profiles
"""

import operator
import os

from absl import app
from absl import flags
from absl import logging
import numpy as np
import pandas as pd
import scoring_utils
from tabulate import tabulate

from scoring import performance_profile

flags.DEFINE_string(
    'submission_directory',
    None,
    'Path to submission directory containing experiment directories.')
flags.DEFINE_string('output_dir',
                    'scoring_results',
                    'Path to save performance profile table and plot.')
flags.DEFINE_boolean('compute_performance_profiles',
                     False,
                     'Whether or not to compute the performance profiles.')
flags.DEFINE_boolean(
    'strict',
    False,
    'Whether to enforce scoring criteria on variant performance and on'
    '5-trial median performance. Note that during official scoring this '
    'flag will be set to True.')
flags.DEFINE_boolean(
    'self_tuning_ruleset',
    False,
    'Whether to score on self-tuning ruleset or externally tuned ruleset')
FLAGS = flags.FLAGS


def get_summary_df(workload, workload_df, include_test_split=False):
  validation_metric, validation_target = scoring_utils.get_workload_metrics_and_targets(workload, split='validation')

  is_minimized = performance_profile.check_if_minimized(validation_metric)
  target_op = operator.le if is_minimized else operator.ge
  best_op = min if is_minimized else max
  idx_op = np.argmin if is_minimized else np.argmax

  summary_df = pd.DataFrame()
  summary_df['workload'] = workload_df['workload']
  summary_df['trial'] = workload_df['trial'].apply(lambda x: x[0])
  summary_df['val target metric name'] = validation_metric
  summary_df['val target metric value'] = validation_target

  summary_df['val target reached'] = workload_df[validation_metric].apply(
      lambda x: target_op(x, validation_target)).apply(np.any)
  summary_df['best metric value on val'] = workload_df[validation_metric].apply(
      lambda x: best_op(x))
  workload_df['index best eval on val'] = workload_df[validation_metric].apply(
      lambda x: idx_op(x))
  # (base)
  summary_df['time to best eval on val (s)'] = workload_df.apply(
      lambda x: x['accumulated_submission_time'][x['index best eval on val']],
      axis=1)
  summary_df['time to target on val (s)'] = summary_df.apply(
      lambda x: x['time to best eval on val (s)']
      if x['val target reached'] else np.inf,
      axis=1)
  if include_test_split:
    test_metric, test_target = scoring_utils.get_workload_metrics_and_targets(workload, split='test')
    summary_df['test target metric name'] = test_metric
    summary_df['test target metric value'] = test_target
    summary_df['test target reached'] = workload_df[test_metric].apply(
        lambda x: target_op(x, test_target)).apply(np.any)
    summary_df['best metric value on test'] = workload_df[test_metric].apply(
        lambda x: best_op(x))
    workload_df['index best eval on test'] = workload_df[test_metric].apply(
        lambda x: idx_op(x))
    summary_df['time to best eval on test (s)'] = workload_df.apply(
        lambda x: x['accumulated_submission_time'][x['index best eval on test']],
        axis=1)
    summary_df['time to target on test (s)'] = summary_df.apply(
        lambda x: x['time to best eval on test (s)']
        if x['test target reached'] else np.inf,
        axis=1)
  
  # # (nico):
  # summary_df['submission time (s)'] = workload_df.apply(
  #     lambda x: x['accumulated_submission_time'][x['index best eval']], axis=1)
  # summary_df['submission time to target (s)'] = summary_df.apply(
  #     lambda x: x['submission time (s)'] if x['target reached'] else np.inf,
  #     axis=1)
  # target_reached_step_indicator = workload_df[validation_metric].apply(
  #     lambda x: target_op(x, validation_target))
  # workload_df['index target reached'] = target_reached_step_indicator.apply(
  #      lambda x: np.argmax(x))
  # summary_df['submission time'] = workload_df.apply(
  #     lambda x: x['accumulated_submission_time'][-1], axis=1)
  # summary_df['submission_time_at_target'] = workload_df.apply(
  #     lambda x: x['accumulated_submission_time'][x['index target reached']], axis=1)
  # summary_df['score'] = summary_df.apply(
  #     lambda x: x['submission_time_at_target'] if x['target reached'] else np.inf, axis=1)
  # summary_df.drop('submission_time_at_target', axis=1, inplace=True)
  # summary_df['step_at_target'] = workload_df.apply(
  #     lambda x: x['global_step'][x['index target reached']], axis=1)
  # summary_df['step_at_target'] = summary_df.apply(
  #     lambda x: x['step_at_target'] if x['target reached'] else np.nan, axis=1)
  # summary_df['global_step'] = workload_df.apply(
  #     lambda x: x['global_step'][-1], axis=1)
  # summary_df.sort_values(['workload', 'trial'], inplace=True)
  # summary_df.reset_index(inplace=True, drop=True)
  
  return summary_df


def print_submission_summary(df, include_test_split=True):
  dfs = []
  for workload, group in df.groupby('workload'):
    summary_df = get_summary_df(
        workload, group, include_test_split=include_test_split)
    dfs.append(summary_df)

  df = pd.concat(dfs)
  logging.info('\n' + tabulate(df, headers='keys', tablefmt='psql'))

# # (nico): save scores to csv: DEPRECATED UNDER NEW CODE
# def save_submission_summary(df, submission, output_dir):
#   dfs = []
#   for workload, group in df.groupby('workload'):
#     summary_df = get_summary_df(workload, group)
#     dfs.append(summary_df)
#   df = pd.concat(dfs)

#   if not os.path.exists(output_dir):
#     os.mkdir(output_dir)
#   csv_file_path = os.path.join(output_dir, 'scores_{}.csv'.format(submission))
#   df.to_csv(csv_file_path, index=False)

def main(_):
  results = {}

  for submission in os.listdir(FLAGS.submission_directory):
    experiment_path = os.path.join(FLAGS.submission_directory, submission)
    df = scoring_utils.get_experiment_df(experiment_path)
    results[submission] = df
    summary_df = print_submission_summary(df)
    with open(os.path.join(FLAGS.output_dir, f'{submission}_summary.csv'),
              'w') as fout:
      summary_df.to_csv(fout)
    # # (nico) DEPRECATED UNDER NEW CODE
    # save_submission_summary(df, submission, FLAGS.output_dir) 

  if not FLAGS.strict:
    logging.warning(
        'You are running with strict=False. This will relax '
        'scoring criteria on the held-out workloads, number of trials and number '
        'of studies. Your score may not be an accurate representation '
        'under competition scoring rules. To enforce the criteria set strict=True.'
    )
  if FLAGS.compute_performance_profiles:
    performance_profile_df = performance_profile.compute_performance_profiles(
        results,
        time_col='score',
        min_tau=1.0,
        max_tau=None,
        reference_submission_tag=None,
        num_points=100,
        scale='linear',
        verbosity=0,
        self_tuning_ruleset=FLAGS.self_tuning_ruleset,
        strict=FLAGS.strict)
    if not os.path.exists(FLAGS.output_dir):
      os.mkdir(FLAGS.output_dir)
    performance_profile.plot_performance_profiles(
        performance_profile_df, 'score', save_dir=FLAGS.output_dir)
    perf_df = tabulate(
        performance_profile_df.T, headers='keys', tablefmt='psql')
    logging.info(f'Performance profile:\n {perf_df}')


if __name__ == '__main__':
  # flags.mark_flag_as_required('submission_directory')
  app.run(main)
