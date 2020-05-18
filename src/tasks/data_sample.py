"""Prepare subset of the data for modelling and evaluation.
"""
import os
import random
import sys
import time
from dotenv import find_dotenv, load_dotenv
from pathlib import Path

import click
import numpy as np
from eolearn.core import EOExecutor, FeatureType, LinearWorkflow, LoadTask, \
    OverwritePermission, SaveTask
from eolearn.geometry import PointSamplingTask

from src.utils import config, logging, misc
from src.eolearn.tasks import TrainTestSplit


# Global variables
log = None
report = {}


def init_workflow(cfg, input_dir, output_dir):
    """Initialize tasks and build linear workflow.

    :param cfg: Configuration
    :type cfg: dict
    :param input_dir: Input directory, where raw data patches are stored.
    :type input_dir: Path
    :param output_dir: Output directory, where sampled data patches will be
        stored.
    :type output_dir: Path
    :return: Workflow
    :rtype: LinearWorkflow
    """

    # EOTask: Load data
    # =================

    load = LoadTask(
        str(input_dir),
        lazy_loading=True
    )

    # EOTask: Sample data
    # ===================
    classes = cfg['reference_data']['classes']
    sample_data = PointSamplingTask(
        n_samples=cfg['sampling']['n'],
        ref_mask_feature='REF_MORPHED',
        ref_labels=[int(k) for k, v in classes.items() if len(v)],
        sample_features=[
            (FeatureType.DATA, 'FEATURES'),
            (FeatureType.MASK_TIMELESS, 'REF_MORPHED')
        ],
        return_new_eopatch=True
    )

    # EOTask: Split patches
    # =====================
    # Assign each patch subset id, i.e. 1 (train set) or 0 (test set)

    # split = cfg['sampling']['split']
    # args = {
    #     'feature_name': 'subset'
    # }

    # if isinstance(split, float):
    #     args['split'] = split

    # train_test_split = TrainTestSplit(**args)
    train_test_split = TrainTestSplit(feature_name='subset')

    # EOTask: Save data
    # =================

    save = SaveTask(
        str(output_dir),
        overwrite_permission=OverwritePermission.OVERWRITE_PATCH
    )

    # EOWorkflow
    # ==========

    workflow = LinearWorkflow(
        load,
        sample_data,
        train_test_split,
        save
    )

    return workflow


def init_execution_args(cfg, tasks, input_dir, output_dir):
    """Define list of argument dictionaries for tasks to be fed to
    executioner together with workflow.

    :param cfg: Configuration
    :type cfg: dict
    :param tasks: Dictionary of tasks
    :type tasks: dict[str, EOTask]
    :param input_dir: Input data directory
    :type input_dir: Path
    :param output_dir: Output data directory
    :type output_dir: Path
    :return: List of argument distionaries
    :rtype: dict[str, object]
    """
    execution_args = []
    eopatch_dirs = list(input_dir.iterdir())
    split = cfg['sampling']['split']

    if isinstance(split, float):
        total_len = len(eopatch_dirs)
        train_len = round(total_len * split)
        train_indices = random.sample(list(range(total_len)), k=train_len)
        split = np.zeros((total_len,), dtype=np.uint8)
        split[train_indices] = 1
        split = list(split)

    for i, path in enumerate(eopatch_dirs):
        eopatch_folder = path.name
        patch_meta = output_dir / eopatch_folder / 'meta_info.pkl'

        # Skip existing patches.
        if patch_meta.is_file():
            log.info(f'Patch {eopatch_folder} already exists.')
            continue

        execution_args.append({
            tasks['LoadTask']: {
                'eopatch_folder': eopatch_folder
            },
            tasks['TrainTestSplit']: {
                'subset': split[i]
            },
            tasks['SaveTask']: {
                'eopatch_folder': eopatch_folder
            }
        })

    return execution_args


def sample_data(cfg, log_dir):
    """Sample processed data for given configuration.

    :param cfg: Configuration
    :type cfg: dict
    :param log_dir: Logging directory
    :type log_dir: Path
    """
    global report

    # Get input directory.
    processed_data_dir = misc.get_processed_data_dir(cfg)
    input_dir = processed_data_dir / 'patches'

    # Get output directory.
    sampled_data_dir = misc.get_sampled_data_dir(cfg)
    output_dir = sampled_data_dir / 'patches'

    if not output_dir.is_dir():
        output_dir.mkdir(parents=True)

    # Initialize workflow.
    workflow = init_workflow(
        cfg,
        input_dir,
        output_dir
    )

    # Initialize workflow execution arguments.
    execution_args = init_execution_args(
        cfg,
        workflow.get_tasks(),
        input_dir,
        output_dir
    )

    total_executions = len(execution_args)

    if total_executions:
        # Initialize executor.
        executor = EOExecutor(
            workflow,
            execution_args,
            save_logs=True,
            logs_folder=str(log_dir)
        )

        ex_workers = int(os.getenv('EX_WORKERS'))
        ex_multiprocess = os.getenv('EX_MULTIPROCESS').lower() == 'true'

        log.info(
            f'Attempting to sample {total_executions} patches from: '
            f'{input_dir}'
        )

        # Execute workflow.
        ex_time = time.time()
        executor.run(
            workers=ex_workers,
            multiprocess=ex_multiprocess
        )
        ex_time = time.time() - ex_time

        successful_executions = len(executor.get_successful_executions())
        log.info(
            f'Sampled {successful_executions} / {total_executions} '
            f'patches in {ex_time:.2f} seconds'
        )
        log.info(f'Results saved: {output_dir}')

        # Report.
        executor.make_report()
    else:
        ex_time = 0.0
        log.info('All patches already sampled.')

    report['sampled_patches'] = {
        'quantity': total_executions,
        'time': ex_time,
    }


@click.command()
@click.argument('config_name', type=click.STRING)
@click.option(
    '--timestamp',
    type=click.FLOAT,
    default=time.time(),
    help='Unix timestamp'
)
def main(config_name, timestamp):
    """Sample data for given configuration.

    :param config_name: Configuration name, i.e. file name of the
        configuration without the `json` extension.
    :type config_name: str
    """
    global log
    global report

    total_time = time.time()

    # Initialize logging.
    log = logging.get_logger(__file__, config_name, timestamp)
    log_dir = Path(log.handlers[1].baseFilename).parent

    # Initialize environment.
    load_dotenv(find_dotenv())

    exit_code = 0
    try:
        # Load configuration.
        cfg = config.load(config_name, log=log)

        # Sample data.
        sample_data(cfg, log_dir)

    except Exception as e:
        log.error(str(e))
        exit_code = 1

    # Save results.
    report['total_time'] = time.time() - total_time
    misc.save_results(report, __file__, config_name, timestamp)
    log.info(f'Report saved: {log_dir.parent}')

    return exit_code


if __name__ == '__main__':
    sys.exit(main())
