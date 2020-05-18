"""Train models on subset of sampled data.
"""
import sys
import time
from dotenv import find_dotenv, load_dotenv
from pathlib import Path

import click
import joblib
import numpy as np
from lightgbm import LGBMClassifier # noqa
from sklearn import metrics
# from sklearn import preprocessing
# from skmultiflow.trees import HoeffdingTree

import src.lib.streamdm as stream # noqa
from src.utils import config, logging, misc


# Global variables
log = None
report = {}


def init_model(cfg, labels_unique):
    """Initialize model with given parameters.

    :param cfg: Configuration
    :type cfg: dict
    :param labels_unique: List of unique labels
    :type labels_unique: list[int]
    :return: Model instance.
    :rtype: object
    """
    name = cfg['name']
    params = {**cfg['params']}

    # Overwrite default params if neccessary
    if name == 'LGBMClassifier':
        params['num_class'] = len(labels_unique)

    return eval(name)(**params)


def train_models(cfg, log_dir):
    """Train models for given configuration.

    :param cfg: Configuration
    :type cfg: dict
    :param log_dir: Logging directory
    :type log_dir: Path
    """
    global report

    # Get input directory.
    sampled_data_dir = misc.get_sampled_data_dir(cfg)
    input_dir = sampled_data_dir / 'patches'

    # Get output directory.
    output_dir = misc.get_models_dir(cfg)

    if not output_dir.is_dir():
        output_dir.mkdir(parents=True)

    # Load train set.
    log.info(f'Loading train set: {input_dir}')
    X_test, y_test = misc.load_sample_subset(input_dir, 1)
    log.info(f'Train set loaded: X{X_test.shape}, y{y_test.shape}')

    labels_unique = np.unique(y_test)

    models = {}
    report['train_times'] = {}

    for model_cfg in cfg['modelling']:
        model_name = model_cfg['name']

        try:
            model = init_model(model_cfg, labels_unique)

            log.info(f'Training model: {model_name}')

            train_time = time.time()
            model.fit(X_test, y_test)
            train_time = time.time() - train_time

            models[model_name] = model

            log.info(f'Training finished after {train_time:.2f} seconds')
            report['train_times'][model_name] = train_time

            if hasattr(model, 'export_json'):
                # StreamDM models are exported to JSON.
                model_path = output_dir / f'{model_name}.json'
                model.export_json(str(model_path))
            else:
                # Other models are dumped in pickle.
                model_path = output_dir / f'{model_name}.pkl'
                joblib.dump(model, str(model_path))

            log.info(f'Model saved: {model_path}')
        except Exception as e:
            log.warning(f'Training {model_name} failed!')
            log.debug(e)

    log.info('Training complete!')

    # Evaluate
    # ========

    log.info(f'Loading test set: {input_dir}')
    X_test, y_test = misc.load_sample_subset(input_dir, 0)
    log.info(f'Test set loaded: X{X_test.shape}, y{y_test.shape}')

    for model_name, model in models.items():
        log.info(f'Evaluating {model_name}:')

        y_pred = model.predict(X_test)

        # labels = np.unique(y_test)
        mask = np.in1d(y_pred, y_test)
        lbls = y_test[mask]
        pred = y_pred[mask]

        accuracy_score = metrics.accuracy_score(lbls, pred)
        f1_score = metrics.f1_score(lbls, pred, average='weighted')

        log.info(f'Classification accuracy: {accuracy_score}')
        log.info(f'Classification F1-score: {f1_score}')

    log.info('Evaluation complete!')


@click.command()
@click.argument('config_name', type=click.STRING)
@click.option(
    '--timestamp',
    type=click.FLOAT,
    default=time.time(),
    help='Unix timestamp'
)
def main(config_name, timestamp):
    """Train models for given configuration.

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

        # Train models.
        train_models(cfg, log_dir)

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
