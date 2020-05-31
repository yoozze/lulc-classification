"""Evaluate models on subset of sampled data.
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


def evaluate_models(cfg, log_dir):
    """Evaluate models for given configuration.

    :param cfg: Configuration
    :type cfg: dict
    :param log_dir: Logging directory
    :type log_dir: Path
    """
    global report

    # Get input directory.
    sampled_data_dir = misc.get_sampled_data_dir(cfg)
    data_dir = sampled_data_dir / 'patches'

    # Get output directory.
    models_dir = misc.get_models_dir(cfg)

    # Load test set.
    log.info(f'Loading test set: {data_dir}')
    X_test, y_test = misc.load_sample_subset(data_dir, 0)
    log.info(f'Test set loaded: X{X_test.shape}, y{y_test.shape}')

    report['predict_times'] = {}
    report['scores'] = {}

    # Evaluate models.
    for model_cfg in cfg['modelling']:
        model_name = model_cfg['name']

        try:
            model = None
            model_path = models_dir / f'{model_name}.pkl'

            if model_path.is_file():
                model = joblib.load(str(model_path))
            else:
                # StreamDM models are exported to JSON.
                model_path = models_dir / f'{model_name}.json'
                model = eval(model_name)()
                model.import_json(str(model_path))

            log.info(f'Predicting with model: {model_name}')

            predict_time = time.time()
            y_pred = model.predict(X_test)
            predict_time = time.time() - predict_time

            log.info(f'Predicting finished after {predict_time:.2f} seconds')
            report['predict_times'][model_name] = predict_time

            # labels = np.unique(y_test)
            mask = np.in1d(y_pred, y_test)
            lbls = y_test[mask]
            pred = y_pred[mask]

            accuracy_score = metrics.accuracy_score(lbls, pred)
            f1_score = metrics.f1_score(lbls, pred, average='weighted')

            report['scores'][model_name] = {
                'accuracy': accuracy_score,
                'f1': f1_score
            }

            log.info(f'Classification accuracy: {accuracy_score}')
            log.info(f'Classification F1-score: {f1_score}')
        except Exception as e:
            log.warning(f'Predicting with {model_name} failed!')
            log.debug(e)

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
    """Evaluate models for given configuration.

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
        report['config'] = cfg
        misc.print_header(cfg, log)

        # Evaluate models.
        evaluate_models(cfg, log_dir)

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
