"""Visualize results.
"""
# import datetime
# import json
# import os
import sys
import time
from dotenv import find_dotenv, load_dotenv
from pathlib import Path

import click
# import numpy as np
# import geopandas as gpd
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap, BoundaryNorm
# from tqdm.auto import tqdm
# from eolearn.core import EOPatch
# from sentinelhub import DataSource

from src.utils import config, const, logging, misc


# Global variables
log = None
report = {}


def generate_report(cfg):
    pass


@click.command()
@click.argument('config_name', type=click.STRING)
@click.option(
    '--timestamp',
    type=click.FLOAT,
    default=time.time(),
    help='Unix timestamp'
)
def main(config_name, timestamp):
    """Visualize results for given configuration.

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

        # Generate report.
        generate_report(cfg)

    except Exception as e:
        log.error(str(e))
        raise e
        exit_code = 1

    # Save results.
    report['total_time'] = time.time() - total_time
    misc.save_results(report, __file__, config_name, timestamp)
    log.info(f'Report saved: {log_dir.parent}')

    return exit_code


if __name__ == '__main__':
    sys.exit(main())
