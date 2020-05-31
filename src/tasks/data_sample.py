"""Prepare subset of the data for modelling and evaluation.
"""
import collections
import os
import random
import re
import sys
import time
from dotenv import find_dotenv, load_dotenv
from pathlib import Path

import click
import numpy as np
import pandas as pd
from eolearn.core import EOExecutor, EOPatch, FeatureType, LinearWorkflow, \
    LoadTask, OverwritePermission, SaveTask
from eolearn.geometry import PointSamplingTask
from sklearn.utils import resample

from src.utils import config, logging, misc
from src.eolearn.tasks import TrainTestSplit


# Global variables
log = None
report = {}


def sample_patches(
    path,
    patches,
    no_samples,
    class_feature,
    mask_feature,
    features,
    weak_classes,
    samples_per_class=None,
    debug=False,
    seed=None,
    class_frequency=False
):
    """
    :param path: Path to folder containing all patches, folders need to be
        named eopatch_{number: 0 to no_patches-1}
    :type path: Path
    :param patches: List of patch IDs, e.g. [0, 1, ...]
    :type patches: list[int]
    :param no_samples: Number of samples taken per patch
    :type no_samples: int
    :param class_feature: Name of feature that contains class number.
        The numbers in the array can be float or nan for no class
    :type class_feature: (FeatureType, string)
    :param mask_feature: Feature that defines the area from where samples are
        taken, if None the whole image is used
    :type mask_feature: (FeatureType, String) or None
    :param features: Features to include in returned dataset for each pixel
        sampled
    :type features: array of type [(FeatureType, string), ...]
    :param samples_per_class: Number of samples per class returned after
        balancing. If the number is higher than minimal number of samples for
        the smallest class then those numbers are upsampled by repetition.
        If the argument is None then number is set to the size of the number of
        samples of the smallest class
    :type samples_per_class: int or None
    :param debug: If set to True patch id and coordinates are included in
        returned DataFrame
    :param seed: Seed for random generator
    :return: pandas DataFrame with columns
        [class feature, features, patch_id, x coord, y coord].
        id,x and y are used for testing
    :param class_frequency: If set to True, the function also return
        dictionary of each class frequency before balancing
    :type class_frequency: boolean
    :param weak_classes: Classes that when found also the neighbouring regions
        will be checked and added if they contain one of the weak classes.
        Used to enrich the samples
    :type weak_classes: int list
    """
    if seed is not None:
        random.seed(seed)

    columns = [class_feature[1]] + [x[1] for x in features]
    if debug:
        columns = columns + ['patch_no', 'x', 'y']

    class_name = class_feature[1]
    sample_dict = []

    for patch_id in patches:
        eopatch = EOPatch.load(
            str(path / f'eopatch_{patch_id}'),
            lazy_loading=True
        )

        if class_feature in eopatch.get_feature_list():
            log.info(f'Sampling eopatch_{patch_id}')
        else:
            log.warning(f'No feature {class_feature} in eopatch_{patch_id}')
            continue

        _, height, width, _ = eopatch.data['BANDS'].shape
        mask = eopatch[mask_feature[0]][mask_feature[1]].squeeze()
        no_samples = min(height * width, no_samples)

        # Finds all the pixels which are not masked
        subsample_id = []
        for h in range(height):
            for w in range(width):
                # Check if pixel has any NaNs.
                has_nan = np.isnan(eopatch.data['BANDS'][:, h, w]).any()

                # Skip pixels with NaNs and masked pixels.
                if not has_nan and (mask is None or mask[h][w] == 1):
                    subsample_id.append((h, w))

        # First sampling
        subsample_id = random.sample(
            subsample_id,
            min(no_samples, len(subsample_id))
        )
        # print(f'Actual patch sample size: {len(subsample_id)}')

        for h, w in subsample_id:
            class_value = eopatch[class_feature[0]][class_feature[1]][h][w][0]

            # Skip class 0 (= no data)
            if not class_value:
                continue

            array_for_dict = [(class_name, class_value)] \
                + [(f[1], float(eopatch[f[0]][f[1]][h][w])) for f in features]

            if debug:
                array_for_dict += [('patch_no', patch_id), ('x', w), ('y', h)]
            sample_dict.append(dict(array_for_dict))

            # Enrichment
            if class_value in weak_classes:  # TODO check duplicates
                neighbours = [-3, -2, -1, 0, 1, 2, 3]
                for x in neighbours:
                    for y in neighbours:
                        if x != 0 or y != 0:
                            h0 = h + x
                            w0 = w + y
                            max_h, max_w = height, width
                            if h0 >= max_h or w0 >= max_w \
                               or h0 <= 0 or w0 <= 0:
                                continue

                            val = eopatch[class_feature[0]][class_feature[1]][h0][w0][0]
                            if val in weak_classes:
                                array_for_dict = [(class_name, val)] \
                                    + [(f[1], float(eopatch[f[0]][f[1]][h0][w0])) for f in features]
                                if debug:
                                    array_for_dict += [
                                        ('patch_no', patch_id),
                                        ('x', w0),
                                        ('y', h0)
                                    ]
                                sample_dict.append(dict(array_for_dict))

    df = pd.DataFrame(sample_dict, columns=columns)
    df.dropna(axis=0, inplace=True)

    class_dictionary = collections.Counter(df[class_feature[1]])
    class_count = class_dictionary.most_common()
    least_common = class_count[-1][1]
    # print(f'Least common: {least_common}')

    # Balancing
    replace = False
    if samples_per_class is not None:
        least_common = samples_per_class
        replace = True
    df_downsampled = pd.DataFrame(columns=columns)
    names = [name[0] for name in class_count]
    dfs = [df[df[class_name] == x] for x in names]
    for d in dfs:
        nd = resample(
            d,
            replace=replace,
            n_samples=least_common,
            random_state=seed
        )
        # print(f'Actual sample size per class: {len(nd.index)}')
        df_downsampled = df_downsampled.append(nd)

    if class_frequency:
        return df_downsampled, class_dictionary

    return df_downsampled


def get_patches(path, n=0):
    """Get selected number of patch IDs from given directory path.
    If number is not provided, i.e. is zero, all patch IDs are returned.

    :param path: Directory path where patches are
    :type path: Path
    :param n: Number of patch IDs to retrieve, defaults to 0
    :type n: int, optional
    :return: List of patch IDs
    :rtype: list[int]
    """
    patches = [patch.name for patch in path.glob('eopatch_*')]
    ids = []

    for patch in patches:
        match = re.match(r'^eopatch_(\d+)$', patch)
        if match:
            ids.append(int(match.group(1)))

    ids.sort()
    return random.sample(ids, n) if n else ids


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

    sampl_cfg = cfg['sampling']

    patches = None
    if isinstance(sampl_cfg['patches'], list):
        patches = sampl_cfg['patches']
    else:
        patches = get_patches(input_dir, sampl_cfg['patches'])

    no_patches = len(patches)

    if no_patches:
        log.info(
            f'Attempting to sample {no_patches} patches of class '
            f'{sampl_cfg["class_feature"]} from: {input_dir}'
        )

        # Execute workflow.
        ex_time = time.time()
        samples, class_freq = sample_patches(
            path=input_dir,
            patches=patches,
            no_samples=sampl_cfg['no_samples'],
            class_feature=(
                FeatureType.MASK_TIMELESS,
                sampl_cfg['class_feature']
            ),
            mask_feature=(FeatureType.MASK_TIMELESS, 'EDGES_INV'),
            features=[],
            weak_classes=sampl_cfg['weak_classes'],
            debug=True,
            seed=None,
            class_frequency=True
        )
        ex_time = time.time() - ex_time

        no_samples = len(samples.index)
        log.info(
            f'Sampled {no_patches} patches in {ex_time:.2f} seconds'
        )
        log.info(f'All samples: {no_samples}')
        log.info(f'Class frequency: {class_freq}')

        # Get output directory.
        output_dir = misc.get_sampled_data_dir(cfg)

        if not output_dir.is_dir():
            output_dir.mkdir(parents=True)

        output_file = output_dir / 'samples.csv'
        samples.to_csv(output_file, index=False)

        log.info(f'Results saved: {output_file}')
    else:
        ex_time = 0.0
        log.info('No patches to sample.')

    report['sampled_patches'] = {
        'quantity': no_patches,
        'samples': no_samples if no_patches else 0,
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
        report['config'] = cfg
        misc.print_header(cfg, log)

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
