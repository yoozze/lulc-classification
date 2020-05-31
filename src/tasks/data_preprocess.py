"""Preproces EO and reference data.
"""
import os
import sys
import time
from dotenv import find_dotenv, load_dotenv
from pathlib import Path

import click
import cv2
import geopandas as gpd
import numpy as np
from eolearn.core import EOExecutor, FeatureType, LinearWorkflow, \
    OverwritePermission, LoadTask, SaveTask
from eolearn.features import LinearInterpolation, SimpleFilterTask
from eolearn.geometry import VectorToRaster, ErosionTask
from eolearn.mask import AddCloudMaskTask, get_s2_pixel_cloud_detector, \
    AddValidDataMaskTask
from sentinelhub import DataSource

from src.eolearn.predicates import SentinelHubValidData, \
    ValidDataFractionPredicate
from src.eolearn.tasks import AddBaseFeatures, AddGradientTask, CleanMeta, \
    ExtractEdgesTask
# AddStreamTemporalFeaturesTask
from src.utils import config, const, logging, misc


# Global variables
log = None
report = {}


def init_workflow(cfg, input_dir, output_dir):
    """Initialize tasks and build linear workflow.

    :param cfg: Configuration
    :type cfg: dict
    :param input_dir: Input directory, where raw data patches are stored.
    :type input_dir: Path
    :param output_dir: Output directory, where processed data patches will be
        stored.
    :type output_dir: Path
    :return: Workflow
    :rtype: LinearWorkflow
    """

    tasks = []

    # EOTask: Load data
    # =================

    tasks.append(LoadTask(
        str(input_dir),
        lazy_loading=True
    ))

    input_cfg = config.get_sh_input_config(cfg, DataSource.SENTINEL2_L1C)
    tasks.append(CleanMeta(
        input_cfg,
        cfg['time_interval']
    ))

    # EOTask: Add cloud mask
    # ======================
    # Use Sentinel Hub's `S2PixelCloudDetector` classifier.

    if 'cloud_detection' in cfg:
        clouds_cfg = cfg['cloud_detection']
        cloud_classifier = get_s2_pixel_cloud_detector(
            **clouds_cfg['s2_pixel_cloud_detector']
        )
        tasks.append(AddCloudMaskTask(
            cloud_classifier,
            **clouds_cfg['cloud_mask']
        ))

    # EOTask: Add valid data mask
    # ===========================
    # Validate pixels using SentinelHub's cloud mask and Sen2Corr's
    # classification map.

    if 'valid_data' in cfg:
        tasks.append(AddValidDataMaskTask(
            SentinelHubValidData(),
            'IS_VALID'
        ))

    # EOTask: Filter valid frames
    # ===========================
    # Keep frames with valid coverage above given threshold.

    if 'filter' in cfg:
        tasks.add(SimpleFilterTask(
            (FeatureType.MASK, 'IS_VALID'),
            ValidDataFractionPredicate(cfg['filter']['threshold'])
        ))

    # EOTask: Interpolate
    # ===================
    # Interpolate invalid pixels of timeseries and resample it to new uniform
    # time sequence.

    if 'interpolation' in cfg:
        bands_feature = config.get_feature_name(cfg, DataSource.SENTINEL2_L1C)
        tasks.add(LinearInterpolation(
            feature=bands_feature,
            mask_feature=(FeatureType.MASK, 'IS_VALID'),
            copy_features=[
                tuple(FeatureType[t[0]], t[1])
                for t in cfg['interpolation']['copy_features']
            ],
            bounds_error=False
        ))

    # EOTask: Add features
    # ====================-
    # Add indices and time series based (stream) features.

    if 'features' in cfg:
        bands_feature = config.get_feature_name(cfg, DataSource.SENTINEL2_L1C)
        band_names = config.get_band_names(cfg, DataSource.SENTINEL2_L1C)

        tasks.append(AddBaseFeatures(
            bands_feature,
            band_names,
            cfg['features']
        ))

    # add_stream_features = [
    #     AddStreamTemporalFeaturesTask(data_feature=feature)
    #     for feature in cfg['features']
    # ]

    # EOTask: Add gradient
    # ====================
    # Calculate inclination from DEM and add it to timeless masks.

    if 'gradient' in cfg:
        tasks.append(AddGradientTask(
            (FeatureType.DATA_TIMELESS, 'DEM'),
            (FeatureType.DATA_TIMELESS, 'INCLINATION'),
            sigma=cfg['gradient']['sigma']
        ))

    # EOTask: Extract edges
    # =====================

    if 'edges' in cfg:
        tasks.append(ExtractEdgesTask(
            edge_features=[
                {
                    'FeatureType': FeatureType[f['feature_type']],
                    'FeatureName': f['feature_name'],
                    'CannyThresholds': tuple(f['canny_thresholds']),
                    'BlurArguments': tuple(
                        tuple(f['blur_arguments'][0]),
                        f['blur_arguments'][1]
                    )
                }
                for f in cfg['edges']['edge_features']
            ],
            structuring_element=cfg['edges']['structuring_element'],
            excluded_features=[],
            dilation_mask=cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                tuple(cfg['edges']['dilation_mask'])
            ),
            erosion_mask=cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                tuple(cfg['edges']['erosion_mask'])
            ),
            output_feature=(FeatureType.MASK_TIMELESS, 'EDGES_INV'),
            adjust_function=lambda x: cv2.GaussianBlur(x, (9, 9), 5),
            adjust_threshold=cfg['edges']['adjust_threshold'],
            yearly_low_threshold=cfg['edges']['yearly_low_threshold']
        ))

    # EOTask: Rasterize reference data
    # ================================
    # Rasterize reference data to a new timeless mask.

    if 'raster' in cfg:
        reference_dir = const.DATA_EXTERNAL_DIR / 'reference'

        for reference_cfg in cfg['reference_data']:
            reference_name = reference_cfg['name']
            reference_file = \
                reference_dir / reference_name / 'data.shp'

            # Load reference data for selected AOI.
            log.info(f'Loading {reference_file}')
            reference_data = gpd.read_file(
                reference_file,
                # bbox=misc.get_aoi_bbox(cfg)
            )
            log.info('Reference data loaded.')

            tasks.append(VectorToRaster(
                reference_data,
                (FeatureType.MASK_TIMELESS, reference_name),
                values_column='CLASS',
                raster_shape=(FeatureType.MASK, 'IS_DATA'),
                raster_dtype=np.uint8
            ))
            tasks.append(VectorToRaster(
                reference_data,
                (FeatureType.MASK_TIMELESS, f'{reference_name}_G'),
                values_column='CLASS_G',
                raster_shape=(FeatureType.MASK, 'IS_DATA'),
                raster_dtype=np.uint8
            ))

            if cfg['raster']['erosion']:
                tasks.append(ErosionTask(
                    mask_feature=(
                        FeatureType.MASK_TIMELESS,
                        reference_name,
                        f'{reference_name}_E'
                    ),
                    disk_radius=1
                ))
                tasks.append(ErosionTask(
                    mask_feature=(
                        FeatureType.MASK_TIMELESS,
                        f'{reference_name}_G',
                        f'{reference_name}_G_E'
                    ),
                    disk_radius=1
                ))

    # EOTask: Save data
    # =================
    features = ...

    if 'preprocess_save' in cfg:
        features = []
        for feature in cfg['preprocess_save']:
            if isinstance(feature, list):
                features.append(tuple([FeatureType[feature[0]], feature[1]]))
            else:
                features.append(FeatureType[feature])

    tasks.append(SaveTask(
        str(output_dir),
        features=features,
        # compress_level=1,
        overwrite_permission=OverwritePermission.OVERWRITE_PATCH
    ))

    # EOWorkflow
    # ==========

    workflow = LinearWorkflow(*tasks)

    return workflow


def init_execution_args(tasks, input_dir, output_dir):
    """Define list of argument dictionaries for tasks to be fed to
    executioner together with workflow.

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
    # interval = [1041, 1061]
    # idxs = [501, 502]
    # for idx in range(*interval):
    # for idx in idxs:
    for path in input_dir.iterdir():
        # eopatch_folder = f'eopatch_{idx}'
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
            tasks['SaveTask']: {
                'eopatch_folder': eopatch_folder
            }
        })

    return execution_args


def preprocess_data(cfg, log_dir):
    """Preprocess EO data.

    :param cfg: Configuration
    :type cfg: dict
    :param log_dir: Logging directory
    :type log_dir: Path
    """
    global report

    # Get input directory.
    raw_data_dir = misc.get_raw_data_dir(cfg)
    input_dir = raw_data_dir / 'patches'

    # Get output directory.
    processed_data_dir = misc.get_processed_data_dir(cfg)
    output_dir = processed_data_dir / 'patches'

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
            f'Attempting to process {total_executions} patches from: '
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
            f'Processed {successful_executions} / {total_executions} '
            f'patches in {ex_time:.2f} seconds'
        )
        log.info(f'Results saved: {output_dir}')

        # Report.
        executor.make_report()
    else:
        ex_time = 0.0
        log.info('All patches already processed.')

    report['process_patches'] = {
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
    """Preprocess data for given configuration.

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

        # Preproces data.
        preprocess_data(cfg, log_dir)

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
