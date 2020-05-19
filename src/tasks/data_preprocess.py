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
    OverwritePermission, LoadTask, SaveTask  # , MergeFeatureTask
# from eolearn.features import LinearInterpolation, SimpleFilterTask
from eolearn.geometry import VectorToRaster, ErosionTask
from eolearn.mask import AddCloudMaskTask, get_s2_pixel_cloud_detector, \
    AddValidDataMaskTask
from sentinelhub import DataSource

from src.eolearn.predicates import SentinelHubValidData
# ValidDataFractionPredicate
from src.eolearn.tasks import AddBaseFeatures, AddGray, \
    AddStreamTemporalFeaturesTask, AddGradientTask, CleanMeta, ExtractEdgesTask
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

    # EOTask: Load data
    # =================

    load = LoadTask(
        str(input_dir),
        lazy_loading=True
    )

    input_cfg = config.get_sh_input_config(cfg, DataSource.SENTINEL2_L1C)
    clean_meta = CleanMeta(
        input_cfg['service_type'].lower(),
        cfg['time_interval']
    )

    # EOTask: Add gradient
    # ====================
    # Calculate inclination from DEM and add it to timeless masks.

    add_gradient = AddGradientTask(
        (FeatureType.DATA_TIMELESS, 'DEM'),
        (FeatureType.DATA_TIMELESS, 'INCLINATION')
    )

    # EOTask: Add cloud mask
    # ======================
    # Use Sentinel Hub's `S2PixelCloudDetector` classifier.

    clouds_cfg = cfg['cloud_detection']
    cloud_classifier = get_s2_pixel_cloud_detector(
        **clouds_cfg['s2_pixel_cloud_detector']
    )
    add_cloud_mask = AddCloudMaskTask(
        cloud_classifier,
        **clouds_cfg['cloud_mask']
    )

    # EOTask: Add valid data mask
    # ===========================
    # Validate pixels using SentinelHub's cloud mask and Sen2Corr's
    # classification map.

    add_valid_data_mask = AddValidDataMaskTask(
        SentinelHubValidData(),
        'IS_VALID'
    )

    # EOTask: Add features
    # ====================-
    # Add indices and time series based (stream) features.

    input_cfg = config.get_sh_input_config(cfg, DataSource.SENTINEL2_L1C)
    band_names = config.get_band_names(cfg, DataSource.SENTINEL2_L1C)
    bands_feature = input_cfg['feature']

    add_base_features = AddBaseFeatures(
        bands_feature,
        band_names,
        cfg['features']
    )

    # add_stream_features = [
    #     AddStreamTemporalFeaturesTask(data_feature=feature)
    #     for feature in cfg['features']
    # ]

    # EOTask: Rasterize reference data
    # ================================
    # Rasterize reference data to a new timeless mask.

    reference_dir = const.DATA_EXTERNAL_DIR / 'reference'
    reference_file = reference_dir / 'data.shp'

    # Load reference data for selected AOI.
    log.info(f'Loading {reference_file}')
    reference_data = gpd.read_file(
        reference_dir / 'data.shp',
        # bbox=misc.get_aoi_bbox(cfg)
    )
    log.info('Reference data loaded.')

    rasterize_reference_data = VectorToRaster(
        reference_data,
        (FeatureType.MASK_TIMELESS, 'REF'),
        values_column='CLASS',
        raster_shape=(FeatureType.MASK, 'IS_VALID'),
        raster_dtype=np.uint8
    )

    # EOTask: Erode reference mask
    # ============================

    erode_reference_mask = ErosionTask(
        mask_feature=(
            FeatureType.MASK_TIMELESS,
            'REF',
            'REF_MORPHED'
        ),
        disk_radius=1
    )

    # EOTask: Extract edges
    # =====================

    add_gray = AddGray(bands_feature)

    extract_edges = ExtractEdgesTask(
        edge_features=[
            {
                "FeatureType": FeatureType.DATA,
                "FeatureName": 'EVI',
                "CannyThresholds": (40, 80),
                "BlurArguments": ((5, 5), 2)
            },
            {
                "FeatureType": FeatureType.DATA,
                "FeatureName": 'ARVI',
                "CannyThresholds": (40, 80),
                "BlurArguments": ((5, 5), 2)
            },
            {
                "FeatureType": FeatureType.DATA,
                "FeatureName": 'NDVI',
                "CannyThresholds": (40, 100),
                "BlurArguments": ((5, 5), 2)
            },
            {
                "FeatureType": FeatureType.DATA,
                "FeatureName": 'GRAY',
                "CannyThresholds": (5, 40),
                "BlurArguments": ((3, 3), 2)
            }
        ],
        structuring_element=[
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]
        ],
        excluded_features=[],
        dilation_mask=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        erosion_mask=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        output_feature=(FeatureType.MASK_TIMELESS, 'EDGES_INV'),
        adjust_function=lambda x: cv2.GaussianBlur(x, (9, 9), 5),
        adjust_threshold=0.05,
        yearly_low_threshold=0.8
    )

    # EOTask: Merge features
    # ======================

    # bands_features = [input_cfg['feature'] for input_cfg in cfg['sh_inputs']]
    # merge_features = MergeFeatureTask(
    #     {FeatureType.DATA: [*bands_features, *cfg['features']]},
    #     (FeatureType.DATA, 'FEATURES')
    # )

    # EOTask: Filter valid frames
    # ===========================
    # Keep frames with valid coverage above given threshold.

    # filter_valid_frames = SimpleFilterTask(
    #     (FeatureType.MASK, 'IS_VALID'),
    #     ValidDataFractionPredicate(cfg['filtering']['threshold'])
    # )

    # EOTask: Interpolate
    # ===================
    # Interpolate invalid pixels of timeseries and resample it to new uniform
    # time sequence.

    # interpolate_invalid_data = LinearInterpolation(
    #     'FEATURES',
    #     mask_feature=(FeatureType.MASK, 'IS_VALID'),
    #     copy_features=[
    #         (FeatureType.MASK_TIMELESS, 'REF'),
    #         (FeatureType.MASK_TIMELESS, 'REF_MORPHED'),
    #     ],
    #     resample_range=(*cfg['time_interval'], cfg['interpolation']['step']),
    #     bounds_error=False
    # )

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
        clean_meta,
        # add_gradient,
        # add_cloud_mask,
        # add_valid_data_mask,
        # add_base_features,
        # *add_stream_features,
        rasterize_reference_data,
        erode_reference_mask,
        # add_gray,
        # extract_edges,
        # merge_features,
        # filter_valid_frames,
        # interpolate_invalid_data,
        save
    )

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

    for path in input_dir.iterdir():
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
