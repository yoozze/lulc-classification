"""Download EO and reference data.
"""
import os
import sys
import time
from dotenv import find_dotenv, load_dotenv
from pathlib import Path

import click
import requests
import patoolib
import geopandas as gpd
import numpy as np
from eolearn.core import EOExecutor, LinearWorkflow, OverwritePermission, \
    SaveTask
from eolearn.io import SentinelHubOGCInput
from sentinelhub import BBox, BBoxSplitter, CRS
from shapely.geometry import Polygon
from tqdm.auto import tqdm

from src.utils import config, const, logging, misc


# Global variables
log = None
report = {}


def download_tqdm(url, file_path):
    """Download file from the given URL and save it to the given file path.

    :param url: URL of the file to be downloaded.
    :type url: str
    :param file_path: File system path to where file should be saved.
    :type file_path: Path
    :return: Total downloaded size in bytes.
    :rtype: int
    :raises e: Download failed.
    """
    response = requests.get(url, stream=True)

    try:
        total = int(response.headers.get('content-length', 0))
        with tqdm.wrapattr(
            open(file_path, "wb"),
            "write",
            miniters=1,
            total=total,
            ascii=True
            # desc=file_path
        ) as fout:
            for chunk in response.iter_content(chunk_size=4096):
                fout.write(chunk)

        return total
    except Exception as e:
        log.error(f'Failed to download: {url}')
        raise e


def get_country(country_code):
    """Get country data (shape, name, ...).
    If data doesn't exist, try to download it first.

    :return: Country data in form of GeoDataFrame.
    :rtype: GeoDataFrame
    """
    global report

    countries_dir = const.DATA_EXTERNAL_DIR / 'countries'
    country_path = countries_dir / f'{country_code}.geojson'

    # If country file is already prepared, load it.
    if country_path.is_file():
        return gpd.read_file(country_path), True

    countries_base_name = const.COUNTRIES_BASE_URL.split('/')[-1]

    if not countries_dir.is_dir():
        countries_dir.mkdir(parents=True)

    report['dl_countries'] = {
        'time': 0.0,
        'size': 0,
        'quantity': 0
    }

    for ext in ['cpg', 'dbf', 'prj', 'shp', 'shx']:
        countries_file = countries_dir / f'{countries_base_name}.{ext}'

        # Download files if neccessary.
        if not countries_file.is_file():
            log.info(f'Downloading countries file: {str(countries_file)}')
            dl_time = time.time()
            dl_size = download_tqdm(
                f'{const.COUNTRIES_BASE_URL}.{ext}',
                countries_file
            )
            dl_time = time.time() - dl_time

            report['dl_countries']['time'] += dl_time
            report['dl_countries']['size'] += dl_size
            report['dl_countries']['quantity'] += 1
        else:
            log.info(f'Skipping download: {str(countries_file)}')

    dl_time = report['dl_countries']['time']
    dl_size = report['dl_countries']['size']
    if report['dl_countries']['size']:
        formatted_total_size = misc.format_data_size(dl_size)
        log.info(f'Downloaded {formatted_total_size} in {dl_time:.2f} seconds')

    # Load downloaded data.
    countries = gpd.read_file(countries_dir / f'{countries_base_name}.shp')
    return countries.loc[countries['SOV_A3'] == country_code], False


def define_AOI(cfg, data_dir):
    """Split selected country's bounding box into grid of smaller bounding
    boxes and select intersecting subset. Resulting subset is further reduced,
    if subarea is defined in configuration.

    :param cfg: Configuration
    :type cfg: dict
    :param data_dir: Data directory
    :type data_dir: Path
    :return: Zip of bounding boxes and their indices covering selected AIO.
    :rtype: zip((int, BBox))
    """
    aoi_cfg = cfg['AOI']

    # Get country data.
    country_code = aoi_cfg['country']
    country, is_prepared = get_country(country_code)

    if country.shape[0] < 1:
        log.error(f'Invalid country code: {country_code}')
        raise Exception('Invalid configuration')
    else:
        country_name = country.iloc[0]['SOVEREIGNT']

        log.info(
            f'Selected country: {country_name} ({country_code})'
        )

    # Buffer it a little bit.
    if is_prepared:
        geometry = country.geometry.tolist()
    else:
        geometry = country.geometry.buffer(aoi_cfg['buffer']).tolist()

    country = gpd.GeoDataFrame(
        {
            'name': [country_name]
        },
        crs=country.crs,
        geometry=geometry
    )

    # Convert CRS.
    try:
        country_crs = CRS[aoi_cfg['crs']]
        log.info(f'Selected CRS: {country_crs} ({aoi_cfg["crs"]})')
    except Exception as e:
        log.error(f'Invalid CRS: {aoi_cfg["crs"]}')
        raise e

    country = country.to_crs(crs=country_crs.pyproj_crs())

    # Split country bounding box to a grid of smaller bounding boxes.
    country_shape = country.geometry.tolist()[-1]
    country_width = country_shape.bounds[2] - country_shape.bounds[0]
    country_height = country_shape.bounds[3] - country_shape.bounds[1]

    log.info(
        f'Country bounding box surface: '
        f'{country_width:.0f} x {country_height:.0f} m2'
    )

    aoi_dir = const.DATA_EXTERNAL_DIR / 'AOI'

    if not aoi_dir.is_dir():
        # Create new split.
        bbox_splitter = BBoxSplitter(
            [country_shape],
            country_crs,
            tuple(aoi_cfg['grid'])
        )

        bbox_list = np.array(bbox_splitter.get_bbox_list())
        info_list = np.array(bbox_splitter.get_info_list())
    else:
        # Load existing AOI split
        bbox_list = []
        info_list = []
        aoi = gpd.read_file(aoi_dir / 'aoi.shp')

        for idx, row in aoi.iterrows():
            info_list.append({
                'index_x': row['index_x'],
                'index_y': row['index_y'],
            })
            bbox_list.append(BBox(row.geometry.bounds, country_crs))

        bbox_list = np.array(bbox_list)
        info_list = np.array(info_list)

    tile_shape = bbox_list[0].geometry
    tile_width = tile_shape.bounds[2] - tile_shape.bounds[0]
    tile_height = tile_shape.bounds[3] - tile_shape.bounds[1]

    log.info(f'Tile surface: {tile_width:.0f} x {tile_height:.0f} m2')
    log.info(f'Number of tiles: {len(bbox_list)}')

    # Store bounding box grid to GeoDataFrame.
    geometry = [Polygon(bbox.get_polygon()) for bbox in bbox_list]
    indices_x = np.array([info['index_x'] for info in info_list])
    indices_y = np.array([info['index_y'] for info in info_list])

    if not len(aoi_cfg['regions']):
        # If no region is selected in config file, mark all tiles as selected.
        selected = np.ones(len(geometry))
    else:
        # Mark tiles of regions defined in config file as selected.
        selected_x = np.zeros(len(geometry), dtype=bool)
        selected_y = np.zeros(len(geometry), dtype=bool)

        for box in aoi_cfg['regions']:
            for i in range(box[0][0], box[1][0] + 1):
                selected_x = np.logical_or(selected_x, indices_x == i)

            for i in range(box[0][1], box[1][1] + 1):
                selected_y = np.logical_or(selected_y, indices_y == i)

        selected = np.logical_and(selected_x, selected_y)

    log.info(f'Number of selected tiles: {sum(selected)}')

    aoi = gpd.GeoDataFrame(
        {
            'index_x': indices_x,
            'index_y': indices_y,
            'selected': selected
        },
        crs=country_crs.pyproj_crs(),
        geometry=geometry
    )

    # Save country/AOI bounding box grid.
    aoi_dir = data_dir / 'AOI'

    if not aoi_dir.is_dir():
        aoi_dir.mkdir(parents=True)

    # country.to_file(aoi_dir / 'country.geojson', driver='GeoJSON')
    country.to_file(aoi_dir / 'country.shp')

    # aoi_gdf.to_file(aoi_dir / 'aoi.geojson', driver='GeoJSON')
    aoi.to_file(aoi_dir / 'aoi.shp')

    return zip(aoi.index[selected].tolist(), bbox_list[selected].tolist())


def init_workflow(cfg, output_dir):
    """Initialize tasks and build linear workflow.

    :param cfg: Configuration
    :type cfg: dict
    :param output_dir: Output directory, where processed data patches will be
        stored.
    :type output_dir: Path
    :return: Workflow
    :rtype: LinearWorkflow
    """

    # EOTask: Download EO data
    # ========================
    # Use Sentinel Hub's OGC service.

    sh_instance_id = os.getenv('SH_INSTANCE_ID')
    add_data = []

    # Support multiple input services.
    for sh_input_cfg in cfg['sh_inputs']:
        # Generate input service configuration.
        input_cfg = config.eval_sh_input(sh_input_cfg)
        input_cfg['instance_id'] = sh_instance_id

        # Initialize service.
        ogc_input = SentinelHubOGCInput(**input_cfg)
        add_data.append(ogc_input)

    # EOTask: Save data
    # =================

    save = SaveTask(
        str(output_dir),
        overwrite_permission=OverwritePermission.OVERWRITE_PATCH
    )

    # EOWorkflow
    # ==========

    workflow = LinearWorkflow(
        *add_data,
        save
    )

    return workflow


def init_execution_args(cfg, tasks, output_dir):
    """Define list of argument dictionaries for tasks to be fed to
    executioner together with workflow.

    :param tasks: Dictionary of tasks
    :type tasks: dict[str, EOTask]
    :param output_dir: Output data directory
    :type output_dir: Path
    :return: List of argument distionaries
    :rtype: dict[str, object]
    """

    # Load AOI.
    bboxes = define_AOI(cfg, output_dir.parent)

    # Prepare workflow execution arguments.
    time_interval = cfg['time_interval']
    log.info(f'Time interval: {time_interval}')

    execution_args = []

    for idx, bbox in bboxes:
        eopatch_folder = f'eopatch_{idx}'
        patch_meta = output_dir / eopatch_folder / 'meta_info.pkl'

        # Skip existing patches.
        if patch_meta.is_file():
            log.info(f'Patch {eopatch_folder} already exists.')
            continue

        args = {}

        for task_name, task in tasks.items():
            if task_name == 'SaveTask':
                args[task] = {
                    'eopatch_folder': eopatch_folder
                }
            else:
                args[task] = {
                    'bbox': bbox,
                    'time_interval': time_interval
                }

        execution_args.append(args)

    return execution_args


def download_eo_data(cfg, log_dir):
    """Download EO data into EOPatches for AOI defined in configuration.

    :param cfg: Configuration
    :type cfg: dict
    :param log_dir: Logging directory
    :type log_dir: [type]
    """
    global report

    data_dir = misc.get_raw_data_dir(cfg)
    output_dir = data_dir / 'patches'

    if not output_dir.is_dir():
        output_dir.mkdir(parents=True)

    # Initialize workflow.
    workflow = init_workflow(
        cfg,
        output_dir
    )

    # Initialize workflow execution arguments.
    execution_args = init_execution_args(
        cfg,
        workflow.get_tasks(),
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

        log.info(f'Attempting to download {total_executions} patches.')

        # Execute workflow.
        ex_time = time.time()
        executor.run(
            workers=ex_workers,
            multiprocess=ex_multiprocess
        )
        ex_time = time.time() - ex_time

        successful_executions = len(executor.get_successful_executions())
        log.info(
            f'Downloaded {successful_executions} / {total_executions} '
            f'patches in {ex_time:.2f} seconds'
        )
        log.info(f'Results saved: {output_dir}')

        # Report.
        executor.make_report()
    else:
        ex_time = 0.0
        log.info('All patches already downloaded.')

    report['dl_patches'] = {
        'quantity': total_executions,
        'time': ex_time,
    }


def download_reference_data(cfg):
    """Download reference data if it doesn't exist.

    :param cfg: Configuration
    :type cfg: dict
    """
    global report

    reference_dir = const.DATA_EXTERNAL_DIR / 'reference'
    reference_url = cfg['reference_data']['url']
    reference_base_name = reference_url.split('/')[-1]

    if not reference_dir.is_dir():
        reference_dir.mkdir(parents=True)

    report['dl_reference'] = {
        'time': 0.0,
        'size': 0,
        'quantity': 0
    }

    # Search for `shp` files.
    shape_files = reference_dir.glob('*.shp')

    # If `shp` file doesn't exist, try to download reference data.
    if not len(list(shape_files)):
        reference_file = reference_dir / reference_base_name

        if reference_file.is_file():
            log.info('Reference data archive already exists.')
        else:
            log.info(f'Downloading reference data: {reference_file}')
            dl_time = time.time()
            dl_size = download_tqdm(
                reference_url,
                reference_file
            )
            dl_time = time.time() - dl_time

            report['dl_reference']['time'] += dl_time
            report['dl_reference']['size'] += dl_size
            report['dl_reference']['quantity'] += 1

            formatted_total_size = misc.format_data_size(dl_size)
            log.info(
                f'Downloaded {formatted_total_size} in {dl_time:.2f} seconds'
            )

        # Try to extract downloaded data.
        try:
            log.info(f'Extracting: {reference_file}')
            patoolib.extract_archive(
                str(reference_file),
                outdir=str(reference_dir)
            )
            log.info('Extraction competed.')
        except Exception as e:
            log.error('Automatic extraction failed.')
            log.info(
                f'Please try to extract data manually to: '
                f'{reference_dir}'
            )
            raise e


def prepare_reference_data(cfg):
    """Prepare reference data by converting CRS and grouping classes.

    :param cfg: Configuration
    :type cfg: dict
    :raises Exception: File not found!
    """
    reference_dir = const.DATA_EXTERNAL_DIR / 'reference'
    data_file = reference_dir / 'data.shp'

    if not data_file.is_file():
        shape_files = list(reference_dir.glob('*.shp'))

        if len(shape_files) < 1:
            log.error('Couldn\'t find `shp` file.')
            raise Exception('File not found!')

        log.info(f'Loading {shape_files[0]}')
        ref_data = gpd.read_file(shape_files[0])

        ref_cfg = cfg['reference_data']

        # Map classes.
        log.info('Mapping classes...')
        class_column = ref_cfg['class_column']
        classes = ref_cfg['classes']

        class_map = {}
        for id_new, group in classes.items():
            for id_old in group:
                class_map[id_old] = int(id_new)

        ref_data[class_column] = ref_data[class_column].map(class_map)

        # Rename class column.
        columns = np.array(list(ref_data.columns))
        columns[columns == class_column] = 'CLASS'
        ref_data.columns = columns

        # Convert CRS.
        log.info('Converting CRS...')
        aoi_cfg = cfg['AOI']
        country_crs = CRS[aoi_cfg['crs']]
        ref_data = ref_data.to_crs(crs=country_crs.pyproj_crs())

        # Save data.
        log.info(f'Saving {data_file}')
        ref_data.to_file(data_file)

    log.info('Reference data ready!')


@click.command()
@click.argument('config_name', type=click.STRING)
@click.option(
    '--timestamp',
    type=click.FLOAT,
    default=time.time(),
    help='Unix timestamp'
)
def main(config_name, timestamp):
    """Download data for given configuration.

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

        # Download EO data.
        download_eo_data(cfg, log_dir)

        # Download reference data.
        download_reference_data(cfg)

        # Prepare reference data.
        prepare_reference_data(cfg)
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
