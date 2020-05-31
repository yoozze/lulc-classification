"""Miscellaneous utilities
"""
import hashlib
import json
import math
import os
import re
import time
from datetime import datetime
# from pathlib import Path

import geopandas as gpd
import numpy as np
from eolearn.core import EOPatch

from src.utils import const


def get_hash(d):
    """Calculates hash from given dictionary `d`.

    :param d: Dictionary from which hash will be calculated.
    :type d: dict
    :return: Hexadecimal hash string.
    :rtype: str
    """
    json_str = json.dumps(d, sort_keys=True).encode('utf-8')

    return hashlib.md5(json_str).hexdigest()


def get_raw_data_dir(cfg):

    """Get path to hash directory inside of raw data directory. Hash is
    based on given configuration.

    :param cfg: Configuration
    :type cfg: dict
    :return: Raw data directory.
    :rtype: Path
    """
    hash_dir = get_hash([
        cfg.get('AOI', None),
        cfg.get('time_interval', None),
        cfg.get('sh_inputs', None)
    ])

    return const.DATA_RAW_DIR / hash_dir


def get_processed_data_dir(cfg):
    """Get path to hash directory inside of processed data directory. Hash is
    based on given configuration.

    :param cfg: Configuration
    :type cfg: dict
    :return: Processed data directory.
    :rtype: Path
    """
    hash_dir = get_hash([
        cfg.get('AOI', None),
        cfg.get('time_interval', None),
        cfg.get('sh_inputs', None),
        cfg.get('reference_data', None),
        cfg.get('cloud_detection', None),
        cfg.get('valid_data', None),
        cfg.get('filter', None),
        cfg.get('interpolation', None),
        cfg.get('features', None),
        cfg.get('gradient', None),
        cfg.get('edges', None),
        cfg.get('raster', None),
        cfg.get('preprocess_save', None),
    ])

    return const.DATA_PROCESSED_DIR / hash_dir


def get_sampled_data_dir(cfg):
    """Get path to hash directory inside of sampled data directory. Hash is
    based on given configuration.

    :param cfg: Configuration
    :type cfg: dict
    :return: Sampled data directory.
    :rtype: Path
    """
    hash_dir = get_hash([
        cfg.get('AOI', None),
        cfg.get('time_interval', None),
        cfg.get('sh_inputs', None),
        cfg.get('reference_data', None),
        cfg.get('cloud_detection', None),
        cfg.get('valid_data', None),
        cfg.get('filter', None),
        cfg.get('interpolation', None),
        cfg.get('features', None),
        cfg.get('gradient', None),
        cfg.get('edges', None),
        cfg.get('raster', None),
        cfg.get('preprocess_save', None),
        cfg.get('sampling', None),
    ])

    return const.DATA_SAMPLED_DIR / hash_dir


def get_final_data_dir(cfg):
    """Get path to hash directory inside of final data directory. Hash is
    based on given configuration.

    :param cfg: Configuration
    :type cfg: dict
    :return: Sampled data directory.
    :rtype: Path
    """
    hash_dir = get_hash([
        cfg.get('AOI', None),
        cfg.get('time_interval', None),
        cfg.get('sh_inputs', None),
        cfg.get('reference_data', None),
        cfg.get('cloud_detection', None),
        cfg.get('valid_data', None),
        cfg.get('filter', None),
        cfg.get('interpolation', None),
        cfg.get('features', None),
        cfg.get('gradient', None),
        cfg.get('edges', None),
        cfg.get('raster', None),
        cfg.get('preprocess_save', None),
        cfg.get('sampling', None),
        cfg.get('timeless_features', None),
    ])

    return const.DATA_FINAL_DIR / hash_dir


def get_models_dir(cfg):
    """Get path to hash directory inside of models directory. Hash is based on
    given configuration.

    :param cfg: Configuration
    :type cfg: dict
    :return: Models directory.
    :rtype: Path
    """
    hash_dir = get_hash([
        cfg.get('AOI', None),
        cfg.get('time_interval', None),
        cfg.get('sh_inputs', None),
        cfg.get('reference_data', None),
        cfg.get('cloud_detection', None),
        cfg.get('valid_data', None),
        cfg.get('filter', None),
        cfg.get('interpolation', None),
        cfg.get('features', None),
        cfg.get('gradient', None),
        cfg.get('edges', None),
        cfg.get('raster', None),
        cfg.get('preprocess_save', None),
        cfg.get('sampling', None),
        cfg.get('timeless_features', None),
        cfg.get('modelling', None),
    ])

    return const.MODELS_DIR / hash_dir


def get_date_time_dir_name(timestamp=time.time()):
    """Format (given) timestamp with file system safe characters.
    If timestamp is not provided, current time is used.

    :param timestamp: Unix timestamp, defaults to time.time()
    :type timestamp: float, optional
    :return: File system safe date-time string
    :rtype: str
    """
    date_time = datetime.fromtimestamp(timestamp)

    return date_time.strftime('%Y-%m-%d_%H-%M-%S')


def get_report_subdir(config_name, timestamp, subdir):
    """Get subdirectory in reports directory with configuration name and
    timestamp parents.

    :param config_name: Configuration name
    :type config_name: string
    :param timestamp: Unix timestamp
    :type timestamp: float
    :param subdir: Subdirectory name
    :type subdir: string
    :return: Figures directory path
    :rtype: Path
    """
    return const.REPORTS_DIR.joinpath(
        config_name,
        get_date_time_dir_name(timestamp),
        subdir
    )


def save_results(results, name, config_name, timestamp):
    """Save results to file defined by name, directory name and timestamp.

    :param results: Results dictionary.
    :type results: dict
    :param name: Results file name.
        If file path is used, e.g. `__file__`, only base name without extension
        will be used.
    :type name: str
    :param config_name: Configuration name
    :type config_name: str
    :param timestamp: Unix timestamp
    :type timestamp: float
    """
    results_name = os.path.splitext(os.path.basename(name))[0]
    results_dir = get_report_subdir(config_name, timestamp, 'results')

    if not results_dir.is_dir():
        results_dir.mkdir(parents=True)

    json_path = results_dir / f'{results_name}.json'
    with open(json_path, 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file)


def format_data_size(size_bytes):
    """Format given data size.

    :param size_bytes: Size in bytes.
    :type size_bytes: int
    :return: Formated size
    :rtype: str
    """
    if size_bytes == 0:
        return "0B"

    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)

    return "%s %s" % (s, size_name[i])


def get_dir_size(start_path='.'):
    """Calculaet size of given directory.

    :param start_path: Directory path, defaults to '.'
    :type start_path: str, optional
    :return: Size in bytes of given directory, including all subdirectories.
    :rtype: int
    """
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)

    return total_size


def eval_obj(obj, globals, locals):
    """Evaluate given object.
    Only keys and values enclosed in `${...}` are evaluated.

    :param obj: Object to be evaluated.
    :type obj: object
    :param globals: Dictionary of global variables to be used for evaluation.
    :type globals: dict[str, object]
    :param locals: Dictionary of local variables to be used for evaluation.
    :type locals: dict[str, object]
    :return: Evaluated object
    :rtype: object
    """
    if isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            new_key = eval_obj(k, globals, locals)
            new_obj[new_key] = eval_obj(v, globals, locals)
        obj = new_obj
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            obj[i] = eval_obj(v, globals, locals)
    else:
        if isinstance(obj, str):
            match = re.match(r'^(\$\{)([^}]+)\}$', obj)
            if match:
                obj = eval(match.group(2), globals, locals)

    return obj


def get_region_paths(cfg, region, data_dir=None):
    """Get matrix of raw eopatch paths, defined by selected region.

    :param cfg: Configuration
    :type cfg: dict
    :param region: Region defined by range of tile indices.
    :type region: list[list[int, int], list[int, int]]
    :param data_dir: Data directory with eopatches. Defaults to `None`.
    :type data_dir: Path, optional
    :return: Matrix of paths to eopatches.
    :rtype: np.array(Path)
    """
    raw_data_dir = get_raw_data_dir(cfg)
    aoi_dir = raw_data_dir / 'AOI'
    aoi = gpd.read_file(aoi_dir / 'aoi.shp')

    # Query selected tiles within given region.
    indices = aoi.index[
        (aoi['selected'] == 1)
        & aoi['index_x'].between(region[0][0], region[1][0])
        & aoi['index_y'].between(region[0][1], region[1][1])
    ]

    if not data_dir:
        data_dir = raw_data_dir

    patches_dir = data_dir / 'patches'

    # Create matrix.
    rows = region[1][1] - region[0][1] + 1
    cols = region[1][0] - region[0][0] + 1
    paths = np.array(
        [str(patches_dir / f'eopatch_{idx}') for idx in indices]
    ).reshape(rows, cols)

    # Reorder tiles.
    if (rows > 1) and (cols > 1):
        paths = np.transpose(np.fliplr(paths))
    elif cols > 1:
        paths = np.transpose(paths)
    else:
        paths = np.transpose(np.flipud(paths))

    return paths


def get_aoi_bbox(cfg, aoi=None):
    """Get bounding box of AOI.

    :param cfg: Configuration
    :type cfg: dict
    :param aoi: GeoDataFrame of AOI, defaults to None
    :type aoi: GeoDataFrame, optional
    :return: Bounding box of AOI.
    :rtype: tuple[float, float, float, float]
    """
    if not aoi:
        aoi = gpd.read_file(str(get_raw_data_dir(cfg) / 'AOI'))

    # Get bounds of all selected bounding boxes.
    bounds = aoi[aoi.selected == 1].bounds

    return (
        bounds['minx'].min(), bounds['miny'].min(),
        bounds['maxx'].max(), bounds['maxy'].max()
    )


def get_bounds_from_values(values):
    """Get list of bounds from given list of numeric values.

    :param values: List of numeric values
    :type values: list[int or float]
    :return: List of bounds
    :rtype: list[float]
    """
    bounds = []
    for i in range(len(values)):
        if i < len(values) - 1:
            if i == 0:
                diff = (values[i + 1] - values[i]) / 2
                bounds.append(values[i] - diff)
            diff = (values[i + 1] - values[i]) / 2
            bounds.append(values[i] + diff)
        else:
            diff = (values[i] - values[i - 1]) / 2
            bounds.append(values[i] + diff)
    return bounds


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """
    return np.isnan(y), lambda z: z.nonzero()[0]


def interpolate_array(y):
    # If nothing to interpolate, return original.
    if not np.isnan(y).any():
        return y

    nan_count = 0
    y_len = len(y)

    # Check if array is reparable.
    for i, yi in enumerate(y):
        if math.isnan():
            nan_count += 1
        else:
            nan_count = 0

        # If more than two nans in a seqquence, don't interpolate.
        if nan_count > 2:
            return None

        # If array starts or ends with more than one nans, don't interpolate.
        if (i == 1 and nan_count == 2) or (i == y_len - 1 and nan_count == 2):
            return None

    # Interpolate
    nans, x = nan_helper(y)
    y[nans] = np.interp(x(nans), x(~nans), y[~nans])

    return y


def load_sample_subset(input_dir, subset_id):
    eopatches = []

    for path in input_dir.iterdir():
        eopatch = EOPatch.load(str(path), lazy_loading=True)

        if eopatch.meta_info['subset'] == subset_id:
            eopatches.append(eopatch)
        else:
            del eopatch

    eopatches = np.array(eopatches)

    X = np.array([
        eopatch.data['FEATURES_SAMPLED']
        for eopatch in eopatches
    ])
    y = np.array([
        eopatch.mask_timeless['REF_PROCESSED_SAMPLED']
        for eopatch in eopatches
    ])

    # Reshape to 2-dimensional arrays
    p, t, w, h, f = X.shape
    X = np.moveaxis(X, 1, 3).reshape(
        p * w * h,
        t * f
    )
    y = np.moveaxis(y, 1, 2).reshape(
        p * w * h,
        1
    ).squeeze()

    # Remove points with no reference data (class = 0)
    mask = y == 0
    X = X[~mask]
    y = y[~mask]

    return X, y


def print_header(cfg, log):
    # print(get_raw_data_dir(cfg))
    # print(get_processed_data_dir(cfg))
    # print(get_sampled_data_dir(cfg))
    # print(get_models_dir(cfg))
    pass
