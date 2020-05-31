"""Visualize results.
"""
import datetime
# import json
# import os
import sys
import time
from dotenv import find_dotenv, load_dotenv
from pathlib import Path

import click
import numpy as np
import geopandas as gpd
# import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from tqdm.auto import tqdm
from eolearn.core import EOPatch
from sentinelhub import DataSource

from src.utils import config, const, logging, misc


# Global variables
figures_dir = const.REPORTS_DIR
log = None
report = {}


def save_fig(plot, name):
    """Save figure.

    :param plot: Matplotlib's pyplot.
    :type plot: matplotlib.pyplot
    :param name: File name.
    :type name: str
    """
    path = figures_dir / f'{name}.png'
    plot.savefig(path)
    log.info(f'Figure saved: {path}')

    path = figures_dir / f'{name}.svg'
    plot.savefig(path)
    log.info(f'Figure saved: {path}')


def plot_aoi(cfg, patches_dir=None):
    """Visualize AOI.

    :param cfg: Configuration.
    :type cfg: dict
    """
    log.info('Plotting AOI...')

    aoi_cfg = cfg['AOI']
    aoi_dir = misc.get_raw_data_dir(cfg) / 'AOI'

    country = gpd.read_file(aoi_dir / 'country.shp')
    aoi = gpd.read_file(aoi_dir / 'aoi.shp')
    selected = aoi[aoi.selected == 1]

    aoi_grid_x = aoi_cfg['grid'][0]
    aoi_grid_y = aoi_cfg['grid'][1]

    # TODO: Better aproximation with actual bounding box.
    ratio = aoi_grid_y / aoi_grid_x

    fig, ax = plt.subplots(figsize=(25, 25 * ratio))
    country.plot(ax=ax, facecolor='#eef1f5', edgecolor='#666', alpha=1.0)
    # aoi.plot(ax=ax, facecolor='#eef1f5', edgecolor='#666', alpha=0.4)
    aoi.plot(ax=ax, facecolor='#ffffff', edgecolor='#666', alpha=0.4)
    # selected.plot(ax=ax, facecolor='None', edgecolor='#f86a6a', alpha=1.0)
    # selected.plot(ax=ax, facecolor='#f86a6a', edgecolor='#f86a6a', alpha=0.2)
    selected.plot(ax=ax, facecolor='#f86a6a', edgecolor='#f86a6a', alpha=0.2)

    for idx, row in aoi.iterrows():
        bbox = row.geometry
        ax.text(
            bbox.centroid.x,
            bbox.centroid.y,
            idx,
            ha='center',
            va='center'
        )

    country_name = country.iloc[0]['name']
    ax.set_title(
        f'{country_name} tiled in a {aoi_grid_x} x {aoi_grid_y} grid '
        f'with {len(aoi.geometry)} tiles ({len(selected.geometry)} selected)',
        fontdict={'fontsize': 20}
    )

    plt.axis('off')
    plt.tight_layout()
    save_fig(fig, 'aoi')
    plt.close()


def plot_map(
    patch_dirs,
    func,
    func_args={},
    imshow_args={},
    title=None,
    colorbar=None
):
    """Plot map from array of patches, defined by partch directories.

    :param patch_dirs: Array of patch directories
    :type patch_dirs: np.array
    :param func: FUnction which calculates map to be plotted from eopatch.
    :type func: function
    :param func_args: Function arguments, defaults to {}
    :type func_args: dict, optional
    :param imshow_args: imshow arguments, defaults to {}
    :type imshow_args: dict, optional
    :param title: Plot title, defaults to None
    :type title: str or None, defaults to None
    :param colorbar: Colorbar settings, optional
    :type colorbar: dict or None, defaults to None
    """
    patch_ids = [int(Path(d).name.split('_')[1]) for d in patch_dirs.ravel()]
    log.info(f'Plotting {func.__name__} of {patch_ids} ...')

    region_shape = patch_dirs.shape
    aspect_ratio = 0

    fig, axs = plt.subplots(nrows=region_shape[0], ncols=region_shape[1])

    # patch_ids = []
    pbar = tqdm(total=len(patch_dirs))
    for i, patch_dir in enumerate(patch_dirs.ravel()):
        # patch_ids.append(int(patch_dir.name.split('_')[1]))

        # Load each patch separately.
        eopatch = EOPatch.load(patch_dir, lazy_loading=True)

        # Plot image.
        ax = axs[i // region_shape[1]][i % region_shape[0]]
        im = ax.imshow(func(eopatch, **func_args), **imshow_args)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('auto')

        # Set aspect ratio based on patch and region shapes.
        if not aspect_ratio:
            data_key = list(eopatch.data.keys())[0]
            patch_shape = eopatch.data[data_key].shape
            width = patch_shape[2] * region_shape[1]
            height = patch_shape[1] * region_shape[0]
            aspect_ratio = width / height

        del eopatch
        pbar.update(1)

    # fig.set_size_inches(20, 20 * aspect_ratio)
    fig.set_size_inches(20, 20)
    plt.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)

    # if title:
    #     fig.suptitle(title, va='top', fontsize=20)

    # Legend
    if colorbar:
        cb = fig.colorbar(
            im,
            ax=axs.ravel().tolist(),
            orientation='horizontal',
            pad=0.01,
            aspect=100
        )
        cb.ax.tick_params(labelsize=20)

        if isinstance(colorbar, dict):
            if 'ticks' in colorbar:
                cb.set_ticks(colorbar['ticks'])
            if 'labels' in colorbar:
                cb.ax.set_xticklabels(
                    colorbar['labels'],
                    ha='right',
                    rotation_mode='anchor',
                    rotation=45,
                    fontsize=15
                )

    save_fig(fig, func.__name__)
    plt.close()


def map_rgb(eopatch, feature, bands, date=None):
    """Creates RGB map, ready for plotting.

    :param eopatch: EOPatch
    :type eopatch: EOPatch
    :param feature: Name of eopatch data feature with given bands
    :type features: str
    :param bands: List of bands, e.g. [`B01`, `B02`, ...]
    :type bands: list[str]
    :param date: Date represented as `%Y-%m-%d` string, defaults to None
    :type date: str, optional
    :return: (Layered) map.
    :rtype: np.array
    """
    r = bands.index('B04')
    g = bands.index('B03')
    b = bands.index('B02')

    time_frame_idx = 0

    if date:
        # Get time frame index closest to the given date.
        date = datetime.datetime.strptime(date, '%Y-%m-%d')
        dates = np.array(eopatch.timestamp)
        time_frame_idx = np.argsort(abs(date - dates))[0]

    # TODO: Choose the right indices for merged features!?
    image = eopatch.data[feature][time_frame_idx][..., [r, g, b]]

    return np.clip(image * 3.5, 0, 1)


def map_reference(patch):
    return patch.mask_timeless['REF'].squeeze()


def map_reference_morphed(patch):
    return patch.mask_timeless['REF_MORPHED'].squeeze()


def plot_maps(cfg):
    """Visualize list of maps as defined in configuration.

    :param cfg: Configuration
    :type cfg: dict
    """
    if 'visualization' not in cfg:
        return

    vis_cfg = cfg['visualization']
    ref_cfg = cfg['reference_data']

    if 'maps' not in vis_cfg:
        return

    bands = config.get_band_names(cfg, DataSource.SENTINEL2_L1C)
    feature = config.get_feature_name(cfg, DataSource.SENTINEL2_L1C)
    ref_classes = [int(k) for k in ref_cfg['classes'].keys()]
    ref_labels = [props['label'] for props in vis_cfg['classes'].values()]
    ref_bounds = misc.get_bounds_from_values(ref_classes)
    ref_cmap = ListedColormap(
        [props['color'] for props in vis_cfg['classes'].values()],
        name='ref_map'
    )
    ref_norm = BoundaryNorm(ref_bounds, ref_cmap.N)

    # Plot all maps defined in config.
    for map_cfg in vis_cfg['maps']:
        # paths_raw = misc.get_region_paths(
        #     cfg,
        #     map_cfg['region'],
        #     misc.get_raw_data_dir(cfg)
        # )
        paths_processed = misc.get_region_paths(
            cfg,
            map_cfg['region'],
            misc.get_processed_data_dir(cfg)
        )

        # True color map
        if map_cfg['type'] == 'rgb':
            rgb_args = {
                'feature': feature,
                'bands': bands,
                'date': map_cfg['date']
            }
            plot_map(
                paths_processed,
                map_rgb,
                func_args=rgb_args
            )

        # Reference map
        elif map_cfg['type'] == 'reference':
            imshow_args = {
                'cmap': ref_cmap,
                'norm': ref_norm
            }
            colorbar = {
                'ticks': ref_classes,
                'labels': ref_labels
            }
            plot_map(
                paths_processed,
                map_reference,
                imshow_args=imshow_args,
                colorbar=colorbar
            )

        # Morphed Reference map
        elif map_cfg['type'] == 'reference_morphed':
            imshow_args = {
                'cmap': ref_cmap,
                'norm': ref_norm
            }
            colorbar = {
                'ticks': ref_classes,
                'labels': ref_labels
            }
            plot_map(
                paths_processed,
                map_reference_morphed,
                imshow_args=imshow_args,
                colorbar=colorbar
            )


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
    global figures_dir
    global log
    global report

    total_time = time.time()

    # Initialize logging.
    log = logging.get_logger(__file__, config_name, timestamp)
    log_dir = Path(log.handlers[1].baseFilename).parent

    # Initialize environment.
    load_dotenv(find_dotenv())

    # Initialize figures directory.
    figures_dir = misc.get_report_subdir(config_name, timestamp, 'figures')

    if not figures_dir.is_dir():
        figures_dir.mkdir(parents=True)

    exit_code = 0
    try:
        # Load configuration.
        cfg = config.load(config_name, log=log)
        report['config'] = cfg
        misc.print_header(cfg, log)

        # Visualize ROI.
        plot_aoi(cfg)

        # Visualize maps.
        plot_maps(cfg)

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
