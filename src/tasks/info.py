"""Print configuration info.
"""
import click

from src.utils import config, misc


@click.command()
@click.argument('config_name', type=click.STRING)
def main(config_name):
    """Print configuration info.

    :param config_name: Configuration name, i.e. file name of the
        configuration without the `json` extension.
    :type config_name: str
    """
    cfg = config.load(config_name)

    print(f'Name: {cfg["name"]}')
    print(f'Description: {cfg["description"]}')
    print(f'Time interval: {cfg["time_interval"]}')
    print(f'Country: {cfg["AOI"]["country"]}')
    print('Directories:')
    print(misc.get_raw_data_dir(cfg))
    print(misc.get_processed_data_dir(cfg))
    print(misc.get_sampled_data_dir(cfg))
    print(misc.get_final_data_dir(cfg))
    print(misc.get_models_dir(cfg))


if __name__ == '__main__':
    main()
