"""Configuration management.
"""
import datetime
import json
import re

from eolearn.core import FeatureType
from sentinelhub import CustomUrlParam, DataSource, ServiceType

from src.utils import const


def exists(name):
    """Check if configuration with given name exists.

    :param name: Configuration name, i.e. name of the configuration file
        without `json` extension.
    :type name: str
    :return: `True` if configuration exists, `False` otherwise.
    :rtype: bool
    """
    json_path = const.CONFIGS_DIR / f'{name}.json'

    return json_path.is_file()


def load(name, log=None):
    """Load configuration with given name. JSON files with C-style comments and
    trailing commas are supported.

    :param name: Configuration name, i.e. name of the configuration file
        without `json` extension.
    :type name: str
    :param log: Python logger. Defaults to None.
    :type log: Logger, optional
    :return: Configuration dictionary if connfiguration with given name exists,
        `None` otherwise.
    :rtype: Any
    """
    json_path = const.CONFIGS_DIR / f'{name}.json'

    try:
        with open(json_path, 'r', encoding='utf-8') as json_file:
            # Clean JSON of C-style comments and trailing commas before
            # parsing.
            json_content = json_file.read()
            json_content = remove_comments(json_content)
            json_content = remove_trailing_commas(json_content)

            return json.loads(json_content)
    except Exception as e:
        if log:
            log.error(f'Failed to load configuration: {name}.json')
        raise e


def remove_comments(json_like):
    """Removes C-style comments from `json_like` and returns the result.

    :param json_like: JSON like string with (possible) C-style comments.
    :type json_like: str
    :return: JSON like string without comments.
    :rtype: str

    Example::
        >>> test_json = '''\
        {
            "foo": "bar", // This is a single-line comment
            "baz": "blah" /* Multi-line
            Comment */
        }'''
        >>> remove_comments('{"foo":"bar","baz":"blah",}')
        '{\n    "foo":"bar",\n    "baz":"blah"\n}'
    """
    def replacer(match):
        s = match.group(0)
        if s[0] == '/':
            return ""
        return s

    comments_re = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )

    return comments_re.sub(replacer, json_like)


def remove_trailing_commas(json_like):
    """Removes trailing commas from `json_like` and returns the result.

    :param json_like: JSON like string with (possible) trailing commas.
    :type json_like: str
    :return: JSON like string without trailing commas.
    :rtype: str

    Example::
        >>> remove_trailing_commas('{"foo":"bar","baz":["blah",],}')
        '{"foo":"bar","baz":["blah"]}'
    """
    trailing_object_commas_re = re.compile(
        r'(,)\s*}(?=([^"\\]*(\\.|"([^"\\]*\\.)*[^"\\]*"))*[^"]*$)'
    )
    trailing_array_commas_re = re.compile(
        r'(,)\s*\](?=([^"\\]*(\\.|"([^"\\]*\\.)*[^"\\]*"))*[^"]*$)'
    )

    # Fix objects {} first
    objects_fixed = trailing_object_commas_re.sub("}", json_like)

    # Now fix arrays/lists [] and return the result
    return trailing_array_commas_re.sub("]", objects_fixed)


def eval_sh_input(sh_input):
    """Evaluate `sh_inputs` config.

    :param sh_input: Sentinel Hub input configurations.
    :type sh_input: dict
    :return: Evaluated Sentinel Hub input configurations.
    :rtype: dict
    """
    if 'custom_url_params' in sh_input:
        evaluated_url_params = {}
        for k, v in sh_input['custom_url_params'].items():
            evaluated_url_params[CustomUrlParam[k]] = v
        sh_input['custom_url_params'] = evaluated_url_params

    if 'service_type' in sh_input:
        sh_input['service_type'] = ServiceType[sh_input['service_type']]

    if 'data_source' in sh_input:
        sh_input['data_source'] = DataSource[sh_input['data_source']]

    if 'time_difference' in sh_input:
        sh_input['time_difference'] = datetime.timedelta(
            seconds=sh_input['time_difference']
        )

    if 'feature' in sh_input:
        if isinstance(sh_input['feature'], list):
            sh_input['feature'] = (
                FeatureType[sh_input['feature'][0]],
                sh_input['feature'][1]
            )

    return sh_input


def get_sh_input_config(cfg, data_source):
    """Get Sentinel Hub OGC configuration for given data source.

    :param cfg: COnfiguration
    :type cfg: dict
    :param data_source: Sentinel Hub's data source
    :type data_source: DataSource
    :return: Sentinel Hub OGC configuration
    :rtype: [type]
    """
    for sh_input in cfg['sh_inputs']:
        if sh_input['data_source'] == data_source.name:
            return sh_input

    return None


def get_feature_name(cfg, data_source):
    """Get feature name for given data source.

    :param cfg: Configuration
    :type cfg: dict
    :param data_source: Data source
    :type data_source: DataSource
    :return: Feature name
    :rtype: str
    """
    for sh_input in cfg['sh_inputs']:
        if sh_input['data_source'] == data_source.name:
            if 'feature' in sh_input:
                return sh_input['feature']
            else:
                break

    return 'BANDS'


def get_band_names(cfg, data_source):
    """Get lis of band names from configuration for given data source.

    :param cfg: Configuration
    :type cfg: dict
    :param data_source: Data source
    :type data_source: DataSource
    :return: List of band names
    :rtype: list[str]
    """
    bands = None
    # TODO: Find better way to get band names.
    for sh_input in cfg['sh_inputs']:
        if sh_input['data_source'] == data_source.name:
            try:
                s = sh_input['custom_url_params']['EVALSCRIPT']
                match = re.match(r'.*\[\s*((B\w+\s*,\s*)*B\w+)\s*\].*', s)
                if match:
                    bands = re.split(r'\s*,\s*', match.group(1).strip())
            except KeyError:
                pass

    return bands
