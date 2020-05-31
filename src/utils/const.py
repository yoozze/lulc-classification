"""Commonly used constatnts.
"""
from pathlib import Path


# Directories
PROJECT_DIR = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_DIR / 'configs'
DATA_DIR = PROJECT_DIR / 'data'
DATA_RAW_DIR = DATA_DIR / 'raw'
DATA_PROCESSED_DIR = DATA_DIR / 'processed'
DATA_SAMPLED_DIR = DATA_DIR / 'sampled'
DATA_FINAL_DIR = DATA_DIR / 'final'
DATA_EXTERNAL_DIR = DATA_DIR / 'external'
MODELS_DIR = PROJECT_DIR / 'models'
REPORTS_DIR = PROJECT_DIR / 'reports'

# URLs
COUNTRIES_BASE_URL = 'https://github.com/nvkelso/natural-earth-vector/raw/master/10m_cultural/ne_10m_admin_0_countries' # noqa
