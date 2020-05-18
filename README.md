# Land Use/Land Cover classification

LULC/LPIS classification of Sentinel 2 imagery using incremental algorithms.

## Prerequisites

- python 3.7 or higher
- conda 4.7 or higher (for environment management)

## Installation

Create new conda environment:
```
conda env create -f environment.yml
conda activate lulc
```
NOTE: Environment name defaults to `lulc`. You can change it by updating `name` property in `environment.yml` accordingly.

Install this project as a pip package by [invoking](https://github.com/pyinvoke/invoke) `install` task:
```
invoke install
```

## Getting Started

### Sentinel Hub account

In order to use Sentinel Hub services you will need a Sentinel Hub account. If you do not have one yet, you can create a free trial account at [Sentinel Hub webpage](https://services.sentinel-hub.com/oauth/subscription). If you are a researcher you can even apply for a free non-commercial account at [ESA OSEO page](https://earth.esa.int/aos/OSEO).

Once you have the account set up, login to [Sentinel Hub Configurator](https://apps.sentinel-hub.com/configurator/). Inside there will already exist one configuration with an **instance ID** (alpha-numeric code of length 36). You can use this one or create a new one (`"Add new configuration"`). 

### Define Sentinel Hub layers

TODO

### Set environment variables

Rename `sample.env` to `.env` and replace `INSTANCE_ID` with your own SentinelHub's instance id.

### Run experiments

To run the whole experiment simply invoke `run` command:
```
inv run lulc-svn
```
where `lulc-svn` is the name of the config in `configs` directory.

You can also run different stages of the experiment separately:
```
inv download lulc-svn
inv preprocess lulc-svn
inv sample lulc-svn
inv train lulc-svn
inv evaluate lulc-svn
inv visualize lulc-svn
inv report lulc-svn
```
Each stage of the same experiment accepts the same configuration.

All available commands can be listed with:
```
inv -l

download     Download data for given configuration.
evaluate     Evaluate model for given configuration.
install      Install this project as a pip package.
lint         Lint python source code using flake8.
preprocess   Preprocess data for given configuration.
report       Generate reports for given configuration.
run          Run the entire workflow:
sample       Sample data for given configuration.
train        Train model for given configuration.
visualize    Visualize data and results for given configuration.
```

## Project Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


## Links

- Sentinel Hub: https://github.com/sentinel-hub/sentinelhub-py
- EO Learn: https://github.com/sentinel-hub/eo-learn
- Cookiecutter Data Science: https://github.com/drivendata/cookiecutter-data-science
- Flake8: https://github.com/PyCQA/flake8
- Coverage: https://github.com/nedbat/coveragepy
- Click: https://github.com/pallets/click/
- Invoke: https://github.com/pyinvoke/invoke
