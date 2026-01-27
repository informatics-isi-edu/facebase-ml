"""
Configuration Package for Deriva ML Model Template
===================================================

This package contains hydra-zen configuration modules that define the parameters
for model execution. Each module registers its configurations with the hydra-zen
store when imported.

Configuration Groups
--------------------
- deriva: Connection settings for the Deriva catalog
- datasets: Dataset specifications (which data to use)
- assets: Asset RIDs (model weights, checkpoints, etc.)
- workflow: Workflow definitions
- model: Base model configurations
- experiments: Preset experiment configurations

Adding New Configurations
-------------------------
1. Create a new Python file in this directory (e.g., `my_config.py`)
2. Use `store(group="group_name")` to register configs with hydra-zen
3. The module will be automatically loaded by `load_all_configs()`

Example config module::

    from hydra_zen import store

    my_store = store(group="model_config")
    my_store(MyModelConfig(...), name="my_model")
"""

import importlib
import pkgutil
from pathlib import Path


def load_all_configs() -> list[str]:
    """
    Dynamically import all configuration modules in this package.

    This function iterates over all Python files in the configs directory
    (excluding __init__.py) and imports them. Each module is expected to
    register its configurations with the hydra-zen store as a side effect
    of being imported.

    Returns
    -------
    list[str]
        Names of the modules that were successfully loaded.

    Notes
    -----
    Modules that fail to import will raise an exception. This is intentional
    to surface configuration errors early rather than silently ignoring them.

    Examples
    --------
    >>> from configs import load_all_configs
    >>> loaded = load_all_configs()
    >>> print(loaded)
    ['assets', 'cifar10_cnn', 'datasets', 'deriva', 'experiments', 'model', 'workflow']
    """
    loaded_modules = []
    package_dir = Path(__file__).parent

    # Iterate over all modules in this package
    for module_info in pkgutil.iter_modules([str(package_dir)]):
        module_name = module_info.name

        # Import the module, which triggers its store() registrations
        importlib.import_module(f"configs.{module_name}")
        loaded_modules.append(module_name)

    return sorted(loaded_modules)
