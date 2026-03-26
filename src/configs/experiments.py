"""Define experiments.

Experiments compose configuration groups together for specific runs.
Use with: uv run deriva-ml-run +experiment=<name>

Pattern:
- Group is "experiment" (singular), matching the +experiment= CLI syntax
- package="_global_" on the store constructor so fields merge at root level
- Experiments use the primary config class (from model.py) as their base,
  but we must get a version WITHOUT hydra_defaults to avoid conflicts
"""

from hydra_zen import make_config, store

# Import the primary config class from our model module.
# This is the same class used for "deriva_model", ensuring OmegaConf
# merge validation passes.
from configs.model import deriva_model as PrimaryConfig

experiment_store = store(group="experiment", package="_global_")

# E15.5 dev sample — 100 random files for quick iteration
experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /deriva_ml": "dev_facebase"},
            {"override /datasets": "none"},
            {"override /script_config": "e155_dev_sample"},
            {"override /workflow": "dataset_generation"},
        ],
        bases=(PrimaryConfig,),
    ),
    name="e155_dev_sample",
)
