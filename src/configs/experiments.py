"""Define experiments.

Experiments compose configuration groups together for specific runs.
Use with: uv run deriva-ml-run +experiment=<name>

Pattern:
- Group is "experiment" (singular), matching the +experiment= CLI syntax
- package="_global_" on the store constructor so fields merge at root level
- Use bases=(PrimaryConfig,) for schema validation — PrimaryConfig must NOT
  have its own hydra_defaults or the base's defaults will shadow yours
- Set inherited fields to MISSING so Hydra fills them from the defaults list
  rather than using the base's default values
"""

from hydra_zen import make_config, store, MISSING

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
        script_config=MISSING,
        bases=(PrimaryConfig,),
    ),
    name="e155_dev_sample",
)

# E15.5 training sample — 200 random files for training experiments
experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /deriva_ml": "dev_facebase"},
            {"override /datasets": "none"},
            {"override /script_config": "e155_training_sample"},
            {"override /workflow": "dataset_generation"},
        ],
        script_config=MISSING,
        bases=(PrimaryConfig,),
    ),
    name="e155_training_sample",
)

# Annotate Phenotype — classify genotypes on E15.5 dataset
experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /deriva_ml": "dev_facebase"},
            {"override /datasets": "none"},
            {"override /script_config": "annotate_phenotype"},
            {"override /workflow": "dataset_generation"},
        ],
        script_config=MISSING,
        bases=(PrimaryConfig,),
    ),
    name="annotate_phenotype_e155",
)
