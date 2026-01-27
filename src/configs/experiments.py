""" Define experiments

These will be stored in the experiment store and can be run with the --multirun +experiment=experiment_name.
    python deriva_run.py --multirun +experiment=run1, run2
"""

from hydra_zen import make_config, store

# Experiment extends the base configuration, so we need to get it from the store.
app_config = store[None]
app_name = next(iter(app_config))
deriva_model_config = store[None][app_name]
experiment_store = store(group="experiments")

# Define your experiments here.  We can pick a specific dataset, asset, and model configuration to run.

experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /datasets": "test2"},
            {"override /assets": "weights_1"},
            {"override /model_config": "epochs_20"},
        ],
        bases=(deriva_model_config,)
   ),
    name="run2",
)

# CIFAR-10 CNN experiments
# These experiments use the CIFAR-10 CNN model with different configurations.
# Make sure you have CIFAR-10 datasets configured in configs/datasets.py

experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model_config": "cifar10_quick"},
        ],
        bases=(deriva_model_config,)
    ),
    name="cifar10_quick",
)

experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model_config": "cifar10_default"},
        ],
        bases=(deriva_model_config,)
    ),
    name="cifar10_default",
)

experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model_config": "cifar10_extended"},
        ],
        bases=(deriva_model_config,)
    ),
    name="cifar10_extended",
)
