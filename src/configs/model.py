"""
Model Configuration Module
===========================

Registers the top-level 'deriva_model' config and a placeholder 'default_model'
in the model_config group. The 'deriva_model' config is required by the
deriva-ml-run CLI as its primary Hydra config name.

Projects should add their own model configs to the model_config group by
creating builds() entries for their model functions (in src/models/) and
registering them here or in a dedicated config module.
"""

from hydra_zen import store, builds

from deriva_ml import DerivaML
from deriva_ml.execution.runner import run_model, create_model_config

# ---------------------------------------------------------------------------
# Top-level config: 'deriva_model'
# ---------------------------------------------------------------------------
# Required by `deriva-ml-run` CLI (--config-name defaults to "deriva_model").
# This composes the default config groups together.
deriva_model = create_model_config(
    DerivaML,
    hydra_defaults=[
        "_self_",
        {"deriva_ml": "default_deriva"},
        {"datasets": "default_dataset"},
        {"assets": "default_asset"},
        {"workflow": "default_workflow"},
        {"model_config": "default_model"},
        {"optional script_config": None},
    ],
)
store(deriva_model, name="deriva_model")

# ---------------------------------------------------------------------------
# Placeholder default_model
# ---------------------------------------------------------------------------
# The base_defaults require model_config/default_model to exist.
# This no-op placeholder satisfies the requirement. Projects should replace
# this with a real model config when they have ML model functions to run.


def _noop_model(**kwargs):
    """Placeholder model that does nothing. Replace with a real model."""
    pass


NoopModel = builds(_noop_model, populate_full_signature=True, zen_partial=True)
model_store = store(group="model_config")
model_store(NoopModel, name="default_model")
