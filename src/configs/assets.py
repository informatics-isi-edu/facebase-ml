"""
Asset Configuration Module
==========================

Defines configurations for execution assets (model weights, checkpoints, etc.).

In DerivaML, assets are additional files needed beyond the dataset. They are
specified as Resource IDs (RIDs) and automatically downloaded when the
execution is initialized.

Typical assets include:
- Pre-trained model weights
- Model checkpoints
- Configuration files
- Reference data files

Configuration Format
--------------------
Assets are specified as a simple list of RID strings::

    my_assets = ["3RA", "3R8"]
    asset_store(my_assets, name="my_asset_config")

The RIDs reference files stored in the Deriva catalog's asset table.
"""

from hydra_zen import store

# ---------------------------------------------------------------------------
# Asset Configurations
# ---------------------------------------------------------------------------
# Define asset sets as lists of RID strings. Each RID references a file
# in the Deriva catalog that will be downloaded for the execution.

assets = []

# ---------------------------------------------------------------------------
# Register with Hydra-Zen Store
# ---------------------------------------------------------------------------
# The group name "assets" must match the parameter name in run_model()

asset_store = store(group="assets")
asset_store(assets, name="default_asset")
